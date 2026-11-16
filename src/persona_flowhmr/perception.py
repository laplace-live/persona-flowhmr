"""Streaming perception: batched YOLOX-L + online ByteTrack + compiled SAM 3D Body Lite."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import supervision as sv
import torch

from .config import MODEL_FPS, SAM_TOKEN_DIM, ResolvedHandMode
from .paths import (
    SAM3D_CKPT_PATH,
    SAM3D_MHR_PATH,
    YOLOX_CKPT_PATH,
    bootstrap_runtime,
)


class Sam3DCheckpointMissingError(FileNotFoundError):
    """Raised when gated `facebook/sam-3d-body-dinov3` weights have not been downloaded yet."""


def default_camera_intrinsics(height: int, width: int) -> np.ndarray:
    """FlowHMR's diagonal-focal-length static camera matrix `K` `(3, 3)`."""
    bootstrap_runtime()
    from flowhmr.utils.runtime_video_processing import VideoProcessor

    return VideoProcessor.default_K_from_hw(height, width)


def _empty_tokens() -> torch.Tensor:
    return torch.empty((0, SAM_TOKEN_DIM), dtype=torch.float32)


def _center_box(height: int, width: int) -> np.ndarray:
    """`(1, 4)` centered person-sized box used for warmup / neutral-pose passes."""
    return np.array([[width * 0.25, height * 0.1, width * 0.75, height * 0.9]], dtype=np.float32)


def _best_box_index(dets: sv.Detections) -> int:
    """Index of the largest, most confident box (`area * confidence`)."""
    conf = dets.confidence if dets.confidence is not None else 1.0
    return int(np.argmax(dets.box_area * conf))


def _install_local_dinov3_hub_hook() -> None:
    """Redirect `torch.hub.load('facebookresearch/dinov3', ...)` to vendored `dinov3` offline."""
    bootstrap_runtime()
    orig_load = torch.hub.load
    if getattr(orig_load, "_persona_flowhmr_patched", False):
        return

    def _patched_hub_load(repo_or_dir, model, *args, **kwargs):
        if isinstance(repo_or_dir, str) and "dinov3" in repo_or_dir:
            from dinov3.hub import backbones as dinov3_backbones

            builder = getattr(dinov3_backbones, model)
            filtered_kw = {
                k: v
                for k, v in kwargs.items()
                if k not in ("source", "trust_repo", "force_reload", "skip_validation", "verbose")
            }
            return builder(*args, **filtered_kw)
        return orig_load(repo_or_dir, model, *args, **kwargs)

    _patched_hub_load._persona_flowhmr_patched = True  # type: ignore[attr-defined]
    torch.hub.load = _patched_hub_load


class BatchedYoloxTracker:
    """Batched BF16 YOLOX-L human detector + online single-person ByteTrack stabilizer."""

    def __init__(
        self,
        ckpt_path: str | Path = YOLOX_CKPT_PATH,
        device: str | torch.device = "cuda",
        test_size: tuple[int, int] = (416, 416),
        yolo_stride: int = 2,
        bbox_thr: float = 0.45,
        nms_thr: float = 0.30,
        ema_alpha: float = 0.65,
    ) -> None:
        bootstrap_runtime()
        ckpt = Path(ckpt_path)
        if not ckpt.is_file():
            raise FileNotFoundError(f"YOLOX-L checkpoint missing: {ckpt}")

        from yolox.exp import get_exp
        from yolox.utils import postprocess

        self.device = torch.device(device)
        self.bbox_thr = float(bbox_thr)
        self.nms_thr = float(nms_thr)
        self.ema_alpha = float(ema_alpha)
        self._postprocess = postprocess

        exp = get_exp(exp_name="yolox-l")
        model = exp.get_model()
        state = torch.load(str(ckpt), map_location="cpu", weights_only=True)
        model.load_state_dict(state["model"])
        model.eval()
        use_bf16 = self.device.type == "cuda" and torch.cuda.get_device_capability(self.device)[0] >= 8
        self._dtype = torch.bfloat16 if use_bf16 else torch.float32
        self.model = model.to(device=self.device, dtype=self._dtype)
        self.num_classes = int(exp.num_classes)

        self.configure(test_size=test_size, yolo_stride=yolo_stride)

    def configure(self, *, test_size: tuple[int, int], yolo_stride: int) -> None:
        """Set the detector input size and frame stride, and restart tracking for a new stream."""
        self.test_size = (int(test_size[0]), int(test_size[1]))
        self.yolo_stride = max(1, int(yolo_stride))
        self.reset_tracker()

    def reset_tracker(self) -> None:
        self._tracker = sv.ByteTrack(frame_rate=int(MODEL_FPS))
        self._locked_track_id: int | None = None
        self._smoothed_bbox: np.ndarray | None = None
        self._missed_frames = 0

    def _letterbox_canvas(self, batch_size: int) -> torch.Tensor:
        """YOLOX's gray (114) padded input batch `(B, 3, H, W)` in the model dtype."""
        return torch.full((batch_size, 3, *self.test_size), 114.0, device=self.device, dtype=self._dtype)

    @torch.no_grad()
    def warmup(self, batch_size: int = 4) -> None:
        """Pre-initialize cuDNN workspace and kernels before real-time capture starts."""
        self.model(self._letterbox_canvas(max(1, int(batch_size))))
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    @torch.no_grad()
    def detect_and_track_batch(
        self,
        rgb_frames: Sequence[np.ndarray],
        *,
        reset_flags: Sequence[bool] | None = None,
    ) -> np.ndarray:
        """Run GPU-preprocessed batched YOLOX-L + ByteTrack over `N` RGB frames -> `(N, 4)`."""
        n = len(rgb_frames)
        if n == 0:
            return np.zeros((0, 4), dtype=np.float32)

        h, w = rgb_frames[0].shape[:2]
        ratio = min(self.test_size[0] / float(h), self.test_size[1] / float(w))
        new_h = max(1, int(h * ratio))
        new_w = max(1, int(w * ratio))

        stride = self.yolo_stride
        if stride > 1 and n >= 2:
            det_indices_set = set(range(0, n, stride))
            det_indices_set.add(n - 1)
            if reset_flags is not None:
                for idx, is_rst in enumerate(reset_flags):
                    if is_rst:
                        det_indices_set.add(idx)
                        if idx > 0:
                            det_indices_set.add(idx - 1)
            det_indices = sorted(det_indices_set)
        else:
            det_indices = list(range(n))

        raw_np = np.stack([rgb_frames[i] for i in det_indices], axis=0)
        raw_gpu = torch.from_numpy(raw_np).to(device=self.device, non_blocking=True)
        bgr_nchw = raw_gpu[..., [2, 1, 0]].permute(0, 3, 1, 2).to(dtype=self._dtype)
        resized = torch.nn.functional.interpolate(
            bgr_nchw,
            size=(new_h, new_w),
            mode="bilinear",
            align_corners=False,
        )
        batch_t = self._letterbox_canvas(len(det_indices))
        batch_t[:, :, :new_h, :new_w] = resized

        raw_out = self.model(batch_t)
        dets = self._postprocess(
            raw_out.float(),
            self.num_classes,
            self.bbox_thr,
            self.nms_thr,
            class_agnostic=True,
        )

        full_box = np.array([0.0, 0.0, float(w), float(h)], dtype=np.float32)
        key_boxes: dict[int, np.ndarray] = {}
        for idx, det in zip(det_indices, dets):
            if reset_flags is not None and reset_flags[idx]:
                self.reset_tracker()
            key_boxes[idx] = self._step_single_frame(det, ratio, w, h, full_box)

        out_boxes = np.empty((n, 4), dtype=np.float32)
        seg = 0
        for i in range(n):
            if i in key_boxes:
                out_boxes[i] = key_boxes[i]
                continue
            while seg + 1 < len(det_indices) - 1 and i > det_indices[seg + 1]:
                seg += 1
            i0, i1 = det_indices[seg], det_indices[seg + 1]
            alpha = float(i - i0) / float(max(1, i1 - i0))
            out_boxes[i] = (1.0 - alpha) * key_boxes[i0] + alpha * key_boxes[i1]

        return out_boxes

    def _select_person(self, det: torch.Tensor | None, ratio: float, width: int, height: int) -> np.ndarray | None:
        """Pick the locked ByteTrack person (else the largest confident one) from one frame's detections."""
        if det is None or len(det) == 0:
            return None
        det_np = det.detach().cpu().numpy()
        det_np = det_np[det_np[:, 6].astype(np.int32) == 0]
        if len(det_np) == 0:
            return None

        xyxy = det_np[:, :4] / ratio
        xyxy[:, [0, 2]] = np.clip(xyxy[:, [0, 2]], 0.0, float(width))
        xyxy[:, [1, 3]] = np.clip(xyxy[:, [1, 3]], 0.0, float(height))
        scores = (det_np[:, 4] * det_np[:, 5]).astype(np.float32)
        detections = sv.Detections(
            xyxy=xyxy.astype(np.float32),
            confidence=scores,
            class_id=np.zeros(len(scores), dtype=np.int32),
        )
        tracked = self._tracker.update_with_detections(detections)
        if len(tracked) == 0 or tracked.tracker_id is None:
            return detections.xyxy[_best_box_index(detections)]

        t_ids = tracked.tracker_id.astype(int)
        if self._locked_track_id is None or self._locked_track_id not in t_ids:
            best = _best_box_index(tracked)
            self._locked_track_id = int(t_ids[best])
            return tracked.xyxy[best].astype(np.float32)
        return tracked.xyxy[int(np.where(t_ids == self._locked_track_id)[0][0])].astype(np.float32)

    def _step_single_frame(
        self,
        det: torch.Tensor | None,
        ratio: float,
        width: int,
        height: int,
        full_box: np.ndarray,
    ) -> np.ndarray:
        selected = self._select_person(det, ratio, width, height)
        if selected is not None:
            self._missed_frames = 0
            if self._smoothed_bbox is None:
                self._smoothed_bbox = selected.copy()
            else:
                a = self.ema_alpha
                self._smoothed_bbox = (a * selected + (1.0 - a) * self._smoothed_bbox).astype(np.float32)
            return self._smoothed_bbox.copy()

        self._missed_frames += 1
        if self._missed_frames > 45:
            self._locked_track_id = None
        if self._smoothed_bbox is not None:
            return self._smoothed_bbox.copy()
        return full_box.copy()


class Sam3DStreamExtractor:
    """Compiled BF16 SAM 3D Body Lite feature extractor producing `(N, SAM_TOKEN_DIM)` tokens."""

    # Hand decoder runs on every `hand_stride`-th frame of a chunk (`fast` mode widens it to the chunk).
    hand_stride = 2

    def __init__(
        self,
        ckpt_path: str | Path = SAM3D_CKPT_PATH,
        mhr_path: str | Path = SAM3D_MHR_PATH,
        device: str | torch.device = "cuda",
        use_compile: bool = True,
        chunk_size: int = 3,
        hand_mode: ResolvedHandMode = "off",
        body_decoder_layers: int = 3,
        hand_decoder_layers: int = 2,
    ) -> None:
        bootstrap_runtime()
        _install_local_dinov3_hub_hook()

        ckpt = Path(ckpt_path)
        mhr = Path(mhr_path)
        if not (ckpt.is_file() and mhr.is_file()):
            raise Sam3DCheckpointMissingError(
                f"SAM 3D Body checkpoints not found at {ckpt.parent}. "
                "Run `persona-flowhmr setup` to download from ModelScope (`facebook/sam-3d-body-dinov3`)."
            )

        from flowhmr.utils.lite_sam_infer import extract_tokens
        from flowhmr.utils.runtime_sam_features import _maybe_convert_backbone_to_fp16
        from flowhmr.utils.sam_compile import enable_sam_compile
        from sam_3d_body import SAM3DBodyEstimator, load_sam_3d_body

        self.device = torch.device(device)
        self.configure(
            chunk_size=chunk_size,
            hand_mode=hand_mode,
            body_decoder_layers=body_decoder_layers,
            hand_decoder_layers=hand_decoder_layers,
        )
        self._use_compile = bool(use_compile and self.device.type == "cuda")
        self._warmed_signatures: set[tuple[int, str, int, int]] = set()
        self._neutral_hand_tokens: torch.Tensor | None = None
        self._extract_tokens = extract_tokens

        sam_model, sam_cfg = load_sam_3d_body(
            checkpoint_path=str(ckpt),
            mhr_path=str(mhr),
            device=str(self.device),
        )
        _maybe_convert_backbone_to_fp16(sam_model, self.device)
        if self._use_compile:
            enable_sam_compile(sam_model, self.device)
        self.estimator = SAM3DBodyEstimator(sam_3d_body_model=sam_model, model_cfg=sam_cfg)

    def configure(
        self,
        *,
        chunk_size: int,
        hand_mode: ResolvedHandMode,
        body_decoder_layers: int,
        hand_decoder_layers: int,
    ) -> None:
        self.chunk_size = max(1, int(chunk_size))
        self.hand_mode = hand_mode
        self.body_decoder_layers = max(1, int(body_decoder_layers))
        self.hand_decoder_layers = max(1, int(hand_decoder_layers))

    @torch.no_grad()
    def _ensure_neutral_hand_tokens(self, hw: tuple[int, int] = (720, 1280)) -> torch.Tensor:
        if self._neutral_hand_tokens is not None:
            return self._neutral_hand_tokens
        h, w = hw
        blank_rgb = [np.full((h, w, 3), 128, dtype=np.uint8)]
        # Run one unpadded single-frame full pass to capture exact in-distribution LayerNorm hand tokens.
        tok = self._extract_tokens(
            self.estimator,
            blank_rgb,
            _center_box(h, w),
            default_camera_intrinsics(h, w)[None],
            gpu_preprocess=True,
            merge_hands=True,
            hand_stride=1,
            hand_mode="full",
            body_decoder_layers=2,
            hand_decoder_layers=2,
        )
        self._neutral_hand_tokens = tok[0, 1024:].clone().cpu()
        return self._neutral_hand_tokens

    @torch.no_grad()
    def warmup(self, hw: tuple[int, int] = (720, 1280)) -> None:
        """Pre-compile static `B = chunk_size` body (and optional hand) shapes."""
        if self.hand_mode == "off":
            self._ensure_neutral_hand_tokens(hw)
        sig = (self.chunk_size, self.hand_mode, self.body_decoder_layers, self.hand_decoder_layers)
        if sig in self._warmed_signatures:
            return
        h, w = hw
        n = self.chunk_size
        dummy_rgb = [np.zeros((h, w, 3), dtype=np.uint8) for _ in range(n)]
        dummy_box = np.repeat(_center_box(h, w), n, axis=0)
        dummy_k = np.repeat(default_camera_intrinsics(h, w)[None], n, axis=0)
        self.extract_chunk(dummy_rgb, dummy_box, dummy_k)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self._warmed_signatures.add(sig)

    @torch.no_grad()
    def extract_chunk(
        self,
        rgb_frames: Sequence[np.ndarray],
        bboxes_xyxy: np.ndarray,
        cameras_k: np.ndarray,
    ) -> torch.Tensor:
        """Extract `(N, SAM_TOKEN_DIM)` float32 conditioning tokens for `N` frames with static compiled batch shapes."""
        if not rgb_frames:
            return _empty_tokens()
        neutral_hands = (
            self._ensure_neutral_hand_tokens(rgb_frames[0].shape[:2])
            if self.hand_mode == "off"
            else None
        )
        effective_hand_stride = max(self.hand_stride, self.chunk_size) if self.hand_mode == "fast" else self.hand_stride

        pieces: list[torch.Tensor] = []
        n = len(rgb_frames)
        cs = self.chunk_size
        for s in range(0, n, cs):
            e = min(n, s + cs)
            sub_rgb = list(rgb_frames[s:e])
            sub_box = bboxes_xyxy[s:e]
            sub_k = cameras_k[s:e]
            actual = len(sub_rgb)
            if self._use_compile and actual < cs:
                pad = cs - actual
                sub_rgb = sub_rgb + [sub_rgb[-1]] * pad
                sub_box = np.concatenate([sub_box, np.repeat(sub_box[-1:], pad, axis=0)], axis=0)
                sub_k = np.concatenate([sub_k, np.repeat(sub_k[-1:], pad, axis=0)], axis=0)
            toks = self._extract_tokens(
                self.estimator,
                sub_rgb,
                sub_box,
                sub_k,
                gpu_preprocess=True,
                merge_hands=True,
                hand_stride=effective_hand_stride,
                hand_mode=self.hand_mode,
                neutral_hand_tokens=neutral_hands,
                body_decoder_layers=self.body_decoder_layers,
                hand_decoder_layers=self.hand_decoder_layers,
            )
            pieces.append(toks[:actual])
        return torch.cat(pieces, dim=0)


class ReplayTokenExtractor:
    """Replays pre-extracted `(T, SAM_TOKEN_DIM)` `.pt`/`.npz` SAM tokens (or deterministic bbox embeddings)."""

    def __init__(self, replay_path: str | Path | None = None) -> None:
        self._tokens: torch.Tensor | None = None
        if replay_path:
            p = Path(replay_path)
            if not p.is_file():
                raise FileNotFoundError(f"Replay tokens file not found: {p}")
            if p.suffix.lower() == ".npz":
                arr = np.load(str(p))["tokens"].astype(np.float32)
                self._tokens = torch.from_numpy(arr)
            else:
                loaded = torch.load(str(p), map_location="cpu", weights_only=True)
                if isinstance(loaded, dict):
                    loaded = loaded.get("tokens", loaded.get("feature"))
                self._tokens = torch.as_tensor(loaded, dtype=torch.float32)
            if self._tokens.ndim != 2 or self._tokens.shape[1] != SAM_TOKEN_DIM:
                raise ValueError(f"Expected (T, {SAM_TOKEN_DIM}) replay tokens, got {tuple(self._tokens.shape)}")
        self._cursor = 0

    def reset(self) -> None:
        self._cursor = 0

    def extract_chunk(
        self,
        rgb_frames: Sequence[np.ndarray],
        bboxes_xyxy: np.ndarray,
        cameras_k: np.ndarray,
    ) -> torch.Tensor:
        n = len(rgb_frames)
        if n == 0:
            return _empty_tokens()
        if self._tokens is not None:
            t_total = self._tokens.shape[0]
            idxs = [(self._cursor + i) % t_total for i in range(n)]
            self._cursor += n
            return self._tokens[idxs].clone()

        h, w = rgb_frames[0].shape[:2]
        cx = ((bboxes_xyxy[:, 0] + bboxes_xyxy[:, 2]) * 0.5 / max(1.0, float(w))) - 0.5
        cy = ((bboxes_xyxy[:, 1] + bboxes_xyxy[:, 3]) * 0.5 / max(1.0, float(h))) - 0.5
        bw = (bboxes_xyxy[:, 2] - bboxes_xyxy[:, 0]) / max(1.0, float(w))
        bh = (bboxes_xyxy[:, 3] - bboxes_xyxy[:, 1]) / max(1.0, float(h))
        geom = np.stack([cx, cy, bw, bh], axis=-1).astype(np.float32)
        tiled = np.tile(geom, (1, SAM_TOKEN_DIM // 4)) * 0.05
        return torch.from_numpy(tiled)
