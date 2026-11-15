"""Streaming perception: batched YOLOX-L + online ByteTrack + compiled SAM 3D Body Lite."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
import numpy as np
import supervision as sv
import torch

from .paths import (
    SAM3D_CKPT_PATH,
    SAM3D_MHR_PATH,
    YOLOX_CKPT_PATH,
    bootstrap_runtime,
)


class Sam3DCheckpointMissingError(FileNotFoundError):
    """Raised when gated `facebook/sam-3d-body-dinov3` weights have not been downloaded yet."""


def is_sam3d_available(ckpt_path: Path = SAM3D_CKPT_PATH, mhr_path: Path = SAM3D_MHR_PATH) -> bool:
    return ckpt_path.is_file() and mhr_path.is_file()


def default_camera_intrinsics(height: int, width: int) -> np.ndarray:
    """Compute FlowHMR's diagonal-focal-length static camera matrix `K` `(3, 3)`."""
    f = float((height * height + width * width) ** 0.5)
    return np.array(
        [[f, 0.0, width * 0.5], [0.0, f, height * 0.5], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )


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
        test_size: tuple[int, int] = (640, 640),
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
        self.test_size = test_size
        self.bbox_thr = float(bbox_thr)
        self.nms_thr = float(nms_thr)
        self.ema_alpha = float(ema_alpha)
        self._postprocess = postprocess

        exp = get_exp(exp_name="yolox-l")
        model = exp.get_model()
        state = torch.load(str(ckpt), map_location="cpu", weights_only=True)
        model.load_state_dict(state["model"])
        model.eval()
        self._use_bf16 = self.device.type == "cuda" and torch.cuda.get_device_capability(self.device)[0] >= 8
        self.model = model.to(device=self.device, dtype=torch.bfloat16 if self._use_bf16 else torch.float32)
        self.num_classes = int(exp.num_classes)

        self.reset_tracker()

    def reset_tracker(self) -> None:
        self._tracker = sv.ByteTrack(frame_rate=30)
        self._locked_track_id: int | None = None
        self._smoothed_bbox: np.ndarray | None = None
        self._missed_frames = 0

    @torch.no_grad()
    def warmup(self, batch_size: int = 8, hw: tuple[int, int] = (720, 1280)) -> None:
        """Pre-initialize cuDNN workspace and kernels before real-time capture starts."""
        dtype = torch.bfloat16 if self._use_bf16 else torch.float32
        dummy = torch.full(
            (max(1, int(batch_size)), 3, self.test_size[0], self.test_size[1]),
            114.0,
            device=self.device,
            dtype=dtype,
        )
        _ = self.model(dummy)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    @torch.no_grad()
    def detect_and_track_batch(self, rgb_frames: Sequence[np.ndarray]) -> np.ndarray:
        """Run GPU-preprocessed batched YOLOX-L + sequential ByteTrack over `N` RGB frames -> `(N, 4)`."""
        if not rgb_frames:
            return np.zeros((0, 4), dtype=np.float32)

        h, w = rgb_frames[0].shape[:2]
        ratio = min(self.test_size[0] / float(h), self.test_size[1] / float(w))
        new_h = max(1, int(h * ratio))
        new_w = max(1, int(w * ratio))
        dtype = torch.bfloat16 if self._use_bf16 else torch.float32

        # Upload compact uint8 NHWC directly to GPU and resize + pad in BF16 on device.
        raw_np = np.stack(rgb_frames, axis=0)
        raw_gpu = torch.from_numpy(raw_np).to(device=self.device, non_blocking=True)
        bgr_nchw = raw_gpu[..., [2, 1, 0]].permute(0, 3, 1, 2).to(dtype=dtype)
        resized = torch.nn.functional.interpolate(
            bgr_nchw,
            size=(new_h, new_w),
            mode="bilinear",
            align_corners=False,
        )
        batch_t = torch.full(
            (len(rgb_frames), 3, self.test_size[0], self.test_size[1]),
            114.0,
            device=self.device,
            dtype=dtype,
        )
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
        out_boxes = np.empty((len(rgb_frames), 4), dtype=np.float32)

        for i, det in enumerate(dets):
            chosen = self._step_single_frame(det, ratio, w, h, full_box)
            out_boxes[i] = chosen

        return out_boxes

    def _step_single_frame(
        self,
        det: torch.Tensor | None,
        ratio: float,
        width: int,
        height: int,
        full_box: np.ndarray,
    ) -> np.ndarray:
        selected: np.ndarray | None = None

        if det is not None and len(det) > 0:
            det_np = det.detach().cpu().numpy()
            person_mask = det_np[:, 6].astype(np.int32) == 0
            det_np = det_np[person_mask]
            if len(det_np) > 0:
                xyxy = det_np[:, :4] / ratio
                xyxy[:, [0, 2]] = np.clip(xyxy[:, [0, 2]], 0.0, float(width))
                xyxy[:, [1, 3]] = np.clip(xyxy[:, [1, 3]], 0.0, float(height))
                scores = (det_np[:, 4] * det_np[:, 5]).astype(np.float32)

                sv_det = sv.Detections(
                    xyxy=xyxy.astype(np.float32),
                    confidence=scores,
                    class_id=np.zeros(len(scores), dtype=np.int32),
                )
                tracked = self._tracker.update_with_detections(sv_det)

                if len(tracked) > 0 and tracked.tracker_id is not None:
                    t_ids = tracked.tracker_id.astype(int)
                    if self._locked_track_id is not None and self._locked_track_id in t_ids:
                        idx = int(np.where(t_ids == self._locked_track_id)[0][0])
                        selected = tracked.xyxy[idx].astype(np.float32)
                    else:
                        # Lock onto largest-area person track weighted by confidence.
                        areas = (tracked.xyxy[:, 2] - tracked.xyxy[:, 0]) * (
                            tracked.xyxy[:, 3] - tracked.xyxy[:, 1]
                        )
                        confs = tracked.confidence if tracked.confidence is not None else np.ones_like(areas)
                        best = int(np.argmax(areas * confs))
                        self._locked_track_id = int(t_ids[best])
                        selected = tracked.xyxy[best].astype(np.float32)
                else:
                    # First frame before ByteTrack confirms track ID: pick largest confident person.
                    areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
                    best = int(np.argmax(areas * scores))
                    selected = xyxy[best].astype(np.float32)

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
    """Compiled BF16 SAM 3D Body Lite feature extractor producing `(N, 3072)` tokens."""

    def __init__(
        self,
        ckpt_path: str | Path = SAM3D_CKPT_PATH,
        mhr_path: str | Path = SAM3D_MHR_PATH,
        device: str | torch.device = "cuda",
        use_compile: bool = True,
        chunk_size: int = 6,
        hand_stride: int = 2,
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
        self.chunk_size = max(1, int(chunk_size))
        self.hand_stride = max(1, int(hand_stride))
        self._use_compile = bool(use_compile and self.device.type == "cuda")
        self._warmed_chunk_sizes: set[int] = set()
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

    @torch.no_grad()
    def warmup(self, chunk_size: int | None = None, hw: tuple[int, int] = (720, 1280)) -> None:
        """Pre-compile static `B = chunk_size` (body) and `B = 2 * chunk_size` (merged hands) shapes."""
        target = max(1, int(chunk_size or self.chunk_size))
        self.chunk_size = target
        if target in self._warmed_chunk_sizes:
            return
        h, w = hw
        dummy_rgb = [np.zeros((h, w, 3), dtype=np.uint8) for _ in range(target)]
        dummy_box = np.tile(
            np.array([[w * 0.25, h * 0.1, w * 0.75, h * 0.9]], dtype=np.float32),
            (target, 1),
        )
        dummy_k = np.repeat(default_camera_intrinsics(h, w)[None], target, axis=0)
        self.extract_chunk(dummy_rgb, dummy_box, dummy_k)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self._warmed_chunk_sizes.add(target)

    @torch.no_grad()
    def extract_chunk(
        self,
        rgb_frames: Sequence[np.ndarray],
        bboxes_xyxy: np.ndarray,
        cameras_k: np.ndarray,
    ) -> torch.Tensor:
        """Extract `(N, 3072)` float32 conditioning tokens for `N` frames with static compiled batch shapes."""
        if not rgb_frames:
            return torch.empty((0, 3072), dtype=torch.float32)
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
                hand_stride=self.hand_stride,
            )
            pieces.append(toks[:actual])
        return torch.cat(pieces, dim=0)


class ReplayTokenExtractor:
    """Replays pre-extracted `(T, 3072)` `.pt`/`.npz` SAM tokens (or deterministic bbox embeddings)."""

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
            if self._tokens.ndim != 2 or self._tokens.shape[1] != 3072:
                raise ValueError(f"Expected (T, 3072) replay tokens, got {tuple(self._tokens.shape)}")
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
            return torch.empty((0, 3072), dtype=torch.float32)
        if self._tokens is not None:
            t_total = self._tokens.shape[0]
            idxs = [(self._cursor + i) % t_total for i in range(n)]
            self._cursor += n
            return self._tokens[idxs].clone()

        # Deterministic normalized bbox-geometry feature projection (for pipeline integration tests
        # when neither SAM3D weights nor a `.pt` token recording is provided).
        h, w = rgb_frames[0].shape[:2]
        cx = ((bboxes_xyxy[:, 0] + bboxes_xyxy[:, 2]) * 0.5 / max(1.0, float(w))) - 0.5
        cy = ((bboxes_xyxy[:, 1] + bboxes_xyxy[:, 3]) * 0.5 / max(1.0, float(h))) - 0.5
        bw = (bboxes_xyxy[:, 2] - bboxes_xyxy[:, 0]) / max(1.0, float(w))
        bh = (bboxes_xyxy[:, 3] - bboxes_xyxy[:, 1]) / max(1.0, float(h))
        geom = np.stack([cx, cy, bw, bh], axis=-1).astype(np.float32)  # (N, 4)
        tiled = np.tile(geom, (1, 3072 // 4)) * 0.05
        return torch.from_numpy(tiled)


@dataclass
class FrameFeatureItem:
    """Per-frame perception output stored in the engine's sliding-window conditioning history."""

    frame_index: int
    media_time_s: float
    capture_mono_s: float
    bbox_xyxy: np.ndarray  # (4,)
    sam_token: torch.Tensor  # (3072,) CPU float32
