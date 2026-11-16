"""Multi-threaded streaming mocap engine: Source -> Perception -> MMDiT -> Stitcher -> Playout."""

from __future__ import annotations

import contextlib
import ctypes
import queue
import sys
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from .config import CONFIG_FIELDS, LATENCY_PRESETS, MODEL_FPS, EngineConfig, LatencyPreset
from .paths import FLOWHMR_CFG_PATH, FLOWHMR_CKPT_PATH, MOTION_STATS_PATH, YOLOX_CKPT_PATH, is_sam3d_available
from .perception import (
    BatchedYoloxTracker,
    ReplayTokenExtractor,
    Sam3DStreamExtractor,
    default_camera_intrinsics,
)
from .persona_api import PersonaPluginClient
from .playout import PosePlayoutBuffer
from .retarget import SMPLH_PARENTS, CanonicalPoseFrame, pose_to_live2d_params, pose_to_vmc_bones
from .setup_models import check_models_status
from .solver import FlowHMRWindowSolver
from .sources import CameraSource, PreviewRingBuffer, SampledFrame, VideoFileSource
from .stitcher import MotionLatentDecoder, StreamingStitcher
from .vmc_sender import VmcUdpSender


@dataclass
class EngineTelemetry:
    state: str = "stopped"  # stopped | warming_up | buffering | streaming | error
    error: str = ""
    active_backend: str = "none"
    capture_fps: float = 0.0
    playout_fps: float = 0.0
    yolox_ms_per_frame: float = 0.0
    sam_ms_per_frame: float = 0.0
    mmdit_ms_per_window: float = 0.0
    target_latency_ms: float = 0.0
    measured_latency_ms: float = 0.0
    buffer_lead_ms: float = 0.0
    frames_captured: int = 0
    frames_perceived: int = 0
    windows_solved: int = 0
    vmc_packets_sent: int = 0
    underruns: int = 0
    plugin_api_connected: bool = False
    plugin_api_vmc_source_id: str = ""


# Settings applied on the fly; changing any other config field restarts a running stream.
_LIVE_FIELDS = frozenset(
    {
        "playout_fps",
        "mirror",
        "follow_movement",
        "vmc_enabled",
        "vmc_host",
        "vmc_port",
        "plugin_api_enabled",
        "plugin_api_url",
        "plugin_api_token",
        "plugin_api_inject_params",
        "plugin_api_auto_configure_vmc",
    }
)
# Changing these reconnects the Persona Plugin API client.
_PLUGIN_CONNECTION_FIELDS = frozenset({"plugin_api_enabled", "plugin_api_url", "plugin_api_token"})
# Changing these re-runs Persona's VMC receiver setup on the already-connected client.
_PLUGIN_RECEIVER_FIELDS = frozenset({"plugin_api_auto_configure_vmc", "vmc_port", "follow_movement"})

_PRESET_DETAILS = {name: preset.to_dict() for name, preset in LATENCY_PRESETS.items()}
_JPEG_PARAMS = [int(cv2.IMWRITE_JPEG_QUALITY), 80]
# Preview frames kept for the latency-synced view: the longest playout delay plus 2 s of slack.
_PREVIEW_FRAMES = round(max(p.playout_delay_s for p in LATENCY_PRESETS.values()) * MODEL_FPS) + 60

# Body joint parent pairs `(child, parent)` within the first 22 SMPL-H joints for HUD drawing.
_BODY22_LIMBS: tuple[tuple[int, int], ...] = tuple(
    (j, SMPLH_PARENTS[j]) for j in range(1, 22) if SMPLH_PARENTS[j] >= 0
)


_WINMM_TIMER_ENABLED = False


def _enable_windows_high_res_timer() -> None:
    """Request 1 ms OS timer resolution on Windows so 30/60 Hz `time.sleep` avoids 15.6 ms jitter."""
    global _WINMM_TIMER_ENABLED
    if _WINMM_TIMER_ENABLED or sys.platform != "win32":
        return
    try:
        ctypes.windll.winmm.timeBeginPeriod(1)
        _WINMM_TIMER_ENABLED = True
    except Exception:
        pass


@dataclass
class FrameFeatureItem:
    """Per-frame SAM conditioning token kept in the GPU worker's sliding-window history."""

    frame_index: int
    capture_mono_s: float
    sam_token: torch.Tensor  # (SAM_TOKEN_DIM,) CPU float32

    @classmethod
    def from_sample(cls, sample: SampledFrame, token: torch.Tensor) -> FrameFeatureItem:
        return cls(frame_index=sample.frame_index, capture_mono_s=sample.capture_mono_s, sam_token=token)


def _plan_keyframes(
    frame_indices: list[int],
    reset_flags: list[bool],
    prev_kf_idx: int | None,
    stride: int,
) -> list[int]:
    """Batch positions that get a SAM keyframe: every `stride` frames, at resets, and the batch's last frame."""
    positions: list[int] = []
    last_kf = prev_kf_idx
    for pos, (f_idx, is_reset) in enumerate(zip(frame_indices, reset_flags)):
        if is_reset or last_kf is None or (f_idx - last_kf) >= stride:
            positions.append(pos)
            last_kf = f_idx

    last_pos = len(frame_indices) - 1
    if last_pos >= 0 and last_pos not in positions:
        # Keep batch boundary phase-locked across video loop resets without adding padding.
        if positions and not reset_flags[positions[-1]]:
            positions[-1] = last_pos
        else:
            positions.append(last_pos)
    return positions


class _KeyframeInterpolator:
    """Fills frames between sparse SAM keyframes by lerping keyframe tokens, emitting history items in order."""

    def __init__(self) -> None:
        self.prev_kf_idx: int | None = None
        self._prev_kf_tok: torch.Tensor | None = None
        self._pending: list[SampledFrame] = []

    def push(self, sample: SampledFrame, token: torch.Tensor | None, *, is_reset: bool) -> list[FrameFeatureItem]:
        """Add one frame (`token` set for keyframes); returns the history items it completes."""
        out: list[FrameFeatureItem] = []
        if is_reset:
            # Flush pre-cut intermediate frames without blending across a loop rewind.
            out += self.flush()
            self.prev_kf_idx = None
            self._prev_kf_tok = None

        if token is None:
            self._pending.append(sample)
            return out

        if self._prev_kf_tok is not None and self.prev_kf_idx is not None and self._pending:
            span = max(1, sample.frame_index - self.prev_kf_idx)
            for p in self._pending:
                alpha = float(p.frame_index - self.prev_kf_idx) / float(span)
                out.append(FrameFeatureItem.from_sample(p, torch.lerp(self._prev_kf_tok, token, alpha)))
            self._pending.clear()
        out.append(FrameFeatureItem.from_sample(sample, token))
        self.prev_kf_idx = sample.frame_index
        self._prev_kf_tok = token
        return out

    def flush(self) -> list[FrameFeatureItem]:
        """Emit pending frames holding the last keyframe token (at stream resets and end of stream)."""
        if not self._pending or self._prev_kf_tok is None:
            return []
        out = [FrameFeatureItem.from_sample(p, self._prev_kf_tok.clone()) for p in self._pending]
        self._pending.clear()
        return out


def _extract_dense(
    extractor: Sam3DStreamExtractor | ReplayTokenExtractor,
    rgb_list: list[np.ndarray],
    bboxes: np.ndarray,
    cam_k: np.ndarray,
    reset_flags: list[bool],
) -> torch.Tensor:
    """Tokens for every frame of a batch; replay extractors restart their token cursor at stream resets."""
    if not isinstance(extractor, ReplayTokenExtractor) or not any(reset_flags):
        return extractor.extract_chunk(rgb_list, bboxes, cam_k)
    pieces: list[torch.Tensor] = []
    seg_start = 0
    for i, is_reset in enumerate(reset_flags):
        if not is_reset:
            continue
        if i > seg_start:
            pieces.append(extractor.extract_chunk(rgb_list[seg_start:i], bboxes[seg_start:i], cam_k[seg_start:i]))
            seg_start = i
        extractor.reset()
    pieces.append(extractor.extract_chunk(rgb_list[seg_start:], bboxes[seg_start:], cam_k[seg_start:]))
    return torch.cat(pieces, dim=0)


class _RateMeter:
    """Counts events and reports their rate (per second) once every `window_s`."""

    def __init__(self, window_s: float = 0.5) -> None:
        self._window_s = window_s
        self._count = 0
        self._t0 = time.perf_counter()

    def tick(self, now: float) -> float | None:
        self._count += 1
        elapsed = now - self._t0
        if elapsed < self._window_s:
            return None
        rate = round(self._count / elapsed, 1)
        self._count = 0
        self._t0 = now
        return rate


class _JpegCache:
    """Last encoded preview JPEG, re-encoded only when its cache key changes."""

    def __init__(self) -> None:
        self._key: Any = None
        self._data: bytes | None = None

    def clear(self) -> None:
        self._key = None
        self._data = None

    def get(self, key: Any, draw: Callable[[], np.ndarray | None]) -> bytes | None:
        if self._data is not None and self._key == key:
            return self._data
        bgr = draw()
        if bgr is None:
            return None
        ok, buf = cv2.imencode(".jpg", bgr, _JPEG_PARAMS)
        if not ok:
            return None
        self._key, self._data = key, buf.tobytes()
        return self._data


class StreamingMocapEngine:
    """Real-time buffered single-camera mocap engine sending VMC (UDP) and Persona Plugin API."""

    def __init__(
        self,
        config: EngineConfig | None = None,
        *,
        settings_path: str | Path | None = None,
    ) -> None:
        _enable_windows_high_res_timer()
        self.config = config or EngineConfig()
        self._settings_path = Path(settings_path) if settings_path is not None else None
        self._lock = threading.Lock()
        self.telemetry = EngineTelemetry()

        self._preview = PreviewRingBuffer(max_items=_PREVIEW_FRAMES, max_width=640)
        self._playout_buffer = PosePlayoutBuffer(max_frames=900)
        self._latest_emitted_pose: CanonicalPoseFrame | None = None
        self._frame0_mono = 0.0  # capture clock of stream frame 0; anchors the playout timeline
        self._live_jpeg = _JpegCache()
        self._synced_jpeg = _JpegCache()

        self._vmc_sender: VmcUdpSender | None = None
        self._plugin_client: PersonaPluginClient | None = None

        # Cached heavy GPU models across start/stop restarts so switching camera/video or presets
        # takes < 0.2s instead of reloading 2.3 GB of weights.
        self._yolox: BatchedYoloxTracker | None = None
        self._sam3d: Sam3DStreamExtractor | None = None
        self._solver: FlowHMRWindowSolver | None = None
        self._decoder: MotionLatentDecoder | None = None

        self._stop_event = threading.Event()
        self._threads: list[threading.Thread] = []

    @property
    def is_running(self) -> bool:
        return any(t.is_alive() for t in self._threads)

    def update_live_settings(self, patch: dict[str, Any]) -> dict[str, Any]:
        """Apply a (partial) config patch: live fields take effect immediately, others restart a running stream."""
        was_running = self.is_running
        changed: set[str] = set()

        with self._lock:
            for k, v in patch.items():
                if k not in CONFIG_FIELDS:
                    continue
                if k == "plugin_api_token" and v == "" and not patch.get("clear_plugin_api_token", False):
                    # Empty token from redacted UI form keeps existing token unless explicitly cleared.
                    continue
                if getattr(self.config, k) != v:
                    setattr(self.config, k, v)
                    changed.add(k)

            if self._vmc_sender is not None:
                self._vmc_sender.set_target(self.config.vmc_host, int(self.config.vmc_port))

            if changed and self._settings_path is not None:
                with contextlib.suppress(OSError):
                    self.config.save(self._settings_path)

        if was_running and (changed - _LIVE_FIELDS):
            self.stop()
            self.start()  # also reconnects the Plugin API client
        elif changed & _PLUGIN_CONNECTION_FIELDS:
            self._sync_plugin_client()
        elif changed & _PLUGIN_RECEIVER_FIELDS:
            self._configure_plugin_receiver()

        return self.get_status()

    def _sync_plugin_client(self) -> None:
        """(Re)connect the Persona Plugin API client from config, then set up Persona's VMC receiver."""
        if self._plugin_client is not None:
            self._plugin_client.stop()
            self._plugin_client = None
        if not self.config.plugin_api_enabled:
            self.telemetry.plugin_api_vmc_source_id = ""
            return

        client = PersonaPluginClient(
            url=self.config.plugin_api_url,
            token=self.config.plugin_api_token,
        )
        client.start(timeout_s=3.0)
        self._plugin_client = client
        self._configure_plugin_receiver()

    def _configure_plugin_receiver(self) -> None:
        """Ensure Persona has an enabled `vmc` source on our port, bound to the primary VRM avatar."""
        client = self._plugin_client
        if client is None or not client.is_connected or not self.config.plugin_api_auto_configure_vmc:
            return
        try:
            res = client.ensure_vmc_receiver(
                vmc_port=self.config.vmc_port,
                bind_primary_avatar=True,
                follow_movement=self.config.follow_movement,
            )
            self.telemetry.plugin_api_vmc_source_id = str(res.get("sourceId") or "")
        except Exception as exc:
            client.last_error = str(exc)

    def _ensure_models_loaded(self) -> tuple[Sam3DStreamExtractor | ReplayTokenExtractor, str]:
        """Load (or reconfigure the cached) YOLOX-L, FlowHMR MMDiT, latent decoder, and SAM/Replay extractor."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        cfg = self.config
        preset = cfg.get_preset()

        if self._yolox is None:
            self._yolox = BatchedYoloxTracker(ckpt_path=YOLOX_CKPT_PATH, device=device)
        self._yolox.configure(test_size=cfg.effective_yolo_size(), yolo_stride=cfg.yolo_stride)

        if self._solver is None:
            self._solver = FlowHMRWindowSolver(
                ckpt_path=FLOWHMR_CKPT_PATH,
                cfg_path=FLOWHMR_CFG_PATH,
                device=device,
                use_compile=cfg.torch_compile,
            )
        else:
            self._solver.reset_noise_cache()

        if self._decoder is None:
            self._decoder = MotionLatentDecoder(motion_stats_path=MOTION_STATS_PATH, motion_style=self._solver.motion_style)

        backend = cfg.perception_backend
        if backend == "replay" or (backend == "auto" and cfg.replay_tokens_path):
            extractor = ReplayTokenExtractor(cfg.replay_tokens_path or None)
            return extractor, ("replay" if cfg.replay_tokens_path else "bbox-probe")

        if backend == "sam3d" or (backend == "auto" and is_sam3d_available()):
            if self._sam3d is None:
                self._sam3d = Sam3DStreamExtractor(device=device, use_compile=cfg.torch_compile)
            self._sam3d.configure(
                chunk_size=preset.sam_chunk_size,
                hand_mode=cfg.effective_hand_mode(),
                body_decoder_layers=preset.body_decoder_layers,
                hand_decoder_layers=preset.hand_decoder_layers,
            )
            return self._sam3d, "sam3d"

        # `auto` without SAM3D checkpoint raises clear actionable instructions.
        raise FileNotFoundError(
            "Checkpoint `facebook/sam-3d-body-dinov3` is not installed yet. "
            "Run `persona-flowhmr setup` to download it automatically from ModelScope "
            "(or select `replay` backend for pipeline testing)."
        )

    def start(self, *, realtime_source: bool = True) -> None:
        """Start capture, GPU sliding-window inference, and 30/60 Hz VMC playout threads."""
        if self.is_running:
            return

        self._stop_event.clear()
        self._preview.clear()
        self._playout_buffer.clear()
        self._latest_emitted_pose = None
        self._frame0_mono = 0.0
        self._live_jpeg.clear()
        self._synced_jpeg.clear()
        self.telemetry = EngineTelemetry(
            state="warming_up",
            target_latency_ms=self.config.get_preset().playout_delay_s * 1000.0,
        )

        if self._vmc_sender is None:
            self._vmc_sender = VmcUdpSender(self.config.vmc_host, self.config.vmc_port)
        else:
            self._vmc_sender.set_target(self.config.vmc_host, self.config.vmc_port)

        self._sync_plugin_client()

        frame_queue: queue.Queue[SampledFrame | None] = queue.Queue(maxsize=96)
        warmup_done_event = threading.Event()
        first_commit_event = threading.Event()
        self._threads = [
            threading.Thread(
                target=self._capture_worker,
                args=(frame_queue, warmup_done_event, realtime_source),
                name="mocap-capture",
                daemon=True,
            ),
            threading.Thread(
                target=self._gpu_worker,
                args=(frame_queue, warmup_done_event, first_commit_event),
                name="mocap-gpu",
                daemon=True,
            ),
            threading.Thread(
                target=self._playout_worker,
                args=(first_commit_event, realtime_source),
                name="mocap-playout",
                daemon=True,
            ),
        ]
        for t in self._threads:
            t.start()

    def stop(self) -> None:
        self._stop_event.set()
        for t in self._threads:
            if t.is_alive():
                t.join(timeout=3.0)
        self._threads.clear()
        if self._plugin_client is not None:
            self._plugin_client.stop()
            self._plugin_client = None
        if self.telemetry.state != "error":
            self.telemetry.state = "stopped"

    def _put_until_stopped(
        self,
        frame_queue: queue.Queue[SampledFrame | None],
        item: SampledFrame | None,
        *,
        drop_oldest: bool = False,
    ) -> None:
        """Block until `item` is queued or the engine stops; `drop_oldest` evicts stale frames instead of waiting."""
        while not self._stop_event.is_set():
            try:
                frame_queue.put(item, timeout=0.1)
                return
            except queue.Full:
                if drop_oldest:
                    with contextlib.suppress(queue.Empty):
                        frame_queue.get_nowait()

    def _capture_worker(
        self,
        frame_queue: queue.Queue[SampledFrame | None],
        warmup_done_event: threading.Event,
        realtime_source: bool,
    ) -> None:
        source: VideoFileSource | CameraSource | None = None
        try:
            # Wait for model load + Triton warmup before starting the real-time capture clock
            # so frame 0's monotonic timestamp isn't aged by compilation time.
            while not warmup_done_event.wait(timeout=0.2):
                if self._stop_event.is_set():
                    return
            if self._stop_event.is_set() or self.telemetry.state == "error":
                return

            if self.config.source_mode == "camera":
                source = CameraSource(
                    self.config.camera_index,
                    width=self.config.camera_width,
                    height=self.config.camera_height,
                    target_fps=MODEL_FPS,
                )
            else:
                source = VideoFileSource(
                    self.config.video_path,
                    loop=self.config.loop_video,
                    realtime=realtime_source,
                    target_fps=MODEL_FPS,
                )

            # Never stall live camera capture if GPU is still compiling warmup kernels.
            drop_oldest = self.config.source_mode == "camera"
            capture_rate = _RateMeter()
            while not self._stop_event.is_set():
                sample = source.read_next()
                if sample is None:
                    break
                if sample.frame_index == 0:
                    self._frame0_mono = sample.capture_mono_s

                self._preview.record_frame(sample.frame_index, sample.rgb)
                self.telemetry.frames_captured = sample.frame_index + 1
                fps = capture_rate.tick(time.perf_counter())
                if fps is not None:
                    self.telemetry.capture_fps = fps

                self._put_until_stopped(frame_queue, sample, drop_oldest=drop_oldest)
        except Exception as exc:
            self.telemetry.state = "error"
            self.telemetry.error = f"Capture error: {exc}"
            self._stop_event.set()
        finally:
            if source is not None:
                source.close()
            self._put_until_stopped(frame_queue, None)

    def _take_batch(
        self,
        frame_queue: queue.Queue[SampledFrame | None],
        goal: int,
    ) -> tuple[list[SampledFrame], bool]:
        """Collect up to `goal` frames; the flag reports whether the end-of-stream sentinel arrived."""
        batch: list[SampledFrame] = []
        while len(batch) < goal and not self._stop_event.is_set():
            try:
                item = frame_queue.get(timeout=0.05)
            except queue.Empty:
                continue
            if item is None:
                return batch, True
            batch.append(item)
        return batch, False

    def _gpu_worker(
        self,
        frame_queue: queue.Queue[SampledFrame | None],
        warmup_done_event: threading.Event,
        first_commit_event: threading.Event,
    ) -> None:
        try:
            extractor, active_backend = self._ensure_models_loaded()
            self.telemetry.active_backend = active_backend
            assert self._yolox is not None and self._solver is not None and self._decoder is not None
            yolox = self._yolox

            preset = self.config.get_preset()
            kf_chunk = preset.sam_chunk_size
            is_sam = isinstance(extractor, Sam3DStreamExtractor)
            sam_stride = max(1, self.config.effective_sam_stride()) if is_sam else 1
            keyframe_mode = sam_stride > 1  # sparse SAM keyframes, lerped tokens in between
            sync_cuda = torch.cuda.is_available()

            # Pre-compile SAM 3D Body + MMDiT static shapes before buffering starts so live windows
            # never pay Dynamo/Triton compilation latency mid-stream.
            yolox.warmup(max(1, (kf_chunk * sam_stride + yolox.yolo_stride) // yolox.yolo_stride))
            if is_sam:
                extractor.warmup()
            self._solver.warmup(preset.window_frames, steps=2)
            if self._stop_event.is_set():
                warmup_done_event.set()
                return

            self.telemetry.state = "buffering"
            warmup_done_event.set()
            stitcher = StreamingStitcher(self._decoder, fps=MODEL_FPS)
            history: deque[FrameFeatureItem] = deque(maxlen=preset.window_frames + 128)
            keyframes = _KeyframeInterpolator()
            next_emit_frame = 0
            eos_reached = False

            while not self._stop_event.is_set() and not eos_reached:
                if keyframe_mode:
                    first_kf = keyframes.prev_kf_idx is None
                    batch_goal = 1 + (kf_chunk - 1) * sam_stride if first_kf else kf_chunk * sam_stride
                else:
                    have = (history[-1].frame_index + 1) if history else 0
                    batch_goal = min(kf_chunk, max(1, next_emit_frame + preset.future_frames - have))

                batch, eos_reached = self._take_batch(frame_queue, batch_goal)
                if batch:
                    rgb_list = [s.rgb for s in batch]
                    reset_flags = [s.is_stream_reset for s in batch]
                    h, w = rgb_list[0].shape[:2]
                    cam_k = np.repeat(default_camera_intrinsics(h, w)[None], len(batch), axis=0)

                    t0 = time.perf_counter()
                    bboxes = yolox.detect_and_track_batch(rgb_list, reset_flags=reset_flags)
                    if sync_cuda:
                        torch.cuda.synchronize()
                    t1 = time.perf_counter()

                    if keyframe_mode:
                        kf_pos = _plan_keyframes([s.frame_index for s in batch], reset_flags, keyframes.prev_kf_idx, sam_stride)
                        kf_tokens = extractor.extract_chunk([rgb_list[i] for i in kf_pos], bboxes[kf_pos], cam_k[kf_pos])
                    else:
                        tokens = _extract_dense(extractor, rgb_list, bboxes, cam_k, reset_flags)
                    if sync_cuda:
                        torch.cuda.synchronize()
                    t2 = time.perf_counter()

                    for s, bbox in zip(batch, bboxes):
                        self._preview.record_bbox(s.frame_index, bbox)
                    if keyframe_mode:
                        kf_token_at = dict(zip(kf_pos, kf_tokens))
                        for i, s in enumerate(batch):
                            history.extend(keyframes.push(s, kf_token_at.get(i), is_reset=reset_flags[i]))
                    else:
                        history.extend(FrameFeatureItem.from_sample(s, tok) for s, tok in zip(batch, tokens))

                    self.telemetry.yolox_ms_per_frame = round((t1 - t0) * 1000.0 / len(batch), 2)
                    self.telemetry.sam_ms_per_frame = round((t2 - t1) * 1000.0 / len(batch), 2)

                if eos_reached:
                    history.extend(keyframes.flush())
                if history:
                    self.telemetry.frames_perceived = history[-1].frame_index + 1

                next_emit_frame = self._solve_ready_windows(
                    history,
                    next_emit_frame,
                    eos=eos_reached,
                    preset=preset,
                    stitcher=stitcher,
                    first_commit_event=first_commit_event,
                )
        except Exception as exc:
            self.telemetry.state = "error"
            self.telemetry.error = str(exc)
            self._stop_event.set()
            warmup_done_event.set()
            first_commit_event.set()

    def _solve_ready_windows(
        self,
        history: deque[FrameFeatureItem],
        next_emit_frame: int,
        *,
        eos: bool,
        preset: LatencyPreset,
        stitcher: StreamingStitcher,
        first_commit_event: threading.Event,
    ) -> int:
        """Solve and commit every stride window whose frames are available; returns the next frame to emit."""
        assert self._solver is not None
        while history and not self._stop_event.is_set():
            latest_available = history[-1].frame_index
            if latest_available < next_emit_frame + preset.future_frames - 1 and not eos:
                break
            if eos and latest_available < next_emit_frame:
                break

            # Window `[next_emit_frame - H .. next_emit_frame + S + O + L - 1]`, clamped to the oldest/newest
            # history item so window 0 (`frame < 0`) and the EOS tail keep a fixed `preset.window_frames` length.
            by_idx = {it.frame_index: it for it in history}
            first_item, last_item = history[0], history[-1]
            win_start = next_emit_frame - preset.history_frames
            win_indices = list(range(win_start, win_start + preset.window_frames))
            win_tokens = [
                by_idx[f].sam_token if f in by_idx else (first_item if f < first_item.frame_index else last_item).sam_token
                for f in win_indices
            ]
            latent_win = self._solver.solve_window(torch.stack(win_tokens, dim=0), win_indices, ode_steps=preset.ode_steps)
            self.telemetry.mmdit_ms_per_window = round(self._solver.last_solve_ms, 2)
            self.telemetry.windows_solved += 1

            stride = preset.stride_frames
            if eos:
                stride = min(stride, max(0, latest_available - next_emit_frame + 1))
                if stride == 0:
                    break

            committed = stitcher.commit_window(
                latent_win,
                emit_offset_in_window=preset.history_frames,
                stride_frames=stride,
                overlap_frames=0 if eos else preset.overlap_frames,
                first_emit_frame_index=next_emit_frame,
                capture_mono_times=[(by_idx.get(next_emit_frame + i) or last_item).capture_mono_s for i in range(stride)],
            )
            self._playout_buffer.push_many(committed)
            next_emit_frame += stride
            if not first_commit_event.is_set():
                self.telemetry.state = "streaming"
                first_commit_event.set()
        return next_emit_frame

    def _playout_worker(self, first_commit_event: threading.Event, realtime_source: bool) -> None:
        if not first_commit_event.wait(timeout=180.0) or self._stop_event.is_set():
            return
        if self.telemetry.state == "error":
            return

        delay_s = self.config.get_preset().playout_delay_s
        # Anchor playout timeline to `max(now, frame0_mono + delay_s)` so real-time streams hit exact
        # `playout_delay_s` end-to-end latency, while unthrottled benchmark runs start immediately.
        now = time.perf_counter()
        playout_origin_mono = max(now, self._frame0_mono + delay_s) if realtime_source else now
        # Wall-clock deadline of the next tick, advanced by the current period so the FPS can change live.
        next_tick_mono = playout_origin_mono
        playout_rate = _RateMeter()
        last_param_inject_mono = 0.0

        while not self._stop_event.is_set():
            dt_out = 1.0 / float(self.config.playout_fps)
            target_wall = next_tick_mono
            sleep_s = target_wall - time.perf_counter()
            if sleep_s > 0.0012:
                time.sleep(sleep_s - 0.0005)
            # Bounded sub-millisecond spin without monopolizing Python's GIL across OS quanta.
            spin_deadline = time.perf_counter() + 0.0008
            while time.perf_counter() < target_wall and time.perf_counter() < spin_deadline:
                pass
            if self._stop_event.is_set():
                break
            next_tick_mono += dt_out

            now_mono = time.perf_counter()
            target_media_s = now_mono - playout_origin_mono
            if target_media_s < 0.0:
                continue

            was_started = self._latest_emitted_pose is not None
            sample = self._playout_buffer.sample_at_media_time(target_media_s)
            if sample is None:
                continue

            if sample.underrun and was_started and realtime_source:
                # Smoothly slip playout origin by up to one tick if external GPU load momentarily spikes
                # so playout never accumulates catch-up lag or fast-forward lurches.
                slip_s = min(dt_out, max(0.0, target_media_s - sample.pose.media_time_s))
                playout_origin_mono += slip_s
                next_tick_mono += slip_s

            pose = sample.pose
            self._latest_emitted_pose = pose
            self.telemetry.buffer_lead_ms = round(sample.buffer_lead_ms, 1)
            self.telemetry.underruns = self._playout_buffer.underrun_count
            if realtime_source and pose.capture_mono_s > 0.0:
                self.telemetry.measured_latency_ms = round((now_mono - pose.capture_mono_s) * 1000.0, 1)
            else:
                self.telemetry.measured_latency_ms = round(delay_s * 1000.0, 1)

            if self.config.vmc_enabled and self._vmc_sender is not None:
                bones = pose_to_vmc_bones(
                    pose,
                    mirror=self.config.mirror,
                    follow_movement=self.config.follow_movement,
                )
                self._vmc_sender.send_pose(bones)
                self.telemetry.vmc_packets_sent = self._vmc_sender.packets_sent

            client = self._plugin_client
            if (
                self.config.plugin_api_enabled
                and self.config.plugin_api_inject_params
                and client is not None
                and client.is_connected
                and client.has_live2d_model
                and (now_mono - last_param_inject_mono) >= 0.05
            ):
                client.inject_params_fire_and_forget(pose_to_live2d_params(pose, mirror=self.config.mirror))
                last_param_inject_mono = now_mono

            fps = playout_rate.tick(now_mono)
            if fps is not None:
                self.telemetry.playout_fps = fps

    def render_live_jpeg(self) -> bytes | None:
        """Encode the latest live input frame (with detected person bbox) as JPEG."""
        return self._render_latest_frame(self._live_jpeg, "LIVE")

    def render_synced_playout_jpeg(self) -> bytes | None:
        """Encode the latency-synced frame matching the exact VMC pose currently playing out."""
        pose = self._latest_emitted_pose
        if pose is None:
            return self._render_latest_frame(self._synced_jpeg, "BUFFERING")

        mirror = self.config.mirror

        def draw() -> np.ndarray | None:
            got = self._preview.get_synced_bgr(pose.frame_index)
            if got is None:
                return None
            f_idx, bgr, bbox, orig_hw = got
            self._draw_bbox_overlay(bgr, bbox, orig_hw, label=f"SYNCED PLAYOUT #{f_idx}")
            self._draw_skeleton_hud(bgr, pose)
            return cv2.flip(bgr, 1) if mirror else bgr

        return self._synced_jpeg.get(("play", pose.frame_index, round(pose.media_time_s, 4), mirror), draw)

    def _render_latest_frame(self, cache: _JpegCache, label: str) -> bytes | None:
        got = self._preview.get_latest_bgr()
        if got is None:
            return None
        f_idx, bgr, bbox, orig_hw = got

        def draw() -> np.ndarray:
            self._draw_bbox_overlay(bgr, bbox, orig_hw, label=f"{label} #{f_idx}")
            return bgr

        return cache.get((label, f_idx, bbox), draw)

    @staticmethod
    def _draw_bbox_overlay(
        bgr: np.ndarray,
        bbox: tuple[float, float, float, float] | None,
        orig_hw: tuple[int, int],
        label: str,
    ) -> None:
        h, w = bgr.shape[:2]
        oh, ow = orig_hw
        sx, sy = w / max(1.0, float(ow)), h / max(1.0, float(oh))
        if bbox is not None:
            x1 = int(round(bbox[0] * sx))
            y1 = int(round(bbox[1] * sy))
            x2 = int(round(bbox[2] * sx))
            y2 = int(round(bbox[3] * sy))
            cv2.rectangle(bgr, (x1, y1), (x2, y2), (96, 230, 150), 2)
        cv2.putText(bgr, label, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (240, 248, 255), 2, cv2.LINE_AA)

    @staticmethod
    def _draw_skeleton_hud(bgr: np.ndarray, pose: CanonicalPoseFrame) -> None:
        """Render the 22-joint 3D body skeleton inset in the bottom-right corner of the synced preview."""
        if pose.joints_cam_3d is None:
            return
        h, w = bgr.shape[:2]
        pts3 = pose.joints_cam_3d  # (22, 3) in camera/local root-centered meters (+Y up)
        center = pts3[0]
        rel = pts3 - center
        span = max(0.8, float(np.ptp(pts3[:, 1])) * 1.25)

        # Draw clean skeleton card in bottom-right corner so bone poses are always legible even
        # when camera bbox crops or flips.
        card_w, card_h = 170, 200
        ox, oy = w - card_w - 10, h - card_h - 10
        sub = bgr[max(0, oy) : h - 10, max(0, ox) : w - 10]
        if sub.size > 0:
            sub[:] = (sub.astype(np.uint16) * 35 // 100).astype(np.uint8)
        cv2.rectangle(bgr, (ox, oy), (w - 10, h - 10), (90, 110, 140), 1)

        cx = ox + card_w // 2
        cy = oy + int(card_h * 0.56)
        scale = (card_h * 0.78) / span
        pts2 = np.empty((22, 2), dtype=np.int32)
        # Note: in camera space before R_y(180), +X is screen-right and +Y is up.
        for j in range(22):
            pts2[j, 0] = int(round(cx + rel[j, 0] * scale))
            pts2[j, 1] = int(round(cy - rel[j, 1] * scale))

        for child, parent in _BODY22_LIMBS:
            color = (255, 190, 90) if child % 2 == 1 else (110, 235, 255)
            cv2.line(bgr, tuple(pts2[parent]), tuple(pts2[child]), color, 2, cv2.LINE_AA)
        for j in range(22):
            cv2.circle(bgr, tuple(pts2[j]), 3, (255, 255, 255), -1, cv2.LINE_AA)

    def get_status(self) -> dict[str, Any]:
        client = self._plugin_client
        self.telemetry.plugin_api_connected = client is not None and client.is_connected
        return {
            "running": self.is_running,
            "config": self.config.to_dict(),
            "presets": _PRESET_DETAILS,
            "telemetry": vars(self.telemetry).copy(),
            "models": check_models_status(),
        }
