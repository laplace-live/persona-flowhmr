"""Multi-threaded streaming mocap engine: Source -> Perception -> MMDiT -> Stitcher -> Playout."""

from __future__ import annotations

import contextlib
import ctypes
import math
import queue
import sys
import threading
import time
from collections import deque
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from .camera_tracker import (
    BackgroundVisualOdometry,
    FrameCameraObservation,
    ShotCutDetector,
    StreamingCameraSolver,
    camera_to_vmc_transform,
    yup_rel_to_opencv_camera_r,
)
from .config import (
    CONFIG_FIELDS,
    LATENCY_PRESETS,
    MODEL_FPS,
    EngineConfig,
    LatencyPreset,
    sanitize_avatar_scale,
    sanitize_camera_fov_deg,
)
from .motion_export import MotionRecording
from .paths import FLOWHMR_CFG_PATH, FLOWHMR_CKPT_PATH, MOTION_STATS_PATH, YOLOX_CKPT_PATH, is_sam3d_available
from .perception import (
    BatchedYoloxTracker,
    ReplayTokenExtractor,
    Sam3DStreamExtractor,
    camera_intrinsics,
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
    camera_distance_m: float = 0.0
    camera_azimuth_deg: float = 0.0
    camera_elevation_deg: float = 0.0
    camera_fov_deg: float = 0.0
    camera_cuts_detected: int = 0
    plugin_api_camera_syncs: int = 0


# Settings applied on the fly; changing any other config field restarts a running stream.
_LIVE_FIELDS = frozenset(
    {
        "camera_index",
        "playout_fps",
        "mirror",
        "follow_movement",
        "camera_tracking_enabled",
        "camera_avatar_scale",
        "camera_fov_deg",
        "vmc_enabled",
        "vmc_host",
        "vmc_port",
        "vmc_send_camera",
        "plugin_api_enabled",
        "plugin_api_url",
        "plugin_api_token",
        "plugin_api_inject_params",
        "plugin_api_auto_configure_vmc",
        "plugin_api_sync_camera",
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
class _PendingFrameMeta:
    """Per-frame tracking & optional VO metadata awaiting keyframe token/camera interpolation."""

    sample: SampledFrame
    bbox_xyxy: np.ndarray | None = None
    cam_k: np.ndarray | None = None
    image_hw: tuple[int, int] = (0, 0)
    r_vo_yup: np.ndarray | None = None
    vo_inliers: int = 0
    trans_flow_px: float = 0.0
    is_cut: bool = False
    shot_id: int = 0

    def to_feature_item(
        self,
        sam_token: torch.Tensor,
        sam_cam_t: np.ndarray | None = None,
        sam_j3d_cam: np.ndarray | None = None,
    ) -> FrameFeatureItem:
        cam_obs = None
        if (
            sam_cam_t is not None
            and sam_j3d_cam is not None
            and self.bbox_xyxy is not None
            and self.cam_k is not None
            and self.r_vo_yup is not None
        ):
            cam_obs = FrameCameraObservation(
                sam_cam_t=sam_cam_t,
                sam_j3d_cam=sam_j3d_cam,
                cam_k=self.cam_k,
                image_hw=self.image_hw,
                bbox_xyxy=self.bbox_xyxy,
                r_vo_yup=self.r_vo_yup,
                vo_inliers=self.vo_inliers,
                is_cut=self.is_cut,
                shot_id=self.shot_id,
                trans_flow_px=self.trans_flow_px,
            )
        return FrameFeatureItem(
            frame_index=self.sample.frame_index,
            capture_mono_s=self.sample.capture_mono_s,
            sam_token=sam_token,
            is_stream_reset=self.sample.is_stream_reset,
            shot_id=self.shot_id,
            is_cut=self.is_cut,
            cam_obs=cam_obs,
        )


@dataclass
class FrameFeatureItem:
    """Per-frame SAM conditioning token + optional camera cues kept in the GPU worker's sliding-window history."""

    frame_index: int
    capture_mono_s: float
    sam_token: torch.Tensor  # (SAM_TOKEN_DIM,) CPU float32
    is_stream_reset: bool
    shot_id: int = 0
    is_cut: bool = False
    cam_obs: FrameCameraObservation | None = None


def _plan_keyframes(
    frame_indices: list[int],
    reset_flags: list[bool],
    prev_kf_idx: int | None,
    stride: int,
) -> list[int]:
    """Batch positions that get a SAM keyframe: every `stride` frames, at resets/cuts, and the batch's last frame."""
    positions: list[int] = []
    last_kf = prev_kf_idx
    for pos, (f_idx, is_reset) in enumerate(zip(frame_indices, reset_flags)):
        if is_reset or last_kf is None or (f_idx - last_kf) >= stride:
            positions.append(pos)
            last_kf = f_idx

    last_pos = len(frame_indices) - 1
    if last_pos >= 0 and last_pos not in positions:
        # Keep batch boundary phase-locked across video loop resets/cuts without adding padding.
        if positions and not reset_flags[positions[-1]]:
            positions[-1] = last_pos
        else:
            positions.append(last_pos)
    return positions


class _KeyframeInterpolator:
    """Fills frames between sparse SAM keyframes by lerping tokens (and optional 3D camera joints) within each shot."""

    def __init__(self) -> None:
        self.prev_kf_idx: int | None = None
        self._prev_kf: tuple[torch.Tensor, np.ndarray | None, np.ndarray | None] | None = None
        self._pending: list[_PendingFrameMeta] = []

    def push(
        self,
        meta: _PendingFrameMeta,
        kf_data: tuple[torch.Tensor, np.ndarray | None, np.ndarray | None] | None,
        *,
        is_reset: bool,
    ) -> list[FrameFeatureItem]:
        """Add one frame (`kf_data` set for keyframes); returns the history items it completes."""
        out: list[FrameFeatureItem] = []
        if is_reset:
            # Flush pre-cut intermediate frames without blending across a loop rewind or shot cut.
            out += self.flush()
            self.prev_kf_idx = None
            self._prev_kf = None

        if kf_data is None:
            self._pending.append(meta)
            return out

        tok, cam_t, j3d = kf_data
        if self._prev_kf is not None and self.prev_kf_idx is not None and self._pending:
            p_tok, p_cam_t, p_j3d = self._prev_kf
            have_cam = (
                p_cam_t is not None
                and cam_t is not None
                and p_j3d is not None
                and j3d is not None
            )
            span = max(1, meta.sample.frame_index - self.prev_kf_idx)
            for p in self._pending:
                alpha = float(p.sample.frame_index - self.prev_kf_idx) / float(span)
                lerp_cam_t = ((1.0 - alpha) * p_cam_t + alpha * cam_t).astype(np.float32) if have_cam else None
                lerp_j3d = ((1.0 - alpha) * p_j3d + alpha * j3d).astype(np.float32) if have_cam else None
                out.append(p.to_feature_item(torch.lerp(p_tok, tok, alpha), lerp_cam_t, lerp_j3d))
            self._pending.clear()
        out.append(meta.to_feature_item(tok, cam_t, j3d))
        self.prev_kf_idx = meta.sample.frame_index
        self._prev_kf = (tok, cam_t, j3d)
        return out

    def flush(self) -> list[FrameFeatureItem]:
        """Emit pending frames holding the last keyframe token/camera (at cuts/resets and end of stream)."""
        if not self._pending or self._prev_kf is None:
            return []
        p_tok, p_cam_t, p_j3d = self._prev_kf
        out = [
            p.to_feature_item(
                p_tok.clone(),
                p_cam_t.copy() if p_cam_t is not None else None,
                p_j3d.copy() if p_j3d is not None else None,
            )
            for p in self._pending
        ]
        self._pending.clear()
        return out


def _extract_dense(
    extractor: Sam3DStreamExtractor | ReplayTokenExtractor,
    rgb_list: list[np.ndarray],
    bboxes: np.ndarray,
    cam_k: np.ndarray,
    stream_resets: list[bool],
) -> torch.Tensor:
    """Tokens for every frame of a batch; replay extractors restart cursor at stream resets."""
    if not isinstance(extractor, ReplayTokenExtractor) or not any(stream_resets):
        return extractor.extract_chunk(rgb_list, bboxes, cam_k)
    pieces: list[torch.Tensor] = []
    seg_start = 0
    for i, is_reset in enumerate(stream_resets):
        if not is_reset:
            continue
        if i > seg_start:
            pieces.append(extractor.extract_chunk(rgb_list[seg_start:i], bboxes[seg_start:i], cam_k[seg_start:i]))
            seg_start = i
        extractor.reset()
    pieces.append(extractor.extract_chunk(rgb_list[seg_start:], bboxes[seg_start:], cam_k[seg_start:]))
    return torch.cat(pieces, dim=0)


def _extract_dense_with_camera(
    extractor: Sam3DStreamExtractor | ReplayTokenExtractor,
    rgb_list: list[np.ndarray],
    bboxes: np.ndarray,
    cam_k: np.ndarray,
    stream_resets: list[bool],
) -> tuple[torch.Tensor, np.ndarray, np.ndarray]:
    """Tokens + camera cues for every frame of a batch; replay extractors restart cursor at stream resets."""
    if not isinstance(extractor, ReplayTokenExtractor) or not any(stream_resets):
        return extractor.extract_chunk_with_camera(rgb_list, bboxes, cam_k)
    tok_pieces: list[torch.Tensor] = []
    ct_pieces: list[np.ndarray] = []
    j3_pieces: list[np.ndarray] = []
    seg_start = 0
    for i, is_reset in enumerate(stream_resets):
        if not is_reset:
            continue
        if i > seg_start:
            t_p, c_p, j_p = extractor.extract_chunk_with_camera(rgb_list[seg_start:i], bboxes[seg_start:i], cam_k[seg_start:i])
            tok_pieces.append(t_p)
            ct_pieces.append(c_p)
            j3_pieces.append(j_p)
            seg_start = i
        extractor.reset()
    t_p, c_p, j_p = extractor.extract_chunk_with_camera(rgb_list[seg_start:], bboxes[seg_start:], cam_k[seg_start:])
    tok_pieces.append(t_p)
    ct_pieces.append(c_p)
    j3_pieces.append(j_p)
    return torch.cat(tok_pieces, dim=0), np.concatenate(ct_pieces, axis=0), np.concatenate(j3_pieces, axis=0)


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
        # One `(key, jpeg)` tuple read and replaced whole, so concurrent MJPEG renders never mix halves.
        self._entry: tuple[Any, bytes] | None = None

    def clear(self) -> None:
        self._entry = None

    def get(self, key: Any, draw: Callable[[], np.ndarray | None]) -> bytes | None:
        entry = self._entry
        if entry is not None and entry[0] == key:
            return entry[1]
        bgr = draw()
        if bgr is None:
            return None
        ok, buf = cv2.imencode(".jpg", bgr, _JPEG_PARAMS)
        if not ok:
            return None
        data = buf.tobytes()
        self._entry = (key, data)
        return data


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
        # Public start/stop/settings requests can arrive from different dashboard tabs.
        # Workers only take _lock, including when finishing while stop() joins them.
        self._lifecycle_lock = threading.RLock()
        self.telemetry = EngineTelemetry()
        self.recording = MotionRecording()
        self._recording_available = False
        self._video_result: str | None = None

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
        self._first_commit_event: threading.Event | None = None
        self._queue_event = threading.Event()
        self._threads: list[threading.Thread] = []
        self._peeked_frame: SampledFrame | None = None
        self._evicted_lock = threading.Lock()
        self._evicted_switch_min_frame: int | None = None
        self._evicted_reset_min_frame: int | None = None
        self._last_emitted_shot_id: int | None = None
        self._consumed_boundary_frame: int | None = None
        self._prev_commit_cam: bool = False

    @property
    def is_running(self) -> bool:
        return any(t.is_alive() for t in self._threads)

    def begin_recording(self) -> None:
        with self._lock:
            if self._video_result == "recording":
                raise ValueError("A full-video recording is in progress.")
            if not self._recording_available:
                raise ValueError("Start a stream before recording.")
            self._video_result = None
            self.recording.start()

    def finish_recording(self) -> None:
        with self._lock:
            if self._video_result == "recording":
                raise ValueError("Full-video recording finishes at the end of the file. Use Stop to cancel.")
            self.recording.stop()

    def _close_recording(self) -> None:
        with self._lock:
            self._recording_available = False
            self.recording.stop()
            if self._video_result == "recording":
                if self.telemetry.state == "error":
                    self._video_result = "error"
                elif self._stop_event.is_set():
                    self._video_result = "cancelled"
                elif not self.telemetry.frames_captured:
                    self.telemetry.state = self._video_result = "error"
                    self.telemetry.error = "The video contained no readable frames."
                elif self.recording.status()["frames"] != self.telemetry.frames_captured:
                    self.telemetry.state = self._video_result = "error"
                    self.telemetry.error = "The recording ended before all captured poses were committed."
                else:
                    self._video_result = "complete"
                    self.telemetry.state = "stopped"

    def update_live_settings(self, patch: dict[str, Any]) -> dict[str, Any]:
        """Apply a (partial) config patch: live fields take effect immediately, others restart a running stream."""
        with self._lifecycle_lock:
            if self._video_result == "recording" or (self.is_running and self._stop_event.is_set()):
                raise ValueError("Wait for the current recording or shutdown to finish before changing settings.")
            return self._apply_settings(patch)

    def _apply_settings(self, patch: dict[str, Any]) -> dict[str, Any]:
        was_running = self.is_running
        changed: set[str] = set()

        with self._lock:
            for k, v in patch.items():
                if k not in CONFIG_FIELDS:
                    continue
                if k == "plugin_api_token" and v == "" and not patch.get("clear_plugin_api_token", False):
                    # Empty token from redacted UI form keeps existing token unless explicitly cleared.
                    continue
                if k == "camera_avatar_scale":
                    v = sanitize_avatar_scale(v)
                elif k == "camera_fov_deg":
                    v = sanitize_camera_fov_deg(v)
                if getattr(self.config, k) != v:
                    setattr(self.config, k, v)
                    changed.add(k)

            if not self.config.camera_tracking_enabled:
                self.telemetry.camera_distance_m = 0.0
                self.telemetry.camera_azimuth_deg = 0.0
                self.telemetry.camera_elevation_deg = 0.0
                self.telemetry.camera_fov_deg = 0.0

            if self._vmc_sender is not None:
                self._vmc_sender.set_target(self.config.vmc_host, int(self.config.vmc_port))

            if changed and self._settings_path is not None:
                with contextlib.suppress(OSError):
                    self.config.save(self._settings_path)

        if was_running and (changed - _LIVE_FIELDS):
            self._stop_workers(wait=True)
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
        if self.config.plugin_api_auto_configure_vmc:
            self._configure_plugin_receiver()
        elif client.is_connected and self.config.plugin_api_sync_camera:
            client.refresh_vrm_placement()

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

    def start(self, *, realtime_source: bool = True, playout: bool = True) -> None:
        """Start capture, GPU sliding-window inference, and 30/60 Hz VMC playout threads."""
        with self._lifecycle_lock:
            if self._video_result == "recording":
                raise ValueError("A full-video recording is in progress.")
            if self.is_running:
                if self._stop_event.is_set():
                    raise ValueError("The previous stream is still stopping.")
                return
            self._start_workers(realtime_source=realtime_source, playout=playout, full_video=False)

    def record_video(self) -> None:
        """Record one complete video at source timestamps, without wall-clock playout."""
        with self._lifecycle_lock:
            if self.is_running:
                raise ValueError("Stop the current stream before recording a full video.")
            if self.config.source_mode != "video" or not Path(self.config.video_path).is_file():
                raise ValueError("Choose an existing video file to record.")
            self._start_workers(realtime_source=False, playout=False, full_video=True)

    def _start_workers(self, *, realtime_source: bool, playout: bool, full_video: bool) -> None:
        self._stop_event.clear()
        self._preview.clear()
        self._playout_buffer.clear()
        self._latest_emitted_pose = None
        self._frame0_mono = 0.0
        self._peeked_frame = None
        self._last_emitted_shot_id = None
        self._consumed_boundary_frame = None
        self._prev_commit_cam = False
        with self._evicted_lock:
            self._evicted_switch_min_frame = None
            self._evicted_reset_min_frame = None
            self._queue_event.clear()
        self._live_jpeg.clear()
        self._synced_jpeg.clear()
        self.telemetry = EngineTelemetry(
            state="warming_up",
            target_latency_ms=self.config.get_preset().playout_delay_s * 1000.0,
        )

        if playout:
            if self._vmc_sender is None:
                self._vmc_sender = VmcUdpSender(self.config.vmc_host, self.config.vmc_port)
            else:
                self._vmc_sender.set_target(self.config.vmc_host, self.config.vmc_port)
            self._sync_plugin_client()

        preset = self.config.get_preset()
        sam_stride = max(1, self.config.effective_sam_stride())
        queue_cap = max(24, min(42, preset.sam_chunk_size * sam_stride + 18))
        frame_queue: queue.Queue[SampledFrame | None] = queue.Queue(maxsize=queue_cap)
        warmup_done_event = threading.Event()
        first_commit_event = threading.Event()
        self._threads = [
            threading.Thread(
                target=self._capture_worker,
                args=(frame_queue, warmup_done_event, realtime_source, False if full_video else self.config.loop_video),
                name="mocap-capture",
                daemon=True,
            ),
            threading.Thread(
                target=self._gpu_worker,
                args=(frame_queue, warmup_done_event, first_commit_event),
                name="mocap-gpu",
                daemon=True,
            ),
        ]
        if playout:
            self._threads.append(
                threading.Thread(
                    target=self._playout_worker,
                    args=(first_commit_event, realtime_source),
                    name="mocap-playout",
                    daemon=True,
                )
            )
        with self._lock:
            self._recording_available = True
            if full_video:
                self.recording.start()
                self._video_result = "recording"
        self._first_commit_event = first_commit_event
        for t in self._threads:
            t.start()

    def stop(self) -> None:
        with self._lifecycle_lock:
            self._stop_workers()

    def _stop_workers(self, *, wait: bool = False) -> None:
        self._stop_event.set()
        self._close_recording()
        self._queue_event.set()
        if self._first_commit_event is not None:
            self._first_commit_event.set()
        for t in self._threads:
            if t.is_alive():
                t.join(timeout=None if wait else 3.0)
        self._threads = [thread for thread in self._threads if thread.is_alive()]
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
        """Block until `item` is queued or the engine stops; `drop_oldest` evicts stale frames while preserving switch/reset boundaries."""
        while not self._stop_event.is_set():
            try:
                frame_queue.put(item, timeout=0.1)
                with self._evicted_lock:
                    self._queue_event.set()
                return
            except queue.Full:
                if drop_oldest:
                    with self._evicted_lock, contextlib.suppress(queue.Empty):
                        evicted = frame_queue.get_nowait()
                        if evicted is not None and (evicted.is_camera_switch or evicted.is_stream_reset):
                            ef = int(evicted.frame_index)
                            if evicted.is_camera_switch:
                                self._evicted_switch_min_frame = (
                                    ef
                                    if self._evicted_switch_min_frame is None
                                    else min(self._evicted_switch_min_frame, ef)
                                )
                            if evicted.is_stream_reset:
                                self._evicted_reset_min_frame = (
                                    ef
                                    if self._evicted_reset_min_frame is None
                                    else min(self._evicted_reset_min_frame, ef)
                                )

    def _capture_worker(
        self,
        frame_queue: queue.Queue[SampledFrame | None],
        warmup_done_event: threading.Event,
        realtime_source: bool,
        loop_video: bool,
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
                    loop=loop_video,
                    realtime=realtime_source,
                    target_fps=MODEL_FPS,
                )

            # Never stall live camera capture if GPU is still compiling warmup kernels.
            drop_oldest = self.config.source_mode == "camera"
            capture_rate = _RateMeter()
            while not self._stop_event.is_set():
                if not drop_oldest:
                    # Wait for bounded queue headroom BEFORE reading/stamping the next video frame
                    # so `capture_mono_s` never ages inside a stalled queue.
                    while frame_queue.full() and not self._stop_event.is_set():
                        time.sleep(0.004)
                    if self._stop_event.is_set():
                        break

                if isinstance(source, CameraSource) and source.camera_index != int(self.config.camera_index):
                    try:
                        source.switch_device(
                            self.config.camera_index,
                            width=self.config.camera_width,
                            height=self.config.camera_height,
                            target_fps=MODEL_FPS,
                        )
                        if self.telemetry.error.startswith("Camera switch failed:"):
                            self.telemetry.error = ""
                    except Exception as exc:
                        # Keep streaming from the active camera if the newly selected device fails to open.
                        self.config.camera_index = source.camera_index
                        self.telemetry.error = f"Camera switch failed: {exc}"
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
        """Collect up to `goal` same-resolution frames; the flag reports whether the end-of-stream sentinel arrived."""
        batch: list[SampledFrame] = []
        if self._peeked_frame is not None:
            batch.append(self._peeked_frame)
            self._peeked_frame = None
        while len(batch) < goal and not self._stop_event.is_set():
            with self._evicted_lock:
                try:
                    item = frame_queue.get_nowait()
                except queue.Empty:
                    self._queue_event.clear()
                    empty = True
                    item = None
                    carry_sw = False
                    carry_rst = False
                else:
                    empty = False
                    if item is not None:
                        carry_sw = (
                            self._evicted_switch_min_frame is not None
                            and item.frame_index >= self._evicted_switch_min_frame
                        )
                        carry_rst = (
                            self._evicted_reset_min_frame is not None
                            and item.frame_index >= self._evicted_reset_min_frame
                        )
                        if carry_sw:
                            self._evicted_switch_min_frame = None
                        if carry_rst:
                            self._evicted_reset_min_frame = None
                    else:
                        carry_sw = False
                        carry_rst = False
            if empty:
                self._queue_event.wait(0.05)
                continue
            if item is None:
                return batch, True
            if carry_sw or carry_rst:
                item = replace(
                    item,
                    is_camera_switch=item.is_camera_switch or carry_sw,
                    is_stream_reset=item.is_stream_reset or carry_rst,
                )
            if batch and item.rgb.shape != batch[0].rgb.shape:
                # Split batch cleanly at a live camera resolution change so stacked tensors stay homogeneous.
                self._peeked_frame = item
                return batch, False
            batch.append(item)
        return batch, False

    def _gpu_worker(
        self,
        frame_queue: queue.Queue[SampledFrame | None],
        warmup_done_event: threading.Event,
        first_commit_event: threading.Event,
    ) -> None:
        vo_pool: ThreadPoolExecutor | None = None
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
                extractor.warmup(include_camera=bool(self.config.camera_tracking_enabled))
            self._solver.warmup(preset.window_frames, steps=2)
            if self._stop_event.is_set():
                warmup_done_event.set()
                return

            self.telemetry.state = "buffering"
            warmup_done_event.set()
            stitcher = StreamingStitcher(self._decoder, fps=MODEL_FPS)
            cut_detector: ShotCutDetector | None = None
            vo_tracker: BackgroundVisualOdometry | None = None
            camera_solver: StreamingCameraSolver | None = None
            stream_frame0_hips: list[np.ndarray | None] = [None]
            shot_id = 0
            prev_track_cam = bool(self.config.camera_tracking_enabled)
            if prev_track_cam:
                cut_detector = ShotCutDetector()
                vo_tracker = BackgroundVisualOdometry()
                camera_solver = StreamingCameraSolver(
                    avatar_scale=sanitize_avatar_scale(self.config.camera_avatar_scale),
                    calibrated_fov_deg=sanitize_camera_fov_deg(self.config.camera_fov_deg),
                )
                vo_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mocap-vo")

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
                    track_cam = bool(self.config.camera_tracking_enabled)
                    toggled_cam_on = track_cam and not prev_track_cam
                    prev_track_cam = track_cam

                    rgb_list = [s.rgb for s in batch]
                    stream_resets = [s.is_stream_reset for s in batch]
                    h, w = rgb_list[0].shape[:2]
                    fov_cfg = sanitize_camera_fov_deg(self.config.camera_fov_deg)
                    cam_k0 = camera_intrinsics(h, w, fov_deg=fov_cfg)
                    cam_k = np.repeat(cam_k0[None], len(batch), axis=0)

                    cut_flags: list[bool] = []
                    shot_ids: list[int] = []
                    if track_cam:
                        if cut_detector is None or vo_tracker is None or camera_solver is None:
                            cut_detector = ShotCutDetector()
                            vo_tracker = BackgroundVisualOdometry()
                            camera_solver = StreamingCameraSolver(
                                avatar_scale=sanitize_avatar_scale(self.config.camera_avatar_scale),
                                calibrated_fov_deg=fov_cfg,
                            )
                            if vo_pool is None:
                                vo_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mocap-vo")
                        elif toggled_cam_on:
                            cut_detector.reset()
                            vo_tracker.reset()
                            camera_solver.reset(keep_origin=True)
                        if stream_frame0_hips[0] is not None:
                            camera_solver.anchor_stream_origin(stream_frame0_hips[0])

                        for idx_in_b, s in enumerate(batch):
                            if s.is_stream_reset:
                                cut_detector.reset()
                                is_cut = False
                                shot_id += 1
                            else:
                                force = s.is_camera_switch or (toggled_cam_on and idx_in_b == 0)
                                is_cut = cut_detector.step(s.rgb, force_cut=force)
                                if is_cut:
                                    shot_id += 1
                                    if not (toggled_cam_on and idx_in_b == 0):
                                        self.telemetry.camera_cuts_detected += 1
                            cut_flags.append(is_cut)
                            shot_ids.append(shot_id)
                        cut_or_reset = [r or c for r, c in zip(stream_resets, cut_flags)]
                    else:
                        for s in batch:
                            if s.is_stream_reset or s.is_camera_switch:
                                if cut_detector is not None:
                                    cut_detector.reset()
                                if vo_tracker is not None:
                                    vo_tracker.reset()
                                shot_id += 1
                            cut_flags.append(bool(s.is_camera_switch))
                            shot_ids.append(shot_id)
                        cut_or_reset = [s.is_stream_reset or s.is_camera_switch for s in batch]

                    t0 = time.perf_counter()
                    bboxes = yolox.detect_and_track_batch(rgb_list, reset_flags=cut_or_reset)
                    if sync_cuda:
                        torch.cuda.synchronize()
                    t1 = time.perf_counter()

                    if track_cam:
                        assert vo_tracker is not None and vo_pool is not None
                        allow_zoom = fov_cfg <= 0.0
                        cur_zoom = float(vo_tracker.zoom_scale) if allow_zoom else 1.0
                        for idx_f, bnd in enumerate(cut_or_reset):
                            if bnd:
                                cur_zoom = 1.0
                            if abs(cur_zoom - 1.0) > 1e-4:
                                cam_k[idx_f, 0, 0] *= cur_zoom
                                cam_k[idx_f, 1, 1] *= cur_zoom
                        cam_k_pre = cam_k.copy()

                        def _step_vo_batch(
                            vt: BackgroundVisualOdometry = vo_tracker,
                            b_frames: list[SampledFrame] = batch,
                            b_boxes: np.ndarray = bboxes,
                            base_k0: np.ndarray = cam_k0,
                            b_bounds: list[bool] = cut_or_reset,
                            z_allow: bool = allow_zoom,
                        ) -> tuple[list[np.ndarray], list[int], list[float], np.ndarray]:
                            rots: list[np.ndarray] = []
                            inls: list[int] = []
                            flows: list[float] = []
                            ks = np.repeat(base_k0[None], len(b_frames), axis=0)
                            for idx_f, (sf, bx, bnd) in enumerate(zip(b_frames, b_boxes, b_bounds)):
                                step_k = base_k0.copy()
                                if not bnd and z_allow and abs(vt.zoom_scale - 1.0) > 1e-4:
                                    step_k[0, 0] *= vt.zoom_scale
                                    step_k[1, 1] *= vt.zoom_scale
                                r_v, n_i = vt.step(sf.rgb, bx, step_k, is_cut=bnd, allow_zoom=z_allow)
                                rots.append(r_v)
                                inls.append(n_i)
                                flows.append(float(vt.last_trans_flow_px))
                                if z_allow and abs(vt.zoom_scale - 1.0) > 1e-4:
                                    ks[idx_f, 0, 0] = base_k0[0, 0] * vt.zoom_scale
                                    ks[idx_f, 1, 1] = base_k0[1, 1] * vt.zoom_scale
                            return rots, inls, flows, ks

                        # Run CPU OpenCV optical-flow VO concurrently while the GPU executes SAM 3D Body.
                        vo_future = vo_pool.submit(_step_vo_batch)

                        if keyframe_mode:
                            kf_pos = _plan_keyframes(
                                [s.frame_index for s in batch],
                                cut_or_reset,
                                keyframes.prev_kf_idx,
                                sam_stride,
                            )
                            kf_tokens, kf_cam_t, kf_j3d = extractor.extract_chunk_with_camera(
                                [rgb_list[i] for i in kf_pos],
                                bboxes[kf_pos],
                                cam_k_pre[kf_pos],
                            )
                            vo_rots, vo_inliers, vo_flows, cam_k = vo_future.result()
                            if len(kf_pos) > 0:
                                old_tz = kf_cam_t[:, 2].copy()
                                new_tz = old_tz * (cam_k[kf_pos, 0, 0] / np.maximum(cam_k_pre[kf_pos, 0, 0], 1e-6))
                                delta_tz = new_tz - old_tz
                                kf_cam_t[:, 2] = new_tz
                                kf_j3d[:, :, 2] += delta_tz[:, None]
                        else:
                            tokens, cam_ts, j3ds = _extract_dense_with_camera(
                                extractor,
                                rgb_list,
                                bboxes,
                                cam_k_pre,
                                stream_resets,
                            )
                            vo_rots, vo_inliers, vo_flows, cam_k = vo_future.result()
                            if len(cam_ts) > 0:
                                old_tz = cam_ts[:, 2].copy()
                                new_tz = old_tz * (cam_k[:, 0, 0] / np.maximum(cam_k_pre[:, 0, 0], 1e-6))
                                delta_tz = new_tz - old_tz
                                cam_ts[:, 2] = new_tz
                                j3ds[:, :, 2] += delta_tz[:, None]
                    else:
                        if keyframe_mode:
                            kf_pos = _plan_keyframes(
                                [s.frame_index for s in batch],
                                cut_or_reset,
                                keyframes.prev_kf_idx,
                                sam_stride,
                            )
                            kf_tokens = extractor.extract_chunk(
                                [rgb_list[i] for i in kf_pos],
                                bboxes[kf_pos],
                                cam_k[kf_pos],
                            )
                        else:
                            tokens = _extract_dense(extractor, rgb_list, bboxes, cam_k, stream_resets)

                    if sync_cuda:
                        torch.cuda.synchronize()
                    t2 = time.perf_counter()

                    metas: list[_PendingFrameMeta] = []
                    for i, (s, bbox) in enumerate(zip(batch, bboxes)):
                        self._preview.record_bbox(s.frame_index, bbox)
                        if track_cam:
                            metas.append(
                                _PendingFrameMeta(
                                    sample=s,
                                    bbox_xyxy=bbox.copy(),
                                    cam_k=cam_k[i].copy(),
                                    image_hw=(h, w),
                                    r_vo_yup=vo_rots[i],
                                    vo_inliers=vo_inliers[i],
                                    trans_flow_px=vo_flows[i],
                                    is_cut=cut_flags[i],
                                    shot_id=shot_ids[i],
                                )
                            )
                        else:
                            metas.append(
                                _PendingFrameMeta(
                                    sample=s,
                                    is_cut=cut_flags[i],
                                    shot_id=shot_ids[i],
                                )
                            )

                    if keyframe_mode:
                        if track_cam:
                            kf_at: dict[int, tuple[torch.Tensor, np.ndarray | None, np.ndarray | None]] = {
                                pos: (kf_tokens[k], kf_cam_t[k], kf_j3d[k])
                                for k, pos in enumerate(kf_pos)
                            }
                        else:
                            kf_at = {pos: (kf_tokens[k], None, None) for k, pos in enumerate(kf_pos)}
                        for i, meta in enumerate(metas):
                            history.extend(keyframes.push(meta, kf_at.get(i), is_reset=cut_or_reset[i]))
                    else:
                        if track_cam:
                            for i, meta in enumerate(metas):
                                history.extend([meta.to_feature_item(tokens[i], cam_ts[i], j3ds[i])])
                        else:
                            for i, meta in enumerate(metas):
                                history.extend([meta.to_feature_item(tokens[i])])

                    self.telemetry.yolox_ms_per_frame = round((t1 - t0) * 1000.0 / len(batch), 2)
                    self.telemetry.sam_ms_per_frame = round((t2 - t1) * 1000.0 / len(batch), 2)

                if eos_reached:
                    history.extend(keyframes.flush())
                if history:
                    self.telemetry.frames_perceived = history[-1].frame_index + 1

                if camera_solver is not None:
                    camera_solver.avatar_scale = sanitize_avatar_scale(self.config.camera_avatar_scale)
                    camera_solver.calibrated_fov_deg = sanitize_camera_fov_deg(self.config.camera_fov_deg)
                next_emit_frame = self._solve_ready_windows(
                    history,
                    next_emit_frame,
                    eos=eos_reached,
                    preset=preset,
                    stitcher=stitcher,
                    camera_solver=camera_solver,
                    stream_frame0_hips=stream_frame0_hips,
                    first_commit_event=first_commit_event,
                )
        except Exception as exc:
            self.telemetry.state = "error"
            self.telemetry.error = str(exc)
            self._stop_event.set()
            warmup_done_event.set()
            first_commit_event.set()
        finally:
            self._close_recording()
            if vo_pool is not None:
                vo_pool.shutdown(wait=True)

    def _solve_ready_windows(
        self,
        history: deque[FrameFeatureItem],
        next_emit_frame: int,
        *,
        eos: bool,
        preset: LatencyPreset,
        stitcher: StreamingStitcher,
        camera_solver: StreamingCameraSolver | None,
        stream_frame0_hips: list[np.ndarray | None],
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

            by_idx = {it.frame_index: it for it in history}
            first_item = history[0]
            exact_emit = by_idx.get(next_emit_frame)
            forward_gap_boundary = False
            if exact_emit is not None:
                emit_item = exact_emit
            else:
                prev_items = [it for it in history if it.frame_index < next_emit_frame]
                if prev_items:
                    # Keep dropped-frame gaps in the preceding shot until the surviving cut frame arrives.
                    emit_item = prev_items[-1]
                else:
                    emit_item = first_item
                    forward_gap_boundary = self._consumed_boundary_frame != emit_item.frame_index and (
                        (self._last_emitted_shot_id is not None and emit_item.shot_id != self._last_emitted_shot_id)
                        or emit_item.is_cut
                        or emit_item.is_stream_reset
                    )
                    if forward_gap_boundary:
                        self._consumed_boundary_frame = emit_item.frame_index
            target_shot_id = emit_item.shot_id

            stride = preset.stride_frames
            if eos:
                stride = min(stride, max(0, latest_available - next_emit_frame + 1))
                if stride == 0:
                    break

            # End the current stride cleanly right before any hard shot cut or loop reset so
            # neither MMDiT window nor overlap crossfade smears across two camera angles.
            cut_within_stride = False
            for k in range(1, stride):
                cand = by_idx.get(next_emit_frame + k)
                if cand is not None and (cand.shot_id != target_shot_id or cand.is_stream_reset):
                    stride = k
                    cut_within_stride = True
                    break

            next_after = next((it for it in history if it.frame_index >= next_emit_frame + stride), None)
            boundary_at_stride_end = next_after is not None and (
                next_after.shot_id != target_shot_id or next_after.is_stream_reset
            )

            shot_items = [it for it in history if it.shot_id == target_shot_id]
            shot_first = shot_items[0] if shot_items else first_item

            def _resolve_shot_item(f_idx: int) -> FrameFeatureItem:
                cand = by_idx.get(f_idx)
                if cand is not None and cand.shot_id == target_shot_id:
                    return cand
                prev_shot = [it for it in shot_items if it.frame_index < f_idx]
                return prev_shot[-1] if prev_shot else shot_first

            win_start = next_emit_frame - preset.history_frames
            win_indices = list(range(win_start, win_start + preset.window_frames))
            win_items = [_resolve_shot_item(f) for f in win_indices]
            win_tokens = [it.sam_token for it in win_items]

            stride_items: list[FrameFeatureItem] = []
            for i in range(stride):
                f_idx = next_emit_frame + i
                raw_it = by_idx.get(f_idx)
                if raw_it is not None and raw_it.shot_id == target_shot_id:
                    if self._consumed_boundary_frame == f_idx:
                        adj_cam = replace(raw_it.cam_obs, is_cut=False) if raw_it.cam_obs is not None else None
                        stride_items.append(
                            replace(raw_it, is_cut=False, is_stream_reset=False, cam_obs=adj_cam)
                        )
                    else:
                        stride_items.append(raw_it)
                else:
                    ref_it = _resolve_shot_item(f_idx)
                    synth_cut = bool(
                        i == 0
                        and forward_gap_boundary
                        and (
                            emit_item.is_cut
                            or (
                                self._last_emitted_shot_id is not None
                                and target_shot_id != self._last_emitted_shot_id
                            )
                        )
                    )
                    synth_rst = bool(i == 0 and forward_gap_boundary and emit_item.is_stream_reset)
                    synth_mono = ref_it.capture_mono_s + (f_idx - ref_it.frame_index) / MODEL_FPS
                    synth_cam = replace(ref_it.cam_obs, is_cut=synth_cut) if ref_it.cam_obs is not None else None
                    stride_items.append(
                        FrameFeatureItem(
                            frame_index=f_idx,
                            capture_mono_s=synth_mono,
                            sam_token=ref_it.sam_token,
                            is_stream_reset=synth_rst,
                            shot_id=target_shot_id,
                            is_cut=synth_cut,
                            cam_obs=synth_cam,
                        )
                    )

            track_cam = bool(self.config.camera_tracking_enabled)
            stride_cam_obs = [it.cam_obs for it in stride_items if it.cam_obs is not None]
            have_stride_cam = (
                track_cam
                and camera_solver is not None
                and win_items[0].cam_obs is not None
                and len(stride_cam_obs) == len(stride_items)
            )
            if have_stride_cam and not self._prev_commit_cam and camera_solver is not None:
                camera_solver.reset(keep_origin=True)
            self._prev_commit_cam = have_stride_cam

            if have_stride_cam and camera_solver is not None and win_items[0].cam_obs is not None:
                r_win0 = win_items[0].cam_obs.r_vo_yup
                rel_yup = np.stack(
                    [(it.cam_obs.r_vo_yup if it.cam_obs is not None else r_win0) @ r_win0.T for it in win_items],
                    axis=0,
                )
                cam_r_w_9 = yup_rel_to_opencv_camera_r(rel_yup).reshape(-1, 9)
                r_cut_fb = shot_first.cam_obs.r_vo_yup if shot_first.cam_obs is not None else r_win0
                window_vo_yaw = camera_solver.window_vo_yaw_rad(
                    target_shot_id,
                    r_win0,
                    r_cut_fallback=r_cut_fb,
                )
            else:
                cam_r_w_9 = None
                window_vo_yaw = 0.0

            latent_win = self._solver.solve_window(
                torch.stack(win_tokens, dim=0),
                win_indices,
                ode_steps=preset.ode_steps,
                camera_r_w_9=cam_r_w_9,
            )
            self.telemetry.mmdit_ms_per_window = round(self._solver.last_solve_ms, 2)
            self.telemetry.windows_solved += 1

            if have_stride_cam and camera_solver is not None and stream_frame0_hips[0] is not None:
                camera_solver.anchor_stream_origin(stream_frame0_hips[0])

            committed = stitcher.commit_window(
                latent_win,
                emit_offset_in_window=preset.history_frames,
                stride_frames=stride,
                overlap_frames=preset.overlap_frames,
                first_emit_frame_index=next_emit_frame,
                capture_mono_times=[it.capture_mono_s for it in stride_items],
                reset_flags=[it.is_stream_reset for it in stride_items],
                cut_flags=[it.is_cut for it in stride_items],
                window_vo_yaw_rad=window_vo_yaw if have_stride_cam else 0.0,
                camera_observations=stride_cam_obs if have_stride_cam else None,
                camera_solver=camera_solver if have_stride_cam else None,
                follow_movement=self.config.follow_movement,
                anchor_world_yaw_across_cuts=have_stride_cam,
                save_tail=not (eos or cut_within_stride or boundary_at_stride_end),
            )
            self._last_emitted_shot_id = target_shot_id
            for cf, it in zip(committed, stride_items):
                if it.is_stream_reset or stream_frame0_hips[0] is None:
                    stream_frame0_hips[0] = cf.hips_pos_rh.copy()
                    if camera_solver is not None:
                        camera_solver.anchor_stream_origin(cf.hips_pos_rh)

            self._playout_buffer.push_many(committed)
            if self.recording.active:
                self.recording.append(committed, mirror=self.config.mirror, follow=self.config.follow_movement)
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
            elif was_started and realtime_source and sample.pose.capture_mono_s > 0.0:
                # Keep real-time playout latency locked onto `delay_s`:
                # - If slipped above `delay_s` with healthy buffer lead, smoothly catch up (<= 20% rate).
                # - If capture clock re-anchored forward so latency fell below `delay_s`, gently restore
                #   the full `delay_s` pipeline budget before lead runs dry.
                excess_s = (now_mono - sample.pose.capture_mono_s) - delay_s
                if excess_s > 0.05 and sample.buffer_lead_ms > 200.0:
                    lead_headroom_s = max(0.0, (sample.buffer_lead_ms / 1000.0) - 0.18)
                    max_rate = 0.20 if excess_s > 0.25 else 0.10
                    catchup_s = min(dt_out * max_rate, excess_s - 0.025, lead_headroom_s * 0.25)
                    if catchup_s > 0.0:
                        playout_origin_mono -= catchup_s
                elif excess_s < -0.04 and sample.buffer_lead_ms < 160.0:
                    relax_s = min(dt_out * 0.15, -excess_s)
                    playout_origin_mono += relax_s

            pose = sample.pose
            self._latest_emitted_pose = pose
            self.telemetry.buffer_lead_ms = round(sample.buffer_lead_ms, 1)
            self.telemetry.underruns = self._playout_buffer.underrun_count
            if realtime_source and pose.capture_mono_s > 0.0:
                self.telemetry.measured_latency_ms = round((now_mono - pose.capture_mono_s) * 1000.0, 1)
            else:
                self.telemetry.measured_latency_ms = round(delay_s * 1000.0, 1)

            cam_pose = pose.camera if self.config.camera_tracking_enabled else None
            if cam_pose is not None:
                self.telemetry.camera_distance_m = round(cam_pose.orbit_distance, 2)
                self.telemetry.camera_azimuth_deg = round(math.degrees(cam_pose.orbit_azimuth), 1)
                self.telemetry.camera_elevation_deg = round(math.degrees(cam_pose.orbit_elevation), 1)
                self.telemetry.camera_fov_deg = round(cam_pose.fov_deg, 1)
            elif not self.config.camera_tracking_enabled:
                self.telemetry.camera_distance_m = 0.0
                self.telemetry.camera_azimuth_deg = 0.0
                self.telemetry.camera_elevation_deg = 0.0
                self.telemetry.camera_fov_deg = 0.0

            if self.config.vmc_enabled and self._vmc_sender is not None:
                bones = pose_to_vmc_bones(
                    pose,
                    mirror=self.config.mirror,
                    follow_movement=self.config.follow_movement,
                )
                if cam_pose is not None and self.config.vmc_send_camera:
                    vmc_cam = camera_to_vmc_transform(cam_pose, mirror=self.config.mirror)
                    self._vmc_sender.send_pose(bones, camera=vmc_cam)
                else:
                    self._vmc_sender.send_pose(bones)
                self.telemetry.vmc_packets_sent = self._vmc_sender.packets_sent

            client = self._plugin_client
            if client is not None and client.is_connected and self.config.plugin_api_enabled:
                if cam_pose is not None and self.config.plugin_api_sync_camera:
                    client.sync_camera_pose_fire_and_forget(cam_pose, mirror=self.config.mirror)
                    self.telemetry.plugin_api_camera_syncs = client.camera_patches_sent

                if (
                    self.config.plugin_api_inject_params
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
        cam_enabled = self.config.camera_tracking_enabled

        def draw() -> np.ndarray | None:
            got = self._preview.get_synced_bgr(pose.frame_index)
            if got is None:
                return None
            f_idx, bgr, bbox, orig_hw = got
            self._draw_bbox_overlay(bgr, bbox, orig_hw, label=f"SYNCED PLAYOUT #{f_idx}")
            self._draw_skeleton_hud(bgr, pose)
            out = cv2.flip(bgr, 1) if mirror else bgr
            if cam_enabled and pose.camera is not None:
                self._draw_camera_hud(out, pose.camera, mirror=mirror)
            return out

        return self._synced_jpeg.get(("play", pose.frame_index, round(pose.media_time_s, 4), mirror, cam_enabled), draw)

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

    @staticmethod
    def _draw_camera_hud(bgr: np.ndarray, cam: Any, *, mirror: bool = False) -> None:
        """Render the dynamic 6-DoF stage camera badge in the bottom-left corner of the synced preview."""
        h, w = bgr.shape[:2]
        sign = -1.0 if mirror else 1.0
        az_deg = sign * math.degrees(float(cam.orbit_azimuth))
        el_deg = math.degrees(float(cam.orbit_elevation))
        text = f"CAM d={cam.orbit_distance:.2f}m  az={az_deg:+.1f}d  el={el_deg:+.1f}d  fov={cam.fov_deg:.1f}d"
        if cam.is_cut:
            text += " [CUT]"
        pad_x, box_h = 10, 28
        y0, y1 = max(0, h - box_h - 10), h - 10
        x1 = min(w - 10, pad_x + len(text) * 8 + 14)
        sub = bgr[y0:y1, pad_x:x1]
        if sub.size > 0:
            sub[:] = (sub.astype(np.uint16) * 32 // 100).astype(np.uint8)
        border = (90, 200, 255) if cam.is_cut else (90, 110, 140)
        cv2.rectangle(bgr, (pad_x, y0), (x1, y1), border, 1)
        cv2.putText(bgr, text, (pad_x + 8, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (220, 240, 255), 1, cv2.LINE_AA)

    def get_status(self) -> dict[str, Any]:
        client = self._plugin_client
        self.telemetry.plugin_api_connected = client is not None and client.is_connected
        with self._lock:
            recording = self.recording.status() | {
                "available": self._recording_available,
                "full_video": self._video_result == "recording",
                "video_result": self._video_result if self._video_result != "recording" else None,
            }
        return {
            "running": self.is_running,
            "recording": recording,
            "config": self.config.to_dict(),
            "presets": _PRESET_DETAILS,
            "telemetry": vars(self.telemetry).copy(),
            "models": check_models_status(),
        }
