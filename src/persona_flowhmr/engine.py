"""Multi-threaded streaming mocap engine: Source -> Perception -> MMDiT -> Stitcher -> Playout."""

from __future__ import annotations

import ctypes
import queue
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
import torch

from .config import LATENCY_PRESETS, EngineConfig
from .paths import FLOWHMR_CFG_PATH, FLOWHMR_CKPT_PATH, MOTION_STATS_PATH, YOLOX_CKPT_PATH
from .perception import (
    BatchedYoloxTracker,
    FrameFeatureItem,
    ReplayTokenExtractor,
    Sam3DStreamExtractor,
    default_camera_intrinsics,
    is_sam3d_available,
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


class StreamingMocapEngine:
    """Real-time buffered single-camera mocap engine sending VMC (UDP) and Persona Plugin API."""

    def __init__(self, config: EngineConfig | None = None) -> None:
        _enable_windows_high_res_timer()
        self.config = config or EngineConfig()
        self._lock = threading.Lock()
        self.telemetry = EngineTelemetry()

        self._preview = PreviewRingBuffer(max_items=300, max_width=640)
        self._playout_buffer = PosePlayoutBuffer(max_frames=900)
        self._latest_emitted_pose: CanonicalPoseFrame | None = None
        self._live_jpeg_cache: tuple[Any, bytes] | None = None
        self._synced_jpeg_cache: tuple[Any, bytes] | None = None

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
        """Update live toggles immediately; restart stream if source/preset/backend changed."""
        restart_keys = {
            "source_mode",
            "camera_index",
            "video_path",
            "loop_video",
            "preset",
            "perception_backend",
            "sam_stride",
            "hand_mode",
            "yolo_size",
            "yolo_stride",
            "replay_tokens_path",
            "torch_compile",
        }
        needs_restart = False
        was_running = self.is_running

        with self._lock:
            for k, v in patch.items():
                if not hasattr(self.config, k):
                    continue
                if k == "plugin_api_token" and v == "":
                    # Empty token from redacted UI form keeps existing token unless explicitly cleared.
                    if not patch.get("clear_plugin_api_token", False):
                        continue
                old_val = getattr(self.config, k)
                if old_val != v:
                    setattr(self.config, k, v)
                    if k in restart_keys:
                        needs_restart = True

            if self._vmc_sender is not None:
                self._vmc_sender.set_target(self.config.vmc_host, int(self.config.vmc_port))

        if "plugin_api_enabled" in patch or "plugin_api_url" in patch or "plugin_api_token" in patch:
            self._sync_plugin_client()

        if was_running and needs_restart:
            self.stop()
            self.start()

        return self.get_status()

    def _sync_plugin_client(self) -> None:
        if self._plugin_client is not None:
            self._plugin_client.stop()
            self._plugin_client = None
        if not self.config.plugin_api_enabled:
            self.telemetry.plugin_api_connected = False
            self.telemetry.plugin_api_vmc_source_id = ""
            return

        client = PersonaPluginClient(
            url=self.config.plugin_api_url,
            token=self.config.plugin_api_token,
        )
        connected = client.start(timeout_s=3.0)
        self._plugin_client = client
        self.telemetry.plugin_api_connected = connected
        if connected and self.config.plugin_api_auto_configure_vmc:
            try:
                res = client.ensure_vmc_receiver(
                    vmc_port=self.config.vmc_port,
                    bind_primary_avatar=True,
                    follow_movement=self.config.follow_movement,
                )
                self.telemetry.plugin_api_vmc_source_id = str(res.get("sourceId") or "")
            except Exception as exc:
                client.last_error = str(exc)

    def _ensure_models_loaded(self) -> tuple[Any, str]:
        """Load YOLOX-L, FlowHMR MMDiT, MotionLatentDecoder, and selected SAM/Replay extractor."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        preset = self.config.get_preset()
        yolo_hw = self.config.effective_yolo_size()
        yolo_stride = max(1, int(self.config.yolo_stride))

        if self._yolox is None:
            self._yolox = BatchedYoloxTracker(
                ckpt_path=YOLOX_CKPT_PATH,
                device=device,
                test_size=yolo_hw,
                yolo_stride=yolo_stride,
            )
        else:
            self._yolox.test_size = yolo_hw
            self._yolox.yolo_stride = yolo_stride
            self._yolox.reset_tracker()

        if self._solver is None:
            self._solver = FlowHMRWindowSolver(
                ckpt_path=FLOWHMR_CKPT_PATH,
                cfg_path=FLOWHMR_CFG_PATH,
                device=device,
                use_compile=self.config.torch_compile,
            )
        else:
            self._solver.reset_noise_cache()

        if self._decoder is None:
            self._decoder = MotionLatentDecoder(
                motion_stats_path=MOTION_STATS_PATH,
                motion_style=self._solver.motion_style,
                fps=30.0,
            )

        backend = self.config.perception_backend
        if backend == "replay" or (backend == "auto" and self.config.replay_tokens_path):
            extractor = ReplayTokenExtractor(self.config.replay_tokens_path or None)
            active_backend = "replay" if self.config.replay_tokens_path else "bbox-probe"
        elif backend == "sam3d" or (backend == "auto" and is_sam3d_available()):
            eff_hand_mode = self.config.effective_hand_mode()
            if self._sam3d is None:
                self._sam3d = Sam3DStreamExtractor(
                    device=device,
                    use_compile=self.config.torch_compile,
                    chunk_size=preset.sam_chunk_size,
                    hand_mode=eff_hand_mode,
                    body_decoder_layers=preset.body_decoder_layers,
                    hand_decoder_layers=preset.hand_decoder_layers,
                )
            else:
                self._sam3d.configure(
                    chunk_size=preset.sam_chunk_size,
                    hand_mode=eff_hand_mode,
                    body_decoder_layers=preset.body_decoder_layers,
                    hand_decoder_layers=preset.hand_decoder_layers,
                )
            extractor = self._sam3d
            active_backend = "sam3d"
        else:
            # `auto` without SAM3D checkpoint raises clear actionable instructions.
            raise FileNotFoundError(
                "Checkpoint `facebook/sam-3d-body-dinov3` is not installed yet. "
                "Run `persona-flowhmr setup` to download it automatically from ModelScope "
                "(or select `replay` backend for pipeline testing)."
            )

        return extractor, active_backend

    def start(self, *, realtime_source: bool = True) -> None:
        """Start capture, GPU sliding-window inference, and 30/60 Hz VMC playout threads."""
        if self.is_running:
            return

        self._stop_event.clear()
        self._preview.clear()
        self._playout_buffer.clear()
        self._latest_emitted_pose = None
        self._live_jpeg_cache = None
        self._synced_jpeg_cache = None
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
        stream_anchor: dict[str, float] = {"frame0_mono": 0.0}

        t_cap = threading.Thread(
            target=self._capture_worker,
            args=(frame_queue, warmup_done_event, stream_anchor, realtime_source),
            name="mocap-capture",
            daemon=True,
        )
        t_gpu = threading.Thread(
            target=self._gpu_worker,
            args=(frame_queue, warmup_done_event, first_commit_event, realtime_source),
            name="mocap-gpu",
            daemon=True,
        )
        t_play = threading.Thread(
            target=self._playout_worker,
            args=(first_commit_event, stream_anchor, realtime_source),
            name="mocap-playout",
            daemon=True,
        )
        self._threads = [t_cap, t_gpu, t_play]
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

    def _capture_worker(
        self,
        frame_queue: queue.Queue[SampledFrame | None],
        warmup_done_event: threading.Event,
        stream_anchor: dict[str, float],
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
                    target_fps=30.0,
                )
            else:
                source = VideoFileSource(
                    self.config.video_path,
                    loop=self.config.loop_video,
                    realtime=realtime_source,
                    target_fps=30.0,
                )

            fps_counter = 0
            fps_t0 = time.perf_counter()

            while not self._stop_event.is_set():
                sample = source.read_next()
                if sample is None:
                    break
                if sample.frame_index == 0:
                    stream_anchor["frame0_mono"] = sample.capture_mono_s

                self._preview.record_frame(sample.frame_index, sample.rgb)
                self.telemetry.frames_captured = sample.frame_index + 1

                fps_counter += 1
                elapsed = time.perf_counter() - fps_t0
                if elapsed >= 0.5:
                    self.telemetry.capture_fps = round(fps_counter / elapsed, 1)
                    fps_counter = 0
                    fps_t0 = time.perf_counter()

                while not self._stop_event.is_set():
                    try:
                        frame_queue.put(sample, timeout=0.1)
                        break
                    except queue.Full:
                        if self.config.source_mode == "camera":
                            # Never stall live camera capture if GPU is still compiling warmup kernels.
                            try:
                                frame_queue.get_nowait()
                            except queue.Empty:
                                pass
        except Exception as exc:
            self.telemetry.state = "error"
            self.telemetry.error = f"Capture error: {exc}"
            self._stop_event.set()
        finally:
            if source is not None:
                source.close()
            while not self._stop_event.is_set():
                try:
                    frame_queue.put(None, timeout=0.1)
                    break
                except queue.Full:
                    pass
            try:
                frame_queue.put_nowait(None)
            except queue.Full:
                pass

    def _gpu_worker(
        self,
        frame_queue: queue.Queue[SampledFrame | None],
        warmup_done_event: threading.Event,
        first_commit_event: threading.Event,
        realtime_source: bool,
    ) -> None:
        try:
            extractor, active_backend = self._ensure_models_loaded()
            self.telemetry.active_backend = active_backend
            assert self._yolox is not None
            assert self._solver is not None
            assert self._decoder is not None

            preset = self.config.get_preset()
            kf_chunk = preset.sam_chunk_size
            sam_stride = self.config.effective_sam_stride() if isinstance(extractor, Sam3DStreamExtractor) else 1
            sam_stride = max(1, sam_stride)

            # Pre-compile SAM 3D Body + MMDiT static shapes before buffering starts so live windows
            # never pay Dynamo/Triton compilation latency mid-stream.
            yolo_warm_n = max(1, (kf_chunk * sam_stride + self._yolox.yolo_stride) // self._yolox.yolo_stride)
            self._yolox.warmup(yolo_warm_n)
            if isinstance(extractor, Sam3DStreamExtractor):
                extractor.warmup(kf_chunk)
            self._solver.warmup(preset.window_frames, steps=2)
            if self._stop_event.is_set():
                warmup_done_event.set()
                return

            self.telemetry.state = "buffering"
            warmup_done_event.set()
            stitcher = StreamingStitcher(self._decoder, fps=30.0)
            history: deque[FrameFeatureItem] = deque(maxlen=preset.window_frames + 128)
            next_emit_frame = 0
            eos_reached = False
            prev_kf_idx: int | None = None
            prev_kf_tok: torch.Tensor | None = None
            pending_interm: list[tuple[SampledFrame, np.ndarray]] = []

            while not self._stop_event.is_set() and not eos_reached:
                if isinstance(extractor, Sam3DStreamExtractor) and sam_stride > 1:
                    batch_goal = (1 + (kf_chunk - 1) * sam_stride) if prev_kf_idx is None else (kf_chunk * sam_stride)
                else:
                    needed_frames = (
                        next_emit_frame + preset.stride_frames + preset.overlap_frames + preset.lookahead_frames
                    ) - ((history[-1].frame_index + 1) if history else 0)
                    batch_goal = min(kf_chunk, max(1, needed_frames))

                batch_samples: list[SampledFrame] = []
                while len(batch_samples) < batch_goal and not self._stop_event.is_set():
                    try:
                        item = frame_queue.get(timeout=0.05)
                    except queue.Empty:
                        continue
                    if item is None:
                        eos_reached = True
                        break
                    batch_samples.append(item)

                if batch_samples:
                    rgb_list = [s.rgb for s in batch_samples]
                    reset_flags = [bool(s.is_stream_reset and s.frame_index > 0) for s in batch_samples]
                    h, w = rgb_list[0].shape[:2]
                    cam_k = np.repeat(default_camera_intrinsics(h, w)[None], len(rgb_list), axis=0)

                    t0 = time.perf_counter()
                    bboxes = self._yolox.detect_and_track_batch(rgb_list, reset_flags=reset_flags)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    t1 = time.perf_counter()

                    if isinstance(extractor, Sam3DStreamExtractor) and sam_stride > 1:
                        last_planned_kf = prev_kf_idx
                        kf_indices: list[int] = []
                        for i, s in enumerate(batch_samples):
                            if (
                                reset_flags[i]
                                or last_planned_kf is None
                                or (s.frame_index - last_planned_kf) >= sam_stride
                            ):
                                kf_indices.append(i)
                                last_planned_kf = s.frame_index

                        last_batch_pos = len(batch_samples) - 1
                        if last_batch_pos >= 0 and last_batch_pos not in kf_indices:
                            # Keep batch boundary phase-locked across video loop resets without adding padding.
                            if kf_indices and not reset_flags[kf_indices[-1]]:
                                kf_indices[-1] = last_batch_pos
                            else:
                                kf_indices.append(last_batch_pos)

                        kf_tokens = (
                            extractor.extract_chunk(
                                [rgb_list[i] for i in kf_indices],
                                bboxes[kf_indices],
                                cam_k[kf_indices],
                            )
                            if kf_indices
                            else torch.empty((0, 3072), dtype=torch.float32)
                        )
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                        t2 = time.perf_counter()

                        kf_map = {kf_indices[pos]: kf_tokens[pos] for pos in range(len(kf_indices))}
                        for i, s in enumerate(batch_samples):
                            self._preview.record_bbox(s.frame_index, bboxes[i])
                            if reset_flags[i]:
                                # Flush pre-cut intermediate frames without blending across a loop rewind.
                                if pending_interm and prev_kf_tok is not None:
                                    for p_sample, p_bbox in pending_interm:
                                        history.append(
                                            FrameFeatureItem(
                                                frame_index=p_sample.frame_index,
                                                media_time_s=p_sample.media_time_s,
                                                capture_mono_s=p_sample.capture_mono_s,
                                                bbox_xyxy=p_bbox,
                                                sam_token=prev_kf_tok.clone(),
                                            )
                                        )
                                    pending_interm.clear()
                                prev_kf_idx = None
                                prev_kf_tok = None

                            if i in kf_map:
                                cur_tok = kf_map[i]
                                if prev_kf_tok is not None and prev_kf_idx is not None and pending_interm:
                                    span = max(1, s.frame_index - prev_kf_idx)
                                    for p_sample, p_bbox in pending_interm:
                                        alpha = float(p_sample.frame_index - prev_kf_idx) / float(span)
                                        interp_tok = torch.lerp(prev_kf_tok, cur_tok, alpha)
                                        history.append(
                                            FrameFeatureItem(
                                                frame_index=p_sample.frame_index,
                                                media_time_s=p_sample.media_time_s,
                                                capture_mono_s=p_sample.capture_mono_s,
                                                bbox_xyxy=p_bbox,
                                                sam_token=interp_tok,
                                            )
                                        )
                                    pending_interm.clear()
                                history.append(
                                    FrameFeatureItem(
                                        frame_index=s.frame_index,
                                        media_time_s=s.media_time_s,
                                        capture_mono_s=s.capture_mono_s,
                                        bbox_xyxy=bboxes[i],
                                        sam_token=cur_tok,
                                    )
                                )
                                prev_kf_idx = s.frame_index
                                prev_kf_tok = cur_tok
                            else:
                                pending_interm.append((s, bboxes[i]))
                    else:
                        if isinstance(extractor, ReplayTokenExtractor) and any(reset_flags):
                            pieces: list[torch.Tensor] = []
                            seg_start = 0
                            for i, is_rst in enumerate(reset_flags):
                                if is_rst and i > seg_start:
                                    pieces.append(
                                        extractor.extract_chunk(
                                            rgb_list[seg_start:i],
                                            bboxes[seg_start:i],
                                            cam_k[seg_start:i],
                                        )
                                    )
                                    extractor.reset()
                                    seg_start = i
                                elif is_rst:
                                    extractor.reset()
                            pieces.append(
                                extractor.extract_chunk(
                                    rgb_list[seg_start:],
                                    bboxes[seg_start:],
                                    cam_k[seg_start:],
                                )
                            )
                            tokens = torch.cat(pieces, dim=0)
                        else:
                            tokens = extractor.extract_chunk(rgb_list, bboxes, cam_k)
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                        t2 = time.perf_counter()
                        for i, s in enumerate(batch_samples):
                            self._preview.record_bbox(s.frame_index, bboxes[i])
                            history.append(
                                FrameFeatureItem(
                                    frame_index=s.frame_index,
                                    media_time_s=s.media_time_s,
                                    capture_mono_s=s.capture_mono_s,
                                    bbox_xyxy=bboxes[i],
                                    sam_token=tokens[i],
                                )
                            )

                    n_b = len(batch_samples)
                    self.telemetry.yolox_ms_per_frame = round(((t1 - t0) * 1000.0) / n_b, 2)
                    self.telemetry.sam_ms_per_frame = round(((t2 - t1) * 1000.0) / n_b, 2)
                    if history:
                        self.telemetry.frames_perceived = history[-1].frame_index + 1

                if eos_reached and pending_interm and prev_kf_tok is not None:
                    for p_sample, p_bbox in pending_interm:
                        history.append(
                            FrameFeatureItem(
                                frame_index=p_sample.frame_index,
                                media_time_s=p_sample.media_time_s,
                                capture_mono_s=p_sample.capture_mono_s,
                                bbox_xyxy=p_bbox,
                                sam_token=prev_kf_tok.clone(),
                            )
                        )
                    pending_interm.clear()
                    if history:
                        self.telemetry.frames_perceived = history[-1].frame_index + 1

                # Solve every ready stride window `[next_emit_frame - H .. next_emit_frame + S + O + L - 1]`.
                while history and not self._stop_event.is_set():
                    required_last = (
                        next_emit_frame
                        + preset.stride_frames
                        + preset.overlap_frames
                        + preset.lookahead_frames
                        - 1
                    )
                    latest_available = history[-1].frame_index
                    if latest_available < required_last and not eos_reached:
                        break
                    if eos_reached and latest_available < next_emit_frame:
                        break

                    # Map frame_index -> FrameFeatureItem with left/right boundary clamping so
                    # window 0 (`frame < 0`) and EOS tail maintain fixed length `preset.window_frames`.
                    by_idx = {it.frame_index: it for it in history}
                    first_item = history[0]
                    last_item = history[-1]

                    win_start = next_emit_frame - preset.history_frames
                    win_indices = list(range(win_start, win_start + preset.window_frames))
                    win_tokens: list[torch.Tensor] = []
                    for f_idx in win_indices:
                        if f_idx in by_idx:
                            win_tokens.append(by_idx[f_idx].sam_token)
                        elif f_idx < first_item.frame_index:
                            win_tokens.append(first_item.sam_token)
                        else:
                            win_tokens.append(last_item.sam_token)

                    stacked_tokens = torch.stack(win_tokens, dim=0)
                    latent_win = self._solver.solve_window(
                        stacked_tokens,
                        win_indices,
                        ode_steps=preset.ode_steps,
                    )
                    self.telemetry.mmdit_ms_per_window = round(self._solver.last_solve_ms, 2)
                    self.telemetry.windows_solved += 1

                    actual_stride = preset.stride_frames
                    if eos_reached:
                        actual_stride = min(actual_stride, max(0, latest_available - next_emit_frame + 1))
                        if actual_stride == 0:
                            break

                    cap_times = [
                        (by_idx[next_emit_frame + i].capture_mono_s if (next_emit_frame + i) in by_idx else last_item.capture_mono_s)
                        for i in range(actual_stride)
                    ]
                    committed = stitcher.commit_window(
                        latent_win,
                        emit_offset_in_window=preset.history_frames,
                        stride_frames=actual_stride,
                        overlap_frames=preset.overlap_frames if not eos_reached else 0,
                        first_emit_frame_index=next_emit_frame,
                        capture_mono_times=cap_times,
                    )
                    self._playout_buffer.push_many(committed)
                    next_emit_frame += actual_stride
                    if not first_commit_event.is_set():
                        self.telemetry.state = "streaming"
                        first_commit_event.set()
        except Exception as exc:
            self.telemetry.state = "error"
            self.telemetry.error = str(exc)
            self._stop_event.set()
            warmup_done_event.set()
            first_commit_event.set()

    def _playout_worker(
        self,
        first_commit_event: threading.Event,
        stream_anchor: dict[str, float],
        realtime_source: bool,
    ) -> None:
        if not first_commit_event.wait(timeout=180.0) or self._stop_event.is_set():
            return
        if self.telemetry.state == "error":
            return

        preset = self.config.get_preset()
        delay_s = preset.playout_delay_s
        fps_out = float(self.config.playout_fps)
        dt_out = 1.0 / fps_out

        # Anchor playout timeline to `max(now, frame0_mono + delay_s)` so real-time streams hit exact
        # `playout_delay_s` end-to-end latency, while unthrottled benchmark runs start immediately.
        now = time.perf_counter()
        frame0_mono = stream_anchor.get("frame0_mono", now)
        playout_origin_mono = max(now, frame0_mono + delay_s) if realtime_source else now

        tick = 0
        fps_counter = 0
        fps_t0 = time.perf_counter()
        last_param_inject_mono = 0.0

        while not self._stop_event.is_set():
            fps_out = float(self.config.playout_fps)
            dt_out = 1.0 / fps_out
            target_wall = playout_origin_mono + tick * dt_out
            sleep_s = target_wall - time.perf_counter()
            if sleep_s > 0.0012:
                time.sleep(sleep_s - 0.0005)
            # Bounded sub-millisecond spin without monopolizing Python's GIL across OS quanta.
            spin_deadline = time.perf_counter() + 0.0008
            while time.perf_counter() < target_wall and time.perf_counter() < spin_deadline:
                pass
            if self._stop_event.is_set():
                break

            now_mono = time.perf_counter()
            target_media_s = now_mono - playout_origin_mono

            if target_media_s < 0.0:
                tick += 1
                continue

            was_started = self._latest_emitted_pose is not None
            sample = self._playout_buffer.sample_at_media_time(target_media_s)
            if sample is not None:
                if sample.underrun and was_started and realtime_source:
                    # Smoothly slip playout origin by up to one tick if external GPU load momentarily spikes
                    # so playout never accumulates catch-up lag or fast-forward lurches.
                    slip_s = min(dt_out, max(0.0, target_media_s - sample.pose.media_time_s))
                    playout_origin_mono += slip_s

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

                if (
                    self.config.plugin_api_enabled
                    and self.config.plugin_api_inject_params
                    and self._plugin_client is not None
                    and self._plugin_client.is_connected
                    and (now_mono - last_param_inject_mono) >= 0.05
                ):
                    params = pose_to_live2d_params(pose, mirror=self.config.mirror)
                    self._plugin_client.inject_params_fire_and_forget(params)
                    last_param_inject_mono = now_mono

                fps_counter += 1
                elapsed = now_mono - fps_t0
                if elapsed >= 0.5:
                    self.telemetry.playout_fps = round(fps_counter / elapsed, 1)
                    fps_counter = 0
                    fps_t0 = now_mono

            tick += 1

    def render_live_jpeg(self) -> bytes | None:
        """Encode the latest live input frame (with detected person bbox) as JPEG."""
        got = self._preview.get_latest_bgr()
        if got is None:
            return None
        f_idx, bgr, bbox, orig_hw = got
        cache_key = (f_idx, bbox)
        cached = self._live_jpeg_cache
        if cached is not None and cached[0] == cache_key:
            return cached[1]
        self._draw_bbox_overlay(bgr, bbox, orig_hw, label=f"LIVE #{f_idx}")
        ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not ok:
            return None
        data = buf.tobytes()
        self._live_jpeg_cache = (cache_key, data)
        return data

    def render_synced_playout_jpeg(self) -> bytes | None:
        """Encode the latency-synced frame matching the exact VMC pose currently playing out."""
        pose = self._latest_emitted_pose
        if pose is None:
            got = self._preview.get_latest_bgr()
            if got is None:
                return None
            f_idx, bgr, bbox, orig_hw = got
            cache_key: Any = ("buf", f_idx, bbox)
            cached = self._synced_jpeg_cache
            if cached is not None and cached[0] == cache_key:
                return cached[1]
            self._draw_bbox_overlay(bgr, bbox, orig_hw, label=f"BUFFERING #{f_idx}")
        else:
            cache_key = ("play", pose.frame_index, round(pose.media_time_s, 4), self.config.mirror)
            cached = self._synced_jpeg_cache
            if cached is not None and cached[0] == cache_key:
                return cached[1]
            got = self._preview.get_synced_bgr(pose.frame_index)
            if got is None:
                return None
            f_idx, bgr, bbox, orig_hw = got
            self._draw_bbox_overlay(bgr, bbox, orig_hw, label=f"SYNCED PLAYOUT #{f_idx}")
            self._draw_skeleton_hud(bgr, pose, bbox, orig_hw)
            if self.config.mirror:
                bgr = cv2.flip(bgr, 1)

        ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not ok:
            return None
        data = buf.tobytes()
        self._synced_jpeg_cache = (cache_key, data)
        return data

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
    def _draw_skeleton_hud(
        bgr: np.ndarray,
        pose: CanonicalPoseFrame,
        bbox: tuple[float, float, float, float] | None,
        orig_hw: tuple[int, int],
    ) -> None:
        """Render the 22-joint 3D body skeleton inset & bbox-aligned overlay on the synced preview."""
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
        if self._plugin_client is not None:
            self.telemetry.plugin_api_connected = self._plugin_client.is_connected
        return {
            "running": self.is_running,
            "config": self.config.to_dict(),
            "presets": {k: vars(v) | {"window_frames": v.window_frames} for k, v in LATENCY_PRESETS.items()},
            "telemetry": vars(self.telemetry).copy(),
            "models": check_models_status(),
        }
