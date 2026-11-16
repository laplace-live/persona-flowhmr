"""Unified 30.0 Hz frame sources: Live Webcam (`CameraSource`) & Debug Video (`VideoFileSource`).

Resamples arbitrary video frame rates (24/25/30/60 fps) onto FlowHMR's exact 30.0 Hz clock
without an offline FFmpeg pre-transcode wait, and retains a bounded preview ring buffer so the
control UI can display both the live frame and the latency-synced playout frame side-by-side.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class SampledFrame:
    """Single 30.0 Hz RGB frame emitted by a frame source."""

    frame_index: int
    media_time_s: float
    capture_mono_s: float
    rgb: np.ndarray  # HxWx3 uint8 RGB
    is_stream_reset: bool = False
    source_frame_pos: int = 0


class PreviewRingBuffer:
    """Bounded history of downscaled BGR frames + detected bboxes keyed by `frame_index`."""

    def __init__(self, max_items: int = 240, max_width: int = 640) -> None:
        self._lock = threading.Lock()
        self._max_items = int(max_items)
        self._max_width = max_width
        self._order: deque[int] = deque()
        self._frame_map: dict[int, np.ndarray] = {}
        self._bboxes: dict[int, tuple[float, float, float, float]] = {}
        self._orig_hw: tuple[int, int] = (720, 1280)

    def clear(self) -> None:
        with self._lock:
            self._order.clear()
            self._frame_map.clear()
            self._bboxes.clear()

    def record_frame(self, frame_index: int, rgb: np.ndarray) -> None:
        h, w = rgb.shape[:2]
        if w > self._max_width:
            scale = self._max_width / float(w)
            small_rgb = cv2.resize(
                rgb,
                (self._max_width, max(1, int(round(h * scale)))),
                interpolation=cv2.INTER_LINEAR,
            )
            bgr = cv2.cvtColor(small_rgb, cv2.COLOR_RGB2BGR)
        else:
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        f_idx = int(frame_index)
        with self._lock:
            self._orig_hw = (h, w)
            if f_idx not in self._frame_map:
                self._order.append(f_idx)
            self._frame_map[f_idx] = bgr
            while len(self._order) > self._max_items:
                evicted = self._order.popleft()
                self._frame_map.pop(evicted, None)
                self._bboxes.pop(evicted, None)

    def record_bbox(self, frame_index: int, bbox_xyxy: np.ndarray | tuple[float, float, float, float]) -> None:
        with self._lock:
            self._bboxes[int(frame_index)] = (
                float(bbox_xyxy[0]),
                float(bbox_xyxy[1]),
                float(bbox_xyxy[2]),
                float(bbox_xyxy[3]),
            )

    def get_latest_bgr(self) -> tuple[int, np.ndarray, tuple[float, float, float, float] | None, tuple[int, int]] | None:
        with self._lock:
            if not self._order:
                return None
            f_idx = self._order[-1]
            bgr = self._frame_map.get(f_idx)
            if bgr is None:
                return None
            return f_idx, bgr.copy(), self._bboxes.get(f_idx), self._orig_hw

    def get_synced_bgr(
        self, target_frame_index: int
    ) -> tuple[int, np.ndarray, tuple[float, float, float, float] | None, tuple[int, int]] | None:
        target = int(target_frame_index)
        with self._lock:
            if not self._order:
                return None
            exact = self._frame_map.get(target)
            if exact is not None:
                return target, exact.copy(), self._bboxes.get(target), self._orig_hw
            best_idx = min(self._order, key=lambda idx: abs(idx - target))
            best_bgr = self._frame_map[best_idx]
            return best_idx, best_bgr.copy(), self._bboxes.get(best_idx), self._orig_hw


class VideoFileSource:
    """Real-time (or unthrottled benchmark) 30.0 Hz resampler over a debug video file."""

    def __init__(
        self,
        video_path: str | Path,
        *,
        loop: bool = True,
        realtime: bool = True,
        target_fps: float = 30.0,
    ) -> None:
        self.video_path = str(Path(video_path).resolve())
        if not Path(self.video_path).is_file():
            raise FileNotFoundError(f"Debug video not found: {self.video_path}")
        self.loop = bool(loop)
        self.realtime = bool(realtime)
        self.target_fps = float(target_fps)

        self._cap = cv2.VideoCapture(self.video_path)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open debug video: {self.video_path}")

        self.width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        raw_fps = float(self._cap.get(cv2.CAP_PROP_FPS))
        self.native_fps = raw_fps if raw_fps > 1.0 else 30.0
        self.total_native_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self._frame_index = 0
        self._loop_frame_index = 0
        self._decoded_pos = -1
        self._last_rgb: np.ndarray | None = None
        self._start_mono: float | None = None
        self._pending_reset = True

    def read_next(self) -> SampledFrame | None:
        """Return the next 30.0 Hz sampled RGB frame, pacing to wall-clock if `realtime=True`."""
        now = time.perf_counter()
        if self._start_mono is None:
            self._start_mono = now

        target_wall = self._start_mono + (self._frame_index / self.target_fps)
        if self.realtime:
            sleep_s = target_wall - time.perf_counter()
            if sleep_s > 0.001:
                time.sleep(sleep_s)

        # Map 30 Hz tick within current loop pass to native video frame index.
        loop_media_s = self._loop_frame_index / self.target_fps
        desired_native_idx = int(round(loop_media_s * self.native_fps))

        rgb = self._advance_to_native_frame(desired_native_idx)
        if rgb is None:
            if not self.loop:
                return None
            # Rewind cleanly for seamless debug looping.
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self._decoded_pos = -1
            self._loop_frame_index = 0
            self._pending_reset = True
            rgb = self._advance_to_native_frame(0)
            if rgb is None:
                return None

        capture_mono = time.perf_counter()
        sample = SampledFrame(
            frame_index=self._frame_index,
            media_time_s=self._frame_index / self.target_fps,
            capture_mono_s=capture_mono,
            rgb=rgb,
            is_stream_reset=self._pending_reset,
            source_frame_pos=max(0, self._decoded_pos),
        )
        self._pending_reset = False
        self._frame_index += 1
        self._loop_frame_index += 1
        return sample

    def _advance_to_native_frame(self, desired_idx: int) -> np.ndarray | None:
        while self._decoded_pos < desired_idx:
            ok, bgr = self._cap.read()
            if not ok or bgr is None:
                return None
            self._decoded_pos += 1
            self._last_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return self._last_rgb

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()


class CameraSource:
    """Live webcam capture with a low-latency reader thread and steady 30.0 Hz output clock."""

    def __init__(
        self,
        camera_index: int = 0,
        *,
        width: int = 1280,
        height: int = 720,
        target_fps: float = 30.0,
    ) -> None:
        self.camera_index = int(camera_index)
        self.target_fps = float(target_fps)

        # Prefer DirectShow on Windows for fast open, falling back to default backend.
        cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open camera index {self.camera_index}")

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
        cap.set(cv2.CAP_PROP_FPS, float(target_fps))
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        ok, first_bgr = cap.read()
        if not ok or first_bgr is None:
            cap.release()
            raise RuntimeError(f"Camera {self.camera_index} opened but returned no frames")

        self._cap = cap
        self.height, self.width = first_bgr.shape[:2]
        self._lock = threading.Lock()
        self._latest_rgb = cv2.cvtColor(first_bgr, cv2.COLOR_BGR2RGB)
        self._stop = threading.Event()
        self._reader_thread = threading.Thread(target=self._reader_loop, name="camera-reader", daemon=True)
        self._reader_thread.start()

        self._frame_index = 0
        self._start_mono: float | None = None
        self._pending_reset = True

    def _reader_loop(self) -> None:
        while not self._stop.is_set():
            ok, bgr = self._cap.read()
            if not ok or bgr is None:
                time.sleep(0.005)
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            with self._lock:
                self._latest_rgb = rgb

    def read_next(self) -> SampledFrame | None:
        if self._stop.is_set():
            return None
        now = time.perf_counter()
        if self._start_mono is None:
            self._start_mono = now

        target_wall = self._start_mono + (self._frame_index / self.target_fps)
        sleep_s = target_wall - time.perf_counter()
        if sleep_s > 0.001:
            time.sleep(sleep_s)

        with self._lock:
            rgb = self._latest_rgb.copy()

        sample = SampledFrame(
            frame_index=self._frame_index,
            media_time_s=self._frame_index / self.target_fps,
            capture_mono_s=time.perf_counter(),
            rgb=rgb,
            is_stream_reset=self._pending_reset,
            source_frame_pos=self._frame_index,
        )
        self._pending_reset = False
        self._frame_index += 1
        return sample

    def close(self) -> None:
        self._stop.set()
        if self._reader_thread.is_alive():
            self._reader_thread.join(timeout=1.0)
        self._cap.release()


def list_available_cameras(max_probe: int = 4) -> list[dict[str, int | str]]:
    """Probe local camera indices `0..max_probe-1` for the control UI dropdown."""
    found: list[dict[str, int | str]] = []
    for idx in range(max_probe):
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap.release()
            continue
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        found.append({"index": idx, "label": f"Camera {idx} ({w}x{h})", "width": w, "height": h})
    return found
