"""Unified 30.0 Hz frame sources: Live Webcam (`CameraSource`) & Debug Video (`VideoFileSource`).

Resamples arbitrary video frame rates (24/25/30/60 fps) onto FlowHMR's exact 30.0 Hz clock
without an offline FFmpeg pre-transcode wait, and retains a bounded preview ring buffer so the
control UI can display both the live frame and the latency-synced playout frame side-by-side.
"""

from __future__ import annotations

import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

# Prefer DirectShow on Windows for fast camera open; elsewhere let OpenCV pick the backend.
_CAMERA_BACKEND = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY

BBox = tuple[float, float, float, float]
PreviewFrame = tuple[int, np.ndarray, BBox | None, tuple[int, int]]  # (frame_index, bgr, bbox, source_hw)


@dataclass(frozen=True)
class SampledFrame:
    """Single 30.0 Hz RGB frame emitted by a frame source."""

    frame_index: int
    media_time_s: float
    capture_mono_s: float
    rgb: np.ndarray  # HxWx3 uint8 RGB
    is_stream_reset: bool = False  # first frame after a video loop rewind


class _FrameClock:
    """Paces a source onto a fixed output rate and stamps the frames it emits."""

    def __init__(self, fps: float) -> None:
        self.fps = float(fps)
        self.frame_index = 0
        self._start_mono: float | None = None

    def wait_next(self) -> None:
        """Sleep until the wall-clock slot of the next frame."""
        now = time.perf_counter()
        if self._start_mono is None:
            self._start_mono = now
        sleep_s = self._start_mono + (self.frame_index / self.fps) - time.perf_counter()
        if sleep_s > 0.001:
            time.sleep(sleep_s)

    def emit(self, rgb: np.ndarray, *, is_stream_reset: bool = False) -> SampledFrame:
        sample = SampledFrame(
            frame_index=self.frame_index,
            media_time_s=self.frame_index / self.fps,
            capture_mono_s=time.perf_counter(),
            rgb=rgb,
            is_stream_reset=is_stream_reset,
        )
        self.frame_index += 1
        return sample


class PreviewRingBuffer:
    """Bounded history of downscaled BGR frames + detected bboxes keyed by `frame_index`."""

    def __init__(self, max_items: int = 240, max_width: int = 640) -> None:
        self._lock = threading.Lock()
        self._max_items = int(max_items)
        self._max_width = max_width
        self._frames: dict[int, np.ndarray] = {}  # insertion-ordered: oldest first
        self._bboxes: dict[int, BBox] = {}
        self._orig_hw: tuple[int, int] = (720, 1280)

    def clear(self) -> None:
        with self._lock:
            self._frames.clear()
            self._bboxes.clear()

    def record_frame(self, frame_index: int, rgb: np.ndarray) -> None:
        h, w = rgb.shape[:2]
        if w > self._max_width:
            scale = self._max_width / float(w)
            rgb = cv2.resize(
                rgb,
                (self._max_width, max(1, int(round(h * scale)))),
                interpolation=cv2.INTER_LINEAR,
            )
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        with self._lock:
            self._orig_hw = (h, w)
            self._frames[int(frame_index)] = bgr
            while len(self._frames) > self._max_items:
                evicted = next(iter(self._frames))
                del self._frames[evicted]
                self._bboxes.pop(evicted, None)

    def record_bbox(self, frame_index: int, bbox_xyxy: np.ndarray | BBox) -> None:
        with self._lock:
            self._bboxes[int(frame_index)] = (
                float(bbox_xyxy[0]),
                float(bbox_xyxy[1]),
                float(bbox_xyxy[2]),
                float(bbox_xyxy[3]),
            )

    def _snapshot(self, f_idx: int) -> PreviewFrame:
        return f_idx, self._frames[f_idx].copy(), self._bboxes.get(f_idx), self._orig_hw

    def get_latest_bgr(self) -> PreviewFrame | None:
        with self._lock:
            if not self._frames:
                return None
            return self._snapshot(next(reversed(self._frames)))

    def get_synced_bgr(self, target_frame_index: int) -> PreviewFrame | None:
        target = int(target_frame_index)
        with self._lock:
            if not self._frames:
                return None
            if target in self._frames:
                return self._snapshot(target)
            return self._snapshot(min(self._frames, key=lambda idx: abs(idx - target)))


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
        path = Path(video_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Debug video not found: {path}")
        self.video_path = str(path)
        self.loop = bool(loop)
        self.realtime = bool(realtime)
        self._clock = _FrameClock(target_fps)

        self._cap = cv2.VideoCapture(self.video_path)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open debug video: {self.video_path}")
        raw_fps = float(self._cap.get(cv2.CAP_PROP_FPS))
        self.native_fps = raw_fps if raw_fps > 1.0 else 30.0

        self._loop_frame_index = 0
        self._decoded_pos = -1
        self._last_rgb: np.ndarray | None = None

    def read_next(self) -> SampledFrame | None:
        """Return the next 30.0 Hz sampled RGB frame, pacing to wall-clock if `realtime=True`."""
        if self.realtime:
            self._clock.wait_next()

        # Map 30 Hz tick within current loop pass to native video frame index.
        loop_media_s = self._loop_frame_index / self._clock.fps
        rgb = self._advance_to_native_frame(int(round(loop_media_s * self.native_fps)))
        is_reset = False
        if rgb is None:
            if not self.loop:
                return None
            # Rewind cleanly for seamless debug looping.
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self._decoded_pos = -1
            self._loop_frame_index = 0
            is_reset = True
            rgb = self._advance_to_native_frame(0)
            if rgb is None:
                return None

        self._loop_frame_index += 1
        return self._clock.emit(rgb, is_stream_reset=is_reset)

    def _advance_to_native_frame(self, desired_idx: int) -> np.ndarray | None:
        # Skip frames we won't emit without decoding them into RGB (`grab` only).
        while self._decoded_pos < desired_idx - 1:
            if not self._cap.grab():
                return None
            self._decoded_pos += 1
        if self._decoded_pos < desired_idx:
            ok, bgr = self._cap.read()
            if not ok or bgr is None:
                return None
            self._decoded_pos += 1
            self._last_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return self._last_rgb

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()


def _open_camera(index: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(index, _CAMERA_BACKEND)
    if not cap.isOpened() and _CAMERA_BACKEND != cv2.CAP_ANY:
        cap.release()
        cap = cv2.VideoCapture(index)
    return cap


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
        cap = _open_camera(self.camera_index)
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
        self._clock = _FrameClock(target_fps)
        self._lock = threading.Lock()
        # Raw BGR; converted to RGB only for frames actually emitted on the output clock.
        self._latest_bgr: np.ndarray = first_bgr
        self._stop = threading.Event()
        self._reader_thread = threading.Thread(target=self._reader_loop, name="camera-reader", daemon=True)
        self._reader_thread.start()

    def _reader_loop(self) -> None:
        while not self._stop.is_set():
            ok, bgr = self._cap.read()
            if not ok or bgr is None:
                time.sleep(0.005)
                continue
            with self._lock:
                self._latest_bgr = bgr

    def read_next(self) -> SampledFrame | None:
        if self._stop.is_set():
            return None
        self._clock.wait_next()
        with self._lock:
            bgr = self._latest_bgr
        return self._clock.emit(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

    def close(self) -> None:
        self._stop.set()
        if self._reader_thread.is_alive():
            self._reader_thread.join(timeout=1.0)
        self._cap.release()


def list_available_cameras(max_probe: int = 4) -> list[dict[str, int | str]]:
    """Probe local camera indices `0..max_probe-1` for the control UI dropdown."""
    found: list[dict[str, int | str]] = []
    for idx in range(max_probe):
        cap = cv2.VideoCapture(idx, _CAMERA_BACKEND)
        if not cap.isOpened():
            cap.release()
            continue
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        found.append({"index": idx, "label": f"Camera {idx} ({w}x{h})", "width": w, "height": h})
    return found
