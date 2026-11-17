"""Unified 30.0 Hz frame sources: Live Webcam (`CameraSource`) & Debug Video (`VideoFileSource`).

Resamples arbitrary video frame rates (24/25/30/60 fps) onto FlowHMR's exact 30.0 Hz clock
without an offline FFmpeg pre-transcode wait, and retains a bounded preview ring buffer so the
control UI can display both the live frame and the latency-synced playout frame side-by-side.
"""

from __future__ import annotations

import contextlib
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
    is_stream_reset: bool = False   # first frame after a video loop rewind
    is_camera_switch: bool = False  # first frame after hot-swapping live camera device


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
        sleep_s = self._start_mono + (self.frame_index / self.fps) - now
        if sleep_s > 0.001:
            time.sleep(sleep_s)
        elif sleep_s < -0.10:
            # Re-anchor after queue backpressure so video pacing resumes smoothly at 30.0 Hz
            # instead of bursting stale backlog frames.
            self._start_mono = now - (self.frame_index / self.fps)

    def emit(
        self,
        rgb: np.ndarray,
        *,
        is_stream_reset: bool = False,
        is_camera_switch: bool = False,
    ) -> SampledFrame:
        sample = SampledFrame(
            frame_index=self.frame_index,
            media_time_s=self.frame_index / self.fps,
            capture_mono_s=time.perf_counter(),
            rgb=rgb,
            is_stream_reset=is_stream_reset,
            is_camera_switch=is_camera_switch,
        )
        self.frame_index += 1
        return sample


class PreviewRingBuffer:
    """Bounded history of downscaled BGR frames + per-frame source dims + bboxes keyed by `frame_index`."""

    def __init__(self, max_items: int = 240, max_width: int = 640) -> None:
        self._lock = threading.Lock()
        self._max_items = int(max_items)
        self._max_width = max_width
        self._frames: dict[int, tuple[np.ndarray, tuple[int, int]]] = {}  # insertion-ordered: oldest first
        self._bboxes: dict[int, BBox] = {}

    def clear(self) -> None:
        with self._lock:
            self._frames.clear()
            self._bboxes.clear()

    def record_frame(self, frame_index: int, rgb: np.ndarray) -> None:
        h, w = rgb.shape[:2]
        orig_hw = (int(h), int(w))
        if w > self._max_width or h > self._max_width * 2:
            scale = min(self._max_width / float(max(1, w)), (self._max_width * 1.5) / float(max(1, h)))
            tw, th = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
            if scale < 0.5:
                rgb = cv2.resize(rgb, (tw * 2, th * 2), interpolation=cv2.INTER_NEAREST)
            rgb = cv2.resize(rgb, (tw, th), interpolation=cv2.INTER_LINEAR)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        with self._lock:
            self._frames[int(frame_index)] = (bgr, orig_hw)
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
        bgr, orig_hw = self._frames[f_idx]
        return f_idx, bgr.copy(), self._bboxes.get(f_idx), orig_hw

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


def _open_and_prime_camera(
    index: int,
    *,
    width: int,
    height: int,
    target_fps: float,
    timeout_s: float = 2.5,
) -> tuple[cv2.VideoCapture, np.ndarray]:
    """Open `index` and read its first BGR frame on a helper thread with a bounded `timeout_s` wait."""
    box: dict[str, object] = {}
    abandoned = threading.Event()

    def _init_worker() -> None:
        cap: cv2.VideoCapture | None = None
        try:
            cap = _open_camera(index)
            if abandoned.is_set():
                cap.release()
                return
            if not cap.isOpened():
                cap.release()
                box["error"] = RuntimeError(f"Cannot open camera index {index}")
                return
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
            cap.set(cv2.CAP_PROP_FPS, float(target_fps))
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            ok, first_bgr = cap.read()
            if abandoned.is_set():
                cap.release()
                return
            if not ok or first_bgr is None:
                cap.release()
                box["error"] = RuntimeError(f"Camera {index} opened but returned no frames")
                return
            box["cap"] = cap
            box["bgr"] = first_bgr
        except Exception as exc:
            if cap is not None:
                with contextlib.suppress(Exception):
                    cap.release()
            box["error"] = exc

    t = threading.Thread(target=_init_worker, name=f"camera-open-{index}", daemon=True)
    t.start()
    t.join(timeout=max(0.1, float(timeout_s)))
    if t.is_alive():
        abandoned.set()
        raise TimeoutError(f"Timed out opening camera index {index} after {timeout_s:.1f}s")
    if "error" in box:
        err = box["error"]
        raise err if isinstance(err, BaseException) else RuntimeError(str(err))
    return box["cap"], box["bgr"]  # type: ignore[return-value]


class CameraSource:
    """Live webcam capture with a lock-free reader loop, bounded hot-swap, and steady 30.0 Hz clock."""

    def __init__(
        self,
        camera_index: int = 0,
        *,
        width: int = 1280,
        height: int = 720,
        target_fps: float = 30.0,
        open_timeout_s: float = 5.0,
    ) -> None:
        self.camera_index = int(camera_index)
        cap, first_bgr = _open_and_prime_camera(
            self.camera_index,
            width=width,
            height=height,
            target_fps=target_fps,
            timeout_s=open_timeout_s,
        )

        self._cap: cv2.VideoCapture | None = cap
        self._device_gen = 0
        self._retired_caps: list[cv2.VideoCapture] = []
        self._clock = _FrameClock(target_fps)
        self._lock = threading.Lock()
        # Raw BGR; converted to RGB only for frames actually emitted on the output clock.
        self._latest_bgr: np.ndarray = first_bgr
        self._pending_switch = False
        self._stop = threading.Event()
        self._reader_thread = threading.Thread(target=self._reader_loop, name="camera-reader", daemon=True)
        self._reader_thread.start()

    def switch_device(
        self,
        camera_index: int,
        *,
        width: int = 1280,
        height: int = 720,
        target_fps: float = 30.0,
        timeout_s: float = 2.5,
    ) -> None:
        """Hot-swap to another live webcam device with bounded timeout and generation-guarded frame publish."""
        new_idx = int(camera_index)
        if new_idx == self.camera_index:
            return
        new_cap, first_bgr = _open_and_prime_camera(
            new_idx,
            width=width,
            height=height,
            target_fps=target_fps,
            timeout_s=timeout_s,
        )

        with self._lock:
            if self._cap is not None:
                self._retired_caps.append(self._cap)
            self._cap = new_cap
            self._device_gen += 1
            self.camera_index = new_idx
            self._latest_bgr = first_bgr
            self._pending_switch = True

    def _drain_retired_caps(self) -> None:
        with self._lock:
            retired = list(self._retired_caps)
            self._retired_caps.clear()
        for old_cap in retired:
            with contextlib.suppress(Exception):
                old_cap.release()

    def _reader_loop(self) -> None:
        try:
            while not self._stop.is_set():
                self._drain_retired_caps()
                with self._lock:
                    cap = self._cap
                    gen = self._device_gen
                if cap is None:
                    break
                # Perform blocking driver read without holding `_lock` so `switch_device` and `close` never stall.
                ok, bgr = cap.read()
                if not ok or bgr is None:
                    time.sleep(0.005)
                    continue
                with self._lock:
                    # Discard stale frames captured from a superseded device before `switch_device` swapped `_cap`.
                    if gen == self._device_gen and not self._stop.is_set():
                        self._latest_bgr = bgr
        finally:
            self._drain_retired_caps()
            if self._stop.is_set():
                with self._lock:
                    active_cap = self._cap
                    self._cap = None
                if active_cap is not None:
                    with contextlib.suppress(Exception):
                        active_cap.release()

    def read_next(self) -> SampledFrame | None:
        if self._stop.is_set():
            return None
        self._clock.wait_next()
        with self._lock:
            bgr = self._latest_bgr
            switched = self._pending_switch
            self._pending_switch = False
        return self._clock.emit(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), is_camera_switch=switched)

    def close(self) -> None:
        self._stop.set()
        if self._reader_thread.is_alive():
            self._reader_thread.join(timeout=1.0)
        with self._lock:
            pending_caps = list(self._retired_caps)
            self._retired_caps.clear()
            if self._cap is not None:
                pending_caps.append(self._cap)
                self._cap = None
        if pending_caps:
            def _release_all() -> None:
                for c in pending_caps:
                    with contextlib.suppress(Exception):
                        c.release()

            if self._reader_thread.is_alive():
                rel_t = threading.Thread(target=_release_all, name="camera-force-release", daemon=True)
                rel_t.start()
                rel_t.join(timeout=0.5)
            else:
                _release_all()


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
