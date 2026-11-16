"""Timestamped 30 Hz pose ring buffer and 30/60 Hz SLERP playout interpolator."""

from __future__ import annotations

import bisect
import threading
from collections import deque
from dataclasses import dataclass
from operator import attrgetter

import numpy as np

from .retarget import CanonicalPoseFrame, slerp_quat_xyzw

_media_time = attrgetter("media_time_s")


@dataclass(frozen=True)
class PlayoutSample:
    """Interpolated pose emitted on a 30 Hz or 60 Hz playout tick."""

    pose: CanonicalPoseFrame
    buffer_lead_ms: float
    interpolated: bool
    underrun: bool


class PosePlayoutBuffer:
    """Thread-safe ring buffer of committed 30 Hz poses supporting arbitrary-rate SLERP playout."""

    def __init__(self, max_frames: int = 600) -> None:
        self._lock = threading.Lock()
        self._frames: deque[CanonicalPoseFrame] = deque(maxlen=max_frames)
        self.underrun_count = 0

    def clear(self) -> None:
        with self._lock:
            self._frames.clear()
            self.underrun_count = 0

    def push_many(self, poses: list[CanonicalPoseFrame]) -> None:
        with self._lock:
            for p in poses:
                # Replace the tail if frames are re-committed after a reset so indices stay monotonic.
                while self._frames and self._frames[-1].frame_index >= p.frame_index:
                    self._frames.pop()
                self._frames.append(p)

    def sample_at_media_time(self, target_media_time_s: float) -> PlayoutSample | None:
        """Sample or SLERP-interpolate a pose at `target_media_time_s` (seconds on 30 Hz stream axis)."""
        with self._lock:
            if not self._frames:
                return None

            first = self._frames[0]
            last = self._frames[-1]
            lead_ms = (last.media_time_s - target_media_time_s) * 1000.0

            if target_media_time_s <= first.media_time_s:
                return PlayoutSample(pose=first, buffer_lead_ms=lead_ms, interpolated=False, underrun=False)

            if target_media_time_s >= last.media_time_s:
                self.underrun_count += 1
                return PlayoutSample(pose=last, buffer_lead_ms=lead_ms, interpolated=False, underrun=True)

            idx0 = bisect.bisect_right(self._frames, target_media_time_s, key=_media_time) - 1
            f0 = self._frames[idx0]
            f1 = self._frames[idx0 + 1]

            # Trim old frames more than 4 seconds behind playout cursor to keep memory tiny.
            while len(self._frames) > 4 and (target_media_time_s - self._frames[0].media_time_s) > 4.0:
                self._frames.popleft()

        span = f1.media_time_s - f0.media_time_s
        if span <= 1e-6:
            return PlayoutSample(pose=f0, buffer_lead_ms=lead_ms, interpolated=False, underrun=False)

        alpha = float(np.clip((target_media_time_s - f0.media_time_s) / span, 0.0, 1.0))
        if alpha <= 1e-4:
            return PlayoutSample(pose=f0, buffer_lead_ms=lead_ms, interpolated=False, underrun=False)
        if alpha >= 1.0 - 1e-4:
            return PlayoutSample(pose=f1, buffer_lead_ms=lead_ms, interpolated=False, underrun=False)

        quats = slerp_quat_xyzw(f0.quats_rh_xyzw, f1.quats_rh_xyzw, alpha)
        hips = ((1.0 - alpha) * f0.hips_pos_rh + alpha * f1.hips_pos_rh).astype(np.float32)
        joints = (
            ((1.0 - alpha) * f0.joints_cam_3d + alpha * f1.joints_cam_3d).astype(np.float32)
            if (f0.joints_cam_3d is not None and f1.joints_cam_3d is not None)
            else f0.joints_cam_3d
        )
        interp_pose = CanonicalPoseFrame(
            media_time_s=target_media_time_s,
            capture_mono_s=(1.0 - alpha) * f0.capture_mono_s + alpha * f1.capture_mono_s,
            frame_index=f0.frame_index if alpha < 0.5 else f1.frame_index,
            quats_rh_xyzw=quats,
            hips_pos_rh=hips,
            joints_cam_3d=joints,
        )
        return PlayoutSample(pose=interp_pose, buffer_lead_ms=lead_ms, interpolated=True, underrun=False)
