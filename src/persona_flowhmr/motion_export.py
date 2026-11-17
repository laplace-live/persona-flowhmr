"""Record committed source frames and write avatar or camera animation files."""

from __future__ import annotations

import threading
from collections.abc import Callable, Sequence
from dataclasses import replace
from pathlib import Path

import numpy as np

from .recorded_frames import RecordedFrames
from .retarget import MIRRORED_UNITY_BONES, SMPLH_TO_UNITY_BONES, CanonicalPoseFrame

AVATAR_FORMATS = ("vrma", "fbx", "vmd")
_MIRROR_ORDER = [SMPLH_TO_UNITY_BONES.index(name) for name in MIRRORED_UNITY_BONES]
_MIRROR_POS = np.array([-1, 1, 1], dtype=np.float32)
_MIRROR_QUAT = np.array([1, -1, -1, 1], dtype=np.float32)


def _record_pose(pose: CanonicalPoseFrame, mirror: bool, follow: bool) -> CanonicalPoseFrame:
    hips = pose.hips_pos_rh
    rotations = pose.quats_rh_xyzw
    camera = pose.camera
    if mirror:
        hips = hips * _MIRROR_POS
        rotations = rotations[_MIRROR_ORDER] * _MIRROR_QUAT
        if camera is not None:
            camera = replace(
                camera,
                position_rh=camera.position_rh * _MIRROR_POS,
                quat_rh_xyzw=camera.quat_rh_xyzw * _MIRROR_QUAT,
                orbit_target_rh=camera.orbit_target_rh * _MIRROR_POS,
                orbit_azimuth=-camera.orbit_azimuth,
                roll=-camera.roll,
            )
    if not follow:
        hips = np.array([0, hips[1], 0], dtype=np.float32)
    return replace(pose, quats_rh_xyzw=rotations, hips_pos_rh=hips, camera=camera, joints_cam_3d=None)


class MotionRecording:
    """A take is retained for repeated downloads until a new recording begins."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.active = False
        self._has_camera = False
        self._start_frame = 0
        self._end_frame: int | None = None
        self._committed_until = 0
        self._frames = RecordedFrames()

    def start(self, capture_count: Callable[[], int] | None = None) -> None:
        with self._lock:
            if self.active and self._end_frame is not None:
                raise ValueError("Wait for the final frames of the current recording.")
            if not self.active:
                self._frames = RecordedFrames()
                self._has_camera = False
                # Arm before sampling the capture clock so a concurrent commit cannot skip the first frame.
                self.active = True
                self._start_frame = capture_count() if capture_count else 0
                self._end_frame = None
                self._committed_until = self._start_frame

    def append(self, poses: Sequence[CanonicalPoseFrame], *, mirror: bool, follow: bool) -> None:
        with self._lock:
            if not self.active or not poses:
                return
            # The stitcher commits ordered batches, which may straddle either capture boundary.
            self._committed_until = max(self._committed_until, poses[-1].frame_index + 1)
            if poses[0].frame_index < self._start_frame:
                poses = [p for p in poses if p.frame_index >= self._start_frame]
            final_batch = self._end_frame is not None and self._committed_until >= self._end_frame
            if final_batch:
                poses = [p for p in poses if p.frame_index < self._end_frame]
            if poses:
                self._frames.extend(_record_pose(p, mirror, follow) for p in poses)
                if not self._has_camera:
                    self._has_camera = any(p.camera is not None for p in poses)
            if final_batch:
                self.active = False

    def finish(self, capture_count: Callable[[], int]) -> None:
        with self._lock:
            if self.active and self._end_frame is None:
                self._end_frame = capture_count()
                self.active = self._committed_until < self._end_frame

    def stop(self) -> None:
        with self._lock:
            self.active = False

    def frames(self) -> Sequence[CanonicalPoseFrame]:
        with self._lock:
            if self.active and self._end_frame is not None:
                raise ValueError("Processing the final recorded frames.")
            if self.active:
                raise ValueError("Finish recording before exporting the take.")
            if not self._frames:
                raise ValueError("No motion frames were recorded.")
            # A new recording receives its own temporary file. Existing downloads keep this one alive.
            return self._frames

    def status(self) -> dict[str, int | bool]:
        with self._lock:
            return {
                "active": self.active, "finishing": self.active and self._end_frame is not None,
                "frames": len(self._frames), "has_camera": self._has_camera,
            }


def write_motion(path: Path, frames: Sequence[CanonicalPoseFrame], *, camera: bool = False) -> None:
    """Choose a writer by file extension. Format-specific imports stay off the live path."""
    ext = path.suffix.lower()
    if camera:
        if ext != ".vmd":
            raise ValueError("Camera motion requires a .vmd destination.")
        from .motion_vmd import write_camera_vmd

        write_camera_vmd(path, frames)
    elif ext == ".vrma":
        from .motion_vrma import write_vrma

        write_vrma(path, frames)
    elif ext == ".fbx":
        from .motion_fbx import write_fbx

        write_fbx(path, frames)
    elif ext == ".vmd":
        from .motion_vmd import write_avatar_vmd

        write_avatar_vmd(path, frames)
    else:
        raise ValueError("Avatar motion requires a .vrma, .fbx or .vmd destination.")
