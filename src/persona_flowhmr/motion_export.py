"""Record committed source frames and write avatar or camera animation files."""

from __future__ import annotations

import threading
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import numpy as np

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
        self._frames: list[CanonicalPoseFrame] = []

    def start(self) -> None:
        with self._lock:
            if not self.active:
                self._frames = []
                self._has_camera = False
                self.active = True

    def append(self, poses: Sequence[CanonicalPoseFrame], *, mirror: bool, follow: bool) -> None:
        with self._lock:
            if self.active:
                self._frames.extend(_record_pose(p, mirror, follow) for p in poses)
                if not self._has_camera:
                    self._has_camera = any(p.camera is not None for p in poses)

    def stop(self) -> None:
        with self._lock:
            self.active = False

    def frames(self) -> list[CanonicalPoseFrame]:
        with self._lock:
            if self.active:
                raise ValueError("Finish recording before exporting the take.")
            if not self._frames:
                raise ValueError("No motion frames were recorded.")
            # Completed takes are immutable. start() allocates a fresh list.
            return self._frames

    def status(self) -> dict[str, int | bool]:
        with self._lock:
            return {"active": self.active, "frames": len(self._frames), "has_camera": self._has_camera}


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
