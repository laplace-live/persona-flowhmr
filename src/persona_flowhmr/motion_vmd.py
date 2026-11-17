"""Export canonical avatar FK and stage-camera motion to MikuMikuDance VMD 0002."""

from __future__ import annotations

import math
import struct
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from .camera_tracker import CameraPoseFrame
from .motion_skeleton import REFERENCE_HEIGHT_M, REST_HIPS_HEIGHT_M
from .retarget import CanonicalPoseFrame, quat_mul_xyzw

_HEADER = struct.Struct("<30s20s")
_BONE = struct.Struct("<15sI3f4f64s")
_CAMERA = struct.Struct("<If3f3f24sIB")
_SIGNATURE = b"Vocaloid Motion Data 0002"

# Persona's VMD loader assumes 20 MMD units per avatar height and a pelvis
# 56.7% of the way up. Use the same avatar scale for both files.
_METERS_PER_UNIT = REFERENCE_HEIGHT_M / 20.0
_MMD_REST_HIPS = REST_HIPS_HEIGHT_M / _METERS_PER_UNIT
_MMD_POSITION_SCALE = np.array([1.0, 1.0, -1.0]) / _METERS_PER_UNIT
_MMD_QUAT_SIGNS = np.array([-1.0, -1.0, 1.0, 1.0])

# The standard bone block stores four staggered copies of the interpolation
# parameters; camera keys pack (x1, x2, y1, y2) per channel.
_BONE_ROW = bytes([20] * 8 + [107] * 8)
_BONE_LINEAR = b"".join(_BONE_ROW[i:] + bytes(i) for i in range(4))
_CAMERA_LINEAR = bytes([20, 107, 20, 107]) * 6

# SMPL-H joint order. Standard MMD rigs have two torso joints, so fold
# UpperChest into Chest. センター parents both the leg and upper-body chains.
_BODY_NAMES = (
    "センター",
    "左足",
    "右足",
    "上半身",
    "左ひざ",
    "右ひざ",
    "上半身2",
    "左足首",
    "右足首",
    None,
    "左つま先",
    "右つま先",
    "首",
    "左肩",
    "右肩",
    "頭",
    "左腕",
    "右腕",
    "左ひじ",
    "右ひじ",
    "左手首",
    "右手首",
)
_FINGER_NAMES = tuple(
    side + finger + digit
    for side in ("左", "右")
    for finger in ("人指", "中指", "小指", "薬指", "親指")
    for digit in ("０１２" if finger == "親指" else "１２３")
)
_BONE_FIELDS = tuple(
    (i, name.encode("shift_jis")) for i, name in enumerate(_BODY_NAMES + _FINGER_NAMES) if name is not None
)
_IK_NAMES = [f"{side}{joint}ＩＫ".encode("shift_jis") for side in ("左", "右") for joint in ("足", "つま先")]


def _arm_rest_rotations() -> tuple[np.ndarray, np.ndarray]:
    """Rotations from VRM's T-pose to the common 35-degree MMD A-pose."""
    parents = np.tile([0.0, 0.0, 0.0, 1.0], (52, 1))
    inverse_self = parents.copy()
    for side, half_angle in enumerate(np.radians([-35.0, 35.0]) / 2):
        head, elbow, hand = 16 + side, 18 + side, 20 + side
        fingers = range(22 + side * 15, 37 + side * 15)
        delta = [0.0, 0.0, math.sin(half_angle), math.cos(half_angle)]
        parents[[elbow, hand, *fingers]] = delta
        inverse_self[[head, elbow, hand, *fingers]] = [0.0, 0.0, -delta[2], delta[3]]
    return parents, inverse_self


_PARENT_REST, _INVERSE_REST = _arm_rest_rotations()


def _vmd_timeline(frames: Sequence[CanonicalPoseFrame]) -> dict[int, CanonicalPoseFrame]:
    if not frames:
        raise ValueError("VMD export requires at least one motion frame")
    origin = frames[0].media_time_s
    keys = {}
    previous = -math.inf
    for frame in frames:
        time = frame.media_time_s
        if not math.isfinite(time) or time < previous:
            raise ValueError("VMD timestamps must be finite and chronological")
        number = round((time - origin) * 30)
        if number > 0xFFFFFFFF:
            raise ValueError("VMD duration exceeds the 32-bit frame limit")
        # VMD is fixed at 30 fps. Last key wins if multiple samples land on one tick.
        keys[number] = frame
        previous = time
    return keys


def write_avatar_vmd(path: Path, frames: Sequence[CanonicalPoseFrame]) -> None:
    """Write root motion and local bone rotations on the 30 Hz media timeline."""
    keys = _vmd_timeline(frames)
    with Path(path).open("wb") as stream:
        stream.write(_HEADER.pack(_SIGNATURE, b"FlowHMR"))
        stream.write(struct.pack("<I", len(keys) * len(_BONE_FIELDS)))
        for number, pose in keys.items():
            norms = np.linalg.norm(pose.quats_rh_xyzw, axis=1)
            if not np.all(np.isfinite(pose.hips_pos_rh)) or not np.all(np.isfinite(norms) & (norms > 1e-8)):
                raise ValueError("VMD requires finite hips positions and nonzero finite quaternions")
            rotations = quat_mul_xyzw(quat_mul_xyzw(_PARENT_REST, pose.quats_rh_xyzw), _INVERSE_REST)
            rotations[6] = quat_mul_xyzw(pose.quats_rh_xyzw[6], pose.quats_rh_xyzw[9])
            rotations *= _MMD_QUAT_SIGNS
            center = pose.hips_pos_rh * _MMD_POSITION_SCALE
            center[1] -= _MMD_REST_HIPS  # VMD stores a translation from the model's bind pose.
            for index, name in _BONE_FIELDS:
                position = center if index == 0 else (0.0, 0.0, 0.0)
                stream.write(_BONE.pack(name, number, *position, *rotations[index], _BONE_LINEAR))
        # Empty morph/camera/light/shadow sections. Disable the model's default
        # foot and toe IK at frame zero so they cannot override the exported FK.
        stream.write(struct.pack("<5I", 0, 0, 0, 0, 1))
        stream.write(struct.pack("<IBI", 0, 1, len(_IK_NAMES)))
        for name in _IK_NAMES:
            stream.write(struct.pack("<20sB", name, 0))


def write_camera_vmd(path: Path, frames: Sequence[CanonicalPoseFrame]) -> None:
    """Write target, distance, pitch/yaw/roll, and vertical FOV on the avatar timeline."""
    keys: dict[int, CameraPoseFrame] = {}
    previous: CameraPoseFrame | None = None
    previous_number = -1
    for number, pose in _vmd_timeline(frames).items():
        # Live camera output holds its last pose while tracking is disabled.
        # Keep the camera clip aligned to the avatar through the end of the take.
        camera = pose.camera or previous
        if camera is None:
            continue
        if camera.is_cut and previous is not None and number - previous_number > 1:
            # Adjacent camera keys switch instantly in MMD. Preserve the cut even
            # when the tracker skipped samples between the shots.
            keys[number - 1] = previous
        keys[number] = camera
        previous, previous_number = camera, number
    if not keys:
        raise ValueError("Camera VMD export requires tracked camera poses")
    with Path(path).open("wb") as stream:
        stream.write(_HEADER.pack(_SIGNATURE, "カメラ・照明".encode("shift_jis")))
        stream.write(struct.pack("<3I", 0, 0, len(keys)))
        for number, cam in keys.items():
            if not np.all(np.isfinite([
                *cam.orbit_target_rh, cam.orbit_distance, cam.orbit_elevation, cam.orbit_azimuth, cam.roll, cam.fov_deg,
            ])) or not 1 <= cam.fov_deg <= 179:
                raise ValueError("Camera VMD requires a finite orbit and a 1–179 degree field of view")
            stream.write(
                _CAMERA.pack(
                    number,
                    -cam.orbit_distance / _METERS_PER_UNIT,
                    *(cam.orbit_target_rh * _MMD_POSITION_SCALE),
                    -cam.orbit_elevation,
                    cam.orbit_azimuth,
                    -cam.roll,
                    _CAMERA_LINEAR,
                    round(cam.fov_deg),
                    0,
                )
            )
        stream.write(struct.pack("<3I", 0, 0, 0))
