"""SMPL-H (52 joints) -> VMC (Unity HumanBodyBones) & Persona Live2D parameter retargeter.

Coordinate conventions verified against:
  - FlowHMR (`v2m_inference.py:_rotate_smplh_params_y180`, `motion_stats.json`)
  - LAPLACE Persona (`apps/desktop/src/shared/tracking/vmc/protocol.ts` & `shared/vrm/pose.ts`)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from .vmc_sender import BoneTransform, Pos3

if TYPE_CHECKING:
    from .camera_tracker import CameraPoseFrame

# Exact 52 SMPL-H joint order (`SMPLH_JOINT2NUM` in FlowHMR) -> Unity HumanBodyBones name.
# Persona's `unityToVrmBone` (`protocol.ts`) maps `LeftThumbProximal/Intermediate/Distal`
# to VRM 1.0's `leftThumbMetacarpal/Proximal/Distal` automatically.
SMPLH_TO_UNITY_BONES: tuple[str, ...] = (
    "Hips",                      # 0: Pelvis
    "LeftUpperLeg",              # 1: L_Hip
    "RightUpperLeg",             # 2: R_Hip
    "Spine",                     # 3: Spine1
    "LeftLowerLeg",              # 4: L_Knee
    "RightLowerLeg",             # 5: R_Knee
    "Chest",                     # 6: Spine2
    "LeftFoot",                  # 7: L_Ankle
    "RightFoot",                 # 8: R_Ankle
    "UpperChest",                # 9: Spine3
    "LeftToes",                  # 10: L_Foot
    "RightToes",                 # 11: R_Foot
    "Neck",                      # 12: Neck
    "LeftShoulder",              # 13: L_Collar
    "RightShoulder",             # 14: R_Collar
    "Head",                      # 15: Head
    "LeftUpperArm",              # 16: L_Shoulder
    "RightUpperArm",             # 17: R_Shoulder
    "LeftLowerArm",              # 18: L_Elbow
    "RightLowerArm",             # 19: R_Elbow
    "LeftHand",                  # 20: L_Wrist
    "RightHand",                 # 21: R_Wrist
    "LeftIndexProximal",         # 22: L_Index1
    "LeftIndexIntermediate",     # 23: L_Index2
    "LeftIndexDistal",           # 24: L_Index3
    "LeftMiddleProximal",        # 25: L_Middle1
    "LeftMiddleIntermediate",    # 26: L_Middle2
    "LeftMiddleDistal",          # 27: L_Middle3
    "LeftLittleProximal",        # 28: L_Pinky1
    "LeftLittleIntermediate",    # 29: L_Pinky2
    "LeftLittleDistal",          # 30: L_Pinky3
    "LeftRingProximal",          # 31: L_Ring1
    "LeftRingIntermediate",      # 32: L_Ring2
    "LeftRingDistal",            # 33: L_Ring3
    "LeftThumbProximal",         # 34: L_Thumb1
    "LeftThumbIntermediate",     # 35: L_Thumb2
    "LeftThumbDistal",           # 36: L_Thumb3
    "RightIndexProximal",        # 37: R_Index1
    "RightIndexIntermediate",    # 38: R_Index2
    "RightIndexDistal",          # 39: R_Index3
    "RightMiddleProximal",       # 40: R_Middle1
    "RightMiddleIntermediate",   # 41: R_Middle2
    "RightMiddleDistal",         # 42: R_Middle3
    "RightLittleProximal",       # 43: R_Pinky1
    "RightLittleIntermediate",   # 44: R_Pinky2
    "RightLittleDistal",         # 45: R_Pinky3
    "RightRingProximal",         # 46: R_Ring1
    "RightRingIntermediate",     # 47: R_Ring2
    "RightRingDistal",           # 48: R_Ring3
    "RightThumbProximal",        # 49: R_Thumb1
    "RightThumbIntermediate",    # 50: R_Thumb2
    "RightThumbDistal",          # 51: R_Thumb3
)

SMPLH_PARENTS: tuple[int, ...] = (
    -1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12,
    13, 14, 16, 17, 18, 19,
    20, 22, 23, 20, 25, 26, 20, 28, 29, 20, 31, 32, 20, 34, 35,
    21, 37, 38, 21, 40, 41, 21, 43, 44, 21, 46, 47, 21, 49, 50,
)


def _swap_unity_side(name: str) -> str:
    if name.startswith("Left"):
        return "Right" + name[4:]
    if name.startswith("Right"):
        return "Left" + name[5:]
    return name


MIRRORED_UNITY_BONES: tuple[str, ...] = tuple(_swap_unity_side(b) for b in SMPLH_TO_UNITY_BONES)


def rot6d_to_matrix_np(rot6d: np.ndarray) -> np.ndarray:
    """Convert FlowHMR `(..., 6)` 6D rotation (`(..., 3, 2)` column-interleaved) to `(..., 3, 3)` matrices."""
    x = rot6d.reshape(*rot6d.shape[:-1], 3, 2)
    a1 = x[..., 0]
    a2 = x[..., 1]
    b1 = a1 / np.clip(np.linalg.norm(a1, axis=-1, keepdims=True), 1e-8, None)
    dot = np.sum(b1 * a2, axis=-1, keepdims=True)
    b2 = a2 - dot * b1
    b2 = b2 / np.clip(np.linalg.norm(b2, axis=-1, keepdims=True), 1e-8, None)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-1)


def matrix_to_quat_xyzw(mats: np.ndarray) -> np.ndarray:
    """Convert `(..., 3, 3)` proper rotation matrices to normalized `(..., 4)` `(x, y, z, w)` quaternions."""
    orig_shape = mats.shape[:-2]
    m = mats.reshape(-1, 3, 3)
    n = m.shape[0]
    q = np.empty((n, 4), dtype=np.float32)

    trace = m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2]

    mask0 = trace > 0.0
    if np.any(mask0):
        s = np.sqrt(trace[mask0] + 1.0) * 2.0
        q[mask0, 3] = 0.25 * s
        q[mask0, 0] = (m[mask0, 2, 1] - m[mask0, 1, 2]) / s
        q[mask0, 1] = (m[mask0, 0, 2] - m[mask0, 2, 0]) / s
        q[mask0, 2] = (m[mask0, 1, 0] - m[mask0, 0, 1]) / s

    mask1 = (~mask0) & (m[:, 0, 0] > m[:, 1, 1]) & (m[:, 0, 0] > m[:, 2, 2])
    if np.any(mask1):
        s = np.sqrt(1.0 + m[mask1, 0, 0] - m[mask1, 1, 1] - m[mask1, 2, 2]) * 2.0
        q[mask1, 3] = (m[mask1, 2, 1] - m[mask1, 1, 2]) / s
        q[mask1, 0] = 0.25 * s
        q[mask1, 1] = (m[mask1, 0, 1] + m[mask1, 1, 0]) / s
        q[mask1, 2] = (m[mask1, 0, 2] + m[mask1, 2, 0]) / s

    mask2 = (~mask0) & (~mask1) & (m[:, 1, 1] > m[:, 2, 2])
    if np.any(mask2):
        s = np.sqrt(1.0 + m[mask2, 1, 1] - m[mask2, 0, 0] - m[mask2, 2, 2]) * 2.0
        q[mask2, 3] = (m[mask2, 0, 2] - m[mask2, 2, 0]) / s
        q[mask2, 0] = (m[mask2, 0, 1] + m[mask2, 1, 0]) / s
        q[mask2, 1] = 0.25 * s
        q[mask2, 2] = (m[mask2, 1, 2] + m[mask2, 2, 1]) / s

    mask3 = (~mask0) & (~mask1) & (~mask2)
    if np.any(mask3):
        s = np.sqrt(1.0 + m[mask3, 2, 2] - m[mask3, 0, 0] - m[mask3, 1, 1]) * 2.0
        q[mask3, 3] = (m[mask3, 1, 0] - m[mask3, 0, 1]) / s
        q[mask3, 0] = (m[mask3, 0, 2] + m[mask3, 2, 0]) / s
        q[mask3, 1] = (m[mask3, 1, 2] + m[mask3, 2, 1]) / s
        q[mask3, 2] = 0.25 * s

    norms = np.clip(np.linalg.norm(q, axis=-1, keepdims=True), 1e-8, None)
    q = q / norms
    # Canonicalize w >= 0 for smooth downstream SLERP.
    neg_w = q[:, 3] < 0
    q[neg_w] = -q[neg_w]
    return q.reshape(*orig_shape, 4)


def quat_mul_xyzw(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product `q1 * q2` for `(..., 4)` `(x, y, z, w)` quaternions."""
    x1, y1, z1, w1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    x2, y2, z2, w2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    out = np.stack(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ],
        axis=-1,
    )
    norms = np.clip(np.linalg.norm(out, axis=-1, keepdims=True), 1e-8, None)
    out = (out / norms).astype(np.float32)
    out = np.where(out[..., 3:4] < 0.0, -out, out)
    return out


def slerp_quat_xyzw(q0: np.ndarray, q1: np.ndarray, alpha: float | np.ndarray) -> np.ndarray:
    """Vectorized antipodal-safe SLERP between `(..., 4)` `(x, y, z, w)` quaternions."""
    q0 = q0.astype(np.float32, copy=False)
    q1 = q1.astype(np.float32, copy=True)

    dot = np.sum(q0 * q1, axis=-1, keepdims=True)
    flip = dot < 0.0
    q1 = np.where(flip, -q1, q1)
    dot = np.abs(dot)

    t = np.asarray(alpha, dtype=np.float32)
    while t.ndim < dot.ndim:
        t = t[..., None]

    linear_mask = dot > 0.9995
    theta_0 = np.arccos(np.clip(dot, -1.0, 1.0))
    sin_theta_0 = np.clip(np.sin(theta_0), 1e-6, None)
    s0 = np.sin((1.0 - t) * theta_0) / sin_theta_0
    s1 = np.sin(t * theta_0) / sin_theta_0

    interp = np.where(linear_mask, (1.0 - t) * q0 + t * q1, s0 * q0 + s1 * q1)
    norms = np.clip(np.linalg.norm(interp, axis=-1, keepdims=True), 1e-8, None)
    return (interp / norms).astype(np.float32)


# Pre-rotation R_y(180 deg) = (0, 1, 0, 0) in (x, y, z, w), matching FlowHMR's
# `_rotate_smplh_params_y180`: brings camera-facing (-Z) SMPL-H root into +Z-forward VRM space.
_RY_180_QUAT = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
# The same R_y(180) applied to positions: `(x, y, z) -> (-x, +y, -z)`.
_RY_180_POS_SIGN = np.array([-1.0, 1.0, -1.0], dtype=np.float32)
# Unity's left-handed VMC frame stores `(qx, -qy, -qz, qw)` for a right-handed `(qx, qy, qz, qw)`.
_RH_TO_UNITY_QUAT_SIGN = np.array([1.0, -1.0, -1.0, 1.0], dtype=np.float32)


@dataclass(frozen=True)
class CanonicalPoseFrame:
    """Single 30 Hz or interpolated 60 Hz frame in canonical +X-left, +Y-up, +Z-forward space."""

    media_time_s: float
    capture_mono_s: float
    frame_index: int
    quats_rh_xyzw: np.ndarray  # (52, 4) local right-handed SMPL-H quaternions, root already R_y(180)-aligned
    hips_pos_rh: np.ndarray    # (3,) pelvis world position in +X-left, +Y-up, +Z-forward meters
    joints_cam_3d: np.ndarray | None = None  # optional (22, 3) body joints in camera frame for 2D HUD
    camera: CameraPoseFrame | None = None    # optional dynamic 6-DoF stage camera pose & FOV


def canonicalize_smplh_rotations(local_rot_mats: np.ndarray) -> np.ndarray:
    """Convert `(T, 52, 3, 3)` local SMPL-H rotations to `(T, 52, 4)` +Z-forward quats (`R_y(180)` on the root)."""
    quats = matrix_to_quat_xyzw(local_rot_mats)
    ry = np.broadcast_to(_RY_180_QUAT, quats[:, 0].shape)
    quats[:, 0] = quat_mul_xyzw(ry, quats[:, 0])
    return quats


def cam_to_canonical_position(pos_cam: np.ndarray) -> np.ndarray:
    """Apply `R_y(180)` (`(-x, +y, -z)`) to `(..., 3)` camera-space positions -> float32 canonical space."""
    return np.asarray(pos_cam, dtype=np.float32) * _RY_180_POS_SIGN


def canonicalize_smplh_sequence(
    local_rot_mats: np.ndarray,
    pelvis_world_cam: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert `(T, 52, 3, 3)` local SMPL-H rotations + `(T, 3)` pelvis_world to +Z-forward canonical space."""
    return canonicalize_smplh_rotations(local_rot_mats), cam_to_canonical_position(pelvis_world_cam)


def pose_to_vmc_bones(
    pose: CanonicalPoseFrame,
    *,
    mirror: bool = False,
    follow_movement: bool = True,
) -> list[BoneTransform]:
    """Convert a canonical +Z-forward SMPL-H pose into 52 Unity VMC `BoneTransform` tuples.

    Verified against LAPLACE Persona's `unityQuatToThree` (`[qx, -qy, -qz, qw]`)
    and `unityVec3ToThree` (`[-vx, vy, vz]`):
      - `mirror=False`: preserves performer anatomy (`Left*` -> `Left*`), `q_vmc = (qx, -qy, -qz, qw)`,
        `hips_vmc = (-px, py, pz)` so Three.js receives `(px, py, pz)` and `(qx, qy, qz, qw)`.
      - `mirror=True`: reflects across YZ plane (`Left*` <-> `Right*`), `q_vmc = (qx, qy, qz, qw)`,
        `hips_vmc = (+px, py, pz)` so Three.js receives `(-px, py, pz)` and `(-qx, qy, qz, -qw) == (qx, -qy, -qz, qw)`.
    """
    px, py, pz = pose.hips_pos_rh.tolist()
    if mirror:
        names = MIRRORED_UNITY_BONES
        quats = pose.quats_rh_xyzw.tolist()  # (qx, qy, qz, qw) -> Three.js (qx, -qy, -qz, qw) on swapped side!
    else:
        names = SMPLH_TO_UNITY_BONES
        quats = (pose.quats_rh_xyzw * _RH_TO_UNITY_QUAT_SIGN).tolist()
        px = -px
    hips_pos: Pos3 = (px, py, pz) if follow_movement else (0.0, py, 0.0)

    zero_pos: Pos3 = (0.0, 0.0, 0.0)
    return [(name, hips_pos if idx == 0 else zero_pos, tuple(q)) for idx, (name, q) in enumerate(zip(names, quats))]


def _quat_to_euler_deg_yxz(q: np.ndarray) -> tuple[float, float, float]:
    """Extract `(yaw_y, pitch_x, roll_z)` in degrees for Live2D `AngleX/Y/Z` parameters."""
    x, y, z, w = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    # Pitch (X), Yaw (Y), Roll (Z)
    sinp = max(-1.0, min(1.0, 2.0 * (w * x - y * z)))
    pitch = math.degrees(math.asin(sinp))
    yaw = math.degrees(math.atan2(2.0 * (w * y + x * z), 1.0 - 2.0 * (x * x + y * y)))
    roll = math.degrees(math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (x * x + z * z)))
    return yaw, pitch, roll


def pose_to_live2d_params(pose: CanonicalPoseFrame, *, mirror: bool = False) -> list[dict[str, float | str]]:
    """Derive `InjectEntry[]` (`param.inject`) for Live2D models (`FaceAngleX/Y/Z` + `ParamBodyAngleX/Y/Z`)."""
    sign = -1.0 if mirror else 1.0
    q_hips = pose.quats_rh_xyzw[0]
    q_spine = quat_mul_xyzw(pose.quats_rh_xyzw[3], quat_mul_xyzw(pose.quats_rh_xyzw[6], pose.quats_rh_xyzw[9]))
    q_torso = quat_mul_xyzw(q_hips, q_spine)
    q_head_chain = quat_mul_xyzw(q_torso, quat_mul_xyzw(pose.quats_rh_xyzw[12], pose.quats_rh_xyzw[15]))

    body_yaw, body_pitch, body_roll = _quat_to_euler_deg_yxz(q_torso)
    head_yaw, head_pitch, head_roll = _quat_to_euler_deg_yxz(q_head_chain)

    def clamp(v: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, v))

    return [
        {"type": "input", "id": "FaceAngleX", "value": clamp(sign * head_yaw, -30.0, 30.0)},
        {"type": "input", "id": "FaceAngleY", "value": clamp(-head_pitch, -30.0, 30.0)},
        {"type": "input", "id": "FaceAngleZ", "value": clamp(-sign * head_roll, -30.0, 30.0)},
        {"type": "live2d-param", "id": "ParamBodyAngleX", "value": clamp(sign * body_yaw * 0.5, -10.0, 10.0)},
        {"type": "live2d-param", "id": "ParamBodyAngleY", "value": clamp(-body_pitch * 0.5, -10.0, 10.0)},
        {"type": "live2d-param", "id": "ParamBodyAngleZ", "value": clamp(-sign * body_roll * 0.5, -10.0, 10.0)},
    ]
