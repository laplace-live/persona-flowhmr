"""Coordinate system & mirror symmetry verification against Persona's Three.js VRM conventions."""

from __future__ import annotations

import math

import numpy as np

from persona_flowhmr.paths import MOTION_STATS_PATH, bootstrap_runtime
from persona_flowhmr.retarget import (
    CanonicalPoseFrame,
    canonicalize_smplh_sequence,
    pose_to_live2d_params,
    pose_to_vmc_bones,
    rot6d_to_matrix_np,
)


def _unity_quat_to_three(q_vmc: tuple[float, float, float, float]) -> np.ndarray:
    """Exact replica of Persona's `unityQuatToThree(q, +1)` in `apps/desktop/src/shared/vrm/pose.ts`."""
    return np.array([q_vmc[0], -q_vmc[1], -q_vmc[2], q_vmc[3]], dtype=np.float32)


def _unity_vec3_to_three(v_vmc: tuple[float, float, float]) -> np.ndarray:
    """Exact replica of Persona's `unityVec3ToThree(v, +1)` in `apps/desktop/src/shared/vrm/pose.ts`."""
    return np.array([-v_vmc[0], v_vmc[1], v_vmc[2]], dtype=np.float32)


def _quat_rotate_vec(q_xyzw: np.ndarray, vec: np.ndarray) -> np.ndarray:
    x, y, z, w = q_xyzw
    qv = np.array([x, y, z], dtype=np.float32)
    t = 2.0 * np.cross(qv, vec)
    return vec + w * t + np.cross(qv, t)


def test_camera_facing_root_becomes_identity_after_ry180() -> None:
    # FlowHMR's neutral standing root in camera space faces -Z: diag(-1, +1, -1).
    mats = np.repeat(np.eye(3, dtype=np.float32)[None, None], 52, axis=1)
    mats[0, 0] = np.diag([-1.0, 1.0, -1.0]).astype(np.float32)
    pelvis_cam = np.array([[0.3, 0.9, -2.0]], dtype=np.float32)

    quats, hips = canonicalize_smplh_sequence(mats, pelvis_cam)
    assert np.allclose(quats[0, 0], [0.0, 0.0, 0.0, 1.0], atol=1e-6)
    assert np.allclose(hips[0], [-0.3, 0.9, 2.0], atol=1e-6)


def test_arm_raise_pitch_yaw_and_travel_in_both_mirror_modes() -> None:
    quats = np.zeros((52, 4), dtype=np.float32)
    quats[:, 3] = 1.0

    # 1. Spine (idx 3): +30 deg around +X -> forward bow (+Y bends toward +Z)
    a_bow = math.radians(30.0)
    quats[3] = [math.sin(a_bow / 2), 0.0, 0.0, math.cos(a_bow / 2)]

    # 2. Head (idx 15): +30 deg around +Y -> turn toward performer's LEFT (+Z turns toward +X)
    a_yaw = math.radians(30.0)
    quats[15] = [0.0, math.sin(a_yaw / 2), 0.0, math.cos(a_yaw / 2)]

    # 3. LeftUpperArm (idx 16, rest +X): +45 deg around +Z -> raise arm UP toward +Y
    a_arm = math.radians(45.0)
    quats[16] = [0.0, 0.0, math.sin(a_arm / 2), math.cos(a_arm / 2)]

    pose = CanonicalPoseFrame(
        media_time_s=0.0,
        capture_mono_s=0.0,
        frame_index=0,
        quats_rh_xyzw=quats,
        hips_pos_rh=np.array([0.5, 0.9, 0.2], dtype=np.float32),  # +0.5m toward performer's left
    )

    # Unmirrored check
    unmirrored = dict((name, (pos, q)) for name, pos, q in pose_to_vmc_bones(pose, mirror=False))
    hips_three = _unity_vec3_to_three(unmirrored["Hips"][0])
    assert np.allclose(hips_three, [0.5, 0.9, 0.2], atol=1e-6)

    spine_up = _quat_rotate_vec(_unity_quat_to_three(unmirrored["Spine"][1]), np.array([0.0, 1.0, 0.0]))
    assert spine_up[2] > 0.45  # bowed forward (+Z)

    head_fwd = _quat_rotate_vec(_unity_quat_to_three(unmirrored["Head"][1]), np.array([0.0, 0.0, 1.0]))
    assert head_fwd[0] > 0.45  # looking toward avatar's left (+X)

    l_arm_dir = _quat_rotate_vec(_unity_quat_to_three(unmirrored["LeftUpperArm"][1]), np.array([1.0, 0.0, 0.0]))
    assert l_arm_dir[1] > 0.65  # left arm raised UP (+Y)

    # Mirrored check (webcam selfie mirror: performer's left arm raises avatar's RIGHT arm UP,
    # forward bow stays forward, head turn flips side, horizontal travel flips X!)
    mirrored = dict((name, (pos, q)) for name, pos, q in pose_to_vmc_bones(pose, mirror=True))
    hips_three_m = _unity_vec3_to_three(mirrored["Hips"][0])
    assert np.allclose(hips_three_m, [-0.5, 0.9, 0.2], atol=1e-6)

    spine_up_m = _quat_rotate_vec(_unity_quat_to_three(mirrored["Spine"][1]), np.array([0.0, 1.0, 0.0]))
    assert spine_up_m[2] > 0.45  # STILL bows forward (+Z), not backward!

    head_fwd_m = _quat_rotate_vec(_unity_quat_to_three(mirrored["Head"][1]), np.array([0.0, 0.0, 1.0]))
    assert head_fwd_m[0] < -0.45  # mirrored to avatar's right (-X)!

    r_arm_dir_m = _quat_rotate_vec(_unity_quat_to_three(mirrored["RightUpperArm"][1]), np.array([-1.0, 0.0, 0.0]))
    assert r_arm_dir_m[1] > 0.65  # avatar's right arm (-X rest) raised UP (+Y), not down!

    params_u = {p["id"]: p["value"] for p in pose_to_live2d_params(pose, mirror=False)}
    params_m = {p["id"]: p["value"] for p in pose_to_live2d_params(pose, mirror=True)}
    assert params_u["FaceAngleX"] == -params_m["FaceAngleX"]


def test_rot6d_parity_with_upstream_flowhmr_geometry() -> None:
    import json

    import torch

    bootstrap_runtime()
    from flowhmr.core.math.geometry import rot6d_to_rotation_matrix

    stats = json.loads(MOTION_STATS_PATH.read_text(encoding="utf-8"))
    mean_6d = np.asarray(stats["local_rot_data"]["mean"], dtype=np.float32).reshape(52, 6)
    ours = rot6d_to_matrix_np(mean_6d)
    theirs = rot6d_to_rotation_matrix(torch.from_numpy(mean_6d)).numpy()
    assert np.allclose(ours, theirs, atol=1e-5)
    # Dataset mean root faces camera (-Z) -> after R_y(180) aligns with VRM +Z forward!
    quats, _ = canonicalize_smplh_sequence(ours[None], np.zeros((1, 3), dtype=np.float32))
    assert quats[0, 0, 3] > 0.999
