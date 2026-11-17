"""Unit & protocol parity tests for dynamic 6-DoF camera tracking, VMC `/VMC/Ext/Cam`, and Persona `vrmCamera`."""

from __future__ import annotations

import asyncio
import json
import math
import socket
import threading

import cv2
import numpy as np
import websockets

from persona_flowhmr.camera_tracker import (
    BackgroundVisualOdometry,
    FrameCameraObservation,
    ShotCutDetector,
    StreamingCameraSolver,
    camera_to_persona_vrm_camera,
    camera_to_vmc_transform,
    decompose_three_camera_pose,
    extract_smplh_body14_stage,
)
from persona_flowhmr.paths import MOTION_STATS_PATH
from persona_flowhmr.perception import default_camera_intrinsics
from persona_flowhmr.persona_api import PersonaPluginClient
from persona_flowhmr.playout import PosePlayoutBuffer
from persona_flowhmr.retarget import CanonicalPoseFrame, pose_to_vmc_bones
from persona_flowhmr.stitcher import MotionLatentDecoder, StreamingStitcher
from persona_flowhmr.vmc_sender import encode_vmc_bundle
from test_osc_vmc_parity import _decode_osc_bundle_py


def _persona_orbit_to_three_matrix_and_pos(vrm_cam: dict) -> tuple[np.ndarray, np.ndarray]:
    """Exact replica of LAPLACE Persona's `orbitPosition` + `camera.rotation.set(-el, az, roll, 'YXZ')`."""
    orb = vrm_cam["orbit"]
    az = float(orb["azimuth"])
    el = float(orb["elevation"])
    dist = float(orb["distance"])
    roll = float(vrm_cam.get("roll", 0.0))

    r_horiz = math.cos(el) * dist
    pos = np.array(
        [
            orb["targetX"] + math.sin(az) * r_horiz,
            orb["targetY"] + math.sin(el) * dist,
            orb["targetZ"] + math.cos(az) * r_horiz,
        ],
        dtype=np.float32,
    )

    # Three.js Euler(-el, az, roll, 'YXZ') == R_y(az) @ R_x(-el) @ R_z(roll).
    ca, sa = math.cos(az), math.sin(az)
    cx, sx = math.cos(-el), math.sin(-el)
    cz, sz = math.cos(roll), math.sin(roll)
    ry = np.array([[ca, 0.0, sa], [0.0, 1.0, 0.0], [-sa, 0.0, ca]], dtype=np.float32)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=np.float32)
    rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    return pos, ry @ rx @ rz


def test_persona_orbit_camera_roundtrip_across_angles_and_mirror() -> None:
    shots = [
        # (eye_pos, az_deg, el_deg, roll_deg, dist, fov)
        ([0.0, 1.05, 3.8], 0.0, 0.0, 0.0, 3.8, 28.0),
        ([2.1, 0.65, 1.9], 48.0, -14.0, 6.0, 2.7, 32.5),
        ([-1.6, 1.85, 2.2], -37.0, 22.0, -9.5, 2.9, 40.0),
    ]
    for eye, az_d, el_d, roll_d, dist, fov in shots:
        _, orig_rot = _persona_orbit_to_three_matrix_and_pos(
            {
                "orbit": {
                    "azimuth": math.radians(az_d),
                    "elevation": math.radians(el_d),
                    "distance": dist,
                    "targetX": 0.0,
                    "targetY": 0.0,
                    "targetZ": 0.0,
                },
                "roll": math.radians(roll_d),
            }
        )
        cam_pose = decompose_three_camera_pose(np.array(eye, dtype=np.float32), orig_rot, dist, fov)
        vrm_cam = camera_to_persona_vrm_camera(cam_pose, mirror=False)
        rec_pos, rec_rot = _persona_orbit_to_three_matrix_and_pos(vrm_cam)
        assert np.allclose(rec_pos, eye, atol=1e-4)
        assert np.allclose(rec_rot, orig_rot, atol=1e-4)
        assert abs(vrm_cam["fov"] - fov) < 1e-3

        # Mirrored camera reflects X eye coordinate and flips azimuth & roll sign.
        vrm_cam_m = camera_to_persona_vrm_camera(cam_pose, mirror=True)
        rec_pos_m, _ = _persona_orbit_to_three_matrix_and_pos(vrm_cam_m)
        assert np.allclose(rec_pos_m, [-eye[0], eye[1], eye[2]], atol=1e-4)


def test_vmc_ext_cam_osc_bundle_layout_and_unity_handedness() -> None:
    quats = np.zeros((52, 4), dtype=np.float32)
    quats[:, 3] = 1.0
    cam = decompose_three_camera_pose(
        np.array([0.4, 1.1, 3.2], dtype=np.float32),
        np.eye(3, dtype=np.float32),  # front-facing Three.js camera looking along -Z
        3.2,
        28.5,
    )
    pose = CanonicalPoseFrame(0.0, 0.0, 0, quats, np.array([0.0, 0.88, 0.0], dtype=np.float32), camera=cam)
    bones = pose_to_vmc_bones(pose, mirror=False)
    vmc_cam = camera_to_vmc_transform(cam, mirror=False)
    pkt = encode_vmc_bundle(bones, camera=vmc_cam)

    msgs = _decode_osc_bundle_py(pkt)
    assert len(msgs) == 55  # 1 root + 52 bones + 1 Cam + 1 OK
    cam_msg = msgs[-2]
    assert cam_msg[0] == "/VMC/Ext/Cam"
    assert cam_msg[1][0] == "Camera"
    # Front-facing Three.js camera `(0.4, 1.1, 3.2)` maps to Unity `(-0.4, 1.1, 3.2)` and R_y(180) `(0, 1, 0, 0)`.
    assert np.allclose(cam_msg[1][1:4], [-0.4, 1.1, 3.2], atol=1e-5)
    assert np.allclose(cam_msg[1][4:8], [0.0, 1.0, 0.0, 0.0], atol=1e-5)
    assert abs(float(cam_msg[1][8]) - 28.5) < 1e-4

    # Mirrored camera flips Unity X sign to +0.4.
    vmc_cam_m = camera_to_vmc_transform(cam, mirror=True)
    assert np.allclose(vmc_cam_m[1], [0.4, 1.1, 3.2], atol=1e-5)


def test_shot_cut_detector_and_background_vo_pan_recovery() -> None:
    rng = np.random.default_rng(123)
    h, w = 360, 640
    k_mat = default_camera_intrinsics(h, w)

    # Rich textured background canvas (`960 x 540`) from which we render a pure camera pan via K * R * K^-1.
    pano = np.zeros((540, 960, 3), dtype=np.uint8)
    pano[:] = (35, 48, 66)
    for _ in range(180):
        cx, cy = int(rng.integers(40, 920)), int(rng.integers(40, 500))
        col = tuple(int(v) for v in rng.integers(90, 245, size=3))
        cv2.circle(pano, (cx, cy), int(rng.integers(4, 14)), col, -1)

    k_pano = np.array([[k_mat[0, 0], 0.0, 480.0], [0.0, k_mat[1, 1], 270.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    k_inv = np.linalg.inv(k_pano)

    cut_det = ShotCutDetector()
    vo = BackgroundVisualOdometry()
    center_bbox = np.array([250.0, 70.0, 390.0, 330.0], dtype=np.float32)

    cuts: list[bool] = []
    last_r_yup = np.eye(3, dtype=np.float32)
    # 6 frames panning camera rightward by 1.5 deg/frame (total +7.5 deg across 5 steps).
    for step in range(6):
        ang = math.radians(step * 1.5)
        # Camera pans right -> world points shift left in OpenCV camera (`R_cv = R_y(-ang)`).
        c, s = math.cos(-ang), math.sin(-ang)
        r_cv = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)
        homog = k_mat.astype(np.float64) @ r_cv @ k_inv
        frame = cv2.warpPerspective(pano, homog, (w, h), flags=cv2.INTER_LINEAR)
        # Simulate moving foreground performer inside bbox (must NOT trigger cut or corrupt VO).
        fg_x = 290 + (step % 2) * 28
        cv2.rectangle(frame, (fg_x, 110), (fg_x + 45, 300), (240, 80, 120), -1)

        is_cut = cut_det.step(frame)
        cuts.append(is_cut)
        last_r_yup, inl = vo.step(frame, center_bbox, k_mat, is_cut=is_cut)
        if step > 0:
            assert inl >= 10

    assert not any(cuts), f"Unexpected cut during smooth pan: {cuts}"
    # Cumulative camera azimuth in Y-up space (`R_world_from_cam = last_r_yup.T`, azimuth = atan2(R[0,2], R[2,2])).
    r_world_from_cam = last_r_yup.T
    est_pan_deg = abs(math.degrees(math.atan2(float(r_world_from_cam[0, 2]), float(r_world_from_cam[2, 2]))))
    assert 6.0 <= est_pan_deg <= 9.0, f"Expected ~7.5 deg pan, got {est_pan_deg:.2f} deg"

    # Hard cut to a bright warm studio shot triggers `is_cut == True` immediately on the cut frame only.
    shot2 = np.full((h, w, 3), (235, 195, 110), dtype=np.uint8)
    cv2.rectangle(shot2, (40, 20), (600, 120), (250, 240, 210), -1)
    assert cut_det.step(shot2) is True
    assert cut_det.step(shot2) is False


def test_multi_camera_cut_preserves_avatar_heading_and_jumps_3d_stage_camera() -> None:
    decoder = MotionLatentDecoder(MOTION_STATS_PATH, motion_style="basic")
    stitcher = StreamingStitcher(decoder, fps=30.0)
    cam_solver = StreamingCameraSolver(avatar_scale=1.0)
    playout = PosePlayoutBuffer()

    h, w = 720, 1280
    cam_k = default_camera_intrinsics(h, w)
    cv_to_three = np.diag([1.0, -1.0, -1.0]).astype(np.float32)

    # Zero normalized latent -> neutral camera-facing pose (`ljp` = dataset mean).
    lat = np.zeros((32, 491), dtype=np.float32)
    dec0 = decoder.decode_window(lat[:1])
    ljp_neutral = dec0["ljp_cam"][0]
    hips0 = np.array([-float(ljp_neutral[0, 0]), float(dec0["srp_y_cam"][0]), -float(ljp_neutral[0, 2])], dtype=np.float32)
    stage_j14 = extract_smplh_body14_stage(ljp_neutral, hips0)

    def _make_obs(
        dist_m: float,
        az_deg: float,
        el_deg: float,
        *,
        is_cut: bool,
        shot_id: int,
    ) -> FrameCameraObservation:
        vrm_spec = {
            "orbit": {
                "azimuth": math.radians(az_deg),
                "elevation": math.radians(el_deg),
                "distance": dist_m,
                "targetX": 0.0,
                "targetY": float(hips0[1]),
                "targetZ": 0.0,
            },
            "roll": 0.0,
        }
        c_pos, r_w_cam = _persona_orbit_to_three_matrix_and_pos(vrm_spec)
        j14_three_cam = (r_w_cam.T @ (stage_j14 - c_pos[None, :]).T).T
        j14_cv = (j14_three_cam @ cv_to_three.T).astype(np.float32)
        return FrameCameraObservation(
            sam_cam_t=np.array([0.0, 0.9, dist_m], dtype=np.float32),
            sam_j3d_cam=j14_cv,
            cam_k=cam_k,
            image_hw=(h, w),
            bbox_xyxy=np.array([450.0, 100.0, 830.0, 680.0], dtype=np.float32),
            r_vo_yup=np.eye(3, dtype=np.float32),
            vo_inliers=25,
            is_cut=is_cut,
            shot_id=shot_id,
        )

    # Shot 1 (frames 0..7): Wide front shot (4.4m, az=0 deg, el=0 deg)
    obs1 = [_make_obs(4.4, 0.0, 0.0, is_cut=False, shot_id=0) for _ in range(8)]
    c1 = stitcher.commit_window(
        lat,
        emit_offset_in_window=8,
        stride_frames=8,
        overlap_frames=0,
        first_emit_frame_index=0,
        capture_mono_times=[i / 30.0 for i in range(8)],
        cut_flags=[False] * 8,
        camera_observations=obs1,
        camera_solver=cam_solver,
    )

    # Shot 2 (frames 8..15): Hard cut to Close-Up Low-Angle Side Shot (1.85m, el=-14 deg),
    # where in Shot 2's camera view (`WV(shot2)`) the performer appears turned -50 deg
    # (so both `ljp` and `root_6d` in `WV(shot2)` are rotated -50 deg, and world camera azimuth is +50 deg!).
    lat_shot2 = np.zeros((32, 491), dtype=np.float32)
    ang_wv = math.radians(-50.0)
    ca, sa = math.cos(ang_wv), math.sin(ang_wv)
    ry_wv = np.array([[ca, 0.0, sa], [0.0, 1.0, 0.0], [-sa, 0.0, ca]], dtype=np.float32)
    ljp_shot2 = ljp_neutral @ ry_wv.T
    ljp_shot2_norm = ((ljp_shot2 - decoder.ljp_mean) / decoder.ljp_std).reshape(156)
    lat_shot2[:, 3:159] = ljp_shot2_norm[None, :]
    # Neutral root before R_y(180) is diag(-1, 1, -1); apply R_y(-50 deg) -> 6D columns.
    r_root_shot2 = ry_wv @ np.diag([-1.0, 1.0, -1.0]).astype(np.float32)
    root_6d = r_root_shot2[:, :2].reshape(6)
    root_6d_norm = (root_6d - decoder.lrot_mean[0]) / decoder.lrot_std[0]
    lat_shot2[:, 159:165] = root_6d_norm[None, :]

    # In Shot 2's camera view, SAM sees the dancer rotated -50 deg at 1.85m and -14 deg elevation
    # (which in world space with dancer at 0 deg corresponds to camera at az=+50 deg, dist=1.85m, el=-14 deg!).
    obs2 = [_make_obs(1.85, 50.0, -14.0, is_cut=(i == 0), shot_id=1) for i in range(8)]
    c2 = stitcher.commit_window(
        lat_shot2,
        emit_offset_in_window=8,
        stride_frames=8,
        overlap_frames=0,
        first_emit_frame_index=8,
        capture_mono_times=[(8 + i) / 30.0 for i in range(8)],
        cut_flags=[i == 0 for i in range(8)],
        camera_observations=obs2,
        camera_solver=cam_solver,
    )

    playout.push_many(c1 + c2)

    # 1. Avatar root quaternion stays facing +Z across the hard camera cut (no 50 deg spin!).
    assert np.allclose(c1[-1].quats_rh_xyzw[0], c2[0].quats_rh_xyzw[0], atol=0.05)

    # 2. Restored 3D stage camera jumps cleanly from (4.4m, 0 deg, 0 deg) -> (1.85m, +50 deg, -14 deg)!
    cam_before = c1[-1].camera
    cam_after = c2[0].camera
    assert cam_before is not None and cam_after is not None
    assert abs(cam_before.orbit_distance - 4.4) < 0.15
    assert abs(math.degrees(cam_before.orbit_azimuth)) < 3.0
    assert cam_after.is_cut is True
    assert abs(cam_after.orbit_distance - 1.85) < 0.15
    assert abs(math.degrees(cam_after.orbit_azimuth) - 50.0) < 4.0
    assert abs(math.degrees(cam_after.orbit_elevation) - (-14.0)) < 4.0

    # 3. 60 Hz playout midpoint across frame 7 -> 8 (`is_cut=True`) snaps instead of swooping!
    mid_before = playout.sample_at_media_time(7.25 / 30.0)
    mid_after = playout.sample_at_media_time(7.75 / 30.0)
    assert mid_before is not None and mid_after is not None
    assert mid_before.pose.camera is not None and mid_after.pose.camera is not None
    assert abs(mid_before.pose.camera.orbit_distance - 4.4) < 0.15
    assert abs(mid_after.pose.camera.orbit_distance - 1.85) < 0.15


def test_persona_plugin_client_coalesces_vrm_camera_patches() -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]

    patched_cameras: list[dict] = []

    async def handler(ws):
        await ws.send(
            json.dumps(
                {
                    "kind": "hello",
                    "protocol": 4,
                    "app": {"name": "LAPLACE Persona", "version": "0.91.0", "platform": "win32", "capabilities": []},
                }
            )
        )
        async for raw in ws:
            msg = json.loads(raw)
            m = msg["method"]
            params = msg.get("params") or {}
            if m == "scene.patch" and "vrmCamera" in params:
                patched_cameras.append(params["vrmCamera"])
                await asyncio.sleep(0.015)
            await ws.send(json.dumps({"kind": "response", "id": msg["id"], "result": {}}))

    stop_srv = threading.Event()

    def _run_srv() -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        async def _main() -> None:
            async with websockets.serve(handler, "127.0.0.1", port):
                while not stop_srv.is_set():
                    await asyncio.sleep(0.02)

        loop.run_until_complete(_main())

    t = threading.Thread(target=_run_srv, daemon=True)
    t.start()

    client = PersonaPluginClient(url=f"ws://127.0.0.1:{port}", token="tok")
    try:
        assert client.start(timeout_s=5.0) is True
        # Fire a rapid burst of 20 camera poses; coalescing keeps WS backlog <= 2 and lands the final pose.
        for idx in range(20):
            client.send_vrm_camera_fire_and_forget({"orbit": {"distance": 2.0 + idx * 0.1}, "fov": 30.0})
        deadline = asyncio.new_event_loop()
        deadline.close()
        for _ in range(40):
            if patched_cameras and abs(patched_cameras[-1]["orbit"]["distance"] - 3.9) < 1e-6:
                break
            threading.Event().wait(0.02)
        assert patched_cameras
        assert len(patched_cameras) < 20  # coalesced burst
        assert abs(patched_cameras[-1]["orbit"]["distance"] - 3.9) < 1e-6
    finally:
        client.stop()
        stop_srv.set()
        t.join(timeout=2.0)


def test_static_tripod_after_hard_cut_has_zero_post_cut_smooth_drift() -> None:
    """Regression test: after a hard cut to a static tripod shot, body/acrobatic pose changes must not
    cause post-cut smooth camera drift ('jump with smooth').
    """
    decoder = MotionLatentDecoder(MOTION_STATS_PATH, motion_style="basic")
    cam_solver = StreamingCameraSolver(avatar_scale=1.0)
    h, w = 720, 1280
    cam_k = default_camera_intrinsics(h, w)
    cv_to_three = np.diag([1.0, -1.0, -1.0]).astype(np.float32)

    lat = np.zeros((1, 491), dtype=np.float32)
    dec0 = decoder.decode_window(lat)
    ljp_neutral = dec0["ljp_cam"][0]
    hips0 = np.array([-float(ljp_neutral[0, 0]), float(dec0["srp_y_cam"][0]), -float(ljp_neutral[0, 2])], dtype=np.float32)
    stage_j14 = extract_smplh_body14_stage(ljp_neutral, hips0)

    # Shot 1: close-up at 2.2m
    c1_pos, r1_w_cam = _persona_orbit_to_three_matrix_and_pos(
        {"orbit": {"azimuth": 0.0, "elevation": 0.0, "distance": 2.2, "targetX": 0.0, "targetY": float(hips0[1]), "targetZ": 0.0}, "roll": 0.0}
    )
    j14_cv_s1 = ((r1_w_cam.T @ (stage_j14 - c1_pos[None, :]).T).T @ cv_to_three.T).astype(np.float32)
    for f_idx in range(5):
        obs = FrameCameraObservation(
            sam_cam_t=np.array([0.0, 0.9, 2.2], dtype=np.float32),
            sam_j3d_cam=j14_cv_s1,
            cam_k=cam_k,
            image_hw=(h, w),
            bbox_xyxy=np.array([400.0, 80.0, 880.0, 690.0], dtype=np.float32),
            r_vo_yup=np.eye(3, dtype=np.float32),
            vo_inliers=30,
            is_cut=(f_idx == 0),
            shot_id=0,
            trans_flow_px=0.10,
        )
        cam_solver.solve_frame(obs, ljp_cam_52=ljp_neutral, hips_pos_rh=hips0)

    # Shot 2: hard cut to wide locked tripod at 6.5m, az=+25 deg, while SAM single-frame knees/ankles
    # and depth jitter wildly across 40 post-cut frames (simulating vertical leg split -> crouch -> stand).
    az2 = math.radians(25.0)
    stage_j14_s2 = extract_smplh_body14_stage(ljp_neutral, hips0, world_yaw_rad=az2)
    c2_pos, r2_w_cam = _persona_orbit_to_three_matrix_and_pos(
        {"orbit": {"azimuth": az2, "elevation": 0.0, "distance": 6.5, "targetX": 0.0, "targetY": float(hips0[1]), "targetZ": 0.0}, "roll": 0.0}
    )
    base_j14_cv_s2 = ((r2_w_cam.T @ (stage_j14_s2 - c2_pos[None, :]).T).T @ cv_to_three.T).astype(np.float32)

    post_cut_poses = []
    for k in range(40):
        j14_noisy = base_j14_cv_s2.copy()
        # Inject huge single-frame leg-split / arm-swing discrepancy on knees, ankles, wrists + 0.8m depth wobble.
        j14_noisy[10:, 1] -= 0.75 * math.cos(k * 0.15)
        j14_noisy[10:, 2] += 0.45 * math.sin(k * 0.2)
        if k > 0:
            j14_noisy[:, 2] += 0.65 * math.sin(k * 0.12)
        obs2 = FrameCameraObservation(
            sam_cam_t=np.array([0.0, 0.9, 6.5 + (0.65 * math.sin(k * 0.12) if k > 0 else 0.0)], dtype=np.float32),
            sam_j3d_cam=j14_noisy,
            cam_k=cam_k,
            image_hw=(h, w),
            bbox_xyxy=np.array([560.0, 200.0, 740.0, 590.0], dtype=np.float32),
            r_vo_yup=np.eye(3, dtype=np.float32),
            vo_inliers=28,
            is_cut=(k == 0),
            shot_id=1,
            trans_flow_px=0.12,  # locked tripod (< 0.48 px residual flow)
        )
        post_cut_poses.append(
            cam_solver.solve_frame(obs2, ljp_cam_52=ljp_neutral, hips_pos_rh=hips0, world_yaw_rad=az2)
        )

    cut_frame = post_cut_poses[0]
    assert cut_frame.is_cut is True
    assert abs(cut_frame.orbit_distance - 6.5) < 0.15
    assert abs(math.degrees(cut_frame.orbit_azimuth) - 25.0) < 1.0
    assert abs(math.degrees(cut_frame.orbit_elevation)) < 1.0

    # Every subsequent frame of the static tripod shot stays 100% locked (zero post-cut smooth glide).
    for p in post_cut_poses[1:]:
        assert np.allclose(p.position_rh, cut_frame.position_rh, atol=1e-5)
        assert abs(p.orbit_distance - cut_frame.orbit_distance) < 1e-5
        assert abs(p.orbit_azimuth - cut_frame.orbit_azimuth) < 1e-5
        assert abs(p.orbit_elevation - cut_frame.orbit_elevation) < 1e-5


def test_dark_studio_hard_cut_detected_while_strobe_rejected() -> None:
    """Dark-to-dark studio cuts must trigger `ShotCutDetector`, whereas brightness strobes on a fixed camera must not."""
    det = ShotCutDetector(min_gap_frames=4)
    h, w = 360, 640

    def _studio_shot_wide(gain: float = 1.0) -> np.ndarray:
        img = np.full((h, w, 3), int(18 * gain), dtype=np.uint8)
        cv2.rectangle(img, (0, 260), (w, h), (int(28 * gain), int(24 * gain), int(30 * gain)), -1)
        cv2.rectangle(img, (295, 130), (345, 295), (int(150 * gain), int(110 * gain), int(120 * gain)), -1)
        return img

    def _studio_shot_closeup() -> np.ndarray:
        img = np.full((h, w, 3), 20, dtype=np.uint8)
        cv2.circle(img, (210, 145), 75, (175, 135, 145), -1)
        cv2.rectangle(img, (110, 220), (320, h), (45, 40, 52), -1)
        return img

    # Warm up 6 steady wide dark studio frames.
    for _ in range(6):
        assert det.step(_studio_shot_wide(1.0)) is False

    # Strobe brightness pulse on the SAME camera (edges unchanged -> must NOT trigger cut).
    assert det.step(_studio_shot_wide(1.75)) is False
    for _ in range(5):
        assert det.step(_studio_shot_wide(1.0)) is False

    # Hard cut to dark studio close-up (dark border in both shots, but silhouette relocates -> MUST trigger cut).
    assert det.step(_studio_shot_closeup()) is True


def test_camera_tracking_disabled_skips_cut_vo_and_camera_extraction(monkeypatch) -> None:
    """When `camera_tracking_enabled=False`, `_gpu_worker` must add zero cut/VO/camera-head overhead."""
    import queue

    import torch

    from persona_flowhmr import engine
    from persona_flowhmr.config import EngineConfig
    from persona_flowhmr.engine import StreamingMocapEngine
    from persona_flowhmr.sources import SampledFrame

    cfg = EngineConfig(preset="ultra_low", camera_tracking_enabled=False, vmc_enabled=False, plugin_api_enabled=False)
    eng = StreamingMocapEngine(cfg)

    extract_chunk_calls = 0

    class DummyExtractor:
        def extract_chunk(self, rgb_frames, bboxes, cam_k):
            nonlocal extract_chunk_calls
            extract_chunk_calls += 1
            return torch.zeros((len(rgb_frames), 3072), dtype=torch.float32)

        def extract_chunk_with_camera(self, *args, **kwargs):
            raise AssertionError("extract_chunk_with_camera must NOT run when camera_tracking_enabled=False")

    class DummyYolox:
        yolo_stride = 2

        def warmup(self, batch_size: int = 1) -> None:
            pass

        def detect_and_track_batch(self, rgb_frames, *, reset_flags=None):
            return np.tile(np.array([[100.0, 100.0, 400.0, 900.0]], dtype=np.float32), (len(rgb_frames), 1))

    class DummySolver:
        motion_style = "basic"
        last_solve_ms = 2.0

        def warmup(self, window_frames: int, steps: int = 2) -> None:
            pass

        def solve_window(self, tokens, indices, *, ode_steps: int, camera_r_w_9=None):
            assert camera_r_w_9 is None
            return np.zeros((len(indices), 491), dtype=np.float32)

    eng._yolox = DummyYolox()  # type: ignore[assignment]
    eng._solver = DummySolver()  # type: ignore[assignment]
    eng._decoder = MotionLatentDecoder(MOTION_STATS_PATH, motion_style="basic")
    monkeypatch.setattr(eng, "_ensure_models_loaded", lambda: (DummyExtractor(), "replay"))
    monkeypatch.setattr(
        engine,
        "ShotCutDetector",
        lambda *a, **kw: (_ for _ in ()).throw(AssertionError("ShotCutDetector must not be constructed when disabled")),
    )
    monkeypatch.setattr(
        engine,
        "BackgroundVisualOdometry",
        lambda *a, **kw: (_ for _ in ()).throw(
            AssertionError("BackgroundVisualOdometry must not be constructed when disabled")
        ),
    )
    monkeypatch.setattr(
        engine,
        "StreamingCameraSolver",
        lambda *a, **kw: (_ for _ in ()).throw(
            AssertionError("StreamingCameraSolver must not be constructed when disabled")
        ),
    )

    fq: queue.Queue[SampledFrame | None] = queue.Queue()
    dummy_rgb = np.zeros((240, 135, 3), dtype=np.uint8)
    for idx in range(12):
        fq.put(SampledFrame(frame_index=idx, media_time_s=idx / 30.0, capture_mono_s=10.0 + idx / 30.0, rgb=dummy_rgb))
    fq.put(None)

    warmup_done = threading.Event()
    first_commit = threading.Event()
    eng._gpu_worker(fq, warmup_done, first_commit)

    assert eng.telemetry.state != "error", eng.telemetry.error
    assert extract_chunk_calls > 0
    assert first_commit.is_set()
    sampled = eng._playout_buffer.sample_at_media_time(0.1)
    assert sampled is not None
    assert sampled.pose.camera is None


def test_playout_worker_recovers_slipped_latency_when_buffer_headroom_returns(monkeypatch) -> None:
    """Regression test: if transient GPU load slips `playout_origin_mono` upward, latency must self-heal
    back to `playout_delay_s` (`800 ms` on `ultra_low`) once buffer headroom returns, instead of staying stuck.
    """
    from persona_flowhmr import engine
    from persona_flowhmr.config import EngineConfig
    from persona_flowhmr.engine import StreamingMocapEngine
    from test_engine_settings import _FakeClock

    eng = StreamingMocapEngine(EngineConfig(preset="ultra_low", playout_fps=60, camera_tracking_enabled=False))
    identity = np.tile(np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32), (52, 1))

    # Initially push only 6 frames (0.00s..0.167s) so playout experiences a 1.2s GPU stall & underrun slip.
    eng._playout_buffer.push_many(
        [
            CanonicalPoseFrame(
                media_time_s=i / 30.0,
                capture_mono_s=1000.0 + i / 30.0,
                frame_index=i,
                quats_rh_xyzw=identity,
                hips_pos_rh=np.zeros(3, dtype=np.float32),
            )
            for i in range(6)
        ]
    )

    clock = _FakeClock(1000.0)
    monkeypatch.setattr(engine, "time", clock)
    latencies_ms: list[float] = []
    underrun_after_refill: list[int] = []

    class FakeSender:
        packets_sent = 0

        def send_pose(self, bones: object) -> None:
            self.packets_sent += 1
            latencies_ms.append(eng.telemetry.measured_latency_ms)
            if self.packets_sent == 80:
                # At tick 80 (~1.33s after start), latency has slipped well above 800ms due to underruns;
                # now refill buffer up to 15.0s so healthy headroom (> 200ms) returns!
                eng._playout_buffer.push_many(
                    [
                        CanonicalPoseFrame(
                            media_time_s=i / 30.0,
                            capture_mono_s=1000.0 + i / 30.0,
                            frame_index=i,
                            quats_rh_xyzw=identity,
                            hips_pos_rh=np.zeros(3, dtype=np.float32),
                        )
                        for i in range(6, 450)
                    ]
                )
            elif self.packets_sent > 85:
                underrun_after_refill.append(eng.telemetry.underruns)
            if self.packets_sent >= 560:
                eng._stop_event.set()

    eng._vmc_sender = FakeSender()  # type: ignore[assignment]
    eng._frame0_mono = 1000.0
    first_commit = threading.Event()
    first_commit.set()
    eng._playout_worker(first_commit, True)

    peak_slipped_ms = max(latencies_ms[:82])
    recovered_ms = latencies_ms[-1]
    assert peak_slipped_ms > 1600.0, f"Expected stall to slip latency > 1600ms, got {peak_slipped_ms}"
    assert 800.0 <= recovered_ms <= 860.0, f"Expected latency to self-heal back to ~800ms, got {recovered_ms}"
    # Recovery must never drain buffer lead into new underruns.
    assert underrun_after_refill[-1] == underrun_after_refill[0]


def test_yup_rel_to_opencv_camera_r_and_panning_overlap_tail_alignment() -> None:
    """Verify OpenCV `camera_R` sign parity with upstream FlowHMR and seam-free overlap blending during pans."""
    from persona_flowhmr.camera_tracker import yup_rel_to_opencv_camera_r

    # Camera pans right by +15 deg: in OpenCV (`+X right, +Y down, +Z forward`), `R_cv = R_y(-15 deg)`.
    ang = math.radians(-15.0)
    c, s = math.cos(ang), math.sin(ang)
    r_cv = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float32)
    cv_to_three = np.diag([1.0, -1.0, -1.0]).astype(np.float32)
    r_vo_yup = cv_to_three @ r_cv @ cv_to_three
    recovered_cv = yup_rel_to_opencv_camera_r(r_vo_yup[None])[0]
    assert np.allclose(recovered_cv, r_cv, atol=1e-6)

    # Overlap crossfade across a +30 deg camera pan between Window 1 (yaw=0) and Window 2 (yaw=+30 deg)
    # must preserve constant world heading (0 deg) and constant world velocity (+1 m/s along +Z).
    decoder = MotionLatentDecoder(MOTION_STATS_PATH, motion_style="basic")
    stitcher = StreamingStitcher(decoder, fps=30.0)

    def _encode_window(raw_yaw_deg: float, world_vz: float = 1.0) -> np.ndarray:
        lat = np.zeros((16, 491), dtype=np.float32)
        # In camera frame rotated by `window_vo_yaw = -raw_yaw_deg`, a world +Z velocity `(0, world_vz)`
        # has `srv_xz_cam = (-vx_rh, -vz_rh)` where `(vx_rh, vz_rh) = R_y(raw_yaw) @ (0, world_vz)`.
        rad = math.radians(raw_yaw_deg)
        cr, sr = math.cos(rad), math.sin(rad)
        srv_cam = np.array([-(sr * world_vz), -(cr * world_vz)], dtype=np.float32)
        lat[:, 0:2] = ((srv_cam - decoder.srv_mean[[0, 2]]) / decoder.srv_std[[0, 2]])[None, :]
        # Also rotate pelvis local joint offset `ljp_mean[0]` by `raw_yaw_deg` so wobble cancels cleanly.
        ljp_rot = decoder.ljp_mean.copy()
        ry = np.array([[cr, 0.0, sr], [0.0, 1.0, 0.0], [-sr, 0.0, cr]], dtype=np.float32)
        ljp_rot = ljp_rot @ ry.T
        lat[:, 3:159] = ((ljp_rot - decoder.ljp_mean) / decoder.ljp_std).reshape(156)[None, :]
        r_root = ry @ np.diag([-1.0, 1.0, -1.0]).astype(np.float32)
        lat[:, 159:165] = ((r_root[:, :2].reshape(6) - decoder.lrot_mean[0]) / decoder.lrot_std[0])[None, :]
        return lat

    w1 = _encode_window(0.0)
    f1 = stitcher.commit_window(
        w1,
        emit_offset_in_window=4,
        stride_frames=4,
        overlap_frames=2,
        first_emit_frame_index=0,
        capture_mono_times=[i / 30.0 for i in range(4)],
        window_vo_yaw_rad=0.0,
    )
    w2 = _encode_window(-30.0)
    f2 = stitcher.commit_window(
        w2,
        emit_offset_in_window=4,
        stride_frames=4,
        overlap_frames=2,
        first_emit_frame_index=4,
        capture_mono_times=[(4 + i) / 30.0 for i in range(4)],
        window_vo_yaw_rad=math.radians(30.0),
    )
    # Every frame across the 2-frame overlap seam keeps identity world root quaternion (0 deg twitch!).
    for pose in f1 + f2:
        assert np.allclose(pose.quats_rh_xyzw[0], [0.0, 0.0, 0.0, 1.0], atol=1e-4)
    # World X drift across the pan seam stays zero while Z advances steadily at 1 m/s.
    assert abs(float(f2[-1].hips_pos_rh[0] - f1[0].hips_pos_rh[0])) < 1e-3
    assert abs(float(f2[-1].hips_pos_rh[2] - f1[0].hips_pos_rh[2]) - (7.0 / 30.0)) < 1e-2


def test_cut_roll_and_live_framing_rebase_on_static_tripod() -> None:
    """Verify cut-frame roll/steep-tilt recovery, live `avatar_scale` / `follow_movement` tripod rebase,
    and preserved travel origin across `reset(keep_origin=True)`.
    """
    decoder = MotionLatentDecoder(MOTION_STATS_PATH, motion_style="basic")
    cam_solver = StreamingCameraSolver(avatar_scale=1.0)
    h, w = 720, 1280
    cam_k = default_camera_intrinsics(h, w)
    cv_to_three = np.diag([1.0, -1.0, -1.0]).astype(np.float32)

    dec0 = decoder.decode_window(np.zeros((1, 491), dtype=np.float32))
    ljp_neutral = dec0["ljp_cam"][0]
    hips_origin = np.array([-float(ljp_neutral[0, 0]), float(dec0["srp_y_cam"][0]), -float(ljp_neutral[0, 2])], dtype=np.float32)
    cam_solver.anchor_stream_origin(hips_origin)

    # Simulate live-enabling camera tracking after avatar has already walked +1.5m along X:
    cam_solver.reset(keep_origin=True)
    hips_displaced = hips_origin + np.array([1.5, 0.0, 0.0], dtype=np.float32)
    stage_j14 = extract_smplh_body14_stage(ljp_neutral, hips_displaced)

    # Cut to a camera at dist=3.2m, el=-32 deg (steep tilt > 22 deg), roll=+16 deg framing the displaced avatar.
    c_pos, r_w_cam = _persona_orbit_to_three_matrix_and_pos(
        {
            "orbit": {
                "azimuth": 0.0,
                "elevation": math.radians(-32.0),
                "distance": 3.2,
                "targetX": 1.5,
                "targetY": float(hips_origin[1]),
                "targetZ": 0.0,
            },
            "roll": math.radians(16.0),
        }
    )
    j14_cv = ((r_w_cam.T @ (stage_j14 - c_pos[None, :]).T).T @ cv_to_three.T).astype(np.float32)
    obs_cut = FrameCameraObservation(
        sam_cam_t=np.array([0.0, 0.9, 3.2], dtype=np.float32),
        sam_j3d_cam=j14_cv,
        cam_k=cam_k,
        image_hw=(h, w),
        bbox_xyxy=np.array([450.0, 100.0, 830.0, 680.0], dtype=np.float32),
        r_vo_yup=np.eye(3, dtype=np.float32),
        vo_inliers=28,
        is_cut=True,
        shot_id=1,
        trans_flow_px=0.05,
    )
    p_cut = cam_solver.solve_frame(obs_cut, ljp_cam_52=ljp_neutral, hips_pos_rh=hips_displaced, follow_movement=True)
    assert abs(math.degrees(p_cut.orbit_elevation) - (-32.0)) < 2.0
    assert abs(math.degrees(p_cut.roll) - 16.0) < 2.0
    assert abs(float(p_cut.orbit_target_rh[0]) - 1.5) < 0.15

    # Next frame on locked tripod (`vo_inliers=28, trans_flow_px=0.05`), change `avatar_scale` live to 1.5
    # and `follow_movement=False`: camera must immediately rebase instead of staying frozen!
    obs_static = FrameCameraObservation(
        sam_cam_t=obs_cut.sam_cam_t,
        sam_j3d_cam=j14_cv,
        cam_k=cam_k,
        image_hw=(h, w),
        bbox_xyxy=obs_cut.bbox_xyxy,
        r_vo_yup=np.eye(3, dtype=np.float32),
        vo_inliers=28,
        is_cut=False,
        shot_id=1,
        trans_flow_px=0.05,
    )
    cam_solver.avatar_scale = 1.5
    p_rebased = cam_solver.solve_frame(obs_static, ljp_cam_52=ljp_neutral, hips_pos_rh=hips_displaced, follow_movement=False)
    assert p_rebased.orbit_distance > p_cut.orbit_distance * 1.35
    assert abs(float(p_rebased.orbit_target_rh[0])) < 0.25

    # Changing calibrated FOV live on a locked tripod must also rebase camera distance instead of keeping stale distance.
    k_narrow = default_camera_intrinsics(h, w).copy()
    k_narrow[0, 0] *= 1.5
    k_narrow[1, 1] *= 1.5
    j14_cv_narrow = j14_cv.copy()
    j14_cv_narrow[:, 2] *= 1.5
    obs_recal = FrameCameraObservation(
        sam_cam_t=obs_cut.sam_cam_t * np.array([1.0, 1.0, 1.5], dtype=np.float32),
        sam_j3d_cam=j14_cv_narrow,
        cam_k=k_narrow,
        image_hw=(h, w),
        bbox_xyxy=obs_cut.bbox_xyxy,
        r_vo_yup=np.eye(3, dtype=np.float32),
        vo_inliers=28,
        is_cut=False,
        shot_id=1,
        trans_flow_px=0.05,
    )
    cam_solver.calibrated_fov_deg = 25.0
    p_fov_rebased = cam_solver.solve_frame(
        obs_recal,
        ljp_cam_52=ljp_neutral,
        hips_pos_rh=hips_displaced,
        follow_movement=False,
    )
    assert p_fov_rebased.orbit_distance > p_rebased.orbit_distance * 1.35


def test_preview_per_frame_dims_and_eviction_preserves_camera_switch() -> None:
    """Verify `PreviewRingBuffer` keeps per-frame `orig_hw` across resolution changes and `_put_until_stopped`
    carries evicted `is_camera_switch` markers forward.
    """
    import queue

    from persona_flowhmr.engine import StreamingMocapEngine
    from persona_flowhmr.sources import PreviewRingBuffer, SampledFrame

    ring = PreviewRingBuffer(max_items=8, max_width=320)
    ring.record_frame(0, np.zeros((720, 1280, 3), dtype=np.uint8))
    ring.record_frame(1, np.zeros((480, 640, 3), dtype=np.uint8))
    snap0 = ring.get_synced_bgr(0)
    snap1 = ring.get_synced_bgr(1)
    assert snap0 is not None and snap0[3] == (720, 1280)
    assert snap1 is not None and snap1[3] == (480, 640)

    eng = StreamingMocapEngine()
    fq: queue.Queue[SampledFrame | None] = queue.Queue(maxsize=2)
    rgb = np.zeros((64, 64, 3), dtype=np.uint8)
    eng._put_until_stopped(fq, SampledFrame(0, 0.0, 1.0, rgb, is_camera_switch=True), drop_oldest=True)
    eng._put_until_stopped(fq, SampledFrame(1, 0.033, 1.033, rgb), drop_oldest=True)
    # Frame 2 evicts Frame 0 (`is_camera_switch=True`); `_take_batch` must preserve `is_camera_switch` on Frame 1!
    eng._put_until_stopped(fq, SampledFrame(2, 0.066, 1.066, rgb), drop_oldest=True)
    fq.put_nowait(None) if not fq.full() else None
    batch, _ = eng._take_batch(fq, 2)
    assert len(batch) == 2
    assert batch[0].frame_index == 1
    assert batch[0].is_camera_switch is True

    # Evicting frame 11 while consumer already holds frame 10 must NOT mark frame 10; frame 12 must receive it.
    eng2 = StreamingMocapEngine()
    fq2: queue.Queue[SampledFrame | None] = queue.Queue(maxsize=2)
    eng2._put_until_stopped(fq2, SampledFrame(10, 0.33, 1.33, rgb), drop_oldest=True)
    # Simulate producer evicting frame 11 (`_evicted_switch_min_frame = 11`) before consumer inspects frame 10:
    with eng2._evicted_lock:
        eng2._evicted_switch_min_frame = 11
    eng2._put_until_stopped(fq2, SampledFrame(12, 0.40, 1.40, rgb), drop_oldest=True)
    batch2, _ = eng2._take_batch(fq2, 2)
    assert [f.frame_index for f in batch2] == [10, 12]
    assert batch2[0].is_camera_switch is False
    assert batch2[1].is_camera_switch is True


def test_persona_sync_camera_pose_refreshes_placement_and_fails_closed() -> None:
    """Verify `sync_camera_pose_fire_and_forget` loads placement even without `ensure_vmc_receiver`,
    skips sending when `instance.get` fails, and picks up updated placement after an event.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]

    state = {"fail_get": True, "offset_x": 2.5}
    patched: list[dict] = []
    server_ws_holder: list[object] = []

    async def handler(ws):
        server_ws_holder.append(ws)
        await ws.send(
            json.dumps(
                {
                    "kind": "hello",
                    "protocol": 4,
                    "app": {"name": "LAPLACE Persona", "version": "0.91.0", "platform": "win32", "capabilities": []},
                }
            )
        )
        async for raw in ws:
            msg = json.loads(raw)
            m = msg["method"]
            req_id = msg["id"]
            if m == "instance.list":
                res = {"instances": [{"instanceId": "vrm-1", "kind": "model", "format": "vrm"}]}
                await ws.send(json.dumps({"kind": "response", "id": req_id, "result": res}))
            elif m == "instance.get":
                if state["fail_get"]:
                    await ws.send(json.dumps({"kind": "error", "id": req_id, "code": "ERR", "message": "lookup failed"}))
                else:
                    res = {"item": {"vrm": {"x": state["offset_x"], "y": 0.0, "z": 0.0, "scale": 1.0, "rotY": 0.0}}}
                    await ws.send(json.dumps({"kind": "response", "id": req_id, "result": res}))
            elif m == "scene.patch":
                patched.append(msg["params"]["vrmCamera"])
                await ws.send(json.dumps({"kind": "response", "id": req_id, "result": {}}))
            else:
                await ws.send(json.dumps({"kind": "response", "id": req_id, "result": {}}))

    stop_srv = threading.Event()
    srv_loop_holder: list[asyncio.AbstractEventLoop] = []

    def _run_srv() -> None:
        loop = asyncio.new_event_loop()
        srv_loop_holder.append(loop)
        asyncio.set_event_loop(loop)

        async def _main() -> None:
            async with websockets.serve(handler, "127.0.0.1", port):
                while not stop_srv.is_set():
                    await asyncio.sleep(0.02)

        loop.run_until_complete(_main())

    t = threading.Thread(target=_run_srv, daemon=True)
    t.start()

    cam_pose = decompose_three_camera_pose(
        np.array([0.0, 1.0, 3.0], dtype=np.float32),
        np.eye(3, dtype=np.float32),
        3.0,
        30.0,
    )
    client = PersonaPluginClient(url=f"ws://127.0.0.1:{port}", token="tok")
    try:
        assert client.start(timeout_s=5.0) is True
        # 1. While `instance.get` fails, `sync_camera_pose_fire_and_forget` must fail closed (0 patches sent).
        client.sync_camera_pose_fire_and_forget(cam_pose)
        threading.Event().wait(0.12)
        assert patched == []

        # 2. Once `instance.get` succeeds (even without `ensure_vmc_receiver`), camera transforms by `offset_x=2.5`!
        state["fail_get"] = False
        client.sync_camera_pose_fire_and_forget(cam_pose)
        for _ in range(40):
            if patched:
                break
            threading.Event().wait(0.02)
        assert len(patched) == 1
        assert abs(float(patched[-1]["orbit"]["targetX"]) - 2.5) < 1e-4

        # 3. Server event invalidates cached placement -> next sync picks up `offset_x = -1.25` immediately!
        state["offset_x"] = -1.25
        ws_conn = server_ws_holder[-1]
        asyncio.run_coroutine_threadsafe(
            ws_conn.send(json.dumps({"kind": "event", "event": "scene.changed"})),  # type: ignore[attr-defined]
            srv_loop_holder[0],
        ).result(timeout=2.0)
        threading.Event().wait(0.05)
        client.sync_camera_pose_fire_and_forget(cam_pose)
        for _ in range(40):
            if len(patched) >= 2:
                break
            threading.Event().wait(0.02)
        assert len(patched) >= 2
        assert abs(float(patched[-1]["orbit"]["targetX"]) - (-1.25)) < 1e-4

        # 4. Calling legacy `send_vrm_camera_fire_and_forget` right after `sync_camera_pose_fire_and_forget`
        # shares `_drain_camera_queue` and never strands a pending update.
        legacy_cam = {
            "orbit": {
                "azimuth": 0.25,
                "elevation": 0.0,
                "distance": 4.2,
                "targetX": 0.5,
                "targetY": 1.0,
                "targetZ": 0.0,
            },
            "roll": 0.0,
            "fov": 35.0,
        }
        client.sync_camera_pose_fire_and_forget(cam_pose)
        client.send_vrm_camera_fire_and_forget(legacy_cam)
        for _ in range(40):
            if patched and abs(float(patched[-1]["orbit"]["distance"]) - 4.2) < 1e-4:
                break
            threading.Event().wait(0.02)
        assert abs(float(patched[-1]["orbit"]["distance"]) - 4.2) < 1e-4
    finally:
        client.stop()
        stop_srv.set()
        t.join(timeout=2.0)


def test_vo_optical_zoom_vs_wall_dolly_and_pre_cut_incoming_crossfade() -> None:
    """Verify VO accepts optical zoom when foreground+background magnify together, rejects dolly toward a wall
    (where closer foreground expands faster) and calibrated fixed-FOV cameras, and `save_tail=False` preserves
    incoming overlap crossfading before a cut.
    """
    import cv2

    rng = np.random.default_rng(7)
    h, w = 360, 640
    cam_k = default_camera_intrinsics(h, w)
    # Build rich synthetic background texture outside center performer box.
    base = np.zeros((h, w, 3), dtype=np.uint8)
    for _ in range(180):
        cx_i = int(rng.integers(20, w - 20))
        cy_i = int(rng.integers(20, h - 20))
        col = tuple(int(v) for v in rng.integers(80, 255, size=3))
        cv2.circle(base, (cx_i, cy_i), int(rng.integers(4, 9)), col, -1)

    zoom_factor = 1.03
    m_zoom = np.array(
        [
            [zoom_factor, 0.0, (1.0 - zoom_factor) * (w * 0.5)],
            [0.0, zoom_factor, (1.0 - zoom_factor) * (h * 0.5)],
        ],
        dtype=np.float32,
    )
    frame_zoomed = cv2.warpAffine(base, m_zoom, (w, h), flags=cv2.INTER_LINEAR)
    box0 = np.array([260.0, 70.0, 380.0, 310.0], dtype=np.float32)

    def _scale_box(b: np.ndarray, s: float) -> np.ndarray:
        bcx, bcy = 0.5 * (b[0] + b[2]), 0.5 * (b[1] + b[3])
        hw_b, hh_b = 0.5 * (b[2] - b[0]) * s, 0.5 * (b[3] - b[1]) * s
        return np.array([bcx - hw_b, bcy - hh_b, bcx + hw_b, bcy + hh_b], dtype=np.float32)

    # 1. Optical zoom (`allow_zoom=True`): foreground scales by same 1.03x as background -> `zoom_scale` increases!
    vo_zoom = BackgroundVisualOdometry()
    vo_zoom.step(base, box0, cam_k)
    vo_zoom.step(frame_zoomed, _scale_box(box0, zoom_factor), cam_k, allow_zoom=True)
    assert vo_zoom.zoom_scale > 1.02
    assert vo_zoom.last_trans_flow_px < 0.35

    # 2. Calibrated fixed-FOV camera (`allow_zoom=False`): `zoom_scale` stays fixed at 1.0!
    vo_cal = BackgroundVisualOdometry()
    vo_cal.step(base, box0, cam_k, allow_zoom=False)
    vo_cal.step(frame_zoomed, _scale_box(box0, zoom_factor), cam_k, allow_zoom=False)
    assert abs(vo_cal.zoom_scale - 1.0) < 1e-6

    # 3. Dolly toward textured wall (`Z_person < Z_wall`): closer foreground expands 2x faster (1.06x vs 1.03x) -> NOT zoom!
    vo_dolly = BackgroundVisualOdometry()
    vo_dolly.step(base, box0, cam_k, allow_zoom=True)
    vo_dolly.step(frame_zoomed, _scale_box(box0, 1.06), cam_k, allow_zoom=True)
    assert abs(vo_dolly.zoom_scale - 1.0) < 1e-6
    assert vo_dolly.last_trans_flow_px > 0.8

    # 4. `save_tail=False` on shortened pre-cut stride blends incoming tail smoothly while dropping outgoing tail.
    decoder = MotionLatentDecoder(MOTION_STATS_PATH, motion_style="basic")
    stitcher = StreamingStitcher(decoder, fps=30.0)
    w_a = np.zeros((16, 491), dtype=np.float32)
    w_b = np.zeros((16, 491), dtype=np.float32)
    w_b[:, 2] = 2.0  # shifted pelvis height `srp_y` in second window
    f_a = stitcher.commit_window(
        w_a,
        emit_offset_in_window=4,
        stride_frames=4,
        overlap_frames=2,
        first_emit_frame_index=0,
        capture_mono_times=[i / 30.0 for i in range(4)],
        save_tail=True,
    )
    dec_b_raw = decoder.decode_window(w_b[4:6])["srp_y_cam"]
    f_b = stitcher.commit_window(
        w_b,
        emit_offset_in_window=4,
        stride_frames=2,
        overlap_frames=2,
        first_emit_frame_index=4,
        capture_mono_times=[(4 + i) / 30.0 for i in range(2)],
        save_tail=False,
    )
    # Frame 4 was crossfaded between `f_a` pelvis height and `dec_b_raw[0]`, and outgoing `_tail` was cleared!
    assert float(f_a[-1].hips_pos_rh[1]) < float(f_b[0].hips_pos_rh[1]) < float(dec_b_raw[0])
    assert stitcher._tail is None

    # 5. One-sided background (left half black, features only on right half) still recovers optical zoom!
    one_sided = base.copy()
    one_sided[:, : w // 2 + 20] = 0
    one_sided_zoomed = cv2.warpAffine(one_sided, m_zoom, (w, h), flags=cv2.INTER_LINEAR)
    box_left = np.array([60.0, 70.0, 180.0, 310.0], dtype=np.float32)
    vo_oneside = BackgroundVisualOdometry()
    vo_oneside.step(one_sided, box_left, cam_k, allow_zoom=True)
    vo_oneside.step(one_sided_zoomed, _scale_box(box_left, zoom_factor), cam_k, allow_zoom=True)
    assert vo_oneside.zoom_scale > 1.02

    # 6. Sustained sub-threshold slow pan (~0.14 px/frame < 0.28 deadband) accumulates across frames!
    vo_slow = BackgroundVisualOdometry()
    vo_slow.step(base, box0, cam_k, allow_zoom=False)
    for k in range(1, 16):
        dx = -0.18 * k  # ~0.135 px/frame at 480p long-side
        m_pan = np.array([[1.0, 0.0, dx], [0.0, 1.0, 0.0]], dtype=np.float32)
        panned = cv2.warpAffine(base, m_pan, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        r_cur, _ = vo_slow.step(panned, box0, cam_k, allow_zoom=False)
    yaw_accum_deg = abs(math.degrees(math.atan2(float(r_cur[2, 0]), float(r_cur[2, 2]))))
    assert yaw_accum_deg > 0.12


def test_stalled_camera_reader_close_releases_device(monkeypatch) -> None:
    """Verify `CameraSource.close()` releases active capture in bounded time even when `cap.read()` is stalled."""
    import time

    from persona_flowhmr import sources

    released = threading.Event()
    unblock_read = threading.Event()

    class StalledCap:
        def __init__(self) -> None:
            self._primed = False

        def isOpened(self) -> bool:
            return True

        def set(self, *_args) -> bool:
            return True

        def read(self):
            if not self._primed:
                self._primed = True
                return True, np.zeros((64, 64, 3), dtype=np.uint8)
            unblock_read.wait(timeout=5.0)
            return False, None

        def release(self) -> None:
            released.set()

    monkeypatch.setattr(sources.cv2, "VideoCapture", lambda *_a, **_kw: StalledCap())
    cam = sources.CameraSource(0, width=64, height=64, target_fps=30.0)
    time.sleep(0.05)  # ensure `_reader_loop` is blocked inside second `cap.read()`
    t0 = time.perf_counter()
    try:
        cam.close()
        elapsed = time.perf_counter() - t0
        assert released.is_set(), "Expected close() to release capture even while reader is stalled in read()"
        assert elapsed < 2.0
    finally:
        unblock_read.set()
