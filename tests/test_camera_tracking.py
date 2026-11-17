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
        engine.ShotCutDetector,
        "step",
        lambda *a, **kw: (_ for _ in ()).throw(AssertionError("ShotCutDetector.step must not run when disabled")),
    )
    monkeypatch.setattr(
        engine.BackgroundVisualOdometry,
        "step",
        lambda *a, **kw: (_ for _ in ()).throw(AssertionError("BackgroundVisualOdometry.step must not run when disabled")),
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
