"""Streaming FlowHMR latent decoder and overlap crossfade stitcher.

Eliminates `smplh/neutral/model.npz` by decoding FlowHMR's 491-dim latent directly via
`motion_stats.json`, integrating `smooth_root_vel_xz` continuously across window seams,
anchoring world heading across multi-camera cuts, and crossfading all 52 canonical joint
quaternions via antipodal SLERP over `O` overlap frames.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from .config import MODEL_FPS
from .retarget import (
    SMPLH_PARENTS,
    CanonicalPoseFrame,
    canonicalize_smplh_rotations,
    quat_mul_xyzw,
    rot6d_to_matrix_np,
    slerp_quat_xyzw,
)

if TYPE_CHECKING:
    from .camera_tracker import FrameCameraObservation, StreamingCameraSolver


def _global_to_local_mats(global_mats: np.ndarray, parents: tuple[int, ...]) -> np.ndarray:
    """Convert `(T, J, 3, 3)` global rotation matrices to local parent-relative matrices."""
    local = np.empty_like(global_mats)
    local[:, 0] = global_mats[:, 0]
    for j in range(1, global_mats.shape[1]):
        p = parents[j]
        # R_local = R_parent^T @ R_global
        local[:, j] = np.matmul(np.swapaxes(global_mats[:, p], -1, -2), global_mats[:, j])
    return local


def _root_heading_rad(quat_xyzw: np.ndarray) -> float:
    """Horizontal world heading `atan2(fwd_x, fwd_z)` of a canonical root quaternion `(x, y, z, w)`."""
    x, y, z, w = float(quat_xyzw[0]), float(quat_xyzw[1]), float(quat_xyzw[2]), float(quat_xyzw[3])
    fwd_x = 2.0 * (x * z + w * y)
    fwd_z = 1.0 - 2.0 * (x * x + y * y)
    return math.atan2(fwd_x, fwd_z)


def _unwrap_rad(prev: float, target: float) -> float:
    return prev + ((target - prev + math.pi) % (2.0 * math.pi) - math.pi)


class MotionLatentDecoder:
    """NumPy/CPU decoder for FlowHMR's 491-dim motion latent (`basic`, `global`, or `mixed`)."""

    SPLIT_INDICES = (2, 3, 159, 471, 487)  # [2, 1, 156, 312, 16, 4] -> total 491

    def __init__(self, motion_stats_path: str | Path, motion_style: str = "basic") -> None:
        if motion_style not in ("basic", "global", "mixed"):
            raise ValueError(f"Unsupported motion_style: {motion_style}")
        self.motion_style = motion_style

        with open(motion_stats_path, encoding="utf-8") as f:
            stats = json.load(f)

        def _pair(key: str, shape: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
            mean = np.asarray(stats[key]["mean"], dtype=np.float32).reshape(shape)
            std = np.maximum(np.asarray(stats[key]["std"], dtype=np.float32), 1e-5).reshape(shape)
            return mean, std

        self.srv_mean, self.srv_std = _pair("smooth_root_vel", (3,))
        self.srp_mean, self.srp_std = _pair("smooth_root_pos", (3,))
        self.ljp_mean, self.ljp_std = _pair("local_joints_positions", (52, 3))
        self.grot_mean, self.grot_std = _pair("global_rot_data", (52, 6))
        self.lrot_mean, self.lrot_std = _pair("local_rot_data", (52, 6))

    def decode_window(self, latent_t_491: np.ndarray) -> dict[str, np.ndarray]:
        """Decode `(T, 491)` normalized latent into per-frame root motion, local joints, and canonical quats."""
        latent = np.asarray(latent_t_491, dtype=np.float32)
        if latent.ndim != 2 or latent.shape[1] != 491:
            raise ValueError(f"Expected (T, 491) latent, got {latent.shape}")
        t_len = latent.shape[0]

        srv_xz_n, srp_y_n, ljp_n, rot_n, _shapes_n, _foot_contacts_n = np.split(latent, self.SPLIT_INDICES, axis=-1)

        srv_xz = srv_xz_n * self.srv_std[[0, 2]] + self.srv_mean[[0, 2]]  # (T, 2) m/s in camera XZ
        srp_y = (srp_y_n[:, 0] * self.srp_std[1]) + self.srp_mean[1]       # (T,) m in camera Y
        ljp = ljp_n.reshape(t_len, 52, 3) * self.ljp_std + self.ljp_mean   # (T, 52, 3)
        rot_data = rot_n.reshape(t_len, 52, 6)

        if self.motion_style == "basic":
            local_rot6d = rot_data * self.lrot_std + self.lrot_mean
            local_mats = rot6d_to_matrix_np(local_rot6d)
        elif self.motion_style == "global":
            global_rot6d = rot_data * self.grot_std + self.grot_mean
            global_mats = rot6d_to_matrix_np(global_rot6d)
            local_mats = _global_to_local_mats(global_mats, SMPLH_PARENTS)
        else:  # mixed
            body_g6d = rot_data[:, :22] * self.grot_std[:22] + self.grot_mean[:22]
            hand_l6d = rot_data[:, 22:] * self.lrot_std[22:] + self.lrot_mean[22:]
            body_lmats = _global_to_local_mats(rot6d_to_matrix_np(body_g6d), SMPLH_PARENTS[:22])
            hand_lmats = rot6d_to_matrix_np(hand_l6d)
            local_mats = np.concatenate([body_lmats, hand_lmats], axis=1)

        return {
            "srv_xz_cam": srv_xz,                                     # (T, 2)
            "srp_y_cam": srp_y,                                       # (T,)
            "ljp_cam": ljp,                                           # (T, 52, 3); joint 0 = local pelvis
            "quats_rh_xyzw": canonicalize_smplh_rotations(local_mats),  # (T, 52, 4) canonical +Z-forward
        }


class StreamingStitcher:
    """Crossfades overlapping sliding-window solves into a C0-continuous 30 Hz pose stream."""

    def __init__(self, decoder: MotionLatentDecoder, fps: float = MODEL_FPS) -> None:
        self.decoder = decoder
        self.fps = float(fps)
        # Uncommitted `O`-frame tail of the previous window (decoded fields), crossfaded into the next stride.
        self._tail: dict[str, np.ndarray] | None = None
        self._tail_start_frame = 0
        self._tail_vo_yaw_rad = 0.0
        self._cum_root_xz_rh = np.zeros(2, dtype=np.float64)
        self._has_started = False
        self._shot_world_yaw_rad = 0.0
        self._last_world_heading_rad: float | None = None

    def commit_window(
        self,
        latent_window: np.ndarray,
        *,
        emit_offset_in_window: int,
        stride_frames: int,
        overlap_frames: int,
        first_emit_frame_index: int,
        capture_mono_times: list[float],
        reset_flags: list[bool] | None = None,
        cut_flags: list[bool] | None = None,
        window_vo_yaw_rad: float = 0.0,
        camera_observations: list[FrameCameraObservation] | None = None,
        camera_solver: StreamingCameraSolver | None = None,
        follow_movement: bool = True,
        anchor_world_yaw_across_cuts: bool = True,
    ) -> list[CanonicalPoseFrame]:
        """Decode `latent_window`, crossfade first `O` frames with prior tail, and commit `S` frames.

        Window slice layout starting at `emit_offset_in_window`:
          `[0 .. S - 1]`     -> committed and returned on this call (`len == stride_frames`)
          `[S .. S + O - 1]` -> stored as `_tail` to crossfade against `[0 .. O - 1]` of the next window
        """
        s_idx = int(emit_offset_in_window)
        n_commit = int(stride_frames)
        e_tail = s_idx + n_commit + int(overlap_frames)
        if e_tail > latent_window.shape[0]:
            raise ValueError(
                f"Window length {latent_window.shape[0]} too short for emit_offset={s_idx}, "
                f"stride={stride_frames}, overlap={overlap_frames}"
            )
        if len(capture_mono_times) != stride_frames:
            raise ValueError(f"Expected {stride_frames} capture_mono_times, got {len(capture_mono_times)}")

        # Decoding is per-frame, so only the committed stride + overlap tail are decoded.
        decoded = self.decoder.decode_window(latent_window[s_idx:e_tail])
        cur = {k: v[:n_commit].copy() for k, v in decoded.items()}

        starts_with_cut_or_reset = bool(
            (reset_flags and reset_flags[0]) or (cut_flags and cut_flags[0])
        )
        # Blend first `O` frames of this stride against the previous window's overlap tail (within the same shot).
        tail = self._tail
        if (
            tail is not None
            and overlap_frames > 0
            and self._tail_start_frame == first_emit_frame_index
            and not starts_with_cut_or_reset
        ):
            n_blend = min(overlap_frames, stride_frames, tail["quats_rh_xyzw"].shape[0])
            tail_srv = tail["srv_xz_cam"][:n_blend].copy()
            tail_srp = tail["srp_y_cam"][:n_blend]
            tail_ljp = tail["ljp_cam"][:n_blend].copy()
            tail_quats = tail["quats_rh_xyzw"][:n_blend].copy()

            # Rotate previous window's tail into the current window's `window_vo_yaw_rad` reference frame
            # before blending so camera pans never introduce heading or velocity seam errors.
            delta_tail_yaw = self._tail_vo_yaw_rad - float(window_vo_yaw_rad)
            if abs(delta_tail_yaw) > 1e-7:
                cd, sd = math.cos(delta_tail_yaw), math.sin(delta_tail_yaw)
                sx, sz = tail_srv[:, 0].copy(), tail_srv[:, 1].copy()
                tail_srv[:, 0] = cd * sx + sd * sz
                tail_srv[:, 1] = -sd * sx + cd * sz
                jx, jz = tail_ljp[:, :, 0].copy(), tail_ljp[:, :, 2].copy()
                tail_ljp[:, :, 0] = cd * jx + sd * jz
                tail_ljp[:, :, 2] = -sd * jx + cd * jz
                q_delta = np.array(
                    [0.0, math.sin(0.5 * delta_tail_yaw), 0.0, math.cos(0.5 * delta_tail_yaw)],
                    dtype=np.float32,
                )
                tail_quats[:, 0] = quat_mul_xyzw(np.broadcast_to(q_delta, tail_quats[:, 0].shape), tail_quats[:, 0])

            # Strictly interior ramp (1/(N+1) .. N/(N+1)) so frame -1 (prev window) and frame N (new window)
            # join without endpoint derivative snaps.
            alphas = np.linspace(0.0, 1.0, n_blend + 2, dtype=np.float32)[1:-1]
            cur["srv_xz_cam"][:n_blend] = (1.0 - alphas[:, None]) * tail_srv + alphas[:, None] * cur["srv_xz_cam"][:n_blend]
            cur["srp_y_cam"][:n_blend] = (1.0 - alphas) * tail_srp + alphas * cur["srp_y_cam"][:n_blend]
            cur["ljp_cam"][:n_blend] = (
                (1.0 - alphas[:, None, None]) * tail_ljp + alphas[:, None, None] * cur["ljp_cam"][:n_blend]
            )
            cur["quats_rh_xyzw"][:n_blend] = slerp_quat_xyzw(
                tail_quats,
                cur["quats_rh_xyzw"][:n_blend],
                alphas[:, None],
            )

        # Store uncommitted tail `[S .. S + O - 1]` for the next window's crossfade.
        if overlap_frames > 0:
            self._tail = {k: v[n_commit:].copy() for k, v in decoded.items()}
            self._tail_start_frame = first_emit_frame_index + stride_frames
            self._tail_vo_yaw_rad = float(window_vo_yaw_rad)
        else:
            self._tail = None

        srv_xz, srp_y, ljp, quats = cur["srv_xz_cam"], cur["srp_y_cam"], cur["ljp_cam"], cur["quats_rh_xyzw"]
        frames: list[CanonicalPoseFrame] = []
        dt = 1.0 / self.fps
        for i in range(stride_frames):
            is_rst = bool(reset_flags[i]) if reset_flags is not None else False
            is_cut = bool(cut_flags[i]) if cut_flags is not None else False

            if is_rst:
                self._cum_root_xz_rh[:] = 0.0
                self._shot_world_yaw_rad = 0.0
                self._last_world_heading_rad = None
                self._has_started = False
            elif is_cut and anchor_world_yaw_across_cuts and self._last_world_heading_rad is not None:
                raw_heading = _root_heading_rad(quats[i, 0])
                # Subtract `window_vo_yaw_rad` (0 on cut frame) so `total_yaw` makes root heading C0-continuous.
                self._shot_world_yaw_rad = _unwrap_rad(
                    self._shot_world_yaw_rad,
                    self._last_world_heading_rad - raw_heading - float(window_vo_yaw_rad),
                )

            total_yaw = self._shot_world_yaw_rad + float(window_vo_yaw_rad)
            cy, sy = math.cos(total_yaw), math.sin(total_yaw)

            # Convert FlowHMR camera-space XZ (`+X` left-of-camera, `+Z` away) via `R_y(180)` (`(-x, -z)`)
            # and rotate by `total_yaw` into consistent canonical +Z-forward stage space.
            vx_rh = -float(srv_xz[i, 0])
            vz_rh = -float(srv_xz[i, 1])
            vx_world = cy * vx_rh + sy * vz_rh
            vz_world = -sy * vx_rh + cy * vz_rh

            if self._has_started:
                self._cum_root_xz_rh[0] += vx_world * dt
                self._cum_root_xz_rh[1] += vz_world * dt
            else:
                self._has_started = True

            wobble_x_rh = -float(ljp[i, 0, 0])
            wobble_z_rh = -float(ljp[i, 0, 2])
            wx_world = cy * wobble_x_rh + sy * wobble_z_rh
            wz_world = -sy * wobble_x_rh + cy * wobble_z_rh

            hips_pos_rh = np.array(
                [
                    float(self._cum_root_xz_rh[0] + wx_world),
                    float(srp_y[i]),
                    float(self._cum_root_xz_rh[1] + wz_world),
                ],
                dtype=np.float32,
            )

            q_frame = quats[i].copy()
            if abs(total_yaw) > 1e-7:
                q_yaw = np.array([0.0, math.sin(0.5 * total_yaw), 0.0, math.cos(0.5 * total_yaw)], dtype=np.float32)
                q_frame[0] = quat_mul_xyzw(q_yaw, q_frame[0])
            self._last_world_heading_rad = _root_heading_rad(q_frame[0])

            cam_pose = None
            if camera_solver is not None and camera_observations is not None:
                cam_pose = camera_solver.solve_frame(
                    camera_observations[i],
                    ljp_cam_52=ljp[i],
                    hips_pos_rh=hips_pos_rh,
                    follow_movement=follow_movement,
                    world_yaw_rad=total_yaw,
                    is_stream_reset=is_rst,
                )

            f_idx = first_emit_frame_index + i
            frames.append(
                CanonicalPoseFrame(
                    media_time_s=f_idx / self.fps,
                    capture_mono_s=float(capture_mono_times[i]),
                    frame_index=f_idx,
                    quats_rh_xyzw=q_frame,
                    hips_pos_rh=hips_pos_rh,
                    joints_cam_3d=ljp[i, :22].copy(),
                    camera=cam_pose,
                )
            )
        return frames
