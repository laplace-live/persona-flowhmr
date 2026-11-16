"""Streaming FlowHMR latent decoder and overlap crossfade stitcher.

Eliminates `smplh/neutral/model.npz` by decoding FlowHMR's 491-dim latent directly via
`motion_stats.json`, integrating `smooth_root_vel_xz` continuously across window seams,
and crossfading all 52 canonical joint quaternions via antipodal SLERP over `O` overlap frames.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .retarget import (
    SMPLH_PARENTS,
    CanonicalPoseFrame,
    canonicalize_smplh_sequence,
    rot6d_to_matrix_np,
    slerp_quat_xyzw,
)


def _global_to_local_mats(global_mats: np.ndarray, parents: tuple[int, ...]) -> np.ndarray:
    """Convert `(T, J, 3, 3)` global rotation matrices to local parent-relative matrices."""
    local = np.empty_like(global_mats)
    local[:, 0] = global_mats[:, 0]
    for j in range(1, global_mats.shape[1]):
        p = parents[j]
        # R_local = R_parent^T @ R_global
        local[:, j] = np.matmul(np.swapaxes(global_mats[:, p], -1, -2), global_mats[:, j])
    return local


class MotionLatentDecoder:
    """NumPy/CPU decoder for FlowHMR's 491-dim motion latent (`basic`, `global`, or `mixed`)."""

    SPLIT_INDICES = (2, 3, 159, 471, 487)  # [2, 1, 156, 312, 16, 4] -> total 491

    def __init__(self, motion_stats_path: str | Path, motion_style: str = "basic", fps: float = 30.0) -> None:
        if motion_style not in ("basic", "global", "mixed"):
            raise ValueError(f"Unsupported motion_style: {motion_style}")
        self.motion_style = motion_style
        self.fps = float(fps)

        with open(motion_stats_path, encoding="utf-8") as f:
            stats = json.load(f)

        def _pair(key: str, shape: tuple[int, ...] | None = None) -> tuple[np.ndarray, np.ndarray]:
            mean = np.asarray(stats[key]["mean"], dtype=np.float32)
            std = np.maximum(np.asarray(stats[key]["std"], dtype=np.float32), 1e-5)
            if shape is not None:
                mean = mean.reshape(shape)
                std = std.reshape(shape)
            return mean, std

        self.srv_mean, self.srv_std = _pair("smooth_root_vel", (3,))
        self.srp_mean, self.srp_std = _pair("smooth_root_pos", (3,))
        self.ljp_mean, self.ljp_std = _pair("local_joints_positions", (52, 3))
        self.grot_mean, self.grot_std = _pair("global_rot_data", (52, 6))
        self.lrot_mean, self.lrot_std = _pair("local_rot_data", (52, 6))
        self.fc_mean, self.fc_std = _pair("foot_contacts", (4,))

    def decode_window(self, latent_t_491: np.ndarray) -> dict[str, np.ndarray]:
        """Decode `(T, 491)` normalized latent into per-frame velocities, local joints, and canonical quats."""
        latent = np.asarray(latent_t_491, dtype=np.float32)
        if latent.ndim != 2 or latent.shape[1] != 491:
            raise ValueError(f"Expected (T, 491) latent, got {latent.shape}")
        t_len = latent.shape[0]

        srv_xz_n, srp_y_n, ljp_n, rot_n, _shapes_n, fc_n = np.split(latent, self.SPLIT_INDICES, axis=-1)

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

        foot_contacts = fc_n * self.fc_std + self.fc_mean
        # Convert rotations to canonical +Z-forward quats (with dummy zero pelvis; stitcher integrates root).
        quats_rh, _ = canonicalize_smplh_sequence(local_mats, np.zeros((t_len, 3), dtype=np.float32))
        return {
            "srv_xz_cam": srv_xz,            # (T, 2)
            "srp_y_cam": srp_y,              # (T,)
            "pelvis_local_cam": ljp[:, 0],   # (T, 3)
            "ljp_cam": ljp,                  # (T, 52, 3)
            "quats_rh_xyzw": quats_rh,       # (T, 52, 4)
            "foot_contacts": foot_contacts,  # (T, 4)
        }


@dataclass
class _OverlapTail:
    """Uncommitted `O`-frame tail from the previous window used to crossfade the next stride."""

    start_frame_index: int
    srv_xz_cam: np.ndarray        # (O, 2)
    srp_y_cam: np.ndarray         # (O,)
    pelvis_local_cam: np.ndarray  # (O, 3)
    ljp_cam: np.ndarray           # (O, 52, 3)
    quats_rh_xyzw: np.ndarray     # (O, 52, 4)


class StreamingStitcher:
    """Crossfades overlapping sliding-window solves into a C0-continuous 30 Hz pose stream."""

    def __init__(self, decoder: MotionLatentDecoder, fps: float = 30.0) -> None:
        self.decoder = decoder
        self.fps = float(fps)
        self._tail: _OverlapTail | None = None
        self._cum_root_xz_cam = np.zeros(2, dtype=np.float64)
        self._has_started = False

    def reset(self) -> None:
        """Reset continuity state (called on source switch or looped video restart)."""
        self._tail = None
        self._cum_root_xz_cam[:] = 0.0
        self._has_started = False

    def commit_window(
        self,
        latent_window: np.ndarray,
        *,
        emit_offset_in_window: int,
        stride_frames: int,
        overlap_frames: int,
        first_emit_frame_index: int,
        capture_mono_times: list[float],
    ) -> list[CanonicalPoseFrame]:
        """Decode `latent_window`, crossfade first `O` frames with prior tail, and commit `S` frames.

        Window slice layout starting at `emit_offset_in_window`:
          `[0 .. S - 1]`     -> committed and returned on this call (`len == stride_frames`)
          `[S .. S + O - 1]` -> stored as `_tail` to crossfade against `[0 .. O - 1]` of the next window
        """
        decoded = self.decoder.decode_window(latent_window)
        s_idx = int(emit_offset_in_window)
        e_commit = s_idx + int(stride_frames)
        e_tail = e_commit + int(overlap_frames)
        if e_tail > latent_window.shape[0]:
            raise ValueError(
                f"Window length {latent_window.shape[0]} too short for emit_offset={s_idx}, "
                f"stride={stride_frames}, overlap={overlap_frames}"
            )
        if len(capture_mono_times) != stride_frames:
            raise ValueError(f"Expected {stride_frames} capture_mono_times, got {len(capture_mono_times)}")

        srv_xz = decoded["srv_xz_cam"][s_idx:e_commit].copy()
        srp_y = decoded["srp_y_cam"][s_idx:e_commit].copy()
        pelvis_local = decoded["pelvis_local_cam"][s_idx:e_commit].copy()
        ljp = decoded["ljp_cam"][s_idx:e_commit].copy()
        quats = decoded["quats_rh_xyzw"][s_idx:e_commit].copy()

        # Blend first `O` frames of this stride against the previous window's overlap tail.
        if (
            self._tail is not None
            and overlap_frames > 0
            and self._tail.start_frame_index == first_emit_frame_index
        ):
            n_blend = min(overlap_frames, stride_frames, self._tail.quats_rh_xyzw.shape[0])
            # Strictly interior ramp (1/(N+1) .. N/(N+1)) so frame -1 (prev window) and frame N (new window)
            # join without endpoint derivative snaps.
            alphas = np.linspace(0.0, 1.0, n_blend + 2, dtype=np.float32)[1:-1]
            a_col = alphas[:, None]
            a_j = alphas[:, None, None]

            srv_xz[:n_blend] = (1.0 - a_col) * self._tail.srv_xz_cam[:n_blend] + a_col * srv_xz[:n_blend]
            srp_y[:n_blend] = (1.0 - alphas) * self._tail.srp_y_cam[:n_blend] + alphas * srp_y[:n_blend]
            pelvis_local[:n_blend] = (
                (1.0 - a_col) * self._tail.pelvis_local_cam[:n_blend] + a_col * pelvis_local[:n_blend]
            )
            ljp[:n_blend] = (1.0 - a_j) * self._tail.ljp_cam[:n_blend] + a_j * ljp[:n_blend]
            quats[:n_blend] = slerp_quat_xyzw(
                self._tail.quats_rh_xyzw[:n_blend],
                quats[:n_blend],
                alphas[:, None],
            )

        # Store uncommitted tail `[S .. S + O - 1]` for the next window's crossfade.
        if overlap_frames > 0:
            self._tail = _OverlapTail(
                start_frame_index=first_emit_frame_index + stride_frames,
                srv_xz_cam=decoded["srv_xz_cam"][e_commit:e_tail].copy(),
                srp_y_cam=decoded["srp_y_cam"][e_commit:e_tail].copy(),
                pelvis_local_cam=decoded["pelvis_local_cam"][e_commit:e_tail].copy(),
                ljp_cam=decoded["ljp_cam"][e_commit:e_tail].copy(),
                quats_rh_xyzw=decoded["quats_rh_xyzw"][e_commit:e_tail].copy(),
            )
        else:
            self._tail = None

        # Integrate horizontal root velocity continuously across window seams, then apply R_y(180)
        # `(-x, +y, -z)` to match canonical +Z-forward VRM world space.
        frames: list[CanonicalPoseFrame] = []
        dt = 1.0 / self.fps
        for i in range(stride_frames):
            if self._has_started:
                self._cum_root_xz_cam[0] += float(srv_xz[i, 0]) * dt
                self._cum_root_xz_cam[1] += float(srv_xz[i, 1]) * dt
            else:
                self._has_started = True

            px_cam = float(pelvis_local[i, 0] + self._cum_root_xz_cam[0])
            py_cam = float(srp_y[i])  # smooth_root_pos_y in FlowHMR is absolute Y height
            pz_cam = float(pelvis_local[i, 2] + self._cum_root_xz_cam[1])

            # Apply R_y(180): (-x, +y, -z)
            hips_rh = np.array([-px_cam, py_cam, -pz_cam], dtype=np.float32)

            f_idx = first_emit_frame_index + i
            frames.append(
                CanonicalPoseFrame(
                    media_time_s=f_idx / self.fps,
                    capture_mono_s=float(capture_mono_times[i]),
                    frame_index=f_idx,
                    quats_rh_xyzw=quats[i],
                    hips_pos_rh=hips_rh,
                    joints_cam_3d=ljp[i, :22].copy(),
                )
            )
        return frames
