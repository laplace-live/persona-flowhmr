"""Dynamic 6-DoF camera pose & FOV recovery from SAM 3D Body + FlowHMR + background VO.

Recovers real/video camera framing and 3D stage movement across both continuous camera motion
(pan, tilt, orbit, dolly, zoom, handheld) and hard multi-camera cuts / live camera switches:
  1. `ShotCutDetector`: sub-millisecond luma + border-correlation + histogram cut detector.
  2. `BackgroundVisualOdometry`: pyramidal LK optical flow outside the expanded person bbox
     with calibrated unit-ray Kabsch RANSAC for degeneracy-free pan/tilt/roll (`camera_R`).
  3. `StreamingCameraSolver`: VIO-style complementary fusion of background VO rotation with
     visibility-weighted 3D joint Kabsch alignment (`SAM 3D Body` OpenCV camera space <->
     `FlowHMR` canonical Three.js stage space), producing exact Persona `OrbitTransform` +
     `fov` + `roll` and Unity VMC `/VMC/Ext/Cam` packets.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from .retarget import matrix_to_quat_xyzw, slerp_quat_xyzw
from .vmc_sender import Pos3, Quat4

VmcCameraTransform = tuple[str, Pos3, Quat4, float]

# Persona `CameraPoseSchema` FOV bounds (`packages/persona-sdk/src/values/limits.ts`).
CAMERA_FOV_MIN_DEG = 1.0
CAMERA_FOV_MAX_DEG = 179.0
CAMERA_DISTANCE_MIN_M = 0.35
CAMERA_DISTANCE_MAX_M = 25.0

# OpenCV (`+X right, +Y down, +Z forward`) <-> Three.js camera (`+X right, +Y up, -Z forward`).
_CV_TO_THREE_CAM = np.diag([1.0, -1.0, -1.0]).astype(np.float32)


def yup_rel_to_opencv_camera_r(rel_r_yup: np.ndarray) -> np.ndarray:
    """Convert `(..., 3, 3)` Three.js/Y-up relative camera rotation `S @ R_cv_rel @ S` back to OpenCV `R_cv_rel`.

    Upstream FlowHMR (`_vggt_opencv_to_yup_rt` -> `build_feature_dict`) transforms world axes only (`R_cv @ S`),
    so `R_t @ R_0^T` cancels `S` on the right and conditions MMDiT on OpenCV camera-space `R_cv_t @ R_cv_0^T`.
    """
    r = np.asarray(rel_r_yup, dtype=np.float32)
    return (_CV_TO_THREE_CAM @ r @ _CV_TO_THREE_CAM).astype(np.float32, copy=False)

# Corresponding anatomical indices between `mhr70` (`SAM 3D Body`) and `SMPL-H` (`FlowHMR` body22).
# Index 0 in our 14-joint array is synthetic Mid-Hip (`mean(L_Hip, R_Hip)`), followed by 13 joints:
#   [MidHip, L_Hip, R_Hip, Neck, L_Shoulder, R_Shoulder, L_Elbow, R_Elbow, L_Wrist, R_Wrist,
#    L_Knee, R_Knee, L_Ankle, R_Ankle]
MHR70_BODY13_INDICES: tuple[int, ...] = (9, 10, 69, 5, 6, 7, 8, 62, 41, 11, 12, 13, 14)
SMPLH_BODY13_INDICES: tuple[int, ...] = (1, 2, 12, 16, 17, 18, 19, 20, 21, 4, 5, 7, 8)

_BASE_JOINT14_WEIGHTS = np.array(
    [
        2.5,  # 0: Mid-Hip
        1.8,  # 1: L_Hip
        1.8,  # 2: R_Hip
        2.0,  # 3: Neck
        1.8,  # 4: L_Shoulder
        1.8,  # 5: R_Shoulder
        0.9,  # 6: L_Elbow
        0.9,  # 7: R_Elbow
        0.5,  # 8: L_Wrist
        0.5,  # 9: R_Wrist
        1.0,  # 10: L_Knee
        1.0,  # 11: R_Knee
        0.7,  # 12: L_Ankle
        0.7,  # 13: R_Ankle
    ],
    dtype=np.float32,
)
# Core torso indices (`Mid-Hip, L_Hip, R_Hip, Neck, L_Shoulder, R_Shoulder`) used to anchor
# camera translation so fast limb swings never tug the camera pivot.
_TORSO_INDICES = np.array([0, 1, 2, 3, 4, 5], dtype=np.intp)
_TORSO_TRANSLATION_WEIGHTS = np.array([2.5, 1.5, 1.5, 2.0, 1.5, 1.5], dtype=np.float32)


@dataclass(frozen=True)
class CameraPoseFrame:
    """Single 30 Hz or interpolated 60 Hz 6-DoF camera pose in canonical Three.js stage space."""

    position_rh: np.ndarray     # (3,) camera eye `(x, y, z)` in +X-right-of-view, +Y-up, +Z-viewer meters
    quat_rh_xyzw: np.ndarray    # (4,) camera world rotation `(qx, qy, qz, qw)` in Three.js space
    orbit_azimuth: float        # radians (unwrapped within shot)
    orbit_elevation: float      # radians
    orbit_distance: float       # meters (> 0)
    orbit_target_rh: np.ndarray # (3,) `(targetX, targetY, targetZ)` in meters
    roll: float                 # radians about camera-local +Z
    fov_deg: float              # vertical field of view in degrees
    is_cut: bool = False        # True on the first frame of a hard shot cut / camera switch


@dataclass(frozen=True)
class FrameCameraObservation:
    """Per-frame visual camera cues extracted before FlowHMR window solving."""

    sam_cam_t: np.ndarray       # (3,) OpenCV `[tx, ty, tz]` meters
    sam_j3d_cam: np.ndarray     # (14, 3) OpenCV camera-space joints in meters
    cam_k: np.ndarray           # (3, 3) intrinsics in original image pixels
    image_hw: tuple[int, int]   # `(height, width)`
    bbox_xyxy: np.ndarray       # (4,) detected person bbox
    r_vo_yup: np.ndarray        # (3, 3) cumulative within-shot Y-up world-to-camera rotation
    vo_inliers: int             # number of background optical-flow RANSAC inliers
    is_cut: bool                # True if a hard shot cut or camera switch occurred on this frame
    shot_id: int                # monotonically increasing shot identifier
    trans_flow_px: float = 0.0  # non-rotational background optical flow residual (480p-equiv px/frame)


def extract_mhr_body14_cam(pred_keypoints_3d_70: np.ndarray, pred_cam_t: np.ndarray) -> np.ndarray:
    """Build `(..., 14, 3)` OpenCV camera-space joints from `(..., 70, 3)` MHR keypoints + `(..., 3)` `pred_cam_t`."""
    kp = np.asarray(pred_keypoints_3d_70, dtype=np.float32)
    cam_t = np.asarray(pred_cam_t, dtype=np.float32)
    j_cam = kp[..., :70, :] + cam_t[..., None, :]
    mid_hip = 0.5 * (j_cam[..., 9:10, :] + j_cam[..., 10:11, :])
    body13 = j_cam[..., MHR70_BODY13_INDICES, :]
    return np.concatenate([mid_hip, body13], axis=-2).astype(np.float32, copy=False)


def extract_smplh_body14_stage(
    ljp_cam_52: np.ndarray,
    hips_stage_rh: np.ndarray,
    *,
    world_yaw_rad: float = 0.0,
    avatar_scale: float = 1.0,
) -> np.ndarray:
    """Build `(14, 3)` canonical Three.js stage joints from FlowHMR's `(52, 3)` `ljp_cam` + stage pelvis."""
    ljp = np.asarray(ljp_cam_52, dtype=np.float32)
    hips = np.asarray(hips_stage_rh, dtype=np.float32)
    # Relative to pelvis in FlowHMR's WV frame, then R_y(180) `(-x, +y, -z)` into canonical +Z-forward space.
    rel = (ljp - ljp[0:1]) * np.array([-1.0, 1.0, -1.0], dtype=np.float32)
    if abs(world_yaw_rad) > 1e-7:
        rel = ( _rot_y_matrix(world_yaw_rad) @ rel.T ).T
    pts_52 = hips[None, :] + float(avatar_scale) * rel
    mid_hip = 0.5 * (pts_52[1:2] + pts_52[2:3])
    body13 = pts_52[list(SMPLH_BODY13_INDICES)]
    return np.concatenate([mid_hip, body13], axis=0).astype(np.float32, copy=False)


def vertical_fov_deg_from_k(cam_k: np.ndarray, height: int) -> float:
    """Compute vertical field of view in degrees from `(3, 3)` intrinsics `K` and frame `height`."""
    fy = max(1e-4, float(cam_k[1, 1]))
    fov = math.degrees(2.0 * math.atan(0.5 * float(height) / fy))
    return float(np.clip(fov, CAMERA_FOV_MIN_DEG, CAMERA_FOV_MAX_DEG))


def pinhole_camera_fallback(
    bboxes_xyxy: np.ndarray,
    cameras_k: np.ndarray,
    image_hw: tuple[int, int],
    *,
    assumed_height_m: float = 1.68,
) -> tuple[np.ndarray, np.ndarray]:
    """Analytical pinhole `(sam_cam_t, sam_j3d_14_cam)` fallback when running `replay` / `bbox-probe` mode."""
    boxes = np.asarray(bboxes_xyxy, dtype=np.float32).reshape(-1, 4)
    ks = np.asarray(cameras_k, dtype=np.float32).reshape(-1, 3, 3)
    h, w = int(image_hw[0]), int(image_hw[1])
    n = boxes.shape[0]

    bw = np.maximum(boxes[:, 2] - boxes[:, 0], 16.0)
    bh = np.maximum(boxes[:, 3] - boxes[:, 1], 32.0)
    cx = 0.5 * (boxes[:, 0] + boxes[:, 2])

    # Detect cropped upper-body / close-up shots: if box bottom touches frame bottom while aspect is wide,
    # estimate full-body pixel height from shoulder/torso width (`~3.1x` bbox width) or visible fraction.
    bottom_cropped = (boxes[:, 3] >= (h - 10.0)) & ((bw / bh) > 0.48)
    est_full_px = np.where(bottom_cropped, np.maximum(bh, bw * 2.85), bh)

    fy = np.maximum(ks[:, 1, 1], 1.0)
    fx = np.maximum(ks[:, 0, 0], 1.0)
    px = ks[:, 0, 2] if np.any(ks[:, 0, 2] > 0) else np.full(n, w * 0.5, dtype=np.float32)
    py = ks[:, 1, 2] if np.any(ks[:, 1, 2] > 0) else np.full(n, h * 0.5, dtype=np.float32)

    tz = np.clip((fy * float(assumed_height_m)) / est_full_px, CAMERA_DISTANCE_MIN_M, CAMERA_DISTANCE_MAX_M)
    # Top of head is near `boxes[:, 1]`; feet in uncropped standing pose are `est_full_px` below `boxes[:, 1]`.
    feet_v = boxes[:, 1] + est_full_px
    tx = ((cx - px) / fx) * tz
    ty = ((feet_v - py) / fy) * tz
    sam_cam_t = np.stack([tx, ty, tz], axis=-1).astype(np.float32)

    # Canonical upright MHR-like 14-joint template relative to feet `(0, 0, 0)` in OpenCV Y-down meters.
    template_rel = np.array(
        [
            [0.00, -0.88, 0.00],   # MidHip
            [0.09, -0.88, 0.00],   # L_Hip
            [-0.09, -0.88, 0.00],  # R_Hip
            [0.00, -1.40, 0.00],   # Neck
            [0.18, -1.35, 0.00],   # L_Shoulder
            [-0.18, -1.35, 0.00],  # R_Shoulder
            [0.26, -1.08, 0.00],   # L_Elbow
            [-0.26, -1.08, 0.00],  # R_Elbow
            [0.30, -0.84, 0.00],   # L_Wrist
            [-0.30, -0.84, 0.00],  # R_Wrist
            [0.09, -0.48, 0.00],   # L_Knee
            [-0.09, -0.48, 0.00],  # R_Knee
            [0.09, -0.08, 0.00],   # L_Ankle
            [-0.09, -0.08, 0.00],  # R_Ankle
        ],
        dtype=np.float32,
    )
    j3d_cam = template_rel[None, :, :] + sam_cam_t[:, None, :]
    return sam_cam_t, j3d_cam.astype(np.float32)


def _rot_y_matrix(angle_rad: float) -> np.ndarray:
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float32)


def _quat_to_matrix_np(q_xyzw: np.ndarray) -> np.ndarray:
    """Convert normalized `(..., 4)` `(x, y, z, w)` quaternions to `(..., 3, 3)` rotation matrices."""
    q = np.asarray(q_xyzw, dtype=np.float32)
    n = np.clip(np.linalg.norm(q, axis=-1, keepdims=True), 1e-8, None)
    q = q / n
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.stack(
        [
            np.stack([1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)], axis=-1),
            np.stack([2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)], axis=-1),
            np.stack([2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)], axis=-1),
        ],
        axis=-2,
    ).astype(np.float32)


def _unwrap_angle(prev: float, target: float) -> float:
    """Unwrap `target` (radians) onto the nearest 2π branch to `prev`."""
    delta = (target - prev + math.pi) % (2.0 * math.pi) - math.pi
    return prev + delta


def decompose_three_camera_pose(
    position_rh: np.ndarray,
    r_world_from_three_cam: np.ndarray,
    distance: float,
    fov_deg: float,
    *,
    prev_azimuth: float | None = None,
    is_cut: bool = False,
) -> CameraPoseFrame:
    """Convert `(C_world, R_world_from_three_cam, distance, fov)` into Persona's exact `CameraPoseFrame`.

    Matches `three-app.ts`:
      `camera.position = target + distance * [sin(az)*cos(el), sin(el), cos(az)*cos(el)]`
      `camera.rotation.set(-elevation, azimuth, roll, 'YXZ')`
    """
    pos = np.asarray(position_rh, dtype=np.float32).reshape(3)
    rot = np.asarray(r_world_from_three_cam, dtype=np.float32).reshape(3, 3)
    # Re-orthonormalize via SVD to guarantee proper SO(3).
    u, _, vt = np.linalg.svd(rot.astype(np.float64))
    if np.linalg.det(u @ vt) < 0.0:
        u[:, -1] *= -1.0
    rot_so3 = (u @ vt).astype(np.float32)

    sin_el = float(np.clip(rot_so3[1, 2], -0.9999, 0.9999))
    elevation = math.asin(sin_el)
    cos_el = math.cos(elevation)
    if abs(cos_el) > 1e-4:
        azimuth = math.atan2(float(rot_so3[0, 2]), float(rot_so3[2, 2]))
        roll = math.atan2(float(rot_so3[1, 0]), float(rot_so3[1, 1]))
    else:
        azimuth = math.atan2(-float(rot_so3[2, 0]), float(rot_so3[0, 0]))
        roll = 0.0

    if prev_azimuth is not None and not is_cut:
        azimuth = _unwrap_angle(prev_azimuth, azimuth)

    dist = float(np.clip(distance, CAMERA_DISTANCE_MIN_M, CAMERA_DISTANCE_MAX_M))
    back_axis = np.array(
        [
            math.sin(azimuth) * math.cos(elevation),
            math.sin(elevation),
            math.cos(azimuth) * math.cos(elevation),
        ],
        dtype=np.float32,
    )
    target = (pos - dist * back_axis).astype(np.float32)
    quat = matrix_to_quat_xyzw(rot_so3)

    return CameraPoseFrame(
        position_rh=pos,
        quat_rh_xyzw=quat,
        orbit_azimuth=float(azimuth),
        orbit_elevation=float(elevation),
        orbit_distance=dist,
        orbit_target_rh=target,
        roll=float(roll),
        fov_deg=float(np.clip(fov_deg, CAMERA_FOV_MIN_DEG, CAMERA_FOV_MAX_DEG)),
        is_cut=bool(is_cut),
    )


def interpolate_camera_pose(c0: CameraPoseFrame, c1: CameraPoseFrame, alpha: float) -> CameraPoseFrame:
    """Interpolate two 30 Hz `CameraPoseFrame`s at `alpha in [0, 1]`, preserving crisp cuts on `c1.is_cut`."""
    if c1.is_cut:
        return c0 if alpha < 0.5 else c1
    t = float(np.clip(alpha, 0.0, 1.0))
    pos = ((1.0 - t) * c0.position_rh + t * c1.position_rh).astype(np.float32)
    quat = slerp_quat_xyzw(c0.quat_rh_xyzw, c1.quat_rh_xyzw, t)
    dist = (1.0 - t) * c0.orbit_distance + t * c1.orbit_distance
    fov = (1.0 - t) * c0.fov_deg + t * c1.fov_deg
    rot = _quat_to_matrix_np(quat)
    return decompose_three_camera_pose(
        pos,
        rot,
        dist,
        fov,
        prev_azimuth=c0.orbit_azimuth,
        is_cut=False,
    )


def weighted_kabsch_so3(src_pts: np.ndarray, dst_pts: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Solve `dst ≈ R @ src + t` (`R in SO(3)`) minimizing `sum_i w_i ||dst_i - (R src_i + t)||^2`."""
    src = np.asarray(src_pts, dtype=np.float64)
    dst = np.asarray(dst_pts, dtype=np.float64)
    w = np.maximum(np.asarray(weights, dtype=np.float64), 0.0)
    w_sum = float(np.sum(w))
    if w_sum <= 1e-8:
        return np.eye(3, dtype=np.float32), np.zeros(3, dtype=np.float32)
    w_norm = (w / w_sum)[:, None]
    src_mean = np.sum(w_norm * src, axis=0)
    dst_mean = np.sum(w_norm * dst, axis=0)
    src_c = src - src_mean
    dst_c = dst - dst_mean
    cov = (dst_c * w_norm).T @ src_c
    u, _, vt = np.linalg.svd(cov)
    d = 1.0 if np.linalg.det(u @ vt) >= 0.0 else -1.0
    rot = u @ np.diag([1.0, 1.0, d]) @ vt
    trans = dst_mean - rot @ src_mean
    return rot.astype(np.float32), trans.astype(np.float32)


# Outer 64x36 thumbnail border mask (top/bottom 22%, left/right 20%) excluding central performer area.
_THUMB_BORDER_MASK = np.ones((36, 64), dtype=bool)
_THUMB_BORDER_MASK[8:28, 13:51] = False


class ShotCutDetector:
    """Detects hard camera angle / shot cuts in `< 0.15 ms` without false-triggering on dancer or strobe motion."""

    def __init__(
        self,
        *,
        luma_cut_thr: float = 0.11,
        border_cut_thr: float = 0.09,
        hist_l1_thr: float = 0.30,
        corr_max_thr: float = 0.68,
        min_gap_frames: int = 6,
    ) -> None:
        self.luma_cut_thr = float(luma_cut_thr)
        self.border_cut_thr = float(border_cut_thr)
        self.hist_l1_thr = float(hist_l1_thr)
        self.corr_max_thr = float(corr_max_thr)
        self.min_gap_frames = int(min_gap_frames)
        self._prev_thumb: np.ndarray | None = None
        self._prev_border: np.ndarray | None = None
        self._prev_hist: np.ndarray | None = None
        self._prev_edge_mag: np.ndarray | None = None
        self._ema_mad: float | None = None
        self._frames_since_cut = 999

    def reset(self) -> None:
        self._prev_thumb = None
        self._prev_border = None
        self._prev_hist = None
        self._prev_edge_mag = None
        self._ema_mad = None
        self._frames_since_cut = 999

    def step(self, rgb: np.ndarray, *, force_cut: bool = False) -> bool:
        """Inspect one RGB frame and return `True` if a hard shot cut occurred on this frame."""
        h, w = rgb.shape[:2]
        # Fast C++ nearest-neighbor pre-decimation avoids copying non-contiguous views on 1080p/4K frames.
        if h > 288 or w > 512:
            sub = cv2.resize(rgb, (min(w, 256), min(h, 144)), interpolation=cv2.INTER_NEAREST)
        else:
            sub = rgb
        small = cv2.resize(sub, (64, 36), interpolation=cv2.INTER_AREA)
        gray = (
            0.299 * small[:, :, 0].astype(np.float32)
            + 0.587 * small[:, :, 1].astype(np.float32)
            + 0.114 * small[:, :, 2].astype(np.float32)
        ) / 255.0

        # Spatial gradient magnitude map `(35, 63)` distinguishes true camera cuts (edges relocate)
        # from lighting/strobe flashes on the same camera (edges stay correlated).
        gx = gray[:-1, 1:] - gray[:-1, :-1]
        gy = gray[1:, :-1] - gray[:-1, :-1]
        edge_mag = np.hypot(gx, gy)

        border = gray[_THUMB_BORDER_MASK]

        # Normalized 24-bin RGB histogram (`3 x 8` bins).
        q = (small.astype(np.uint16) >> 5)  # 0..7
        hist = np.concatenate(
            [np.bincount(q[:, :, c].ravel(), minlength=8) for c in range(3)]
        ).astype(np.float32)
        hist /= max(1.0, float(hist.sum()))

        is_cut = bool(force_cut)
        if (
            not is_cut
            and self._prev_thumb is not None
            and self._prev_border is not None
            and self._prev_hist is not None
            and self._prev_edge_mag is not None
        ):
            abs_diff = np.abs(gray - self._prev_thumb)
            luma_mad = float(np.mean(abs_diff))
            p90_diff = float(np.percentile(abs_diff, 90))
            border_mad = float(np.mean(np.abs(border - self._prev_border)))
            hist_l1 = float(np.sum(np.abs(hist - self._prev_hist)))

            b0 = self._prev_border - float(np.mean(self._prev_border))
            b1 = border - float(np.mean(border))
            denom = float(np.linalg.norm(b0) * np.linalg.norm(b1))
            border_corr = float(np.dot(b0, b1) / denom) if denom > 1e-5 else 1.0

            e0 = self._prev_edge_mag.ravel() - float(np.mean(self._prev_edge_mag))
            e1 = edge_mag.ravel() - float(np.mean(edge_mag))
            e_denom = float(np.linalg.norm(e0) * np.linalg.norm(e1))
            edge_corr = float(np.dot(e0, e1) / e_denom) if e_denom > 1e-5 else 1.0

            has_ema = self._ema_mad is not None
            ema_ref = float(self._ema_mad) if self._ema_mad is not None else max(0.012, luma_mad)
            spike_ratio = (luma_mad / max(0.006, ema_ref)) if has_ema else 1.0

            if self._frames_since_cut >= self.min_gap_frames:
                strong_visual_jump = (
                    border_mad >= self.border_cut_thr
                    and (luma_mad >= self.luma_cut_thr or hist_l1 >= self.hist_l1_thr)
                    and border_corr <= self.corr_max_thr
                )
                # Catches dark-room / dark-studio hard cuts (where both shots share low border luma)
                # while rejecting lighting/strobe pulses on a fixed camera (`edge_corr > 0.50`) and steady pans.
                temporal_structural_cut = (
                    has_ema
                    and spike_ratio >= 3.8
                    and luma_mad >= 0.055
                    and p90_diff >= 0.16
                    and edge_corr <= 0.48
                )
                extreme_jump = luma_mad >= 0.24 and (hist_l1 >= 0.45 or edge_corr <= 0.55)
                is_cut = strong_visual_jump or temporal_structural_cut or extreme_jump

            if not is_cut:
                if self._ema_mad is None:
                    self._ema_mad = max(0.008, luma_mad)
                elif spike_ratio < 3.0:
                    self._ema_mad = 0.82 * self._ema_mad + 0.18 * luma_mad

        self._prev_thumb = gray
        self._prev_border = border
        self._prev_hist = hist
        self._prev_edge_mag = edge_mag
        if is_cut:
            self._frames_since_cut = 0
        else:
            self._frames_since_cut += 1
        return is_cut


class BackgroundVisualOdometry:
    """Estimates inter-frame camera rotation `R_vo_yup` and translational residual flow from background features."""

    def __init__(self, *, work_width: int = 320, min_inliers: int = 10) -> None:
        self.work_width = int(work_width)
        self.min_inliers = int(min_inliers)
        self.last_trans_flow_px: float = 0.0
        self.zoom_scale: float = 1.0
        self._prev_gray: np.ndarray | None = None
        self._prev_hw: tuple[int, int] | None = None
        self._prev_bbox: np.ndarray | None = None
        self._r_yup = np.eye(3, dtype=np.float32)
        self._rng = np.random.default_rng(42)

    def reset(self) -> None:
        self._prev_gray = None
        self._prev_hw = None
        self._prev_bbox = None
        self._r_yup = np.eye(3, dtype=np.float32)
        self.last_trans_flow_px = 0.0
        self.zoom_scale = 1.0

    def step(
        self,
        rgb: np.ndarray,
        bbox_xyxy: np.ndarray,
        cam_k: np.ndarray,
        *,
        is_cut: bool = False,
        allow_zoom: bool = True,
    ) -> tuple[np.ndarray, int]:
        """Advance VO by one frame; returns `(r_vo_yup_3x3, inlier_count)` and updates `last_trans_flow_px`."""
        h, w = rgb.shape[:2]
        cur_bbox = np.asarray(bbox_xyxy, dtype=np.float32).reshape(4).copy()
        prev_bbox = self._prev_bbox
        if not allow_zoom:
            self.zoom_scale = 1.0
        long_side = max(1, max(h, w))
        scale = min(1.0, float(self.work_width) / float(long_side))
        sw, sh = max(32, int(round(w * scale))), max(32, int(round(h * scale)))
        # Fast C++ nearest pre-downsample -> 1-channel grayscale -> anti-aliased resize to `(sw, sh)`.
        if scale < 0.5:
            sub_rgb = cv2.resize(rgb, (sw * 2, sh * 2), interpolation=cv2.INTER_NEAREST)
            sub_gray = cv2.cvtColor(sub_rgb, cv2.COLOR_RGB2GRAY)
            gray = cv2.resize(sub_gray, (sw, sh), interpolation=cv2.INTER_AREA)
        elif scale < 0.999:
            sub_gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            gray = cv2.resize(sub_gray, (sw, sh), interpolation=cv2.INTER_LINEAR)
        else:
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

        if is_cut or self._prev_gray is None or self._prev_hw != (sh, sw):
            self._prev_gray = gray
            self._prev_hw = (sh, sw)
            self._prev_bbox = cur_bbox
            self._r_yup = np.eye(3, dtype=np.float32)
            self.last_trans_flow_px = 0.0
            self.zoom_scale = 1.0
            return self._r_yup.copy(), 0

        self._prev_bbox = cur_bbox

        # Mask out expanded person bounding box so moving limbs never corrupt background camera motion.
        mask = np.full((sh, sw), 255, dtype=np.uint8)
        bx1, by1, bx2, by2 = (cur_bbox * scale).tolist()
        pad_x = 0.14 * max(1.0, bx2 - bx1)
        pad_y = 0.10 * max(1.0, by2 - by1)
        ix1 = max(0, int(math.floor(bx1 - pad_x)))
        iy1 = max(0, int(math.floor(by1 - pad_y)))
        ix2 = min(sw, int(math.ceil(bx2 + pad_x)))
        iy2 = min(sh, int(math.ceil(by2 + pad_y)))
        mask[iy1:iy2, ix1:ix2] = 0

        pts0 = cv2.goodFeaturesToTrack(
            self._prev_gray,
            maxCorners=90,
            qualityLevel=0.015,
            minDistance=8,
            mask=mask,
        )
        if pts0 is None or len(pts0) < self.min_inliers:
            self._prev_gray = gray
            self.last_trans_flow_px = 0.0
            return self._r_yup.copy(), 0

        lk_kw = dict(winSize=(13, 13), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 12, 0.03))
        pts1, st_fwd, lk_err = cv2.calcOpticalFlowPyrLK(self._prev_gray, gray, pts0, None, **lk_kw)
        self._prev_gray = gray

        if pts1 is None or st_fwd is None:
            self.last_trans_flow_px = 0.0
            return self._r_yup.copy(), 0

        valid = st_fwd.ravel() == 1
        if lk_err is not None:
            valid &= lk_err.ravel() < 24.0
        p0 = pts0.reshape(-1, 2)[valid] / scale
        p1 = pts1.reshape(-1, 2)[valid] / scale

        # Reject points landing inside current expanded bbox as well.
        raw_bx1, raw_by1, raw_bx2, raw_by2 = np.asarray(bbox_xyxy, dtype=np.float32).tolist()
        raw_px, raw_py = 0.14 * max(1.0, raw_bx2 - raw_bx1), 0.10 * max(1.0, raw_by2 - raw_by1)
        outside1 = ~(
            (p1[:, 0] >= raw_bx1 - raw_px)
            & (p1[:, 0] <= raw_bx2 + raw_px)
            & (p1[:, 1] >= raw_by1 - raw_py)
            & (p1[:, 1] <= raw_by2 + raw_py)
        )
        p0, p1 = p0[outside1], p1[outside1]
        if len(p0) < self.min_inliers:
            self.last_trans_flow_px = 0.0
            return self._r_yup.copy(), 0

        # Resolution-invariant 480p-long-side flow normalization for both landscape and portrait inputs.
        norm_480 = 480.0 / float(long_side)
        # Deadband sub-pixel sensor/compression noise on locked tripod cameras so rotation never random-walks.
        raw_flow_480 = float(np.median(np.linalg.norm(p1 - p0, axis=-1))) * norm_480
        if raw_flow_480 < 0.28:
            self.last_trans_flow_px = raw_flow_480
            return self._r_yup.copy(), int(len(p0))

        fx = max(1.0, float(cam_k[0, 0]))
        fy = max(1.0, float(cam_k[1, 1]))
        cx = float(cam_k[0, 2])
        cy = float(cam_k[1, 2])

        # Estimate candidate optical zoom BEFORE pure-rotation RANSAC so radial edge expansion doesn't get
        # misfit as a one-sided pan. Subtracting median displacement cancels first-order pan/tilt shift.
        eff_zoom = 1.0
        eff_k1: np.ndarray | None = None
        du0 = p0[:, 0] - cx
        dv0 = p0[:, 1] - cy
        r0 = np.hypot(du0, dv0)
        far_mask = r0 >= 0.12 * float(long_side)
        if allow_zoom and prev_bbox is not None and int(np.sum(far_mask)) >= self.min_inliers:
            du_far = du0[far_mask]
            both_sides = bool(np.any(du_far < -0.05 * w) and np.any(du_far > 0.05 * w))
            a_far = np.stack([du_far, dv0[far_mask]], axis=-1).astype(np.float64)
            b_far = np.stack([p1[far_mask, 0] - cx, p1[far_mask, 1] - cy], axis=-1).astype(np.float64)
            ac = a_far - np.mean(a_far, axis=0)
            bc = b_far - np.mean(b_far, axis=0)
            denom_z = float(np.sum(ac * ac))
            med_zoom = float(np.sum(ac * bc) / denom_z) if denom_z > 1e-6 else 1.0
            zoom_res_480 = float(np.median(np.linalg.norm(bc - med_zoom * ac, axis=-1))) * norm_480

            # Separate optical lens zoom from physical camera dolly: under optical zoom, an unclipped
            # foreground performer at closer depth `Z_person < Z_bg` magnifies by the SAME factor `med_zoom`,
            # whereas dolly toward a textured wall expands the closer performer faster due to perspective.
            fg_agrees = False
            margin = 8.0
            bw0 = float(prev_bbox[2] - prev_bbox[0])
            bh0 = float(prev_bbox[3] - prev_bbox[1])
            unclipped = (
                prev_bbox[0] > margin
                and prev_bbox[1] > margin
                and prev_bbox[2] < (w - margin)
                and prev_bbox[3] < (h - margin)
                and cur_bbox[0] > margin
                and cur_bbox[1] > margin
                and cur_bbox[2] < (w - margin)
                and cur_bbox[3] < (h - margin)
                and bw0 >= 24.0
                and bh0 >= 48.0
            )
            if unclipped:
                fg_scale = 0.5 * (
                    float(cur_bbox[2] - cur_bbox[0]) / bw0
                    + float(cur_bbox[3] - cur_bbox[1]) / bh0
                )
                fg_agrees = (
                    (fg_scale - 1.0) * (med_zoom - 1.0) > 0.0
                    and abs(fg_scale - med_zoom) <= max(0.0045, 0.28 * abs(med_zoom - 1.0))
                )

            if (
                both_sides
                and fg_agrees
                and zoom_res_480 < 0.85
                and 0.92 <= med_zoom <= 1.08
                and abs(med_zoom - 1.0) >= 0.0025
            ):
                eff_zoom = med_zoom
                eff_k1 = cam_k.copy()
                eff_k1[0, 0] = fx * eff_zoom
                eff_k1[1, 1] = fy * eff_zoom

        delta_r_cv, n_inliers = self._ransac_ray_rotation(p0, p1, cam_k, cam_k1=eff_k1)
        rays0 = np.stack([(p0[:, 0] - cx) / fx, (p0[:, 1] - cy) / fy, np.ones(len(p0), dtype=np.float64)], axis=-1)
        rays0_rot = rays0 @ delta_r_cv.astype(np.float64).T
        z_rot = np.maximum(rays0_rot[:, 2], 1e-4)
        fx_out = fx * eff_zoom
        fy_out = fy * eff_zoom
        u_pred = fx_out * (rays0_rot[:, 0] / z_rot) + cx
        v_pred = fy_out * (rays0_rot[:, 1] / z_rot) + cy
        if eff_k1 is not None and n_inliers >= self.min_inliers:
            self.zoom_scale = float(np.clip(self.zoom_scale * eff_zoom, 0.4, 3.5))

        self.last_trans_flow_px = float(np.median(np.hypot(p1[:, 0] - u_pred, p1[:, 1] - v_pred))) * norm_480

        if n_inliers < self.min_inliers:
            # Even if pure-rotation RANSAC had few inliers due to strong radial dolly, report tracked background count
            # when `last_trans_flow_px` indicates real camera translation.
            return self._r_yup.copy(), int(len(p0)) if self.last_trans_flow_px >= 0.6 else 0

        # Convert OpenCV `cam_t <- cam_{t-1}` into FlowHMR's Y-up `cam_yup` convention: `S @ R_cv @ S`.
        delta_r_yup = _CV_TO_THREE_CAM @ delta_r_cv @ _CV_TO_THREE_CAM
        self._r_yup = (delta_r_yup @ self._r_yup).astype(np.float32)
        # Keep orthonormal across multi-minute shots.
        u, _, vt = np.linalg.svd(self._r_yup.astype(np.float64))
        if np.linalg.det(u @ vt) < 0.0:
            u[:, -1] *= -1.0
        self._r_yup = (u @ vt).astype(np.float32)
        return self._r_yup.copy(), n_inliers

    def _ransac_ray_rotation(
        self,
        p0_px: np.ndarray,
        p1_px: np.ndarray,
        cam_k: np.ndarray,
        *,
        cam_k1: np.ndarray | None = None,
    ) -> tuple[np.ndarray, int]:
        """Estimate `R_cv` (`v1 ≈ R_cv @ v0`) via calibrated unit-ray Kabsch RANSAC (degeneracy-free at `t=0`)."""
        fx0 = max(1.0, float(cam_k[0, 0]))
        fy0 = max(1.0, float(cam_k[1, 1]))
        cx0 = float(cam_k[0, 2])
        cy0 = float(cam_k[1, 2])
        k1 = cam_k1 if cam_k1 is not None else cam_k
        fx1 = max(1.0, float(k1[0, 0]))
        fy1 = max(1.0, float(k1[1, 1]))
        cx1 = float(k1[0, 2])
        cy1 = float(k1[1, 2])

        def _to_rays(pts: np.ndarray, fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
            r = np.stack([(pts[:, 0] - cx) / fx, (pts[:, 1] - cy) / fy, np.ones(len(pts), dtype=np.float64)], axis=-1)
            return r / np.clip(np.linalg.norm(r, axis=-1, keepdims=True), 1e-8, None)

        v0 = _to_rays(p0_px.astype(np.float64), fx0, fy0, cx0, cy0)
        v1 = _to_rays(p1_px.astype(np.float64), fx1, fy1, cx1, cy1)
        n = len(v0)
        # Angular threshold ~3.0 pixels at focal length `fx1`.
        cos_thr = math.cos(max(0.0025, 3.0 / fx1))

        n_iters = 28
        # Sample 3 distinct rays per RANSAC hypothesis without replacement.
        idx = np.argsort(self._rng.random((n_iters, n)), axis=1)[:, :3]
        cov = np.einsum("kni,knj->kij", v1[idx], v0[idx])
        u, _, vt = np.linalg.svd(cov)
        dets = np.linalg.det(u @ vt)
        diag = np.ones((n_iters, 3), dtype=np.float64)
        diag[:, 2] = np.where(dets >= 0.0, 1.0, -1.0)
        r_cands = (u * diag[:, None, :]) @ vt
        dots = np.einsum("ni,kij,nj->kn", v1, r_cands, v0)
        inl_mat = dots >= cos_thr
        counts = np.sum(inl_mat, axis=-1)
        best_k = int(np.argmax(counts))
        best_count = int(counts[best_k])
        best_inliers = inl_mat[best_k]

        if best_count < self.min_inliers:
            return np.eye(3, dtype=np.float32), best_count

        cov_refit = v1[best_inliers].T @ v0[best_inliers]
        u, _, vt = np.linalg.svd(cov_refit)
        d = 1.0 if np.linalg.det(u @ vt) >= 0.0 else -1.0
        r_final = (u @ np.diag([1.0, 1.0, d]) @ vt).astype(np.float32)
        # Clamp unrealistic >25 deg/frame single-step spikes (e.g. moving foreground occlusion).
        trace_cos = float(np.clip((np.trace(r_final) - 1.0) * 0.5, -1.0, 1.0))
        if math.acos(trace_cos) > math.radians(25.0):
            return np.eye(3, dtype=np.float32), 0
        return r_final, best_count


class StreamingCameraSolver:
    """Fuses per-frame SAM 3D Body + background VO with FlowHMR stage poses into drift-free 6-DoF camera frames."""

    def __init__(self, *, avatar_scale: float = 1.0, calibrated_fov_deg: float = 0.0) -> None:
        self.avatar_scale = max(0.1, float(avatar_scale)) if math.isfinite(float(avatar_scale)) else 1.0
        self.calibrated_fov_deg = float(calibrated_fov_deg) if math.isfinite(float(calibrated_fov_deg)) else 0.0
        self._shot_base_rot: np.ndarray | None = None
        self._shot_cut_vo_yup: np.ndarray | None = None
        self._r_world_from_cam: np.ndarray | None = None
        self._pos_rh: np.ndarray | None = None
        self._dist_m: float = 3.0
        self._fov_deg: float = 30.0
        self._prev_azimuth: float | None = None
        self._frame0_hips_rh: np.ndarray | None = None
        self._last_avatar_scale: float | None = None
        self._last_follow_movement: bool | None = None
        self._last_calibrated_fov_deg: float | None = None

    def anchor_stream_origin(self, hips_pos_rh: np.ndarray) -> None:
        """Record stream frame-0 pelvis origin so live-enabling camera tracking preserves prior XZ travel."""
        if self._frame0_hips_rh is None:
            self._frame0_hips_rh = np.asarray(hips_pos_rh, dtype=np.float32).copy()

    def reset(self, *, keep_origin: bool = False) -> None:
        self._shot_base_rot = None
        self._shot_cut_vo_yup = None
        self._r_world_from_cam = None
        self._pos_rh = None
        self._prev_azimuth = None
        self._last_avatar_scale = None
        self._last_follow_movement = None
        self._last_calibrated_fov_deg = None
        if not keep_origin:
            self._frame0_hips_rh = None

    def solve_frame(
        self,
        obs: FrameCameraObservation,
        *,
        ljp_cam_52: np.ndarray,
        hips_pos_rh: np.ndarray,
        follow_movement: bool = True,
        world_yaw_rad: float = 0.0,
        is_stream_reset: bool = False,
    ) -> CameraPoseFrame:
        """Solve one committed 30 Hz `CameraPoseFrame` synchronized with FlowHMR's stage pose."""
        if is_stream_reset or self._frame0_hips_rh is None:
            self._frame0_hips_rh = np.asarray(hips_pos_rh, dtype=np.float32).copy()

        follow_flag = bool(follow_movement)
        cal_fov = float(self.calibrated_fov_deg) if math.isfinite(float(self.calibrated_fov_deg)) else 0.0
        framing_changed = (
            (self._last_avatar_scale is not None and abs(self.avatar_scale - self._last_avatar_scale) > 1e-6)
            or (self._last_follow_movement is not None and follow_flag != self._last_follow_movement)
            or (self._last_calibrated_fov_deg is not None and abs(cal_fov - self._last_calibrated_fov_deg) > 1e-4)
        )
        self._last_avatar_scale = float(self.avatar_scale)
        self._last_follow_movement = follow_flag
        self._last_calibrated_fov_deg = cal_fov

        # Match Persona's `VrmPoseDriver.applyTranslation` (`hipsRest + (hips_t - hips_0)`):
        # At stream start, the avatar stands at `(0, hips_0_y * scale, 0)`, and travels by `(hips_t - hips_0)`.
        delta_hips = np.asarray(hips_pos_rh, dtype=np.float32) - self._frame0_hips_rh
        fx_travel = float(delta_hips[0]) if follow_flag else 0.0
        fz_travel = float(delta_hips[2]) if follow_flag else 0.0
        stage_hips = np.array(
            [
                fx_travel * self.avatar_scale,
                (float(self._frame0_hips_rh[1]) + float(delta_hips[1])) * self.avatar_scale,
                fz_travel * self.avatar_scale,
            ],
            dtype=np.float32,
        )

        stage_j14 = extract_smplh_body14_stage(
            ljp_cam_52,
            stage_hips,
            world_yaw_rad=world_yaw_rad,
            avatar_scale=self.avatar_scale,
        )
        # Convert OpenCV camera-space 3D joints into Three.js camera-local space (`+X right, +Y up, -Z forward`).
        sam_j14_cv = np.asarray(obs.sam_j3d_cam, dtype=np.float32) * self.avatar_scale
        sam_j14_three_cam = (sam_j14_cv @ _CV_TO_THREE_CAM.T).astype(np.float32)
        h, _w = obs.image_hw

        is_cut = bool(
            obs.is_cut
            or is_stream_reset
            or self._shot_base_rot is None
            or self._shot_cut_vo_yup is None
            or self._pos_rh is None
        )
        if is_cut:
            # Initialize new shot orientation ONCE at the cut boundary:
            # - Azimuth comes directly from FlowHMR's temporal shot yaw (`world_yaw_rad`).
            # - Elevation & camera roll come from rigid 6-joint core torso alignment (never knees/ankles/arms)
            #   when the torso is upright and well-conditioned; otherwise level (`0 deg`).
            spine_stage_y = float(stage_j14[3, 1] - stage_j14[0, 1])
            spine_cam_y = float(sam_j14_three_cam[3, 1] - sam_j14_three_cam[0, 1])
            torso_span_cam_x = max(
                abs(float(sam_j14_three_cam[4, 0] - sam_j14_three_cam[5, 0])),
                abs(float(sam_j14_three_cam[1, 0] - sam_j14_three_cam[2, 0])),
            )
            ankles_below_hips = float(np.max(sam_j14_three_cam[12:14, 1])) < float(sam_j14_three_cam[0, 1])
            min_upright_m = 0.25 * max(0.25, self.avatar_scale)
            min_lateral_m = 0.05 * max(0.25, self.avatar_scale)
            if (
                spine_stage_y >= min_upright_m
                and spine_cam_y >= min_upright_m
                and torso_span_cam_x >= min_lateral_m
                and ankles_below_hips
            ):
                r_cam_from_world, _ = weighted_kabsch_so3(
                    stage_j14[_TORSO_INDICES],
                    sam_j14_three_cam[_TORSO_INDICES],
                    _TORSO_TRANSLATION_WEIGHTS,
                )
                r_kab_torso = r_cam_from_world.T
                raw_el = math.asin(float(np.clip(r_kab_torso[1, 2], -0.9999, 0.9999)))
                cos_el = math.cos(raw_el)
                raw_roll = math.atan2(float(r_kab_torso[1, 0]), float(r_kab_torso[1, 1])) if abs(cos_el) > 1e-4 else 0.0
                # Suppress small `< 4 deg` spine-lean / `< 3 deg` shoulder-shrug discrepancy on flat cuts.
                el_cut = (
                    0.0
                    if abs(raw_el) < math.radians(4.0)
                    else float(np.clip(raw_el, math.radians(-75.0), math.radians(75.0)))
                )
                roll_cut = (
                    0.0
                    if abs(raw_roll) < math.radians(3.0)
                    else float(np.clip(raw_roll, math.radians(-75.0), math.radians(75.0)))
                )
            else:
                el_cut = 0.0
                roll_cut = 0.0

            ce, se = math.cos(-el_cut), math.sin(-el_cut)
            cr, sr = math.cos(roll_cut), math.sin(roll_cut)
            rx_neg_el = np.array([[1.0, 0.0, 0.0], [0.0, ce, -se], [0.0, se, ce]], dtype=np.float32)
            rz_roll = np.array([[cr, -sr, 0.0], [sr, cr, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
            self._shot_base_rot = (_rot_y_matrix(float(world_yaw_rad)) @ rx_neg_el @ rz_roll).astype(np.float32)
            self._shot_cut_vo_yup = obs.r_vo_yup.copy()
            r_fused = self._shot_base_rot.copy()
        else:
            assert self._shot_base_rot is not None and self._shot_cut_vo_yup is not None
            # Within a continuous shot, camera rotation evolves strictly from background visual odometry
            # (`0.00 deg` post-cut drift on tripod shots; smooth lag-free pan/tilt/roll on moving cameras).
            rel_vo_yup = obs.r_vo_yup @ self._shot_cut_vo_yup.T
            r_fused = (self._shot_base_rot @ rel_vo_yup.T).astype(np.float32)

        # Solve camera world position from core torso centroid so screen projection is pixel-locked
        # and limb swings don't tug the camera.
        torso_w = _TORSO_TRANSLATION_WEIGHTS / float(np.sum(_TORSO_TRANSLATION_WEIGHTS))
        p_stage_torso = np.sum(torso_w[:, None] * stage_j14[_TORSO_INDICES], axis=0)
        p_cam_torso = np.sum(torso_w[:, None] * sam_j14_three_cam[_TORSO_INDICES], axis=0)
        raw_pos = (p_stage_torso - (r_fused @ p_cam_torso)).astype(np.float32)
        raw_dist = float(np.clip(-p_cam_torso[2], CAMERA_DISTANCE_MIN_M, CAMERA_DISTANCE_MAX_M))
        raw_fov = vertical_fov_deg_from_k(obs.cam_k, h)

        if is_cut or framing_changed:
            pos_fused = raw_pos
            dist_fused = raw_dist
            fov_fused = raw_fov
        else:
            assert self._pos_rh is not None
            step_m = float(np.linalg.norm(raw_pos - self._pos_rh))
            if obs.vo_inliers >= 12 and obs.trans_flow_px < 0.48:
                # Background features prove the camera's 3D optical center is stationary (static, pan, or optical zoom):
                # lock virtual tripod position & orbit distance completely (`0.0 m` post-cut drift).
                pos_fused = self._pos_rh.copy()
                dist_fused = self._dist_m
                fov_fused = 0.75 * self._fov_deg + 0.25 * raw_fov if abs(raw_fov - self._fov_deg) > 0.1 else self._fov_deg
            elif obs.vo_inliers >= 12:
                # Active physical camera translation / dolly / zoom observed in background parallax.
                motion_gain = float(np.clip((obs.trans_flow_px - 0.42) / 1.4, 0.08, 1.0))
                alpha_pos = motion_gain * float(np.clip(0.16 + 1.6 * step_m, 0.16, 0.65))
                pos_fused = ((1.0 - alpha_pos) * self._pos_rh + alpha_pos * raw_pos).astype(np.float32)
                dist_fused = (1.0 - alpha_pos) * self._dist_m + alpha_pos * raw_dist
                fov_fused = 0.80 * self._fov_deg + 0.20 * raw_fov
            else:
                # Tight close-up or featureless studio background (`vo_inliers < 12`):
                # deadband minor pose-induced depth wobble while smoothly tracking real dollies.
                alpha_pos = 0.02 if step_m < 0.14 else float(np.clip(0.14 + 1.3 * (step_m - 0.14), 0.14, 0.55))
                pos_fused = ((1.0 - alpha_pos) * self._pos_rh + alpha_pos * raw_pos).astype(np.float32)
                dist_fused = (1.0 - alpha_pos) * self._dist_m + alpha_pos * raw_dist
                fov_fused = 0.85 * self._fov_deg + 0.15 * raw_fov

        self._r_world_from_cam = r_fused
        self._pos_rh = pos_fused
        self._dist_m = float(dist_fused)
        self._fov_deg = float(fov_fused)

        cam_pose = decompose_three_camera_pose(
            pos_fused,
            r_fused,
            dist_fused,
            fov_fused,
            prev_azimuth=self._prev_azimuth,
            is_cut=is_cut,
        )
        self._prev_azimuth = cam_pose.orbit_azimuth
        return cam_pose


def camera_to_vmc_transform(
    cam: CameraPoseFrame,
    *,
    mirror: bool = False,
    name: str = "Camera",
) -> VmcCameraTransform:
    """Convert Three.js `CameraPoseFrame` into Unity VMC `/VMC/Ext/Cam` `(name, pos_unity, quat_unity, fov_deg)`.

    Verified against Persona's `unityVec3ToThree` (`[-vx, vy, vz]`):
      - `R_unity = diag(-1, 1, 1) @ R_three_effective @ diag(1, 1, -1)` (`det == +1`, front camera -> `R_y(180)`).
      - `pos_unity = (-px, py, pz)` (`mirror=False`) or `(+px, py, pz)` (`mirror=True`).
    """
    rot_three = _quat_to_matrix_np(cam.quat_rh_xyzw)
    px, py, pz = float(cam.position_rh[0]), float(cam.position_rh[1]), float(cam.position_rh[2])
    if mirror:
        # Mirror camera across stage YZ plane (`x -> -x`), then map into Unity VMC frame.
        fx = np.diag([-1.0, 1.0, 1.0]).astype(np.float32)
        rot_three = fx @ rot_three @ fx
        px = -px

    r_unity = np.diag([-1.0, 1.0, 1.0]).astype(np.float32) @ rot_three @ np.diag([1.0, 1.0, -1.0]).astype(np.float32)
    q_unity = matrix_to_quat_xyzw(r_unity)
    pos_unity: Pos3 = (-px, py, pz)
    quat_unity: Quat4 = (float(q_unity[0]), float(q_unity[1]), float(q_unity[2]), float(q_unity[3]))
    return (name, pos_unity, quat_unity, float(cam.fov_deg))


def camera_to_persona_vrm_camera(
    cam: CameraPoseFrame,
    *,
    mirror: bool = False,
    vrm_placement: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build LAPLACE Persona's `SceneCamera` (`scene.patch -> vrmCamera`) dict from `CameraPoseFrame`."""
    sign = -1.0 if mirror else 1.0
    azimuth = sign * float(cam.orbit_azimuth)
    elevation = float(cam.orbit_elevation)
    distance = float(cam.orbit_distance)
    roll = sign * float(cam.roll)
    tx = sign * float(cam.orbit_target_rh[0])
    ty = float(cam.orbit_target_rh[1])
    tz = float(cam.orbit_target_rh[2])

    # Transform camera orbit into world space if the primary VRM instance has a non-origin stage placement.
    if vrm_placement:
        scale = float(vrm_placement.get("scale") or 1.0)
        rot_y = float(vrm_placement.get("rotY") or 0.0)
        ox = float(vrm_placement.get("x") or 0.0)
        oy = float(vrm_placement.get("y") or 0.0)
        oz = float(vrm_placement.get("z") or 0.0)
        c, s = math.cos(rot_y), math.sin(rot_y)
        sx, sy, sz = tx * scale, ty * scale, tz * scale
        tx = ox + c * sx + s * sz
        ty = oy + sy
        tz = oz - s * sx + c * sz
        distance = float(np.clip(distance * scale, CAMERA_DISTANCE_MIN_M, CAMERA_DISTANCE_MAX_M))
        azimuth += rot_y

    return {
        "orbit": {
            "azimuth": round(azimuth, 6),
            "elevation": round(elevation, 6),
            "distance": round(distance, 6),
            "targetX": round(tx, 6),
            "targetY": round(ty, 6),
            "targetZ": round(tz, 6),
        },
        "fov": round(float(cam.fov_deg), 4),
        "roll": round(roll, 6),
        "projection": "perspective",
        "clipAssetId": None,
        "follow": None,
        "aim": None,
    }
