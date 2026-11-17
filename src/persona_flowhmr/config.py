"""Streaming latency presets and runtime configuration."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Literal, get_args

from .paths import DEFAULT_DEMO_VIDEO, SETTINGS_FILE, resolve_worktree_seed_settings

CAMERA_AVATAR_SCALE_MIN = 0.4
CAMERA_AVATAR_SCALE_MAX = 2.5
CAMERA_CALIBRATED_FOV_MIN_DEG = 10.0
CAMERA_CALIBRATED_FOV_MAX_DEG = 140.0


def sanitize_avatar_scale(value: object, default: float = 1.0) -> float:
    """Parse and clamp `camera_avatar_scale` into `[0.4, 2.5]`, falling back cleanly on malformed inputs."""
    try:
        parsed = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(parsed) or parsed <= 0.0:
        return float(default)
    return float(min(CAMERA_AVATAR_SCALE_MAX, max(CAMERA_AVATAR_SCALE_MIN, parsed)))


def sanitize_camera_fov_deg(value: object, default: float = 0.0) -> float:
    """Parse calibrated vertical FOV (`0.0` = auto/estimated, else clamped to `[10.0, 140.0]` deg)."""
    try:
        parsed = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(parsed) or parsed <= 0.0:
        return 0.0
    return float(min(CAMERA_CALIBRATED_FOV_MAX_DEG, max(CAMERA_CALIBRATED_FOV_MIN_DEG, parsed)))

# FlowHMR's native motion rate: sources resample to it and windows/stitching run on its frame grid.
MODEL_FPS = 30.0
# Width of one SAM 3D Body conditioning token (1024 body + 2048 hand features).
SAM_TOKEN_DIM = 3072

PresetName = Literal["ultra_low", "low", "balanced", "high", "max"]
SourceMode = Literal["video", "camera"]
PerceptionBackend = Literal["auto", "sam3d", "replay"]
ResolvedHandMode = Literal["off", "fast", "full"]
HandMode = Literal["auto", ResolvedHandMode]  # auto = use preset default_hand_mode
DetectorSize = Literal[416, 640]
YoloSize = Literal[0, DetectorSize]  # 0 = use preset default_yolo_size
YoloStride = Literal[1, 2, 3]
SamStride = Literal[0, 1, 2, 3, 4, 5, 6]  # 0 = use preset default_sam_stride
PlayoutFps = Literal[30, 60]


@dataclass(frozen=True)
class LatencyPreset:
    """Sliding-window FlowHMR parameters satisfying both GPU budget and causal playout delay."""

    name: PresetName
    label: str
    history_frames: int
    stride_frames: int
    overlap_frames: int
    lookahead_frames: int
    ode_steps: int
    sam_chunk_size: int
    default_sam_stride: int
    default_hand_mode: ResolvedHandMode
    body_decoder_layers: int
    hand_decoder_layers: int
    default_yolo_size: DetectorSize
    playout_delay_s: float

    @property
    def future_frames(self) -> int:
        """Frames a stride needs from its first emitted frame onward (`S + O + L`)."""
        return self.stride_frames + self.overlap_frames + self.lookahead_frames

    @property
    def window_frames(self) -> int:
        """Total frames passed to FlowHMR MMDiT per sliding-window solve."""
        return self.history_frames + self.future_frames

    def to_dict(self) -> dict[str, Any]:
        return asdict(self) | {"window_frames": self.window_frames}


LATENCY_PRESETS: dict[PresetName, LatencyPreset] = {
    "ultra_low": LatencyPreset(
        name="ultra_low",
        label="Ultra Low (~0.80s · Fast Body)",
        history_frames=38,
        stride_frames=6,
        overlap_frames=2,
        lookahead_frames=2,
        ode_steps=4,
        sam_chunk_size=2,
        default_sam_stride=3,
        default_hand_mode="off",
        body_decoder_layers=3,
        hand_decoder_layers=2,
        default_yolo_size=416,
        playout_delay_s=0.80,
    ),
    "low": LatencyPreset(
        name="low",
        label="Low Latency (~1.05s · Fast Body)",
        history_frames=44,
        stride_frames=9,
        overlap_frames=3,
        lookahead_frames=4,
        ode_steps=5,
        sam_chunk_size=3,
        default_sam_stride=3,
        default_hand_mode="off",
        body_decoder_layers=3,
        hand_decoder_layers=2,
        default_yolo_size=416,
        playout_delay_s=1.05,
    ),
    "balanced": LatencyPreset(
        name="balanced",
        label="Balanced (~1.45s · Fast Body+Hands)",
        history_frames=48,
        stride_frames=12,
        overlap_frames=4,
        lookahead_frames=6,
        ode_steps=6,
        sam_chunk_size=4,
        default_sam_stride=3,
        default_hand_mode="fast",
        body_decoder_layers=4,
        hand_decoder_layers=2,
        default_yolo_size=416,
        playout_delay_s=1.45,
    ),
    "high": LatencyPreset(
        name="high",
        label="High Smoothness (~2.15s · Full Hands)",
        history_frames=60,
        stride_frames=15,
        overlap_frames=6,
        lookahead_frames=10,
        ode_steps=10,
        sam_chunk_size=3,
        default_sam_stride=5,
        default_hand_mode="full",
        body_decoder_layers=6,
        hand_decoder_layers=6,
        default_yolo_size=640,
        playout_delay_s=2.15,
    ),
    "max": LatencyPreset(
        name="max",
        label="Max Context (~3.20s · Offline Quality)",
        history_frames=60,
        stride_frames=20,
        overlap_frames=8,
        lookahead_frames=28,
        ode_steps=16,
        sam_chunk_size=4,
        default_sam_stride=5,
        default_hand_mode="full",
        body_decoder_layers=6,
        hand_decoder_layers=6,
        default_yolo_size=640,
        playout_delay_s=3.20,
    ),
}


@dataclass
class EngineConfig:
    """Mutable runtime configuration for StreamingMocapEngine."""

    source_mode: SourceMode = "video"
    camera_index: int = 0
    camera_width: int = 1280
    camera_height: int = 720
    video_path: str = str(DEFAULT_DEMO_VIDEO)
    loop_video: bool = True

    preset: PresetName = "low"
    playout_fps: PlayoutFps = 60
    mirror: bool = False
    follow_movement: bool = True
    torch_compile: bool = True
    sam_stride: SamStride = 0
    hand_mode: HandMode = "auto"
    yolo_size: YoloSize = 0
    yolo_stride: YoloStride = 2
    perception_backend: PerceptionBackend = "auto"
    replay_tokens_path: str = ""

    camera_tracking_enabled: bool = True
    camera_avatar_scale: float = 1.0
    camera_fov_deg: float = 0.0  # 0.0 = auto intrinsics (+ optical-zoom VO scaling); >0 = calibrated vertical FOV deg

    vmc_enabled: bool = True
    vmc_host: str = "127.0.0.1"
    vmc_port: int = 39539
    vmc_send_camera: bool = True

    plugin_api_enabled: bool = False
    plugin_api_url: str = "ws://127.0.0.1:25034"
    plugin_api_token: str = ""
    plugin_api_inject_params: bool = False
    plugin_api_auto_configure_vmc: bool = True
    plugin_api_sync_camera: bool = True

    def get_preset(self) -> LatencyPreset:
        return LATENCY_PRESETS[self.preset]

    def effective_sam_stride(self) -> int:
        return int(self.sam_stride) if self.sam_stride > 0 else self.get_preset().default_sam_stride

    def effective_hand_mode(self) -> ResolvedHandMode:
        if self.hand_mode in get_args(ResolvedHandMode):
            return self.hand_mode
        return self.get_preset().default_hand_mode

    def effective_yolo_size(self) -> tuple[int, int]:
        sz = int(self.yolo_size) if self.yolo_size in get_args(DetectorSize) else self.get_preset().default_yolo_size
        return (sz, sz)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["has_plugin_api_token"] = bool(self.plugin_api_token)
        data["plugin_api_token"] = ""
        data["effective_sam_stride"] = self.effective_sam_stride()
        data["effective_hand_mode"] = self.effective_hand_mode()
        data["effective_yolo_size"] = self.effective_yolo_size()[0]
        data["preset_details"] = self.get_preset().to_dict()
        return data

    def save(self, path: str | Path) -> None:
        """Atomically persist configuration to a local JSON file."""
        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".tmp")
        tmp.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        tmp.replace(dest)

    @classmethod
    def load(cls, path: str | Path) -> EngineConfig:
        """Load persisted configuration from JSON if present, falling back to defaults."""
        src = Path(path)
        cfg = cls()
        if not src.is_file() and src.resolve() == SETTINGS_FILE.resolve():
            seed = resolve_worktree_seed_settings()
            if seed is not None and seed.is_file():
                src = seed
        if not src.is_file():
            return cfg
        try:
            raw = json.loads(src.read_text(encoding="utf-8"))
        except Exception:
            return cfg
        if not isinstance(raw, dict):
            return cfg
        for k, v in raw.items():
            if k in CONFIG_FIELDS and v is not None:
                setattr(cfg, k, v)
        cfg.camera_avatar_scale = sanitize_avatar_scale(cfg.camera_avatar_scale)
        cfg.camera_fov_deg = sanitize_camera_fov_deg(cfg.camera_fov_deg)
        return cfg


CONFIG_FIELDS: frozenset[str] = frozenset(f.name for f in fields(EngineConfig))
