"""Streaming latency presets and runtime configuration."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

from .paths import DEFAULT_DEMO_VIDEO

PresetName = Literal["ultra_low", "low", "balanced", "high", "max"]
SourceMode = Literal["video", "camera"]
PerceptionBackend = Literal["auto", "sam3d", "replay"]
HandMode = Literal["auto", "off", "fast", "full"]


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
    default_hand_mode: Literal["off", "fast", "full"]
    body_decoder_layers: int
    hand_decoder_layers: int
    default_yolo_size: Literal[416, 640]
    playout_delay_s: float

    @property
    def window_frames(self) -> int:
        """Total frames passed to FlowHMR MMDiT per sliding-window solve."""
        return self.history_frames + self.stride_frames + self.overlap_frames + self.lookahead_frames

    @property
    def causal_wait_s(self) -> float:
        """Media-time span (seconds at 30 Hz) needed ahead of the first frame of a stride."""
        return (self.stride_frames + self.overlap_frames + self.lookahead_frames - 1) / 30.0


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
    playout_fps: Literal[30, 60] = 60
    mirror: bool = False
    follow_movement: bool = True
    torch_compile: bool = True
    sam_stride: Literal[0, 1, 2, 3, 4, 5, 6] = 0  # 0 = use preset default_sam_stride
    hand_mode: HandMode = "auto"  # auto = use preset default_hand_mode
    yolo_size: Literal[0, 416, 640] = 0  # 0 = use preset default_yolo_size
    yolo_stride: Literal[1, 2, 3] = 2
    perception_backend: PerceptionBackend = "auto"
    replay_tokens_path: str = ""

    vmc_enabled: bool = True
    vmc_host: str = "127.0.0.1"
    vmc_port: int = 39539

    plugin_api_enabled: bool = False
    plugin_api_url: str = "ws://127.0.0.1:25034"
    plugin_api_token: str = ""
    plugin_api_inject_params: bool = False
    plugin_api_auto_configure_vmc: bool = True

    def get_preset(self) -> LatencyPreset:
        return LATENCY_PRESETS[self.preset]

    def effective_sam_stride(self) -> int:
        return int(self.sam_stride) if self.sam_stride > 0 else self.get_preset().default_sam_stride

    def effective_hand_mode(self) -> Literal["off", "fast", "full"]:
        if self.hand_mode in ("off", "fast", "full"):
            return self.hand_mode
        return self.get_preset().default_hand_mode

    def effective_yolo_size(self) -> tuple[int, int]:
        sz = int(self.yolo_size) if self.yolo_size in (416, 640) else self.get_preset().default_yolo_size
        return (sz, sz)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["has_plugin_api_token"] = bool(self.plugin_api_token)
        data["plugin_api_token"] = ""
        data["effective_sam_stride"] = self.effective_sam_stride()
        data["effective_hand_mode"] = self.effective_hand_mode()
        data["effective_yolo_size"] = self.effective_yolo_size()[0]
        data["preset_details"] = asdict(self.get_preset())
        return data
