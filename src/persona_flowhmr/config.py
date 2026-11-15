"""Streaming latency presets and runtime configuration."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

from .paths import DEFAULT_DEMO_VIDEO

PresetName = Literal["low", "balanced", "high", "max"]
SourceMode = Literal["video", "camera"]
PerceptionBackend = Literal["auto", "sam3d", "replay"]


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
    "low": LatencyPreset(
        name="low",
        label="Low Latency (~2.20s)",
        history_frames=60,
        stride_frames=15,
        overlap_frames=4,
        lookahead_frames=6,
        ode_steps=8,
        sam_chunk_size=6,
        playout_delay_s=2.20,
    ),
    "balanced": LatencyPreset(
        name="balanced",
        label="Balanced (~2.75s)",
        history_frames=60,
        stride_frames=30,
        overlap_frames=6,
        lookahead_frames=12,
        ode_steps=10,
        sam_chunk_size=6,
        playout_delay_s=2.75,
    ),
    "high": LatencyPreset(
        name="high",
        label="High Smoothness (~2.95s)",
        history_frames=60,
        stride_frames=30,
        overlap_frames=8,
        lookahead_frames=18,
        ode_steps=20,
        sam_chunk_size=6,
        playout_delay_s=2.95,
    ),
    "max": LatencyPreset(
        name="max",
        label="Max Narrowband Context (~3.95s)",
        history_frames=60,
        stride_frames=30,
        overlap_frames=10,
        lookahead_frames=46,
        ode_steps=20,
        sam_chunk_size=6,
        playout_delay_s=3.95,
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

    preset: PresetName = "balanced"
    playout_fps: Literal[30, 60] = 60
    mirror: bool = False
    follow_movement: bool = True
    torch_compile: bool = True
    sam_stride: Literal[1, 2, 3, 5, 6] = 5
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

    def to_dict(self) -> dict:
        data = asdict(self)
        # Redact bearer token in status responses while reporting whether one is set.
        data["has_plugin_api_token"] = bool(self.plugin_api_token)
        data["plugin_api_token"] = ""
        data["preset_details"] = asdict(self.get_preset())
        return data
