"""Regression tests for CLI settings overlay, live-settings dispatch, and live playout-rate switching."""

from __future__ import annotations

import argparse
import threading
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from persona_flowhmr import cli, engine
from persona_flowhmr.config import EngineConfig
from persona_flowhmr.engine import StreamingMocapEngine
from persona_flowhmr.paths import DEFAULT_DEMO_VIDEO
from persona_flowhmr.retarget import CanonicalPoseFrame


def _parse_engine_flags(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    cli._add_common_engine_args(parser)
    return parser.parse_args(argv)


def test_cli_flags_overlay_saved_settings_without_dropping_unflagged_fields() -> None:
    saved = EngineConfig(
        camera_width=640,
        camera_height=480,
        plugin_api_auto_configure_vmc=False,
        mirror=True,
        preset="max",
        video_path="",
    )
    cfg = cli._config_from_args(_parse_engine_flags(["--preset", "low", "--no-loop"]), base=saved)

    # Settings without a CLI flag keep their saved values instead of resetting to defaults.
    assert (cfg.camera_width, cfg.camera_height, cfg.plugin_api_auto_configure_vmc) == (640, 480, False)
    # Given flags override the saved values; flags left unset keep them.
    assert (cfg.preset, cfg.loop_video, cfg.mirror) == ("low", False, True)
    # An empty saved video path still falls back to the demo clip.
    assert cfg.video_path == str(DEFAULT_DEMO_VIDEO)


def test_live_settings_act_only_on_changed_values(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    eng = StreamingMocapEngine(EngineConfig(), settings_path=tmp_path / "settings.json")
    actions: list[str] = []
    monkeypatch.setattr(eng, "_sync_plugin_client", lambda: actions.append("reconnect"))
    monkeypatch.setattr(eng, "_configure_plugin_receiver", lambda: actions.append("receiver"))

    def post_form(**change: object) -> None:
        # The dashboard posts every field on each change, with the saved token redacted to "".
        eng.update_live_settings(asdict(eng.config) | {"plugin_api_token": ""} | change)

    post_form(mirror=True)
    assert actions == []
    post_form(vmc_port=40000)
    assert actions == ["receiver"]
    post_form(plugin_api_url="ws://127.0.0.1:1")
    assert actions == ["receiver", "reconnect"]

    # Keys that are not config fields are ignored instead of overwriting config attributes.
    eng.update_live_settings({"get_preset": 1, "save": None})
    assert eng.get_status()["config"]["preset"] == "low"


class _FakeClock:
    """Deterministic stand-in for the engine's `time`: sleeps advance it, reads tick it forward slightly."""

    def __init__(self, t0: float) -> None:
        self.t = t0

    def perf_counter(self) -> float:
        self.t += 1e-7
        return self.t

    def sleep(self, seconds: float) -> None:
        self.t += max(0.0, seconds)


def test_switching_playout_fps_live_keeps_a_steady_tick_cadence(monkeypatch: pytest.MonkeyPatch) -> None:
    eng = StreamingMocapEngine(EngineConfig(playout_fps=60))
    identity = np.tile(np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32), (52, 1))
    eng._playout_buffer.push_many(
        [
            CanonicalPoseFrame(
                media_time_s=i / 30,
                capture_mono_s=1000.0 + i / 30,
                frame_index=i,
                quats_rh_xyzw=identity,
                hips_pos_rh=np.zeros(3, dtype=np.float32),
            )
            for i in range(300)
        ]
    )
    clock = _FakeClock(1000.0)
    monkeypatch.setattr(engine, "time", clock)
    sent_at: list[float] = []

    class FakeSender:
        packets_sent = 0

        def send_pose(self, bones: object) -> None:
            sent_at.append(clock.t)
            if len(sent_at) == 60:
                eng.config.playout_fps = 30
            if len(sent_at) == 120:
                eng._stop_event.set()

    eng._vmc_sender = FakeSender()
    eng._frame0_mono = 1000.0
    first_commit = threading.Event()
    first_commit.set()
    eng._playout_worker(first_commit, True)

    gaps = np.diff(sent_at)
    # 60 Hz until the switch, then 30 Hz: no output freeze when slowing down, no burst when speeding up.
    assert np.allclose(gaps[:60], 1 / 60, atol=1e-4)
    assert np.allclose(gaps[60:], 1 / 30, atol=1e-4)
