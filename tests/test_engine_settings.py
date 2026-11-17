"""Regression tests for CLI settings overlay, live-settings dispatch, and live playout-rate switching."""

from __future__ import annotations

import argparse
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from persona_flowhmr import cli, engine
from persona_flowhmr.config import EngineConfig
from persona_flowhmr.engine import StreamingMocapEngine
from persona_flowhmr.paths import DEFAULT_DEMO_VIDEO
from persona_flowhmr.retarget import CanonicalPoseFrame
from persona_flowhmr.web_ui import create_app


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


def test_manual_take_rejects_restart_before_applying_any_settings(tmp_path, monkeypatch):
    path = tmp_path / "settings.json"
    config = EngineConfig(preset="low", mirror=False, follow_movement=True, plugin_api_token="original")
    config.save(path)
    eng = StreamingMocapEngine(config, settings_path=path)
    monkeypatch.setattr(eng, "_recording_available", True)
    monkeypatch.setattr(StreamingMocapEngine, "is_running", property(lambda self: True))
    with TestClient(create_app(eng)) as client:
        assert client.post("/api/recording/start").status_code == 200
        original = path.read_bytes()
        change = {"mirror": True, "plugin_api_token": "replacement", "preset": "balanced"}
        blocked = client.post("/api/config", json=change)
        assert blocked.status_code == 409 and "Finish the current take" in blocked.json()["detail"]
        assert eng.recording.active and not config.mirror and config.preset == "low"
        assert config.plugin_api_token == "original" and path.read_bytes() == original

        form = asdict(config) | {"mirror": True, "follow_movement": False, "plugin_api_token": ""}
        allowed = client.post("/api/config", json=form)
        assert allowed.status_code == 200 and allowed.json()["recording"]["active"]
        live = allowed.json()["live_config_fields"]
        assert "mirror" in live and "follow_movement" in live and "preset" not in live
        assert config.mirror and not config.follow_movement and config.plugin_api_token == "original"
        assert EngineConfig.load(path) == config


def test_manual_start_rejects_a_concurrent_stream_shutdown(monkeypatch):
    eng = StreamingMocapEngine()
    eng._recording_available = True
    entered = threading.Event()
    release = threading.Event()
    stop_workers = eng._stop_workers

    def paused_stop():
        entered.set()
        assert release.wait(timeout=5)
        stop_workers()

    monkeypatch.setattr(eng, "_stop_workers", paused_stop)
    with ThreadPoolExecutor(max_workers=1) as pool:
        stopping = pool.submit(eng.stop)
        try:
            assert entered.wait(timeout=5)
            with pytest.raises(ValueError, match="Stream controls are busy"):
                eng.begin_recording()
            assert not eng.recording.active
        finally:
            release.set()
        stopping.result(timeout=5)


class _FakeClock:
    """Deterministic stand-in for the engine's `time`: sleeps advance it, reads tick it forward slightly."""

    def __init__(self, t0: float) -> None:
        self.t = t0

    def perf_counter(self) -> float:
        self.t += 1e-7
        return self.t

    def sleep(self, seconds: float) -> None:
        self.t += max(0.0, seconds)


@pytest.mark.parametrize(("start_fps", "new_fps"), [(60, 30), (30, 60)])
def test_switching_playout_fps_live_keeps_a_steady_tick_cadence(
    monkeypatch: pytest.MonkeyPatch, start_fps: int, new_fps: int
) -> None:
    eng = StreamingMocapEngine(EngineConfig(playout_fps=start_fps))
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
                eng.config.playout_fps = new_fps
            if len(sent_at) == 120:
                eng._stop_event.set()

    eng._vmc_sender = FakeSender()
    eng._frame0_mono = 1000.0
    first_commit = threading.Event()
    first_commit.set()
    eng._playout_worker(first_commit, True)

    gaps = np.diff(sent_at)
    # The old rate until the switch, then the new one: no output freeze when slowing down, no burst when speeding up.
    assert np.allclose(gaps[:60], 1 / start_fps, atol=1e-4)
    assert np.allclose(gaps[60:], 1 / new_fps, atol=1e-4)
