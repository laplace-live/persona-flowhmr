"""Tests for Web UI header button layout and native system video file picker endpoint."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from persona_flowhmr import web_ui
from persona_flowhmr.config import EngineConfig
from persona_flowhmr.engine import StreamingMocapEngine


def test_dashboard_single_line_buttons_and_pick_video_endpoint(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    app = web_ui.create_app(StreamingMocapEngine(EngineConfig()))

    with TestClient(app) as client:
        html = client.get("/").text
        assert "white-space: nowrap" in html
        assert '<div class="actions">' in html
        assert 'id="browseVideoBtn"' in html
        assert "pickVideoFile()" in html

        fake_video = tmp_path / "picked_clip.mp4"
        fake_video.write_bytes(b"0000")
        captured_initial: list[str | None] = []

        def _fake_dialog(current_path: str | None = None) -> str:
            captured_initial.append(current_path)
            return str(fake_video)

        monkeypatch.setattr(web_ui, "_open_system_video_dialog", _fake_dialog)

        res = client.post("/api/pick-video", json={"current_path": str(tmp_path)})
        assert res.status_code == 200
        assert res.json() == {"path": str(fake_video)}
        assert captured_initial == [str(tmp_path)]

        # Cancelled dialog returns empty string without error.
        monkeypatch.setattr(web_ui, "_open_system_video_dialog", lambda _cur=None: "")
        res_cancel = client.post("/api/pick-video", json={"current_path": ""})
        assert res_cancel.status_code == 200
        assert res_cancel.json() == {"path": ""}


def test_settings_persisted_across_refresh_and_restart(tmp_path: Path) -> None:
    settings_file = tmp_path / "settings.json"
    eng1 = StreamingMocapEngine(EngineConfig(), settings_path=settings_file)
    app1 = web_ui.create_app(eng1)

    with TestClient(app1) as client1:
        html = client1.get("/").text
        assert "scheduleSaveConfig" in html
        res = client1.post(
            "/api/config",
            json={
                "preset": "balanced",
                "mirror": True,
                "camera_index": 2,
                "video_path": "D:/videos/custom.mp4",
                "plugin_api_token": "sk-lp-v1-test-token",
            },
        )
        assert res.status_code == 200

    assert settings_file.is_file()
    loaded = EngineConfig.load(settings_file)
    assert loaded.preset == "balanced"
    assert loaded.mirror is True
    assert loaded.camera_index == 2
    assert loaded.video_path == "D:/videos/custom.mp4"
    assert loaded.plugin_api_token == "sk-lp-v1-test-token"

    # Simulated page refresh / server restart keeps redacted token intact when UI sends empty string.
    eng2 = StreamingMocapEngine(loaded, settings_path=settings_file)
    app2 = web_ui.create_app(eng2)
    with TestClient(app2) as client2:
        st = client2.get("/api/status").json()["config"]
        assert st["preset"] == "balanced"
        assert st["mirror"] is True
        assert st["camera_index"] == 2
        assert st["has_plugin_api_token"] is True

        client2.post("/api/config", json={"playout_fps": 30, "plugin_api_token": ""})

    reloaded = EngineConfig.load(settings_file)
    assert reloaded.playout_fps == 30
    assert reloaded.plugin_api_token == "sk-lp-v1-test-token"
