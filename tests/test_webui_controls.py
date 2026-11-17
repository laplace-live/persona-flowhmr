"""Tests for Web UI header button layout and native system video file picker endpoint."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from persona_flowhmr import web_ui
from persona_flowhmr.camera_tracker import decompose_three_camera_pose
from persona_flowhmr.config import EngineConfig
from persona_flowhmr.engine import StreamingMocapEngine
from persona_flowhmr.retarget import CanonicalPoseFrame


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


def test_recording_controls_and_download_cleanup(monkeypatch, tmp_path):
    eng = StreamingMocapEngine(EngineConfig(), settings_path=tmp_path / "settings.json")
    with TestClient(web_ui.create_app(eng)) as client:
        assert client.post("/api/recording/start").status_code == 409
        monkeypatch.setattr(eng, "_recording_available", True)
        manual = {"available": True, "full_video": False, "video_result": None, "has_camera": False}
        assert client.post("/api/recording/start").json()["recording"] == manual | {"active": True, "frames": 0}
        assert client.get("/api/recording/download/avatar.vmd").status_code == 409
        assert client.post("/api/recording/stop").json()["recording"]["active"] is False
        assert client.get("/api/recording/download/avatar.vmd").status_code == 409

        client.post("/api/recording/start")
        frame = CanonicalPoseFrame(0., 100., 0, np.tile([0., 0., 0., 1.], (52, 1)), np.array([0., 1., 0.]))
        eng.recording.append([frame], mirror=False, follow=True)
        assert client.post("/api/recording/stop").json()["recording"] == manual | {"active": False, "frames": 1}
        temp_dir = tmp_path / "downloads"
        temp_dir.mkdir()
        monkeypatch.setattr(web_ui.tempfile, "tempdir", str(temp_dir))
        for suffix in ("vrma", "fbx", "vmd"):
            filename = f"avatar.{suffix}"
            response = client.get(f"/api/recording/download/{filename}")
            assert response.status_code == 200
            assert len(response.content) > 100
            assert response.headers["content-disposition"] == f'attachment; filename="{filename}"'
            assert not list(temp_dir.iterdir())
        assert client.get("/api/recording/download/camera.vmd").status_code == 422
        assert not list(temp_dir.iterdir())
        assert client.get("/api/recording/download/camera.fbx").status_code == 422
        assert client.get("/api/status").json()["recording"]["frames"] == 1

        client.post("/api/recording/start")
        camera = decompose_three_camera_pose(np.array([0, 1, 3]), np.eye(3), 3., 50.)
        eng.recording.append([replace(frame, camera=camera)], mirror=False, follow=True)
        client.post("/api/recording/stop")
        response = client.get("/api/recording/download/camera.vmd")
        assert response.status_code == 200
        assert response.content[30:50].rstrip(b"\0").decode("shift_jis") == "カメラ・照明"
        assert not list(temp_dir.iterdir())
