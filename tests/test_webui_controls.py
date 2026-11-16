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
