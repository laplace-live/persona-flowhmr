"""Full-video recording uses one finite run and finalizes on the producer's EOF."""

import threading
from dataclasses import asdict
from unittest.mock import Mock

import numpy as np
import pytest
from fastapi.testclient import TestClient

from persona_flowhmr import engine, recorded_frames, web_ui
from persona_flowhmr.config import EngineConfig
from persona_flowhmr.sources import SampledFrame


@pytest.fixture
def video_engine(monkeypatch, tmp_path):
    path = tmp_path / "dancer.mp4"
    path.touch()
    instance = engine.StreamingMocapEngine(
        EngineConfig(video_path=str(path), loop_video=True, camera_tracking_enabled=False)
    )
    monkeypatch.setattr(engine.torch.cuda, "is_available", lambda: False)
    instance._yolox = Mock(yolo_stride=1)
    instance._yolox.detect_and_track_batch.side_effect = lambda rgb, **_: np.tile([1, 1, 31, 31], (len(rgb), 1))
    instance._solver = Mock(last_solve_ms=0.0)
    instance._solver.solve_window.side_effect = lambda tokens, indices, **_: np.zeros((len(indices), 491))
    instance._decoder = Mock()
    instance._decoder.decode_window.side_effect = lambda window: {
        "quats_rh_xyzw": np.tile([0, 0, 0, 1.0], (len(window), 52, 1)),
        "ljp_cam": np.zeros((len(window), 52, 3)),
        "srp_y_cam": np.ones(len(window)),
        "srv_xz_cam": np.zeros((len(window), 2)),
    }
    monkeypatch.setattr(instance, "_ensure_models_loaded", lambda: (engine.ReplayTokenExtractor(), "replay"))
    yield instance
    instance.stop()


def _wait_for_end(instance):
    for thread in instance._threads:
        thread.join(timeout=5)
    assert not instance.is_running, "The full-video workers must exit at EOF"
    return instance.get_status()


def test_short_video_starts_at_zero_flushes_at_eof_and_preserves_normal_loop(video_engine, monkeypatch):
    rgb = np.zeros((32, 32, 3), dtype=np.uint8)
    samples = [SampledFrame(i, i / 30, 1 + i / 30, rgb) for i in range(5)]
    source = Mock()
    source.read_next.side_effect = [*samples, None]
    open_video = Mock(return_value=source)
    monkeypatch.setattr(engine, "VideoFileSource", open_video)

    video_engine.record_video()
    status = _wait_for_end(video_engine)
    assert status["recording"]["video_result"] == "complete", status["telemetry"]["error"]
    assert not status["recording"]["full_video"] and not status["recording"]["active"]
    assert status["telemetry"]["state"] == "stopped"
    frames = video_engine.recording.frames()
    assert [f.frame_index for f in frames] == list(range(5))
    assert [f.media_time_s for f in frames] == pytest.approx([i / 30 for i in range(5)])
    open_video.assert_called_once_with(video_engine.config.video_path, loop=False, realtime=False, target_fps=30)
    source.close.assert_called_once()

    assert video_engine.config.loop_video
    source.read_next.side_effect = [None]
    video_engine.start(realtime_source=True, playout=False)
    _wait_for_end(video_engine)
    open_video.assert_called_with(video_engine.config.video_path, loop=True, realtime=True, target_fps=30)
    assert video_engine.recording.frames() is frames


@pytest.mark.parametrize("failure", [None, RuntimeError("decoder failed")])
def test_empty_or_unreadable_video_finishes_with_error(video_engine, monkeypatch, failure):
    source = Mock()
    source.read_next.side_effect = [failure]
    monkeypatch.setattr(engine, "VideoFileSource", Mock(return_value=source))
    video_engine.record_video()
    status = _wait_for_end(video_engine)
    assert status["recording"]["video_result"] == "error"
    assert status["recording"]["frames"] == 0 and not status["recording"]["active"]
    assert status["telemetry"]["state"] == "error"
    assert ("decoder failed" if failure else "no readable frames") in status["telemetry"]["error"]
    source.close.assert_called_once()


def test_final_spool_write_failure_is_reported_before_full_video_completes(video_engine, monkeypatch):
    source = Mock()
    source.read_next.side_effect = [SampledFrame(0, 0, 1, np.zeros((32, 32, 3), dtype=np.uint8)), None]
    monkeypatch.setattr(engine, "VideoFileSource", Mock(return_value=source))
    spool = Mock()
    spool.write.side_effect = OSError("disk full")
    monkeypatch.setattr(recorded_frames, "TemporaryFile", Mock(return_value=spool))
    video_engine.record_video()
    status = _wait_for_end(video_engine)
    assert status["recording"]["video_result"] == "error"
    assert status["recording"]["frames"] == 0
    assert "disk full" in status["telemetry"]["error"]


def test_cancelling_armed_video_and_conflicting_controls(video_engine, monkeypatch):
    entered_read = threading.Event()

    def wait_for_cancel():
        entered_read.set()
        assert video_engine._stop_event.wait(timeout=5)
        return None

    source = Mock()
    source.read_next.side_effect = wait_for_cancel
    open_video = Mock(return_value=source)
    monkeypatch.setattr(engine, "VideoFileSource", open_video)

    video_engine.record_video()
    assert entered_read.wait(timeout=5)
    status = video_engine.get_status()
    assert status["recording"]["active"] and status["recording"]["full_video"]
    assert status["recording"]["frames"] == 0
    for competing in (video_engine.record_video, video_engine.start, video_engine.begin_recording,
                      video_engine.finish_recording):
        with pytest.raises(ValueError):
            competing()
    with pytest.raises(ValueError):
        video_engine.update_live_settings({"source_mode": "camera"})
    assert video_engine.config.source_mode == "video"
    with pytest.raises(ValueError):
        video_engine.record_video({"video_path": "competing.mp4", "mirror": True})
    assert video_engine.config.video_path != "competing.mp4" and not video_engine.config.mirror
    video_engine.stop()
    status = video_engine.get_status()
    assert status["recording"]["video_result"] == "cancelled"
    assert not status["running"] and not status["recording"]["active"]
    open_video.assert_called_once()


def test_web_record_request_binds_file_and_settings_without_live_output(video_engine, monkeypatch, tmp_path):
    chosen = video_engine.config.video_path
    other = tmp_path / "other-tab.mp4"
    other.touch()
    # Another client changed the defaults after this dashboard selected its file.
    video_engine.update_live_settings({"video_path": str(other)})
    video_engine.config.plugin_api_token = "saved-token"
    connect = Mock()
    configure = Mock()
    monkeypatch.setattr(video_engine, "_sync_plugin_client", connect)
    monkeypatch.setattr(video_engine, "_configure_plugin_receiver", configure)
    rgb = np.zeros((32, 32, 3), dtype=np.uint8)
    source = Mock()
    source.read_next.side_effect = [SampledFrame(0, 0, 1, rgb), None]
    open_video = Mock(return_value=source)
    monkeypatch.setattr(engine, "VideoFileSource", open_video)
    with TestClient(web_ui.create_app(video_engine)) as client:
        result = client.post("/api/recording/video", json={
            "video_path": chosen, "mirror": True, "camera_avatar_scale": 100,
            "plugin_api_enabled": True, "plugin_api_token": "", "vmc_port": 39540,
        })
        assert result.status_code == 200
        assert result.json()["config"]["video_path"] == chosen
        assert result.json()["config"]["mirror"]
        assert "saved-token" not in result.text
        assert _wait_for_end(video_engine)["recording"]["video_result"] == "complete"
        open_video.assert_called_once_with(chosen, loop=False, realtime=False, target_fps=30)
        assert video_engine.config.plugin_api_token == "saved-token"
        assert video_engine.config.camera_avatar_scale == 2.5
        # A client that already configured the engine can still make a bodyless request.
        source.read_next.side_effect = [SampledFrame(0, 0, 1, rgb), None]
        assert client.post("/api/recording/video").status_code == 200
        assert _wait_for_end(video_engine)["recording"]["video_result"] == "complete"
    connect.assert_not_called()
    configure.assert_not_called()


@pytest.mark.parametrize("invalid", [{"video_path": "missing-file.mp4"}, {"source_mode": "camera"}, {"preset": "typo"}])
def test_invalid_record_request_preserves_previous_configuration(video_engine, invalid):
    previous = asdict(video_engine.config)
    with TestClient(web_ui.create_app(video_engine)) as client:
        result = client.post("/api/recording/video", json={"mirror": True, **invalid})
        assert result.status_code == 409
        assert asdict(video_engine.config) == previous
        assert not video_engine.is_running
