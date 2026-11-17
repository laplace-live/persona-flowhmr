from dataclasses import replace
from io import BytesIO
from zipfile import ZipFile

import numpy as np
import pytest
from fastapi.testclient import TestClient

from persona_flowhmr import web_ui
from persona_flowhmr.camera_tracker import decompose_three_camera_pose
from persona_flowhmr.engine import StreamingMocapEngine
from persona_flowhmr.retarget import CanonicalPoseFrame


@pytest.fixture
def recorded_engine():
    engine = StreamingMocapEngine()
    camera = decompose_three_camera_pose(np.array([0, 1, 3]), np.eye(3), 3.0, 50.0)
    pose = CanonicalPoseFrame(
        0.0, 100.0, 0, np.tile([0.0, 0.0, 0.0, 1.0], (52, 1)), np.array([0.0, 1.0, 0.0]), camera=camera
    )
    engine.recording.start()
    engine.recording.append([pose, replace(pose, frame_index=1, media_time_s=1 / 30)], mirror=False, follow=True)
    engine.recording.stop()
    return engine


@pytest.mark.parametrize("ext", ["vrma", "fbx", "vmd"])
def test_paired_download_uses_same_take_as_individual_files(recorded_engine, ext, monkeypatch, tmp_path):
    monkeypatch.setattr(web_ui.tempfile, "tempdir", str(tmp_path))
    base = "/api/recording/download/"
    with TestClient(web_ui.create_app(recorded_engine)) as client:
        names = [f"avatar.{ext}", "camera.vmd"]
        expected = {name: client.get(base + name).content for name in names}
        filename = f"avatar-{ext}-camera.zip"
        # Start another recording between the two writers. Both files in the
        # previous take must remain downloadable from the same captured frames.
        writer = web_ui.write_motion

        def write_then_start_new_take(path, frames, *, camera):
            writer(path, frames, camera=camera)
            if camera:
                recorded_engine.recording.start()

        monkeypatch.setattr(web_ui, "write_motion", write_then_start_new_take)
        response = client.get(base + filename)
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/zip"
        assert response.headers["content-disposition"] == f'attachment; filename="{filename}"'
        with ZipFile(BytesIO(response.content)) as bundle:
            assert set(bundle.namelist()) == set(names)
            for name, data in expected.items():
                assert bundle.read(name) == data
        assert not list(tmp_path.iterdir())


def test_paired_download_requires_camera_and_cleans_partial_archive(recorded_engine, monkeypatch, tmp_path):
    monkeypatch.setattr(web_ui.tempfile, "tempdir", str(tmp_path))
    frames = [replace(frame, camera=None) for frame in recorded_engine.recording.frames()]
    recorded_engine.recording.start()
    recorded_engine.recording.append(frames, mirror=False, follow=True)
    recorded_engine.recording.stop()
    with TestClient(web_ui.create_app(recorded_engine)) as client:
        response = client.get("/api/recording/download/avatar-vrma-camera.zip")
        assert response.status_code == 422
        assert "camera" in response.json()["detail"].lower()
        assert not list(tmp_path.iterdir())
