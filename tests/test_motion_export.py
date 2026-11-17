import json
import struct
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from persona_flowhmr import cli, engine, motion_vmd, recorded_frames
from persona_flowhmr.camera_tracker import CameraPoseFrame
from persona_flowhmr.config import EngineConfig
from persona_flowhmr.motion_export import MotionRecording
from persona_flowhmr.retarget import CanonicalPoseFrame
from persona_flowhmr.sources import SampledFrame


def _frame(i: int) -> CanonicalPoseFrame:
    rotations = np.tile([0.0, 0.0, 0.0, 1.0], (52, 1))
    rotations[1] = [0.5, -0.5, 0.5, 0.5]
    camera = CameraPoseFrame(
        position_rh=np.array([2, 3, 4]),
        quat_rh_xyzw=np.array([0.5, 0.5, -0.5, 0.5]),
        orbit_target_rh=np.array([4.0, 1.5, 7.0]),
        orbit_azimuth=0.9,
        orbit_elevation=0.1,
        orbit_distance=3.2,
        roll=-0.2,
        fov_deg=38.0,
        is_cut=True,
    )
    return CanonicalPoseFrame(i / 30, i / 30, i, rotations, np.array([0.25, 0.9, 1.0]), camera=camera)


@pytest.mark.parametrize("follow", [True, False])
@pytest.mark.parametrize("mirror", [True, False])
def test_recording_mirrors_anatomy_travel_and_camera(mirror, follow):
    source = _frame(10)
    recording = MotionRecording()
    recording.start()
    recording.append([source], mirror=mirror, follow=follow)
    recording.stop()
    (pose,) = recording.frames()
    assert pose.media_time_s == source.media_time_s
    expected_x = -0.25 if mirror else 0.25
    np.testing.assert_allclose(pose.hips_pos_rh, [expected_x if follow else 0, 0.9, 1 if follow else 0])
    left, right = pose.quats_rh_xyzw[1:3]
    np.testing.assert_allclose(left, [0, 0, 0, 1] if mirror else [0.5, -0.5, 0.5, 0.5])
    np.testing.assert_allclose(right, [0.5, 0.5, -0.5, 0.5] if mirror else [0, 0, 0, 1])
    cam = pose.camera
    np.testing.assert_allclose(cam.position_rh, [-2 if mirror else 2, 3, 4])
    np.testing.assert_allclose(cam.orbit_target_rh, [-4 if mirror else 4, 1.5, 7])
    np.testing.assert_allclose(cam.quat_rh_xyzw, [0.5, -0.5, 0.5, 0.5] if mirror else [0.5, 0.5, -0.5, 0.5])
    assert cam.orbit_azimuth == (-0.9 if mirror else 0.9)
    assert cam.roll == (0.2 if mirror else -0.2)
    assert (cam.orbit_elevation, cam.orbit_distance, cam.fov_deg, cam.is_cut) == (0.1, 3.2, 38.0, True)
    np.testing.assert_array_equal(source.hips_pos_rh, [0.25, 0.9, 1])
    np.testing.assert_array_equal(source.quats_rh_xyzw[1], [0.5, -0.5, 0.5, 0.5])


def test_recording_lifecycle_preserves_finished_take_during_next_recording():
    rec = MotionRecording()
    rec.append([_frame(0)], mirror=False, follow=True)
    with pytest.raises(ValueError, match="No motion"):
        rec.frames()
    rec.start()
    rec.append([_frame(1)], mirror=False, follow=True)
    rec.start()
    with pytest.raises(ValueError, match="Finish recording"):
        rec.frames()
    rec.stop()
    first = rec.frames()
    rec.append([_frame(2)], mirror=False, follow=True)
    rec.start()
    rec.append([_frame(3)], mirror=False, follow=True)
    rec.stop()
    assert [f.frame_index for f in first] == [1]
    assert [f.frame_index for f in rec.frames()] == [3]


@pytest.mark.parametrize("failure", [0, OSError("disk full")])
def test_spool_retries_short_writes_and_preserves_committed_frames(monkeypatch, tmp_path, failure):
    frames = recorded_frames.RecordedFrames()
    with (tmp_path / "spool").open("w+b", buffering=0) as raw:
        spool = Mock(wraps=raw)
        factory = Mock(return_value=spool)
        monkeypatch.setattr(recorded_frames, "TemporaryFile", factory)
        spool.write.side_effect = lambda view: raw.write(view[:97])
        frames.extend([_frame(0), _frame(1)])
        factory.assert_called_once_with(buffering=0)
        assert spool.write.call_count > 1
        expected_size = raw.tell()

        def incomplete_batch(view):
            if raw.tell() == expected_size:
                return raw.write(view[:97])
            if isinstance(failure, OSError):
                raise failure
            return failure

        spool.write.side_effect = incomplete_batch
        with pytest.raises(OSError):
            frames.extend([_frame(2)])
        assert raw.seek(0, 2) == expected_size
        assert [frame.frame_index for frame in frames] == [0, 1]
        spool.write.side_effect = raw.write
        frames.extend([_frame(3)])
        assert [frame.frame_index for frame in frames] == [0, 1, 3]


def test_manual_capture_interval_filters_delayed_batches_and_finishes_without_another_click():
    stream = engine.StreamingMocapEngine(EngineConfig())
    stream._recording_available = True
    stream.telemetry.frames_captured = 3
    stream.begin_recording()
    stream.telemetry.frames_captured = 7
    poses = [replace(_frame(i), camera=None) if 3 <= i < 7 else _frame(i) for i in range(10)]
    stream.recording.append(poses[:5], mirror=False, follow=True)
    stream.finish_recording()
    assert stream.recording.status() == {"active": True, "finishing": True, "frames": 2, "has_camera": False}
    stream.telemetry.frames_captured = len(poses)
    stream.finish_recording()  # A retried request keeps the original upper boundary.
    with pytest.raises(ValueError, match="final frames"):
        stream.begin_recording()
    stream.recording.append(poses[5:], mirror=False, follow=True)
    assert stream.recording.status() == {"active": False, "finishing": False, "frames": 4, "has_camera": False}
    assert [p.frame_index for p in stream.recording.frames()] == [3, 4, 5, 6]


def test_empty_manual_interval_finishes_immediately_despite_uncommitted_frames():
    stream = engine.StreamingMocapEngine(EngineConfig())
    stream._recording_available = True
    stream.telemetry.frames_captured = 3
    stream.begin_recording()
    stream.finish_recording()
    assert not stream.recording.active
    stream.recording.append([_frame(i) for i in range(3)], mirror=False, follow=True)
    with pytest.raises(ValueError, match="No motion"):
        stream.recording.frames()


def test_stopping_stream_releases_partially_drained_manual_take():
    stream = engine.StreamingMocapEngine(EngineConfig())
    stream._recording_available = True
    stream.begin_recording()
    stream.telemetry.frames_captured = 3
    stream.finish_recording()
    stream.recording.append([_frame(0)], mirror=False, follow=True)
    assert stream.recording.status()["finishing"]
    stream.stop()
    assert not stream.recording.status()["finishing"]
    stream.recording.append([_frame(1)], mirror=False, follow=True)
    assert [p.frame_index for p in stream.recording.frames()] == [0]


@pytest.fixture
def source_file(tmp_path):
    video = tmp_path / "source.mp4"
    video.touch()
    return video


@pytest.fixture
def inference(monkeypatch):
    recording = MotionRecording()
    instance = Mock(recording=recording, is_running=False, telemetry=SimpleNamespace(state="stopped"))
    def record_video():
        recording.start()
        recording.append([_frame(i) for i in range(5)], mirror=False, follow=True)

    instance.record_video.side_effect = record_video
    instance.stop.side_effect = recording.stop
    factory = Mock(return_value=instance)
    monkeypatch.setattr(engine, "StreamingMocapEngine", factory)
    return factory


def test_cli_exports_all_tracks_from_one_solve(source_file, tmp_path, inference):
    folder = tmp_path / "take"
    args = ["export", "--video", str(source_file)]
    avatars = [folder / filename for filename in ("avatar.VRMA", "avatar.fbx", "avatar.vmd")]
    for avatar in avatars:
        args.extend(["--avatar", str(avatar)])
    camera = folder / "camera.vmd"
    assert cli.main([*args, "--camera-motion", str(camera)]) == 0
    inference.assert_called_once()
    cfg = inference.call_args.args[0]
    assert Path(cfg.video_path) == source_file and cfg.source_mode == "video"
    assert not cfg.loop_video and not cfg.vmc_enabled and not cfg.plugin_api_enabled
    inference.return_value.record_video.assert_called_once_with()
    inference.return_value.stop.assert_called_once()
    vrma, fbx, vmd = [path.read_bytes() for path in avatars]
    assert vrma.startswith(b"glTF") and fbx.startswith(b"Kaydara FBX Binary")
    assert vmd.startswith(b"Vocaloid Motion Data") and int.from_bytes(vmd[50:54], "little") > 0
    data = camera.read_bytes()
    assert int.from_bytes(data[50:54], "little") == 0 and int.from_bytes(data[58:62], "little") == 5


def test_offline_engine_flushes_last_partial_window_on_eof(source_file, tmp_path, monkeypatch):
    # Five frames end before even one full lookahead window. Run the actual
    # worker threads, queue, window scheduling, stitcher and recorder on CPU.
    rgb = np.zeros((32, 32, 3), dtype=np.uint8)
    samples = [SampledFrame(i, i / 30, 1 + i / 30, rgb) for i in range(5)]
    source = Mock()
    source.read_next.side_effect = [*samples, None]
    open_source = Mock(return_value=source)
    monkeypatch.setattr(engine, "VideoFileSource", open_source)
    monkeypatch.setattr(engine.torch.cuda, "is_available", lambda: False)

    def load_models(instance):
        instance._yolox = Mock(yolo_stride=1)
        instance._yolox.detect_and_track_batch.side_effect = lambda rgb, **_: np.tile([1, 1, 31, 31], (len(rgb), 1))
        instance._solver = Mock(last_solve_ms=0.0)
        instance._solver.solve_window.side_effect = lambda tokens, indices, **_: np.zeros((len(indices), 491))
        instance._decoder = Mock()
        instance._decoder.decode_window.side_effect = lambda indices: {
            "quats_rh_xyzw": np.tile(np.array([0, 0, 0, 1], dtype=np.float32), (len(indices), 52, 1)),
            "ljp_cam": np.zeros((len(indices), 52, 3)),
            "srp_y_cam": np.ones(len(indices)),
            "srv_xz_cam": np.zeros((len(indices), 2)),
        }
        return engine.ReplayTokenExtractor(), "replay"

    monkeypatch.setattr(engine.StreamingMocapEngine, "_ensure_models_loaded", load_models)
    path = tmp_path / "avatar.vrma"
    assert cli.main(["export", "--video", str(source_file), "--avatar", str(path), "--no-camera-track"]) == 0
    open_source.assert_called_once_with(str(source_file), loop=False, realtime=False, target_fps=30)
    source.close.assert_called_once()
    raw = path.read_bytes()
    (length,) = struct.unpack_from("<I", raw, 12)
    doc = json.loads(raw[20 : 20 + length])
    time = doc["accessors"][doc["animations"][0]["samplers"][0]["input"]]
    assert time["count"] == 5
    assert time["max"][0] == pytest.approx(4 / 30)


def test_cli_writer_failure_preserves_previous_files(source_file, tmp_path, inference, monkeypatch, capsys):
    avatar, camera = tmp_path / "take.vrma", tmp_path / "camera.vmd"
    avatar.write_bytes(b"previous take")
    monkeypatch.setattr(motion_vmd, "write_camera_vmd", Mock(side_effect=OSError("disk full")))
    assert (
        cli.main(["export", "--video", str(source_file), "--avatar", str(avatar), "--camera-motion", str(camera)]) == 1
    )
    assert "disk full" in capsys.readouterr().err
    assert avatar.read_bytes() == b"previous take"
    assert set(tmp_path.iterdir()) == {source_file, avatar}
    inference.return_value.stop.assert_called_once()


@pytest.mark.parametrize("phase", ["backup", "install"])
def test_cli_rolls_back_published_outputs(source_file, tmp_path, inference, monkeypatch, capsys, phase):
    avatar = tmp_path / "avatar.vrma"
    new_avatar = tmp_path / "new-avatar.fbx"
    camera = tmp_path / "camera.vmd"
    unrelated = tmp_path / "avatar.bak"
    avatar.write_bytes(b"earlier avatar")
    camera.write_bytes(b"earlier camera")
    unrelated.write_bytes(b"unrelated backup")
    replace_file = Path.replace

    def fail_on_camera(path, target):
        if (phase == "backup" and path == camera) or (phase == "install" and target == camera and path.suffix != ".bak"):
            assert avatar.read_bytes().startswith(b"glTF")
            assert new_avatar.exists()
            raise PermissionError("camera locked")
        return replace_file(path, target)

    monkeypatch.setattr(Path, "replace", fail_on_camera)
    args = ["export", "--video", str(source_file), "--avatar", str(avatar), "--avatar", str(new_avatar)]
    assert cli.main([*args, "--camera-motion", str(camera)]) == 1
    assert "camera locked" in capsys.readouterr().err
    assert avatar.read_bytes() == b"earlier avatar"
    assert camera.read_bytes() == b"earlier camera"
    assert unrelated.read_bytes() == b"unrelated backup"
    assert set(tmp_path.iterdir()) == {source_file, avatar, camera, unrelated}


def test_cli_warns_on_locked_backup_after_publishing_all_outputs(source_file, tmp_path, inference, monkeypatch, capsys):
    avatar, camera = tmp_path / "avatar.vrma", tmp_path / "camera.vmd"
    avatar.write_bytes(b"earlier avatar")
    camera.write_bytes(b"earlier camera")
    unlink_file = Path.unlink
    backups = []

    def lock_first_backup(path, **kwargs):
        if path.suffix == ".bak":
            backups.append(path)
            if len(backups) == 1:
                raise PermissionError("backup locked")
        return unlink_file(path, **kwargs)

    monkeypatch.setattr(Path, "unlink", lock_first_backup)
    args = ["export", "--video", str(source_file), "--avatar", str(avatar), "--camera-motion", str(camera)]
    assert cli.main(args) == 0
    output = capsys.readouterr()
    assert len(backups) == 2
    assert "Warning" in output.err and "backup locked" in output.err and str(backups[0]) in output.err
    assert str(avatar) in output.out and str(camera) in output.out
    assert avatar.read_bytes().startswith(b"glTF")
    assert camera.read_bytes().startswith(b"Vocaloid Motion Data")
    assert backups[0].read_bytes() == b"earlier avatar"
    assert set(tmp_path.iterdir()) == {source_file, avatar, camera, backups[0]}


@pytest.mark.parametrize(
    "outputs",
    [
        [],
        ["--avatar", "same.vmd", "--camera-motion", "same.vmd"],
        ["--avatar", "clip.fbx", "--avatar", "./clip.fbx"],
        ["--avatar", "motion.bvh"],
        ["--camera-motion", "camera.vrma"],
        ["--no-camera-tracking", "--camera-motion", "camera.vmd"],
    ],
)
def test_cli_rejects_invalid_exports_before_inference(source_file, outputs, inference):
    assert cli.main(["export", "--video", str(source_file), *outputs]) == 1
    inference.assert_not_called()
