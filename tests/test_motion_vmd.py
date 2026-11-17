import struct
from dataclasses import replace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from persona_flowhmr.camera_tracker import CameraPoseFrame
from persona_flowhmr.motion_vmd import write_avatar_vmd, write_camera_vmd
from persona_flowhmr.retarget import CanonicalPoseFrame


def _pose():
    return CanonicalPoseFrame(
        media_time_s=4.0,
        capture_mono_s=108.0,
        frame_index=120,
        hips_pos_rh=np.array([0.21, 0.84, 0.42]),
        quats_rh_xyzw=np.tile([0.0, 0.0, 0.0, 1.0], (52, 1)),
    )


def _name(field):
    return field.rstrip(b"\0").decode("shift_jis")


def test_avatar_vmd_can_be_decoded_as_mmd_fk(tmp_path):
    pose = _pose()
    rotations = Rotation.from_euler("xyz", [[0.2, 0.3, -0.1], [-0.3, 0.1, 0.12], [0.1, -0.2, -0.3], [0.2, -0.1, -0.4]])
    pose.quats_rh_xyzw[[6, 9, 16, 34]] = rotations.as_quat()
    path = tmp_path / "dance.vmd"
    write_avatar_vmd(path, [pose, replace(pose, media_time_s=4.1)])
    data = path.read_bytes()
    assert data[:30].rstrip(b"\0") == b"Vocaloid Motion Data 0002"
    (count,) = struct.unpack_from("<I", data, 50)
    assert count == 2 * 51
    end = 54 + count * 111
    records = {}
    for name, number, *values, curve in struct.iter_unpack("<15sI7f64s", data[54:end]):
        records[_name(name), number] = (values[:3], values[3:])
        for channel in range(4):
            assert curve[16 * channel : 16 * channel + 16 : 4] == bytes([20, 20, 107, 107])
    assert len(records) == count
    assert {k for _, k in records} == {0, 3}
    units = 1.68 / 20
    x, y, z = records["センター", 3][0]
    np.testing.assert_allclose(np.array([x, 20 * 0.567 + y, -z]) * units, pose.hips_pos_rh)
    np.testing.assert_array_equal(records["左腕", 0][0], [0, 0, 0])

    def read_rotation(name):
        return Rotation.from_quat(np.asarray(records[name, 0][1]) * [-1, -1, 1, 1])

    np.testing.assert_allclose(
        read_rotation("上半身2").as_matrix(), (rotations[0] * rotations[1]).as_matrix(), atol=1e-6
    )
    droop = Rotation.from_euler("z", -35, degrees=True)
    np.testing.assert_allclose((read_rotation("左腕") * droop).as_matrix(), rotations[2].as_matrix(), atol=1e-6)
    thumb = droop.inv() * read_rotation("左親指０") * droop
    np.testing.assert_allclose(thumb.as_matrix(), rotations[3].as_matrix(), atol=1e-6)
    assert struct.unpack_from("<5IIBI", data, end) == (0, 0, 0, 0, 1, 0, 1, 4)
    toggles = {_name(name): state for name, state in struct.iter_unpack("<20sB", data[end + 29 :])}
    assert toggles == {"左足ＩＫ": 0, "右足ＩＫ": 0, "左つま先ＩＫ": 0, "右つま先ＩＫ": 0}


def test_camera_vmd_preserves_orbit_fov_and_cuts_on_shared_timeline(tmp_path):
    pose = _pose()
    cam = CameraPoseFrame(
        position_rh=np.array([0.0, 1.0, 3.0]),
        quat_rh_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
        orbit_azimuth=7.4,
        orbit_elevation=0.3,
        orbit_distance=2.1,
        orbit_target_rh=np.array([0.21, 0.84, -0.42]),
        roll=-0.15,
        fov_deg=54.8,
    )
    path = tmp_path / "camera.vmd"
    write_camera_vmd(
        path,
        [
            pose,
            replace(pose, media_time_s=4.1, camera=cam),
            replace(pose, media_time_s=4.3, camera=replace(cam, is_cut=True, orbit_azimuth=-2.0)),
            replace(pose, media_time_s=4.3 + 1 / 30),
        ],
    )
    data = path.read_bytes()
    assert _name(data[30:50]) == "カメラ・照明"
    assert struct.unpack_from("<3I", data, 50) == (0, 0, 4)
    keys = list(struct.iter_unpack("<I7f24sIB", data[62:-12]))
    assert [k[0] for k in keys] == [3, 8, 9, 10]
    assert keys[2][1:] == keys[3][1:]
    assert keys[0][1:] == keys[1][1:]
    _, distance, x, y, z, pitch, yaw, roll, curves, fov, perspective = keys[0]
    units = 1.68 / 20
    np.testing.assert_allclose(np.array([x, y, -z]) * units, cam.orbit_target_rh)
    assert -distance * units == pytest.approx(cam.orbit_distance)
    assert [yaw, -pitch, -roll] == pytest.approx([cam.orbit_azimuth, cam.orbit_elevation, cam.roll])
    assert keys[-1][6] == -2.0
    assert curves == bytes([20, 107, 20, 107]) * 6
    assert (fov, perspective) == (55, 0)
    assert data[-12:] == bytes(12)


def test_missing_camera_does_not_write_a_file(tmp_path):
    path = tmp_path / "camera.vmd"
    with pytest.raises(ValueError, match="camera poses"):
        write_camera_vmd(path, [_pose()])
    assert not path.exists()


def test_vmd_rejects_invalid_motion_values(tmp_path):
    pose = _pose()
    path = tmp_path / "avatar.vmd"
    with pytest.raises(ValueError, match="chronological"):
        write_avatar_vmd(path, [pose, replace(pose, media_time_s=1.)])
    pose.quats_rh_xyzw[0] = 0
    with pytest.raises(ValueError, match="quaternions"):
        write_avatar_vmd(path, [pose])
