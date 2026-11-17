import json
import struct
from pathlib import Path

import numpy as np
import pytest

from persona_flowhmr.motion_vrma import write_vrma
from persona_flowhmr.retarget import CanonicalPoseFrame


def _pose(i: int) -> CanonicalPoseFrame:
    quats = np.tile([0, 0, 0, 1], (52, 1)).astype(np.float32)
    # The same physical orientation with opposite quaternion signs.
    quats[34] = np.array([0.6, 0, 0.8, 0]) * (-1 if i % 2 else 1)
    return CanonicalPoseFrame(i / 30, i / 30, i, quats, np.array([i / 30, 0.8, -0.2]))


def test_vrma_hierarchy_and_animation_keys(tmp_path: Path):
    path = tmp_path / "motion.vrma"
    poses = [_pose(i) for i in (90, 91, 93)]
    write_vrma(path, poses)
    raw = path.read_bytes()
    assert struct.unpack_from("<4sII", raw) == (b"glTF", 2, len(raw))
    length, tag = struct.unpack_from("<I4s", raw, 12)
    assert tag == b"JSON" and length % 4 == 0
    doc = json.loads(raw[20 : 20 + length])
    data = raw[28 + length :]
    assert struct.unpack_from("<I4s", raw, 20 + length) == (len(data), b"BIN\0")
    assert doc["buffers"] == [{"byteLength": len(data)}]
    extension = doc["extensions"]["VRMC_vrm_animation"]
    assert extension["specVersion"] == "1.0"
    bones = {name: info["node"] for name, info in extension["humanoid"]["humanBones"].items()}
    assert len(bones) == len(doc["nodes"]) == 52
    assert bones["leftThumbProximal"] in doc["nodes"][bones["leftThumbMetacarpal"]]["children"]
    assert bones["rightThumbMetacarpal"] in doc["nodes"][bones["rightHand"]]["children"]
    assert doc["nodes"][bones["hips"]]["translation"][1] > 0
    assert doc["nodes"][bones["leftUpperArm"]]["translation"][0] > 0
    assert doc["nodes"][bones["rightUpperArm"]]["translation"][0] < 0

    def values(accessor):
        info = doc["accessors"][accessor]
        view = doc["bufferViews"][info["bufferView"]]
        assert view["byteOffset"] % 4 == 0 and info["componentType"] == 5126
        array = np.frombuffer(data, "<f4", view["byteLength"] // 4, view["byteOffset"])
        return array.reshape(info["count"], -1)

    anim = doc["animations"][0]
    tracks = {}
    for channel in anim["channels"]:
        target = channel["target"]
        sampler = anim["samplers"][channel["sampler"]]
        np.testing.assert_allclose(values(sampler["input"])[:, 0], [0, 1 / 30, 0.1])
        tracks[target["node"], target["path"]] = values(sampler["output"])
    assert len(tracks) == 53
    np.testing.assert_allclose(tracks[bones["hips"], "translation"], [p.hips_pos_rh for p in poses])
    for i in range(52):
        # Magnitude one and unit absolute dot product imply equal orientations.
        actual = tracks[i, "rotation"]
        expected = np.stack([p.quats_rh_xyzw[i] for p in poses])
        np.testing.assert_allclose(np.linalg.norm(actual, axis=1), 1)
        np.testing.assert_allclose(np.abs((actual * expected).sum(axis=1)), 1)
        assert np.all((actual[1:] * actual[:-1]).sum(axis=1) >= 0)


def test_vrma_rejects_nonmonotonic_time(tmp_path):
    with pytest.raises(ValueError, match="timestamps"):
        write_vrma(tmp_path / "motion.vrma", [_pose(8), _pose(8)])
