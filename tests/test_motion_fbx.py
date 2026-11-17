"""Read back the binary FBX records independently of the exporter."""

import struct
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from persona_flowhmr.motion_fbx import write_fbx
from persona_flowhmr.retarget import CanonicalPoseFrame


def _records(data, offset, end):
    while offset < end:
        stop, count, size, name_len = struct.unpack_from("<IIIB", data, offset)
        if stop == 0:
            break
        offset += 13
        name = data[offset : offset + name_len].decode()
        offset += name_len
        children_at = offset + size
        props = []
        for _ in range(count):
            tag = chr(data[offset])
            offset += 1
            if tag == "S":
                (length,) = struct.unpack_from("<I", data, offset)
                offset += 4
                value = data[offset : offset + length].decode()
                offset += length
            elif tag in "lfi":
                length, encoding, nbytes = struct.unpack_from("<III", data, offset)
                assert encoding == 0
                offset += 12
                value = np.frombuffer(data, {"l": "<i8", "f": "<f4", "i": "<i4"}[tag], length, offset)
                offset += nbytes
            else:
                fmt = {"L": "<q", "D": "<d", "I": "<i"}[tag]
                (value,) = struct.unpack_from(fmt, data, offset)
                offset += struct.calcsize(fmt)
            props.append(value)
        assert offset == children_at <= stop <= len(data)
        yield name, props, list(_records(data, offset, stop))
        offset = stop


def test_fbx_preserves_skeleton_timing_root_translation_and_local_rotations(tmp_path: Path):
    frames = []
    for i, pitch in enumerate((85, 91, 96)):
        quats = np.tile([0.0, 0.0, 0.0, 1.0], (52, 1))
        quats[22] = Rotation.from_euler("xyz", [13, pitch, -52], degrees=True).as_quat()
        index = 75 + i
        frames.append(CanonicalPoseFrame(index / 30, index / 30, index, quats, np.array([i * 0.1, 0.9, 0.0])))
    path = tmp_path / "clip.fbx"
    write_fbx(path, frames)
    data = path.read_bytes()
    assert data.startswith(b"Kaydara FBX Binary  \x00\x1a\x00" + struct.pack("<I", 7400))
    sections = {name: children for name, _, children in _records(data, 27, len(data))}
    objects = {props[0]: (kind, props[1].split("\x00")[0], children) for kind, props, children in sections["Objects"]}
    bones = {name: obj_id for obj_id, (kind, name, _) in objects.items() if kind == "Model"}
    connections = [props for _, props, _ in sections["Connections"]]
    assert len(bones) == 52
    assert ["OO", bones["mixamorigLeftHandIndex1"], bones["mixamorigLeftHand"]] in connections

    def read_track(bone, channel):
        (curve_node,) = [
            child
            for mode, child, dest, *prop in connections
            if mode == "OP" and dest == bones[bone] and prop == [channel]
        ]
        axes = {}
        for _, child, dest, *prop in connections:
            if dest != curve_node:
                continue
            fields = {name: p[0] for name, p, _ in objects[child][2]}
            np.testing.assert_allclose(fields["KeyTime"] / 46_186_158_000, [0, 1 / 30, 2 / 30])
            axes[prop[0]] = fields["KeyValueFloat"]
        return np.column_stack([axes[f"d|{axis}"] for axis in "XYZ"])

    hips = read_track("mixamorigHips", "Lcl Translation")
    np.testing.assert_allclose(hips, [[0, 90, 0], [10, 90, 0], [20, 90, 0]])
    eulers = read_track("mixamorigLeftHandIndex1", "Lcl Rotation")
    np.testing.assert_allclose(eulers, [[13, 85, -52], [13, 91, -52], [13, 96, -52]], atol=1e-4)
