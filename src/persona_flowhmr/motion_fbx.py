"""Write a binary FBX 7.4 animation on the Mixamo skeleton recognized by Persona."""

from __future__ import annotations

import struct
import warnings
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import BinaryIO

import numpy as np
from scipy.spatial.transform import Rotation

from .motion_skeleton import REST_OFFSETS_M
from .retarget import SMPLH_PARENTS, CanonicalPoseFrame

_BODY = (
    "Hips",
    "LeftUpLeg",
    "RightUpLeg",
    "Spine",
    "LeftLeg",
    "RightLeg",
    "Spine1",
    "LeftFoot",
    "RightFoot",
    "Spine2",
    "LeftToeBase",
    "RightToeBase",
    "Neck",
    "LeftShoulder",
    "RightShoulder",
    "Head",
    "LeftArm",
    "RightArm",
    "LeftForeArm",
    "RightForeArm",
    "LeftHand",
    "RightHand",
)
_FINGERS = tuple(
    f"{side}Hand{finger}{joint}"
    for side in ("Left", "Right")
    for finger in ("Index", "Middle", "Pinky", "Ring", "Thumb")
    for joint in (1, 2, 3)
)
_BONES = tuple("mixamorig" + name for name in _BODY + _FINGERS)
_TICKS_PER_SECOND = 46_186_158_000
_HEADER = struct.Struct("<IIIB")
_VERSION = 7400


def _property(value) -> bytes:
    if isinstance(value, str):
        data = value.encode("utf-8")
        return b"S" + struct.pack("<I", len(data)) + data
    if isinstance(value, np.ndarray):
        code = {"<f4": b"f", "<i4": b"i", "<i8": b"l"}[value.dtype.str]
        return code + struct.pack("<III", value.size, 0, value.nbytes) + value.tobytes()
    if isinstance(value, np.int64):
        return b"L" + struct.pack("<q", value)
    if isinstance(value, int):
        return b"I" + struct.pack("<i", value)
    return b"D" + struct.pack("<d", value)


class _Node:
    """One FBX record with its typed properties and nested records."""

    def __init__(self, name: str, *props, children: Sequence[_Node] = ()):
        self.name = name
        self.props = props
        self.children = children

    def write(self, stream: BinaryIO) -> None:
        start = stream.tell()
        name = self.name.encode("ascii")
        data = b"".join(_property(value) for value in self.props)
        stream.write(bytes(_HEADER.size) + name + data)
        for child in self.children:
            child.write(stream)
        if self.children or self.name in ("AnimationLayer", "AnimationStack"):
            stream.write(bytes(_HEADER.size))
        end = stream.tell()
        stream.seek(start)
        stream.write(_HEADER.pack(end, len(self.props), len(data), len(name)))
        stream.seek(end)


def _properties(*rows) -> _Node:
    # Properties70 rows consist of a name, type, subtype, flags, and typed values.
    return _Node("Properties70", children=[_Node("P", *row) for row in rows])


def _continuous_eulers(quaternions: np.ndarray) -> np.ndarray:
    # FBX's default rotation order is extrinsic XYZ. Choose the closest equivalent
    # Euler triple on each frame, including motion through +/-90 degrees of pitch.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Gimbal lock detected")
        eulers = Rotation.from_quat(quaternions.reshape(-1, 4)).as_euler("xyz", degrees=True)
    eulers = eulers.reshape(*quaternions.shape[:-1], 3)
    for i in range(1, len(eulers)):
        prev, current = eulers[i - 1], eulers[i]
        alternate = current * [1, -1, 1] + [180, 180, 180]
        delta = (current - prev + 180) % 360 - 180
        alt_delta = (alternate - prev + 180) % 360 - 180
        use_alt = (alt_delta**2).sum(axis=1) < (delta**2).sum(axis=1)
        current[:] = prev + np.where(use_alt[:, None], alt_delta, delta)
    return eulers


def write_fbx(path: Path, frames: Sequence[CanonicalPoseFrame]) -> None:
    """Write a Y-up centimeter rig with 52 local rotation tracks and a hips position track."""
    if not frames:
        raise ValueError("Cannot export an empty FBX animation")
    seconds = np.array([f.media_time_s - frames[0].media_time_s for f in frames])
    if not np.all(np.isfinite(seconds)) or np.any(np.diff(seconds) <= 0):
        raise ValueError("FBX timestamps must be finite and strictly increasing")
    times = np.rint(seconds * _TICKS_PER_SECOND).astype("<i8")
    angles = _continuous_eulers(np.stack([f.quats_rh_xyzw for f in frames]))
    translations = 100.0 * np.stack([f.hips_pos_rh for f in frames])
    if not np.all(np.isfinite(translations)) or not np.all(np.isfinite(angles)):
        raise ValueError("FBX bone transforms must be finite")
    objects: list[_Node] = []
    connections: list[_Node] = []

    def add(kind, label, class_name, subtype="", children=()):
        obj_id = np.int64(len(objects) + 1)
        objects.append(_Node(kind, obj_id, f"{label}\x00\x01{class_name}", subtype, children=children))
        return obj_id

    def connect(child, parent, prop=None):
        edge = ("OO", child, parent) if prop is None else ("OP", child, parent, prop)
        connections.append(_Node("C", *edge))

    anim_range = _properties(
        ("LocalStart", "KTime", "Time", "", times[0]),
        ("LocalStop", "KTime", "Time", "", times[-1]),
        ("ReferenceStart", "KTime", "Time", "", times[0]),
        ("ReferenceStop", "KTime", "Time", "", times[-1]),
    )
    stack = add("AnimationStack", "Motion", "AnimStack", children=[anim_range])
    layer = add("AnimationLayer", "BaseLayer", "AnimLayer")
    connect(layer, stack)

    def add_track(model, channel, property_name, keys):
        axes = [f"d|{axis}" for axis in "XYZ"]
        defaults = _properties(*[(axis, "Number", "", "A", float(v)) for axis, v in zip(axes, keys[0])])
        curve_node = add("AnimationCurveNode", channel, "AnimCurveNode", children=[defaults])
        connect(curve_node, layer)
        connect(curve_node, model, property_name)
        for axis, values in zip(axes, keys.T):
            data = [
                _Node("Default", float(values[0])),
                _Node("KeyVer", 4008),
                _Node("KeyTime", times),
                _Node("KeyValueFloat", values.astype("<f4")),
                _Node("KeyAttrFlags", np.array([4], dtype="<i4")),  # linear interpolation
                _Node("KeyAttrDataFloat", np.zeros(4, dtype="<f4")),
                _Node("KeyAttrRefCount", np.array([len(times)], dtype="<i4")),
            ]
            curve = add("AnimationCurve", "", "AnimCurve", children=data)
            connect(curve, curve_node, axis)

    models = []
    for i, name in enumerate(_BONES):
        offset = np.asarray(REST_OFFSETS_M[i]) * 100.0
        rest = _properties(
            ("Lcl Translation", "Lcl Translation", "", "A", *offset),
            ("Lcl Rotation", "Lcl Rotation", "", "A", 0.0, 0.0, 0.0),
            ("Lcl Scaling", "Lcl Scaling", "", "A", 1.0, 1.0, 1.0),
        )
        model = add("Model", name, "Model", "LimbNode", [_Node("Version", 232), rest])
        limb = add("NodeAttribute", name, "NodeAttribute", "LimbNode", [_Node("TypeFlags", "Skeleton")])
        connect(limb, model)
        connect(model, models[SMPLH_PARENTS[i]] if i else np.int64(0))
        models.append(model)
        add_track(model, "R", "Lcl Rotation", angles[:, i])
        if i == 0:
            add_track(model, "T", "Lcl Translation", translations)

    global_props = _properties(
        ("UpAxis", "int", "Integer", "", 1),
        ("UpAxisSign", "int", "Integer", "", 1),
        ("FrontAxis", "int", "Integer", "", 2),
        ("FrontAxisSign", "int", "Integer", "", -1),
        ("CoordAxis", "int", "Integer", "", 0),
        ("CoordAxisSign", "int", "Integer", "", 1),
        ("UnitScaleFactor", "double", "Number", "", 1.0),
        ("TimeMode", "enum", "", "", 6),
        ("TimeSpanStart", "KTime", "Time", "", times[0]),
        ("TimeSpanStop", "KTime", "Time", "", times[-1]),
    )
    counts = Counter(obj.name for obj in objects)
    definitions = [_Node("Version", 100), _Node("Count", len(objects))]
    definitions.extend(_Node("ObjectType", k, children=[_Node("Count", v)]) for k, v in counts.items())
    sections = [
        _Node(
            "FBXHeaderExtension",
            children=[
                _Node("FBXHeaderVersion", 1003),
                _Node("FBXVersion", _VERSION),
                _Node("Creator", "persona-flowhmr"),
            ],
        ),
        _Node("GlobalSettings", children=[_Node("Version", 1000), global_props]),
        _Node("Definitions", children=definitions),
        _Node("Objects", children=objects),
        _Node("Connections", children=connections),
    ]
    with path.open("wb") as stream:
        stream.write(b"Kaydara FBX Binary  \x00\x1a\x00" + struct.pack("<I", _VERSION))
        for section in sections:
            section.write(stream)
        stream.write(bytes(_HEADER.size))
        stream.write(bytes.fromhex("fabcab09d0c8d466b176fb831cf7267e") + bytes(4))
        stream.write(bytes(16 - stream.tell() % 16))
        stream.write(struct.pack("<I", _VERSION) + bytes(120) + bytes.fromhex("f85a8c6adef5d97eece90ce3758f290b"))
