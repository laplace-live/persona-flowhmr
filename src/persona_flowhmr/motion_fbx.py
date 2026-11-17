"""Write a binary FBX 7.4 animation on the Mixamo skeleton recognized by Persona."""

from __future__ import annotations

import struct
import warnings
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from tempfile import TemporaryFile
from typing import BinaryIO

import numpy as np
from scipy.spatial.transform import Rotation

from .motion_io import atomic_motion_stream
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
_BLOCK_FRAMES = 512
_MAX_OFFSET = 0xFFFFFFFF


@dataclass(frozen=True)
class _KeyArray:
    """Contiguous, little-endian curve data in the export scratch file."""

    scratch: BinaryIO
    offset: int
    count: int
    tag: bytes

    @property
    def itemsize(self) -> int:
        return 8 if self.tag == b"l" else 4

    def store(self, index: int, values: np.ndarray) -> None:
        self.scratch.seek(self.offset + index * self.itemsize)
        self.scratch.write(values.tobytes())

    def write(self, stream: BinaryIO) -> None:
        remaining = self.count * self.itemsize
        stream.write(self.tag + struct.pack("<III", self.count, 0, remaining))
        self.scratch.seek(self.offset)
        while remaining:
            data = self.scratch.read(min(remaining, 1024 * 1024))
            if not data:
                raise OSError("Incomplete FBX key data")
            stream.write(data)
            remaining -= len(data)


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
        self.props = [p if isinstance(p, _KeyArray) else _property(p) for p in props]
        self.children = children
        self.terminated = bool(children) or name in ("AnimationLayer", "AnimationStack")
        self.prop_size = sum(
            13 + p.count * p.itemsize if isinstance(p, _KeyArray) else len(p) for p in self.props
        )
        self.size = (
            _HEADER.size + len(name) + self.prop_size + sum(child.size for child in children)
            + (_HEADER.size if self.terminated else 0)
        )

    def write(self, stream: BinaryIO) -> None:
        name = self.name.encode("ascii")
        stream.write(_HEADER.pack(stream.tell() + self.size, len(self.props), self.prop_size, len(name)) + name)
        for prop in self.props:
            if isinstance(prop, _KeyArray):
                prop.write(stream)
            else:
                stream.write(prop)
        for child in self.children:
            child.write(stream)
        if self.terminated:
            stream.write(bytes(_HEADER.size))


def _properties(*rows) -> _Node:
    # Properties70 rows consist of a name, type, subtype, flags, and typed values.
    return _Node("Properties70", children=[_Node("P", *row) for row in rows])


def _continuous_eulers(quaternions: np.ndarray, previous: np.ndarray | None) -> np.ndarray:
    # FBX's default rotation order is extrinsic XYZ. Choose the closest equivalent
    # Euler triple on each frame, including motion through +/-90 degrees of pitch.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Gimbal lock detected")
        eulers = Rotation.from_quat(quaternions.reshape(-1, 4)).as_euler("xyz", degrees=True)
    eulers = eulers.reshape(*quaternions.shape[:-1], 3)
    for current in eulers:
        if previous is not None:
            alternate = current * [1, -1, 1] + [180, 180, 180]
            delta = (current - previous + 180) % 360 - 180
            alt_delta = (alternate - previous + 180) % 360 - 180
            use_alt = (alt_delta**2).sum(axis=1) < (delta**2).sum(axis=1)
            current[:] = previous + np.where(use_alt[:, None], alt_delta, delta)
        previous = current
    return eulers


def write_fbx(path: Path, frames: Sequence[CanonicalPoseFrame]) -> None:
    """Write a Y-up centimeter rig with 52 local rotation tracks and a hips position track."""
    if not frames:
        raise ValueError("Cannot export an empty FBX animation")
    count = len(frames)
    axes = (len(_BONES) + 1) * 3  # 52 rotations and one hips translation
    if count * axes * 12 > _MAX_OFFSET:  # Every axis stores both int64 times and float32 values.
        raise ValueError("Animation exceeds the 4 GiB binary FBX 7.4 offset limit")
    with TemporaryFile(dir=path.parent) as scratch:
        times = _KeyArray(scratch, 0, count, b"l")
        curves = [_KeyArray(scratch, (8 + axis * 4) * count, count, b"f") for axis in range(axes)]
        origin = frames[0].media_time_s
        previous_angles = None
        last_tick = np.int64(-1)
        poses = iter(frames)
        for start in range(0, count, _BLOCK_FRAMES):
            batch = list(islice(poses, min(_BLOCK_FRAMES, count - start)))
            seconds = np.array([f.media_time_s - origin for f in batch], dtype=np.float64)
            ticks = np.rint(seconds * _TICKS_PER_SECOND)
            if (
                not np.all(np.isfinite(ticks))
                or np.any(ticks >= 2**63)
                or ticks[0] <= last_tick
                or np.any(np.diff(ticks) <= 0)
            ):
                raise ValueError("FBX timestamps must be finite, strictly increasing and fit in 64-bit ticks")
            ticks = ticks.astype("<i8")
            times.store(start, ticks)
            last_tick = ticks[-1]
            quats = np.stack([f.quats_rh_xyzw for f in batch])
            hips = np.stack([f.hips_pos_rh for f in batch])
            if quats.shape != (len(batch), len(_BONES), 4) or hips.shape != (len(batch), 3):
                raise ValueError("FBX expects 52 bone quaternions and a hips position per frame")
            angles = _continuous_eulers(quats, previous_angles)
            previous_angles = angles[-1].copy()
            values = np.concatenate((angles, 100.0 * hips[:, None, :]), axis=1)
            values = values.reshape(len(batch), axes).astype("<f4")
            if not np.all(np.isfinite(values)):
                raise ValueError("FBX bone transforms must be finite")
            if start == 0:
                defaults = values[0].copy()
            for curve, column in zip(curves, values.T):
                curve.store(start, column)
        _write_scene(path, times, last_tick, curves, defaults)


def _write_scene(
    path: Path, times: _KeyArray, last_tick: np.int64, curves: list[_KeyArray], defaults: np.ndarray
) -> None:
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
        ("LocalStart", "KTime", "Time", "", np.int64(0)),
        ("LocalStop", "KTime", "Time", "", last_tick),
        ("ReferenceStart", "KTime", "Time", "", np.int64(0)),
        ("ReferenceStop", "KTime", "Time", "", last_tick),
    )
    stack = add("AnimationStack", "Motion", "AnimStack", children=[anim_range])
    layer = add("AnimationLayer", "BaseLayer", "AnimLayer")
    connect(layer, stack)

    def add_track(model, channel, property_name, first_axis):
        axes = [f"d|{axis}" for axis in "XYZ"]
        initial = defaults[first_axis:first_axis + 3]
        props = _properties(*[(axis, "Number", "", "A", float(v)) for axis, v in zip(axes, initial)])
        curve_node = add("AnimationCurveNode", channel, "AnimCurveNode", children=[props])
        connect(curve_node, layer)
        connect(curve_node, model, property_name)
        for k, axis in enumerate(axes):
            data = [
                _Node("Default", float(initial[k])),
                _Node("KeyVer", 4008),
                _Node("KeyTime", times),
                _Node("KeyValueFloat", curves[first_axis + k]),
                _Node("KeyAttrFlags", np.array([4], dtype="<i4")),  # linear interpolation
                _Node("KeyAttrDataFloat", np.zeros(4, dtype="<f4")),
                _Node("KeyAttrRefCount", np.array([times.count], dtype="<i4")),
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
        add_track(model, "R", "Lcl Rotation", i * 3)
        if i == 0:
            add_track(model, "T", "Lcl Translation", len(_BONES) * 3)

    global_props = _properties(
        ("UpAxis", "int", "Integer", "", 1),
        ("UpAxisSign", "int", "Integer", "", 1),
        ("FrontAxis", "int", "Integer", "", 2),
        ("FrontAxisSign", "int", "Integer", "", -1),
        ("CoordAxis", "int", "Integer", "", 0),
        ("CoordAxisSign", "int", "Integer", "", 1),
        ("UnitScaleFactor", "double", "Number", "", 1.0),
        ("TimeMode", "enum", "", "", 6),
        ("TimeSpanStart", "KTime", "Time", "", np.int64(0)),
        ("TimeSpanStop", "KTime", "Time", "", last_tick),
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
    if 27 + sum(section.size for section in sections) > _MAX_OFFSET:
        raise ValueError("Animation exceeds the 4 GiB binary FBX 7.4 offset limit")
    with atomic_motion_stream(path) as stream:
        stream.write(b"Kaydara FBX Binary  \x00\x1a\x00" + struct.pack("<I", _VERSION))
        for section in sections:
            section.write(stream)
        stream.write(bytes(_HEADER.size))
        stream.write(bytes.fromhex("fabcab09d0c8d466b176fb831cf7267e") + bytes(4))
        stream.write(bytes(16 - stream.tell() % 16))
        stream.write(struct.pack("<I", _VERSION) + bytes(120) + bytes.fromhex("f85a8c6adef5d97eece90ce3758f290b"))
