"""Export canonical +Z-forward pose frames as VRM Animation 1.0 (binary glTF)."""

from __future__ import annotations

import json
import struct
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from .motion_skeleton import REST_OFFSETS_M
from .retarget import SMPLH_PARENTS, SMPLH_TO_UNITY_BONES, CanonicalPoseFrame


def _vrm_name(unity: str) -> str:
    # VRM 1.0's first two thumb segments have different names from Unity's.
    name = unity.replace("ThumbProximal", "ThumbMetacarpal").replace("ThumbIntermediate", "ThumbProximal")
    return name[:1].lower() + name[1:]


def write_vrma(path: Path, frames: Sequence[CanonicalPoseFrame]) -> None:
    """Write the 52 local rotation tracks and a hips translation track in meters."""
    if not frames:
        raise ValueError("VRMA export requires at least one motion frame")

    names = [_vrm_name(name) for name in SMPLH_TO_UNITY_BONES]
    nodes = [{"name": name, "translation": rest} for name, rest in zip(names, REST_OFFSETS_M)]
    for index, parent in enumerate(SMPLH_PARENTS):
        if parent >= 0:
            nodes[parent].setdefault("children", []).append(index)

    data = bytearray()
    views = []
    accessors = []

    def accessor(array: np.ndarray, kind: str) -> int:
        values = np.asarray(array, dtype="<f4")
        views.append({"buffer": 0, "byteOffset": len(data), "byteLength": values.nbytes})
        data.extend(values.tobytes())
        index = len(accessors)
        entry = {"bufferView": index, "componentType": 5126, "count": len(values), "type": kind}
        if kind == "SCALAR":
            entry.update(min=[float(values[0])], max=[float(values[-1])])
        accessors.append(entry)
        return index

    times = np.array([f.media_time_s - frames[0].media_time_s for f in frames], dtype="<f4")
    if not np.all(np.isfinite(times)) or np.any(np.diff(times) <= 0):
        raise ValueError("VRMA timestamps must be finite and strictly increasing")
    time_index = accessor(times, "SCALAR")
    channels = []
    samplers = []

    def track(node: int, prop: str, output: int) -> None:
        channels.append({"sampler": len(samplers), "target": {"node": node, "path": prop}})
        samplers.append({"input": time_index, "output": output, "interpolation": "LINEAR"})

    positions = np.stack([f.hips_pos_rh for f in frames])
    rotations = np.stack([f.quats_rh_xyzw for f in frames]).astype("<f4", copy=False)
    norms = np.linalg.norm(rotations, axis=2, keepdims=True)
    if not np.all(np.isfinite(positions)) or not np.all(np.isfinite(norms)) or np.any(norms < 1e-8):
        raise ValueError("VRMA requires finite hips positions and nonzero finite quaternions")
    rotations /= norms
    # Keep neighboring keys in the same hemisphere when crossing 180 degrees.
    dots = (rotations[1:] * rotations[:-1]).sum(axis=2)
    rotations[1:] *= np.cumprod(np.where(dots < 0, -1, 1), axis=0)[:, :, None]
    track(0, "translation", accessor(positions, "VEC3"))
    for index in range(len(nodes)):
        track(index, "rotation", accessor(rotations[:, index], "VEC4"))

    extension = "VRMC_vrm_animation"
    human_bones = {name: {"node": i} for i, name in enumerate(names)}
    document = {
        "asset": {"version": "2.0", "generator": "persona-flowhmr"},
        "extensionsUsed": [extension],
        "extensions": {extension: {"specVersion": "1.0", "humanoid": {"humanBones": human_bones}}},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": nodes,
        "animations": [{"name": "FlowHMR", "channels": channels, "samplers": samplers}],
        "accessors": accessors,
        "bufferViews": views,
        "buffers": [{"byteLength": len(data)}],
    }
    metadata = json.dumps(document, separators=(",", ":")).encode("utf-8")
    metadata += b" " * (-len(metadata) % 4)
    with Path(path).open("wb") as stream:
        stream.write(struct.pack("<4sII", b"glTF", 2, 28 + len(metadata) + len(data)))
        stream.write(struct.pack("<I4s", len(metadata), b"JSON"))
        stream.write(metadata)
        stream.write(struct.pack("<I4s", len(data), b"BIN\0"))
        stream.write(data)
