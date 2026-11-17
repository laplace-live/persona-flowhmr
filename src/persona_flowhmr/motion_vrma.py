"""Export canonical +Z-forward pose frames as VRM Animation 1.0 (binary glTF)."""

from __future__ import annotations

import json
import struct
from collections.abc import Sequence
from itertools import islice
from pathlib import Path
from typing import BinaryIO

import numpy as np

from .motion_io import atomic_motion_stream
from .motion_skeleton import REST_OFFSETS_M
from .retarget import SMPLH_PARENTS, SMPLH_TO_UNITY_BONES, CanonicalPoseFrame

_CHUNK_FRAMES = 1024


def _vrm_name(unity: str) -> str:
    # VRM 1.0's first two thumb segments have different names from Unity's.
    name = unity.replace("ThumbProximal", "ThumbMetacarpal").replace("ThumbIntermediate", "ThumbProximal")
    return name[:1].lower() + name[1:]


def _write_keyframes(stream: BinaryIO, frames: Sequence[CanonicalPoseFrame], offsets: list[int]) -> None:
    # glTF tracks occupy contiguous buffer views. Scatter each source batch to the
    # corresponding part of every track so disk-backed takes need only one pass.
    binary_start = stream.tell()
    frame_count = len(frames)
    origin = frames[0].media_time_s
    source = iter(frames)
    previous_time = -np.inf
    previous_rotation = None
    for first in range(0, frame_count, _CHUNK_FRAMES):
        count = min(_CHUNK_FRAMES, frame_count - first)
        batch = list(islice(source, count))
        if len(batch) != count:
            raise ValueError("VRMA source ended before the expected number of frames")
        times = np.fromiter((f.media_time_s - origin for f in batch), dtype="<f4", count=count)
        if times[0] <= previous_time or not np.isfinite(times).all() or (np.diff(times) <= 0).any():
            raise ValueError("VRMA timestamps must be finite and strictly increasing")
        previous_time = times[-1]
        positions = np.stack([f.hips_pos_rh for f in batch]).astype("<f4", copy=False)
        rotations = np.stack([f.quats_rh_xyzw for f in batch]).astype("<f4", copy=False)
        if positions.shape != (count, 3) or rotations.shape != (count, len(REST_OFFSETS_M), 4):
            raise ValueError("VRMA requires a hips translation and 52 local bone quaternions per frame")
        norms = np.linalg.norm(rotations, axis=2, keepdims=True)
        if not np.isfinite(positions).all() or not np.isfinite(norms).all() or (norms < 1e-8).any():
            raise ValueError("VRMA requires finite hips positions and nonzero finite quaternions")
        rotations /= norms
        if previous_rotation is not None:
            flip = (rotations[0] * previous_rotation).sum(axis=1) < 0
            rotations[0, flip] *= -1
        dots = (rotations[1:] * rotations[:-1]).sum(axis=2)
        rotations[1:] *= np.cumprod(np.where(dots < 0, -1, 1), axis=0)[:, :, None]
        previous_rotation = rotations[-1].copy()
        tracks = (times, positions, *rotations.transpose(1, 0, 2))
        for offset, values in zip(offsets, tracks):
            stream.seek(binary_start + offset + first * values[0].nbytes)
            stream.write(values.tobytes())


def write_vrma(path: Path, frames: Sequence[CanonicalPoseFrame]) -> None:
    """Write the 52 local rotation tracks and a hips translation track in meters."""
    if not frames:
        raise ValueError("VRMA export requires at least one motion frame")

    names = [_vrm_name(name) for name in SMPLH_TO_UNITY_BONES]
    nodes = [{"name": name, "translation": rest} for name, rest in zip(names, REST_OFFSETS_M)]
    for index, parent in enumerate(SMPLH_PARENTS):
        if parent >= 0:
            nodes[parent].setdefault("children", []).append(index)

    data_length = 0
    views = []
    accessors = []

    def accessor(kind: str, components: int) -> int:
        nonlocal data_length
        size = len(frames) * components * 4
        views.append({"buffer": 0, "byteOffset": data_length, "byteLength": size})
        data_length += size
        index = len(accessors)
        accessors.append({"bufferView": index, "componentType": 5126, "count": len(frames), "type": kind})
        return index

    time_index = accessor("SCALAR", 1)
    duration = float(np.float32(frames[-1].media_time_s - frames[0].media_time_s))
    accessors[time_index].update(min=[0.0], max=[duration])
    channels = []
    samplers = []

    def track(node: int, prop: str, output: int) -> None:
        channels.append({"sampler": len(samplers), "target": {"node": node, "path": prop}})
        samplers.append({"input": time_index, "output": output, "interpolation": "LINEAR"})

    track(0, "translation", accessor("VEC3", 3))
    for index in range(len(nodes)):
        track(index, "rotation", accessor("VEC4", 4))

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
        "buffers": [{"byteLength": data_length}],
    }
    metadata = json.dumps(document, separators=(",", ":")).encode("utf-8")
    metadata += b" " * (-len(metadata) % 4)
    file_size = 28 + len(metadata) + data_length
    if file_size > 0xFFFFFFFF:
        raise ValueError("VRMA export exceeds the 4 GiB binary glTF size limit")
    with atomic_motion_stream(path) as stream:
        stream.write(struct.pack("<4sII", b"glTF", 2, file_size))
        stream.write(struct.pack("<I4s", len(metadata), b"JSON"))
        stream.write(metadata)
        stream.write(struct.pack("<I4s", data_length, b"BIN\0"))
        _write_keyframes(stream, frames, [view["byteOffset"] for view in views])
