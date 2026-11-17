"""Fixed-size, file-backed source frames shared by downloads of a finished take."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from tempfile import TemporaryFile
from typing import BinaryIO

import numpy as np

from .camera_tracker import CameraPoseFrame
from .retarget import CanonicalPoseFrame

_CAMERA = np.dtype([
    ("position_rh", "f4", 3), ("quat_rh_xyzw", "f4", 4),
    ("orbit_azimuth", "f8"), ("orbit_elevation", "f8"), ("orbit_distance", "f8"),
    ("orbit_target_rh", "f4", 3), ("roll", "f8"), ("fov_deg", "f8"), ("is_cut", "?"),
])
_FRAME = np.dtype([
    ("media", "f8"), ("capture", "f8"), ("index", "i8"), ("quats", "f4", (52, 4)),
    ("hips", "f4", 3), ("tracked", "?"), ("camera", _CAMERA),
])
_EMPTY_CAMERA = np.zeros((), dtype=_CAMERA)


def _pack(pose: CanonicalPoseFrame) -> tuple:
    camera = pose.camera
    cam = _EMPTY_CAMERA if camera is None else tuple(getattr(camera, key) for key in _CAMERA.names)
    return (pose.media_time_s, pose.capture_mono_s, pose.frame_index, pose.quats_rh_xyzw,
            pose.hips_pos_rh, camera is not None, cam)


def _unpack(row: np.void) -> CanonicalPoseFrame:
    camera = None
    if row["tracked"]:
        cam = row["camera"]
        camera = CameraPoseFrame(**{key: cam[key] for key in _CAMERA.names})
    return CanonicalPoseFrame(
        float(row["media"]), float(row["capture"]), int(row["index"]), row["quats"], row["hips"], camera=camera,
    )


class RecordedFrames(Sequence[CanonicalPoseFrame]):
    """Append one committed batch at a time. Readers use a read-only mapping of the completed take."""

    def __init__(self) -> None:
        self._stream: BinaryIO | None = None
        self._count = 0
        self._rows: np.ndarray | None = None

    def extend(self, poses: Iterable[CanonicalPoseFrame]) -> None:
        rows = np.fromiter((_pack(pose) for pose in poses), dtype=_FRAME)
        if not len(rows):
            return
        if self._stream is None:
            self._stream = TemporaryFile(buffering=0)
        if self._rows is not None:
            self._rows = None
            self._stream.seek(self._count * _FRAME.itemsize)
        # A batch is available for download only once every byte has been written.
        remaining = memoryview(rows).cast("B")
        try:
            while remaining:
                written = self._stream.write(remaining)
                if not written:
                    raise OSError("Could not write recorded motion frames")
                remaining = remaining[written:]
        except OSError:
            self._stream.seek(self._count * _FRAME.itemsize)
            self._stream.truncate()
            raise
        self._count += len(rows)

    def __len__(self) -> int:
        return self._count

    def __getitem__(self, index: int | slice) -> CanonicalPoseFrame | list[CanonicalPoseFrame]:
        if self._rows is None:
            if not self._count:
                if isinstance(index, slice):
                    return []
                raise IndexError(index)
            self._rows = np.memmap(self._stream, dtype=_FRAME, mode="r", shape=(self._count,))
        rows = self._rows[index]
        if isinstance(index, slice):
            return [_unpack(row) for row in rows]
        return _unpack(rows)
