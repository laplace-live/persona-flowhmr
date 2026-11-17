"""Atomic file publication for motion exporters."""

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import BinaryIO


@contextmanager
def atomic_motion_stream(path: Path) -> Iterator[BinaryIO]:
    """Publish the staged binary file after every frame has been validated and written."""
    destination = Path(path)
    stream = NamedTemporaryFile(dir=destination.parent, delete=False)
    staged = Path(stream.name)
    try:
        with stream:
            yield stream
        staged.replace(destination)
    finally:
        staged.unlink(missing_ok=True)
