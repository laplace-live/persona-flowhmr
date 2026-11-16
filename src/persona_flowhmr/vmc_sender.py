"""Zero-dependency binary OSC 1.0 #bundle encoder and UDP VMC sender.

Matches LAPLACE Persona's VMC receiver (`apps/desktop/src/shared/tracking/vmc/protocol.ts`
and `osc-encode.ts`): packs `/VMC/Ext/Root/Pos` and all 52 `/VMC/Ext/Bone/Pos` messages
into a single atomic UDP datagram per tick so Persona's 10 ms forward throttle never splits
bones across frames.
"""

from __future__ import annotations

import functools
import logging
import socket
import struct
import time
from collections.abc import Iterable

Pos3 = tuple[float, float, float]
Quat4 = tuple[float, float, float, float]  # (x, y, z, w)
BoneTransform = tuple[str, Pos3, Quat4]

_SIZE = struct.Struct(">I")
_POS7 = struct.Struct(">7f")  # (px, py, pz, qx, qy, qz, qw)
_BUNDLE_HEADER = b"#bundle\x00" + struct.pack(">Q", 1)  # immediate timetag = 1
_RESOLVE_TTL_S = 5.0  # how long a resolved target address is reused before looking the host up again
_log = logging.getLogger(__name__)


def encode_osc_string(value: str) -> bytes:
    """Encode an ASCII OSC string with NUL terminator and 4-byte padding."""
    raw = value.encode("utf-8") + b"\x00"
    pad = (4 - (len(raw) & 3)) & 3
    return raw + (b"\x00" * pad)


_TAGS_POS7 = encode_osc_string(",sfffffff")


def _bundle_element(msg: bytes) -> bytes:
    """Prefix an OSC message with its big-endian byte size, as stored inside a `#bundle`."""
    return _SIZE.pack(len(msg)) + msg


@functools.cache
def _bone_element_head(name: str) -> bytes:
    """Size prefix + `/VMC/Ext/Bone/Pos ,sfffffff <name>`: everything before a bone's 7 floats."""
    head = encode_osc_string("/VMC/Ext/Bone/Pos") + _TAGS_POS7 + encode_osc_string(name)
    return _SIZE.pack(len(head) + _POS7.size) + head


# Identity root transform and `/VMC/Ext/OK` status (`loaded=1, calibrated=3, tracking=1`) never change.
_ROOT_ELEMENT = _bundle_element(
    encode_osc_string("/VMC/Ext/Root/Pos")
    + _TAGS_POS7
    + encode_osc_string("root")
    + _POS7.pack(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
)
_OK_ELEMENT = _bundle_element(encode_osc_string("/VMC/Ext/OK") + encode_osc_string(",iii") + struct.pack(">3i", 1, 3, 1))


def encode_vmc_bundle(bones: Iterable[BoneTransform]) -> bytes:
    """Pack the identity root transform, all humanoid bone transforms, and `/VMC/Ext/OK` into one OSC `#bundle`."""
    parts = [_BUNDLE_HEADER, _ROOT_ELEMENT]
    for name, pos, quat in bones:
        parts.append(_bone_element_head(name))
        parts.append(_POS7.pack(*pos, *quat))
    parts.append(_OK_ELEMENT)
    return b"".join(parts)


class VmcUdpSender:
    """Non-blocking UDP sender for VMC OSC `#bundle` packets."""

    def __init__(self, host: str = "127.0.0.1", port: int = 39539) -> None:
        self.host = host
        self.port = int(port)
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 262144)
        # ((host, port), (ip, port), expires_at): the address `sendto` uses for the current target.
        self._resolved: tuple[tuple[str, int], tuple[str, int], float] | None = None
        self.packets_sent = 0

    def set_target(self, host: str, port: int) -> None:
        self.host = host
        self.port = int(port)

    def send_pose(self, bones: Iterable[BoneTransform]) -> int:
        target = (self.host, self.port)
        now = time.monotonic()
        resolved = self._resolved
        if resolved is None or resolved[0] != target or now >= resolved[2]:
            # Look a host name up every few seconds rather than inside every 60 Hz `sendto`, so DNS changes still land.
            try:
                addr = (socket.gethostbyname(target[0]), target[1])
            except OSError as exc:
                if resolved is None or resolved[0] != target:
                    raise
                # A failed refresh keeps the last working address and retries after the TTL.
                _log.warning("VMC host %r lookup failed (%s); still sending to %s", target[0], exc, resolved[1][0])
                addr = resolved[1]
            resolved = (target, addr, now + _RESOLVE_TTL_S)
            self._resolved = resolved
        sent = self._sock.sendto(encode_vmc_bundle(bones), resolved[1])
        self.packets_sent += 1
        return sent

    def close(self) -> None:
        try:
            self._sock.close()
        except OSError:
            pass
