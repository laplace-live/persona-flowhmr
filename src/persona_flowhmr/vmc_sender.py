"""Zero-dependency binary OSC 1.0 #bundle encoder and UDP VMC sender.

Matches LAPLACE Persona's VMC receiver (`apps/desktop/src/shared/tracking/vmc/protocol.ts`
and `osc-encode.ts`): packs `/VMC/Ext/Root/Pos` and all 52 `/VMC/Ext/Bone/Pos` messages
into a single atomic UDP datagram per tick so Persona's 10 ms forward throttle never splits
bones across frames.
"""

from __future__ import annotations

import socket
import struct
from collections.abc import Iterable, Sequence

Pos3 = tuple[float, float, float]
Quat4 = tuple[float, float, float, float]  # (x, y, z, w)
BoneTransform = tuple[str, Pos3, Quat4]

_BUNDLE_HEADER = b"#bundle\x00" + struct.pack(">Q", 1)  # immediate timetag = 1


def encode_osc_string(value: str) -> bytes:
    """Encode an ASCII OSC string with NUL terminator and 4-byte padding."""
    raw = value.encode("utf-8") + b"\x00"
    pad = (4 - (len(raw) & 3)) & 3
    return raw + (b"\x00" * pad)


_ADDR_ROOT_POS = encode_osc_string("/VMC/Ext/Root/Pos")
_ADDR_BONE_POS = encode_osc_string("/VMC/Ext/Bone/Pos")
_ADDR_OK = encode_osc_string("/VMC/Ext/OK")
_TAGS_POS7 = encode_osc_string(",sfffffff")
_TAGS_OK = encode_osc_string(",iii")
_ROOT_NAME = encode_osc_string("root")
_BONE_NAME_CACHE: dict[str, bytes] = {}


def _cached_bone_name(name: str) -> bytes:
    encoded = _BONE_NAME_CACHE.get(name)
    if encoded is None:
        encoded = encode_osc_string(name)
        _BONE_NAME_CACHE[name] = encoded
    return encoded


def encode_vmc_pos_message(address_bytes: bytes, name_bytes: bytes, pos: Pos3, quat: Quat4) -> bytes:
    """Encode `/VMC/Ext/Root/Pos` or `/VMC/Ext/Bone/Pos` with `(name, px, py, pz, qx, qy, qz, qw)`."""
    payload = struct.pack(
        ">7f",
        float(pos[0]),
        float(pos[1]),
        float(pos[2]),
        float(quat[0]),
        float(quat[1]),
        float(quat[2]),
        float(quat[3]),
    )
    return address_bytes + _TAGS_POS7 + name_bytes + payload


def encode_vmc_ok_message(loaded: int = 1, calibrated: int = 3, tracking: int = 1) -> bytes:
    """Encode `/VMC/Ext/OK` status message (`loaded=1, calibrated=3, tracking=1`)."""
    return _ADDR_OK + _TAGS_OK + struct.pack(">3i", int(loaded), int(calibrated), int(tracking))


def encode_vmc_bundle(
    bones: Iterable[BoneTransform],
    root_pos: Pos3 = (0.0, 0.0, 0.0),
    root_quat: Quat4 = (0.0, 0.0, 0.0, 1.0),
    include_ok: bool = True,
) -> bytes:
    """Pack root transform, all humanoid bone transforms, and `/VMC/Ext/OK` into one OSC `#bundle`."""
    elements: list[bytes] = []

    root_msg = encode_vmc_pos_message(_ADDR_ROOT_POS, _ROOT_NAME, root_pos, root_quat)
    elements.append(struct.pack(">I", len(root_msg)) + root_msg)

    for bone_name, pos, quat in bones:
        bone_msg = encode_vmc_pos_message(_ADDR_BONE_POS, _cached_bone_name(bone_name), pos, quat)
        elements.append(struct.pack(">I", len(bone_msg)) + bone_msg)

    if include_ok:
        ok_msg = encode_vmc_ok_message(1, 3, 1)
        elements.append(struct.pack(">I", len(ok_msg)) + ok_msg)

    return _BUNDLE_HEADER + b"".join(elements)


class VmcUdpSender:
    """Thread-safe non-blocking UDP sender for VMC OSC `#bundle` packets."""

    def __init__(self, host: str = "127.0.0.1", port: int = 39539) -> None:
        self.host = host
        self.port = int(port)
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 262144)
        self.packets_sent = 0
        self.bytes_sent = 0

    def set_target(self, host: str, port: int) -> None:
        self.host = host
        self.port = int(port)

    def send_pose(
        self,
        bones: Sequence[BoneTransform],
        root_pos: Pos3 = (0.0, 0.0, 0.0),
        root_quat: Quat4 = (0.0, 0.0, 0.0, 1.0),
    ) -> int:
        packet = encode_vmc_bundle(bones, root_pos=root_pos, root_quat=root_quat, include_ok=True)
        sent = self._sock.sendto(packet, (self.host, self.port))
        self.packets_sent += 1
        self.bytes_sent += sent
        return sent

    def close(self) -> None:
        try:
            self._sock.close()
        except OSError:
            pass
