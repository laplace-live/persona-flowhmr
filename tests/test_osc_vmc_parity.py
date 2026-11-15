"""Binary OSC #bundle & VMC protocol parity tests against LAPLACE Persona's TypeScript decoder."""

from __future__ import annotations

import base64
import json
from pathlib import Path
import shutil
import struct
import subprocess
import numpy as np
import pytest

from persona_flowhmr.retarget import CanonicalPoseFrame, SMPLH_TO_UNITY_BONES, pose_to_vmc_bones
from persona_flowhmr.vmc_sender import encode_vmc_bundle


def _decode_osc_bundle_py(buf: bytes) -> list[tuple[str, list[str | float | int]]]:
    assert buf[:8] == b"#bundle\x00"
    _timetag = struct.unpack(">Q", buf[8:16])[0]
    at = 16
    messages: list[tuple[str, list[str | float | int]]] = []
    while at < len(buf):
        size = struct.unpack(">I", buf[at : at + 4])[0]
        at += 4
        elem = buf[at : at + size]
        at += size

        def _read_str(b: bytes, off: int) -> tuple[str, int]:
            end = b.index(0, off)
            nxt = off + ((end - off + 4) & ~3)
            return b[off:end].decode("utf-8"), nxt

        addr, p = _read_str(elem, 0)
        tags, p = _read_str(elem, p)
        assert tags.startswith(",")
        args: list[str | float | int] = []
        for t in tags[1:]:
            if t == "s":
                s_val, p = _read_str(elem, p)
                args.append(s_val)
            elif t == "f":
                args.append(struct.unpack(">f", elem[p : p + 4])[0])
                p += 4
            elif t == "i":
                args.append(struct.unpack(">i", elem[p : p + 4])[0])
                p += 4
            else:
                raise AssertionError(f"Unexpected OSC tag: {t}")
        messages.append((addr, args))
    return messages


def test_osc_bundle_binary_layout() -> None:
    quats = np.zeros((52, 4), dtype=np.float32)
    quats[:, 3] = 1.0
    pose = CanonicalPoseFrame(
        media_time_s=0.5,
        capture_mono_s=10.0,
        frame_index=15,
        quats_rh_xyzw=quats,
        hips_pos_rh=np.array([0.25, 0.92, -0.15], dtype=np.float32),
    )
    bones = pose_to_vmc_bones(pose, mirror=False)
    assert len(bones) == 52
    pkt = encode_vmc_bundle(bones)

    msgs = _decode_osc_bundle_py(pkt)
    assert len(msgs) == 54  # 1 root + 52 bones + 1 OK
    assert msgs[0][0] == "/VMC/Ext/Root/Pos"
    assert msgs[0][1][0] == "root"
    assert msgs[-1] == ("/VMC/Ext/OK", [1, 3, 1])

    seen_names = [m[1][0] for m in msgs[1:53]]
    assert tuple(seen_names) == SMPLH_TO_UNITY_BONES
    # Hips X negated in VMC Unity space (-0.25) so Persona's `unityVec3ToThree` recovers +0.25.
    assert np.allclose(msgs[1][1][1:4], [-0.25, 0.92, -0.15], atol=1e-6)


@pytest.mark.skipif(not Path(r"D:\Git\persona\node_modules").exists(), reason="Persona repo not present")
def test_persona_typescript_vmc_decoder_accepts_all_52_bones(tmp_path: Path) -> None:
    rng = np.random.default_rng(7)
    raw_q = rng.normal(size=(52, 4)).astype(np.float32)
    raw_q /= np.linalg.norm(raw_q, axis=-1, keepdims=True)
    pose = CanonicalPoseFrame(
        media_time_s=1.0,
        capture_mono_s=2.0,
        frame_index=30,
        quats_rh_xyzw=raw_q,
        hips_pos_rh=np.array([0.12, 0.88, 0.34], dtype=np.float32),
    )
    pkt = encode_vmc_bundle(pose_to_vmc_bones(pose, mirror=False))
    pkt_file = tmp_path / "bundle.bin"
    pkt_file.write_bytes(pkt)

    script_file = tmp_path / "check_vmc.ts"
    script_file.write_text(
        "import fs from 'node:fs';\n"
        "import { decodeOscPacket } from 'D:/Git/persona/apps/desktop/src/shared/tracking/vmc/osc.ts';\n"
        "import { VmcPose } from 'D:/Git/persona/apps/desktop/src/shared/tracking/vmc/protocol.ts';\n"
        f"const buf = fs.readFileSync({json.dumps(pkt_file.as_posix())});\n"
        "const msgs = decodeOscPacket(new Uint8Array(buf));\n"
        "const vmc = new VmcPose();\n"
        "const changed = vmc.apply(msgs);\n"
        "const f = vmc.frame();\n"
        "console.log(JSON.stringify({changed, count: Object.keys(f.bones).length, bones: Object.keys(f.bones), hips: f.hipsPosition, root: f.rootPosition}));\n",
        encoding="utf-8",
    )
    pnpm_bin = shutil.which("pnpm") or "pnpm.cmd"
    proc = subprocess.run(
        [pnpm_bin, "--filter", "laplace-persona", "exec", "tsx", str(script_file)],
        cwd=r"D:\Git\persona",
        capture_output=True,
        text=True,
        check=True,
    )
    parsed = json.loads(proc.stdout.strip().splitlines()[-1])
    assert parsed["changed"] is True
    assert parsed["count"] == 52
    assert "leftThumbMetacarpal" in parsed["bones"]
    assert "leftThumbProximal" in parsed["bones"]
    assert "leftThumbDistal" in parsed["bones"]
    assert "rightLittleDistal" in parsed["bones"]
    assert np.allclose(parsed["hips"], [-0.12, 0.88, 0.34], atol=1e-5)
