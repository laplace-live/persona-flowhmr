"""Tests for streaming stitcher continuity, 60 Hz SLERP playout, and Persona Plugin API client."""

from __future__ import annotations

import asyncio
import base64
import json
import math
import socket
import threading

import numpy as np
import websockets

from persona_flowhmr.paths import MOTION_STATS_PATH
from persona_flowhmr.persona_api import PersonaPluginClient
from persona_flowhmr.playout import PosePlayoutBuffer
from persona_flowhmr.retarget import CanonicalPoseFrame
from persona_flowhmr.stitcher import MotionLatentDecoder, StreamingStitcher


def test_stitcher_continuous_root_integration_and_overlap_blend() -> None:
    decoder = MotionLatentDecoder(MOTION_STATS_PATH, motion_style="basic", fps=30.0)
    stitcher = StreamingStitcher(decoder, fps=30.0)

    # Window 1 & Window 2 with constant normalized latent = 0 (except positive X velocity in win 1 & 2).
    w1 = np.zeros((40, 491), dtype=np.float32)
    w2 = np.zeros((40, 491), dtype=np.float32)
    w1[:, 0] = 1.5  # non-zero forward/sideways velocity
    w2[:, 0] = 1.5

    c1 = stitcher.commit_window(
        w1,
        emit_offset_in_window=10,
        stride_frames=16,
        overlap_frames=6,
        first_emit_frame_index=0,
        capture_mono_times=[i / 30.0 for i in range(16)],
    )
    c2 = stitcher.commit_window(
        w2,
        emit_offset_in_window=10,
        stride_frames=16,
        overlap_frames=6,
        first_emit_frame_index=16,
        capture_mono_times=[(16 + i) / 30.0 for i in range(16)],
    )
    all_frames = c1 + c2
    xs = np.array([f.hips_pos_rh[0] for f in all_frames])
    diffs = np.diff(xs)
    # Every single step (including across seam 15 -> 16!) must have identical constant step velocity!
    assert np.allclose(diffs, diffs[0], atol=1e-5)
    assert abs(diffs[15] - diffs[0]) < 1e-6


def test_60hz_slerp_playout_midpoint() -> None:
    buf = PosePlayoutBuffer()
    q0 = np.zeros((52, 4), dtype=np.float32)
    q0[:, 3] = 1.0
    q1 = q0.copy()
    ang = math.radians(60.0)
    q1[0] = [0.0, math.sin(ang / 2), 0.0, math.cos(ang / 2)]

    p0 = CanonicalPoseFrame(0.0, 1.0, 0, q0, np.array([0.0, 1.0, 0.0], dtype=np.float32))
    p1 = CanonicalPoseFrame(1.0 / 30.0, 1.0 + 1.0 / 30.0, 1, q1, np.array([0.2, 1.0, 0.0], dtype=np.float32))
    buf.push_many([p0, p1])

    mid = buf.sample_at_media_time(0.5 / 30.0)
    assert mid is not None
    assert mid.interpolated is True
    assert mid.underrun is False
    expected_q = [0.0, math.sin(math.radians(30.0) / 2), 0.0, math.cos(math.radians(30.0) / 2)]
    assert np.allclose(mid.pose.quats_rh_xyzw[0], expected_q, atol=1e-4)
    assert np.allclose(mid.pose.hips_pos_rh, [0.1, 1.0, 0.0], atol=1e-5)


def test_persona_plugin_api_client_wire_contract(tmp_path) -> None:
    # Find an unused ephemeral TCP port
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]

    received_methods: list[tuple[str, dict]] = []
    auth_seen: list[str] = []
    fake_png = b"\x89PNG\r\n\x1a\nFAKE"

    async def handler(ws):
        req_headers = ws.request.headers if hasattr(ws, "request") and ws.request else {}
        auth_seen.append(req_headers.get("Authorization", ""))
        await ws.send(
            json.dumps(
                {
                    "kind": "hello",
                    "protocol": 1,
                    "app": {"name": "LAPLACE Persona", "version": "0.71.2", "platform": "win32", "capabilities": ["pose-movement-options"]},
                }
            )
        )
        async for raw in ws:
            msg = json.loads(raw)
            assert msg["kind"] == "request"
            m = msg["method"]
            params = msg.get("params") or {}
            received_methods.append((m, params))
            if m == "session.identify":
                res = {}
            elif m == "settings.get":
                res = {"settings": {"tracking": {"sources": []}}}
            elif m == "tracking.addSource":
                res = {"source": {"id": "vmc-flowhmr-1", "kind": "vmc", "enabled": True, "port": params["port"]}}
            elif m == "pose.setEnabled":
                res = {}
            elif m == "instance.list":
                res = {"instances": [{"instanceId": "inst-vrm-1", "kind": "model", "format": "vrm", "loaded": True, "capabilities": []}]}
            elif m == "instance.setTrackingSources":
                res = {}
            elif m == "param.inject":
                res = {}
            elif m == "stage.capture":
                res = {
                    "width": 640,
                    "height": 360,
                    "dataUrl": "data:image/png;base64," + base64.b64encode(fake_png).decode("ascii"),
                }
            else:
                res = {}
            await ws.send(json.dumps({"kind": "response", "id": msg["id"], "result": res}))

    stop_srv = threading.Event()

    def _run_srv() -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        async def _main() -> None:
            async with websockets.serve(handler, "127.0.0.1", port):
                while not stop_srv.is_set():
                    await asyncio.sleep(0.05)

        loop.run_until_complete(_main())

    t = threading.Thread(target=_run_srv, daemon=True)
    t.start()

    client = PersonaPluginClient(url=f"ws://127.0.0.1:{port}", token="secret-test-token")
    try:
        assert client.start(timeout_s=5.0) is True
        assert auth_seen == ["Bearer secret-test-token"]
        info = client.ensure_vmc_receiver(vmc_port=39539, bind_primary_avatar=True)
        assert info == {"sourceId": "vmc-flowhmr-1", "boundInstanceId": "inst-vrm-1"}
        out_png = client.capture_stage_png(tmp_path / "stage.png")
        assert out_png.read_bytes() == fake_png
        names = [m for m, _ in received_methods]
        assert names[:6] == [
            "session.identify",
            "settings.get",
            "tracking.addSource",
            "pose.setEnabled",
            "instance.list",
            "instance.setTrackingSources",
        ]
    finally:
        client.stop()
        stop_srv.set()
        t.join(timeout=2.0)
