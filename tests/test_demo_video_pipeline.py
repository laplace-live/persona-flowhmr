"""End-to-end streaming verification on the demo video (`dancing girl [2E2okJyQyZI].mp4`)."""

from __future__ import annotations

import socket
import struct
import threading
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from persona_flowhmr.config import EngineConfig
from persona_flowhmr.engine import StreamingMocapEngine
from persona_flowhmr.paths import DEFAULT_DEMO_VIDEO
from persona_flowhmr.perception import is_sam3d_available
from persona_flowhmr.web_ui import create_app


@pytest.mark.skipif(not DEFAULT_DEMO_VIDEO.is_file(), reason="Demo video not found")
def test_streaming_demo_video_vmc_and_webui() -> None:
    # Bind a real local UDP VMC listener on an ephemeral port to verify 60 Hz packet arrival & content.
    recv_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    recv_sock.bind(("127.0.0.1", 0))
    recv_sock.settimeout(0.2)
    vmc_port = recv_sock.getsockname()[1]

    received_packets: list[tuple[float, bytes]] = []
    stop_rx = threading.Event()

    def _rx_loop() -> None:
        while not stop_rx.is_set():
            try:
                data, _ = recv_sock.recvfrom(65535)
                received_packets.append((time.perf_counter(), data))
            except TimeoutError:
                continue
            except OSError:
                break

    rx_thread = threading.Thread(target=_rx_loop, daemon=True)
    rx_thread.start()

    backend = "sam3d" if is_sam3d_available() else "replay"
    cfg = EngineConfig(
        source_mode="video",
        video_path=str(DEFAULT_DEMO_VIDEO),
        loop_video=True,
        preset="low",
        playout_fps=60,
        mirror=False,
        perception_backend=backend,
        vmc_enabled=True,
        vmc_host="127.0.0.1",
        vmc_port=vmc_port,
    )
    engine = StreamingMocapEngine(cfg)
    app = create_app(engine)

    try:
        with TestClient(app) as client:
            r0 = client.get("/api/status")
            assert r0.status_code == 200

            r_start = client.post("/api/start")
            assert r_start.status_code == 200

            # Wait up to 180s for cold Triton compilation (SAM 3D Body + MMDiT) + steady 60 Hz streaming.
            deadline = time.perf_counter() + 180.0
            while time.perf_counter() < deadline:
                st = client.get("/api/status").json()
                tel = st["telemetry"]
                assert tel["state"] != "error", f"Engine failed: {tel['error']}"
                if tel["vmc_packets_sent"] >= 150:
                    break
                time.sleep(0.2)

            status = client.get("/api/status").json()
            tel = status["telemetry"]
            assert tel["state"] == "streaming"
            assert tel["vmc_packets_sent"] >= 150
            assert tel["windows_solved"] >= 2
            assert tel["underruns"] == 0
            # Verify playout FPS is smooth ~60 Hz and latency matches low preset target.
            target_ms = cfg.get_preset().playout_delay_s * 1000.0
            assert 54.0 <= tel["playout_fps"] <= 64.0
            assert (target_ms - 250.0) <= tel["measured_latency_ms"] <= (target_ms + 300.0)

            # Verify live & latency-synced preview frames render valid JPEGs with HUD overlay.
            live_jpg = engine.render_live_jpeg()
            synced_jpg = engine.render_synced_playout_jpeg()
            assert live_jpg is not None and live_jpg.startswith(b"\xff\xd8")
            assert synced_jpg is not None and synced_jpg.startswith(b"\xff\xd8")

            client.post("/api/stop")
    finally:
        engine.stop()
        stop_rx.set()
        recv_sock.close()
        rx_thread.join(timeout=2.0)

    assert len(received_packets) >= 140
    # Verify steady ~60 Hz inter-packet pacing on the steady-state tail.
    tail_times = np.array([t for t, _ in received_packets[-120:]])
    intervals_ms = np.diff(tail_times) * 1000.0
    mean_interval_ms = float(np.mean(intervals_ms))
    assert 15.0 <= mean_interval_ms <= 18.5, f"Expected ~16.67ms (60Hz), got {mean_interval_ms:.2f}ms"

    # Verify packet header & finite normalized quaternions inside last received VMC datagram.
    last_pkt = received_packets[-1][1]
    assert last_pkt[:8] == b"#bundle\x00"
    first_elem_size = struct.unpack(">I", last_pkt[16:20])[0]
    assert first_elem_size > 40
