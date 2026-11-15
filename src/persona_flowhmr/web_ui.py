"""FastAPI Control & Preview UI for `persona-flowhmr`."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import Any
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

from .config import EngineConfig
from .engine import StreamingMocapEngine
from .setup_models import check_models_status, download_missing_models
from .sources import list_available_cameras


class SetupModelsRequest(BaseModel):
    hf_token: str | None = None


_DASHBOARD_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Persona FlowHMR — External Live Mocap</title>
  <style>
    :root {
      --bg: #0b0e14;
      --panel: #131822;
      --border: #232c3d;
      --text: #e6edf7;
      --muted: #8b9bb4;
      --accent: #4cc9f0;
      --good: #4ade80;
      --warn: #fbbf24;
      --bad: #f87171;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0; font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
      background: var(--bg); color: var(--text); font-size: 14px;
    }
    header {
      display: flex; align-items: center; justify-content: space-between;
      padding: 14px 22px; border-bottom: 1px solid var(--border); background: #0f141d;
    }
    header h1 { margin: 0; font-size: 17px; font-weight: 600; letter-spacing: 0.2px; }
    .badge {
      padding: 4px 10px; border-radius: 999px; font-size: 12px; font-weight: 600;
      background: #1e2636; color: var(--muted);
    }
    .badge.streaming { background: rgba(74, 222, 128, 0.16); color: var(--good); }
    .badge.buffering, .badge.warming_up { background: rgba(251, 191, 36, 0.16); color: var(--warn); }
    .badge.error { background: rgba(248, 113, 113, 0.16); color: var(--bad); }
    main { display: grid; grid-template-columns: 360px 1fr; gap: 16px; padding: 16px 22px; }
    @media (max-width: 1024px) { main { grid-template-columns: 1fr; } }
    .card {
      background: var(--panel); border: 1px solid var(--border); border-radius: 10px; padding: 14px;
    }
    .card h2 { margin: 0 0 12px 0; font-size: 13px; text-transform: uppercase; color: var(--muted); letter-spacing: 0.6px; }
    .field { margin-bottom: 10px; }
    .field label { display: block; font-size: 12px; color: var(--muted); margin-bottom: 4px; }
    input[type="text"], input[type="password"], input[type="number"], select {
      width: 100%; padding: 7px 9px; border-radius: 6px; border: 1px solid var(--border);
      background: #0d1118; color: var(--text); font-size: 13px;
    }
    .row { display: flex; gap: 8px; align-items: center; }
    .row > * { flex: 1; }
    .check { display: flex; align-items: center; gap: 8px; margin: 8px 0; font-size: 13px; cursor: pointer; }
    button {
      padding: 8px 14px; border-radius: 6px; border: 1px solid var(--border);
      background: #1c2434; color: var(--text); font-weight: 600; cursor: pointer; font-size: 13px;
    }
    button.primary { background: #1a6fb0; border-color: #2c8bd4; color: #fff; }
    button.danger { background: #7c2424; border-color: #a63434; color: #fff; }
    .previews { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-bottom: 14px; }
    @media (max-width: 900px) { .previews { grid-template-columns: 1fr; } }
    .video-box {
      background: #07090d; border: 1px solid var(--border); border-radius: 8px; overflow: hidden;
      aspect-ratio: 16 / 9; display: flex; align-items: center; justify-content: center; position: relative;
    }
    .video-box img { width: 100%; height: 100%; object-fit: contain; display: block; }
    .metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(145px, 1fr)); gap: 10px; }
    .metric {
      background: #0e131b; border: 1px solid var(--border); border-radius: 8px; padding: 10px 12px;
    }
    .metric .k { font-size: 11px; color: var(--muted); text-transform: uppercase; }
    .metric .v { font-size: 19px; font-weight: 700; margin-top: 3px; font-variant-numeric: tabular-nums; }
    .alert {
      padding: 10px 12px; border-radius: 8px; margin-bottom: 12px; font-size: 12.5px; line-height: 1.45;
      background: rgba(251, 191, 36, 0.12); border: 1px solid rgba(251, 191, 36, 0.35); color: #fde68a;
    }
    .alert.error { background: rgba(248, 113, 113, 0.12); border-color: rgba(248, 113, 113, 0.35); color: #fecaca; }
  </style>
</head>
<body>
  <header>
    <div style="display:flex;align-items:center;gap:12px;">
      <h1>LAPLACE Persona · FlowHMR External Mocap</h1>
      <span id="stateBadge" class="badge">STOPPED</span>
    </div>
    <div class="row" style="flex:none;">
      <button class="primary" onclick="startEngine()">Start Stream</button>
      <button class="danger" onclick="stopEngine()">Stop</button>
    </div>
  </header>
  <main>
    <div style="display:flex;flex-direction:column;gap:14px;">
      <section class="card">
        <h2>Input Source (30 Hz Capture)</h2>
        <div class="field">
          <label>Source Mode</label>
          <select id="source_mode" onchange="toggleSourceFields()">
            <option value="video">Debug Video File</option>
            <option value="camera">Live Camera (Webcam)</option>
          </select>
        </div>
        <div id="videoFields" class="field">
          <label>Debug Video Path</label>
          <input type="text" id="video_path" />
          <label class="check"><input type="checkbox" id="loop_video" checked /> Loop Debug Video</label>
        </div>
        <div id="cameraFields" class="field" style="display:none;">
          <label>Camera Device</label>
          <div class="row">
            <select id="camera_index"><option value="0">Camera 0</option></select>
            <button style="flex:none;" onclick="refreshCameras()">Scan</button>
          </div>
        </div>
        <div class="field">
          <label>Perception Backend</label>
          <select id="perception_backend">
            <option value="auto">Auto (SAM 3D Body Dinov3)</option>
            <option value="sam3d">SAM 3D Body Dinov3 (Strict)</option>
            <option value="replay">Replay / BBox Probe (Debug)</option>
          </select>
        </div>
      </section>

      <section class="card">
        <h2>Latency &amp; Playout</h2>
        <div class="field">
          <label>Sliding-Window Latency Preset</label>
          <select id="preset">
            <option value="low">Low Latency (~1.15s · W=86 · 10 steps)</option>
            <option value="balanced">Balanced (~1.65s · W=102 · 10 steps)</option>
            <option value="high">High Smoothness (~2.45s · W=124 · 20 steps)</option>
            <option value="max">Max Narrowband (~3.50s · W=156 · 20 steps)</option>
          </select>
        </div>
        <div class="row">
          <div class="field">
            <label>Playout Rate</label>
            <select id="playout_fps">
              <option value="60">60 FPS (SLERP)</option>
              <option value="30">30 FPS (Native)</option>
            </select>
          </div>
        </div>
        <label class="check"><input type="checkbox" id="mirror" /> Mirror Avatar (Webcam Selfie Mode)</label>
        <label class="check"><input type="checkbox" id="follow_movement" checked /> Follow Root Movement (XZ Travel)</label>
      </section>

      <section class="card">
        <h2>Persona Output (VMC + Plugin API)</h2>
        <label class="check"><input type="checkbox" id="vmc_enabled" checked /> Send VMC OSC Bundle over UDP</label>
        <div class="row">
          <div class="field"><label>VMC Host</label><input type="text" id="vmc_host" value="127.0.0.1" /></div>
          <div class="field"><label>VMC Port</label><input type="number" id="vmc_port" value="39539" /></div>
        </div>
        <hr style="border:0;border-top:1px solid var(--border);margin:10px 0;" />
        <label class="check"><input type="checkbox" id="plugin_api_enabled" /> Connect Persona Plugin API (WS)</label>
        <div class="field"><label>Plugin API WebSocket URL</label><input type="text" id="plugin_api_url" value="ws://127.0.0.1:25034" /></div>
        <div class="field"><label>Bearer Token (Settings → Developer)</label><input type="password" id="plugin_api_token" placeholder="Paste token if enabled" /></div>
        <label class="check"><input type="checkbox" id="plugin_api_inject_params" /> Also Inject Live2D Angles (`param.inject`)</label>
        <button class="primary" style="width:100%;margin-top:6px;" onclick="applyConfig()">Apply Settings</button>
      </section>
    </div>

    <div>
      <div id="samAlert" class="alert" style="display:none;">
        <strong>Gated Model Needed (`facebook/sam-3d-body-dinov3`):</strong> Public YOLOX-L + FlowHMR MMDiT weights are installed,
        but Meta's gated SAM 3D Body checkpoint is not yet downloaded. Accept the license at
        <code>https://huggingface.co/facebook/sam-3d-body-dinov3</code>, then paste an HF token below (or run <code>persona-flowhmr setup</code>):
        <div class="row" style="margin-top:8px;">
          <input type="password" id="hf_token" placeholder="hf_..." />
          <button style="flex:none;" onclick="setupSam()">Download SAM 3D Body</button>
        </div>
      </div>
      <div id="errAlert" class="alert error" style="display:none;"></div>

      <div class="previews">
        <div>
          <div style="font-size:12px;color:var(--muted);margin-bottom:5px;">Live Camera / Video Capture (t = now)</div>
          <div class="video-box"><img src="/stream/live.mjpg" alt="Live input preview" /></div>
        </div>
        <div>
          <div style="font-size:12px;color:var(--muted);margin-bottom:5px;">Latency-Synced Playout + 3D Skeleton (t = now − Δ)</div>
          <div class="video-box"><img src="/stream/synced.mjpg" alt="Synced playout preview" /></div>
        </div>
      </div>

      <section class="card">
        <h2>Real-Time Pipeline Telemetry</h2>
        <div class="metrics">
          <div class="metric"><div class="k">Playout FPS</div><div class="v" id="m_play_fps">0.0</div></div>
          <div class="metric"><div class="k">Capture FPS</div><div class="v" id="m_cap_fps">0.0</div></div>
          <div class="metric"><div class="k">End-to-End Latency</div><div class="v" id="m_lat">0 ms</div></div>
          <div class="metric"><div class="k">Buffer Headroom</div><div class="v" id="m_lead">0 ms</div></div>
          <div class="metric"><div class="k">YOLOX-L / Frame</div><div class="v" id="m_yolo">0.0 ms</div></div>
          <div class="metric"><div class="k">SAM 3D / Frame</div><div class="v" id="m_sam">0.0 ms</div></div>
          <div class="metric"><div class="k">MMDiT / Window</div><div class="v" id="m_mmdit">0.0 ms</div></div>
          <div class="metric"><div class="k">VMC Packets Sent</div><div class="v" id="m_vmc">0</div></div>
          <div class="metric"><div class="k">Underruns</div><div class="v" id="m_under">0</div></div>
          <div class="metric"><div class="k">Backend</div><div class="v" id="m_back">—</div></div>
        </div>
      </section>
    </div>
  </main>

  <script>
    let initialized = false;
    function toggleSourceFields() {
      const mode = document.getElementById('source_mode').value;
      document.getElementById('videoFields').style.display = mode === 'video' ? 'block' : 'none';
      document.getElementById('cameraFields').style.display = mode === 'camera' ? 'block' : 'none';
    }
    async function refreshCameras() {
      const r = await fetch('/api/cameras');
      const d = await r.json();
      const sel = document.getElementById('camera_index');
      sel.innerHTML = '';
      (d.cameras.length ? d.cameras : [{index: 0, label: 'Camera 0 (default)'}]).forEach(c => {
        const o = document.createElement('option');
        o.value = c.index; o.textContent = c.label;
        sel.appendChild(o);
      });
    }
    function gatherConfig() {
      return {
        source_mode: document.getElementById('source_mode').value,
        video_path: document.getElementById('video_path').value,
        loop_video: document.getElementById('loop_video').checked,
        camera_index: parseInt(document.getElementById('camera_index').value || '0', 10),
        perception_backend: document.getElementById('perception_backend').value,
        preset: document.getElementById('preset').value,
        playout_fps: parseInt(document.getElementById('playout_fps').value, 10),
        mirror: document.getElementById('mirror').checked,
        follow_movement: document.getElementById('follow_movement').checked,
        vmc_enabled: document.getElementById('vmc_enabled').checked,
        vmc_host: document.getElementById('vmc_host').value,
        vmc_port: parseInt(document.getElementById('vmc_port').value || '39539', 10),
        plugin_api_enabled: document.getElementById('plugin_api_enabled').checked,
        plugin_api_url: document.getElementById('plugin_api_url').value,
        plugin_api_token: document.getElementById('plugin_api_token').value,
        plugin_api_inject_params: document.getElementById('plugin_api_inject_params').checked,
      };
    }
    async function applyConfig() {
      await fetch('/api/config', {
        method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(gatherConfig()),
      });
      await pollStatus();
    }
    async function startEngine() {
      await applyConfig();
      await fetch('/api/start', {method: 'POST'});
      await pollStatus();
    }
    async function stopEngine() {
      await fetch('/api/stop', {method: 'POST'});
      await pollStatus();
    }
    async function setupSam() {
      const tok = document.getElementById('hf_token').value.trim();
      const r = await fetch('/api/setup-models', {
        method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({hf_token: tok || null}),
      });
      const d = await r.json();
      if (d.sam3d_download_error) alert(d.sam3d_download_error);
      await pollStatus();
    }
    async function pollStatus() {
      try {
        const r = await fetch('/api/status');
        const s = await r.json();
        if (!initialized) {
          const c = s.config;
          document.getElementById('source_mode').value = c.source_mode;
          document.getElementById('video_path').value = c.video_path;
          document.getElementById('loop_video').checked = c.loop_video;
          document.getElementById('perception_backend').value = c.perception_backend;
          document.getElementById('preset').value = c.preset;
          document.getElementById('playout_fps').value = String(c.playout_fps);
          document.getElementById('mirror').checked = c.mirror;
          document.getElementById('follow_movement').checked = c.follow_movement;
          document.getElementById('vmc_enabled').checked = c.vmc_enabled;
          document.getElementById('vmc_host').value = c.vmc_host;
          document.getElementById('vmc_port').value = c.vmc_port;
          document.getElementById('plugin_api_enabled').checked = c.plugin_api_enabled;
          document.getElementById('plugin_api_url').value = c.plugin_api_url;
          document.getElementById('plugin_api_inject_params').checked = c.plugin_api_inject_params;
          toggleSourceFields();
          initialized = true;
        }
        const t = s.telemetry;
        const badge = document.getElementById('stateBadge');
        badge.textContent = t.state.toUpperCase();
        badge.className = 'badge ' + t.state;

        document.getElementById('samAlert').style.display = s.models.sam3d_ready ? 'none' : 'block';
        const errEl = document.getElementById('errAlert');
        errEl.style.display = t.error ? 'block' : 'none';
        errEl.textContent = t.error || '';

        document.getElementById('m_play_fps').textContent = t.playout_fps.toFixed(1);
        document.getElementById('m_cap_fps').textContent = t.capture_fps.toFixed(1);
        document.getElementById('m_lat').textContent = Math.round(t.measured_latency_ms) + ' ms';
        document.getElementById('m_lead').textContent = Math.round(t.buffer_lead_ms) + ' ms';
        document.getElementById('m_yolo').textContent = t.yolox_ms_per_frame.toFixed(1) + ' ms';
        document.getElementById('m_sam').textContent = t.sam_ms_per_frame.toFixed(1) + ' ms';
        document.getElementById('m_mmdit').textContent = t.mmdit_ms_per_window.toFixed(1) + ' ms';
        document.getElementById('m_vmc').textContent = t.vmc_packets_sent.toLocaleString();
        document.getElementById('m_under').textContent = t.underruns;
        document.getElementById('m_back').textContent = t.active_backend;
      } catch (_e) {}
    }
    setInterval(pollStatus, 400);
    pollStatus();
  </script>
</body>
</html>
"""


def _placeholder_jpeg(text: str) -> bytes:
    import cv2
    import numpy as np

    canvas = np.zeros((360, 640, 3), dtype=np.uint8)
    canvas[:] = (18, 22, 30)
    cv2.putText(canvas, text, (180, 185), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (140, 160, 190), 2, cv2.LINE_AA)
    ok, buf = cv2.imencode(".jpg", canvas)
    return buf.tobytes() if ok else b""


def create_app(engine: StreamingMocapEngine | None = None) -> FastAPI:
    eng = engine or StreamingMocapEngine(EngineConfig())

    @asynccontextmanager
    async def _lifespan(_app: FastAPI):
        try:
            yield
        finally:
            eng.stop()

    app = FastAPI(title="Persona FlowHMR External Mocap", version="0.1.0", lifespan=_lifespan)
    app.state.engine = eng

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return _DASHBOARD_HTML

    @app.get("/api/status")
    def api_status() -> dict[str, Any]:
        return eng.get_status()

    @app.get("/api/models")
    def api_models() -> dict[str, Any]:
        return check_models_status()

    @app.get("/api/cameras")
    def api_cameras() -> dict[str, Any]:
        return {"cameras": list_available_cameras()}

    @app.post("/api/config")
    def api_config(patch: dict[str, Any]) -> dict[str, Any]:
        return eng.update_live_settings(patch)

    @app.post("/api/start")
    def api_start() -> dict[str, Any]:
        eng.start(realtime_source=True)
        return eng.get_status()

    @app.post("/api/stop")
    def api_stop() -> dict[str, Any]:
        eng.stop()
        return eng.get_status()

    @app.post("/api/setup-models")
    def api_setup_models(req: SetupModelsRequest) -> dict[str, Any]:
        try:
            return download_missing_models(hf_token=req.hf_token)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    async def _mjpeg_generator(kind: str):
        idle_jpg = _placeholder_jpeg("Stream Idle - Click Start Stream")
        while True:
            jpg = eng.render_live_jpeg() if kind == "live" else eng.render_synced_playout_jpeg()
            payload = jpg or idle_jpg
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + payload + b"\r\n"
            await asyncio.sleep(1.0 / 24.0)

    @app.get("/stream/live.mjpg")
    async def stream_live() -> StreamingResponse:
        return StreamingResponse(
            _mjpeg_generator("live"),
            media_type="multipart/x-mixed-replace; boundary=frame",
        )

    @app.get("/stream/synced.mjpg")
    async def stream_synced() -> StreamingResponse:
        return StreamingResponse(
            _mjpeg_generator("synced"),
            media_type="multipart/x-mixed-replace; boundary=frame",
        )

    return app
