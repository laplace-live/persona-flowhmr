"""FastAPI Control & Preview UI for `persona-flowhmr`."""

from __future__ import annotations

import asyncio
import functools
import json
import os
import subprocess
import sys
import tempfile
import threading
from collections.abc import Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from pydantic import BaseModel
from starlette.types import Receive, Scope, Send

from . import __version__
from .engine import StreamingMocapEngine
from .motion_export import AVATAR_FORMATS, write_motion
from .setup_models import check_models_status, download_missing_models
from .sources import list_available_cameras

_DOWNLOAD_TRACKS = {
    "camera.vmd": ("camera.vmd",),
    **{f"avatar.{ext}": (f"avatar.{ext}",) for ext in AVATAR_FORMATS},
    **{f"avatar-{ext}-camera.zip": ("camera.vmd", f"avatar.{ext}") for ext in AVATAR_FORMATS},
}


class SetupModelsRequest(BaseModel):
    hf_token: str | None = None


class PickVideoRequest(BaseModel):
    current_path: str | None = None


class _MotionDownload(FileResponse):
    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            Path(self.path).unlink(missing_ok=True)


_PICKER_LOCK = threading.Lock()

_TK_PICK_VIDEO_SCRIPT = """
import ctypes, json, sys, tkinter as tk
from tkinter import filedialog
if sys.platform == "win32":
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        pass
root = tk.Tk()
root.withdraw()
root.attributes("-topmost", True)
path = filedialog.askopenfilename(
    parent=root,
    initialdir=sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] else None,
    title="Select Debug Video File",
    filetypes=[
        ("Video Files", "*.mp4 *.mkv *.mov *.avi *.webm *.m4v *.ts"),
        ("All Files", "*.*"),
    ],
)
root.destroy()
print(json.dumps({"path": path or ""}))
"""


def _open_system_video_dialog(current_path: str | None = None) -> str:
    """Open the native OS file picker in a clean helper process and return the chosen path (or '')."""
    if not _PICKER_LOCK.acquire(blocking=False):
        return ""
    try:
        initial_dir = Path.home()
        if current_path:
            cand = Path(current_path).expanduser()
            if cand.is_dir():
                initial_dir = cand
            elif cand.parent.is_dir():
                initial_dir = cand.parent

        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        proc = subprocess.run(
            [sys.executable, "-c", _TK_PICK_VIDEO_SCRIPT, str(initial_dir)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            env=env,
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError((proc.stderr or proc.stdout or "Failed to open system file dialog").strip())
        lines = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]
        if not lines:
            return ""
        raw_path = str(json.loads(lines[-1]).get("path") or "").strip()
        return str(Path(raw_path).resolve()) if raw_path else ""
    finally:
        _PICKER_LOCK.release()


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
    .row > * { flex: 1; min-width: 0; }
    .actions { display: flex; gap: 8px; align-items: center; flex-shrink: 0; }
    .check { display: flex; align-items: center; gap: 8px; margin: 8px 0; font-size: 13px; cursor: pointer; }
    button {
      padding: 8px 14px; border-radius: 6px; border: 1px solid var(--border);
      background: #1c2434; color: var(--text); font-weight: 600; cursor: pointer; font-size: 13px;
      white-space: nowrap; flex-shrink: 0;
    }
    button:disabled { opacity: 0.6; cursor: not-allowed; }
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
    <div style="display:flex;align-items:center;gap:12px;min-width:0;">
      <h1>LAPLACE Persona · FlowHMR External Mocap</h1>
      <span id="stateBadge" class="badge">STOPPED</span>
    </div>
    <div class="actions">
      <button id="startStream" class="primary" onclick="startEngine()">Start Stream</button>
      <button id="stopStream" class="danger" onclick="stopEngine()">Stop</button>
    </div>
  </header>
  <main>
    <div style="display:flex;flex-direction:column;gap:14px;">
      <section class="card">
        <h2>Input Source (30 Hz Capture)</h2>
        <div class="field">
          <label>Source Mode</label>
          <select id="source_mode" data-cfg onchange="toggleSourceFields()">
            <option value="video">Debug Video File</option>
            <option value="camera">Live Camera (Webcam)</option>
          </select>
        </div>
        <div id="videoFields" class="field">
          <label>Debug Video Path</label>
          <div class="row">
            <input type="text" id="video_path" data-cfg placeholder="Select or enter video path..." />
            <button id="browseVideoBtn" type="button" style="flex:none;" onclick="pickVideoFile()">Browse…</button>
          </div>
          <label class="check"><input type="checkbox" id="loop_video" data-cfg checked /> Loop during streaming</label>
          <label class="check"><input type="checkbox" id="autoRecordVideo" /> Auto-record when I select a video</label>
          <button id="recordVideo" class="primary" disabled onclick="recordFullVideo()">Record full video</button>
          <div id="videoRecordStatus" role="status" style="margin-top:8px;color:var(--muted);"></div>
        </div>
        <div id="cameraFields" class="field" style="display:none;">
          <label>Camera Device</label>
          <div class="row">
            <select id="camera_index" data-cfg data-int><option value="0">Camera 0</option></select>
            <button style="flex:none;" onclick="refreshCameras()">Scan</button>
          </div>
        </div>
        <div class="field">
          <label>Perception Backend</label>
          <select id="perception_backend" data-cfg>
            <option value="auto">Auto (SAM 3D Body Dinov3)</option>
            <option value="sam3d">SAM 3D Body Dinov3 (Strict)</option>
            <option value="replay">Replay / BBox Probe (Debug)</option>
          </select>
        </div>
        <div class="field">
          <label>SAM 3D Keyframe Stride (30 Hz Lerp)</label>
          <select id="sam_stride" data-cfg data-int>
            <option value="0">Auto · Preset Optimal (Recommended)</option>
            <option value="3">Stride 3 · 10 Hz Keyframes</option>
            <option value="4">Stride 4 · 7.5 Hz Keyframes</option>
            <option value="5">Stride 5 · 6 Hz Keyframes</option>
            <option value="2">Stride 2 · 15 Hz Keyframes</option>
            <option value="1">Stride 1 · 30 Hz Every Frame</option>
          </select>
        </div>
        <div class="row">
          <div class="field">
            <label>Hand Decoder Mode</label>
            <select id="hand_mode" data-cfg>
              <option value="auto">Auto (Preset)</option>
              <option value="off">Body Only (Fastest)</option>
              <option value="fast">Fast Hands (2-Layer)</option>
              <option value="full">Full Hands (6-Layer)</option>
            </select>
          </div>
          <div class="field">
            <label>Detector Resolution</label>
            <select id="yolo_size" data-cfg data-int>
              <option value="0">Auto (Preset)</option>
              <option value="416">416px (3x Faster)</option>
              <option value="640">640px (Full)</option>
            </select>
          </div>
        </div>
      </section>

      <section class="card">
        <h2>Latency &amp; Playout</h2>
        <div class="field">
          <label>Sliding-Window Latency Preset</label>
          <select id="preset" data-cfg>
            <option value="ultra_low">Ultra Low (~0.80s · W=48 · 4 steps · Fast Body)</option>
            <option value="low">Low Latency (~1.05s · W=60 · 5 steps · Fast Body)</option>
            <option value="balanced">Balanced (~1.45s · W=70 · 6 steps · Body+Hands)</option>
            <option value="high">High Smoothness (~2.15s · W=91 · 10 steps · Full)</option>
            <option value="max">Max Context (~3.20s · W=116 · 16 steps · Offline)</option>
          </select>
        </div>
        <div class="row">
          <div class="field">
            <label>Playout Rate</label>
            <select id="playout_fps" data-cfg data-int>
              <option value="60">60 FPS (SLERP)</option>
              <option value="30">30 FPS (Native)</option>
            </select>
          </div>
          <div class="field">
            <label>Camera Scale</label>
            <input type="number" id="camera_avatar_scale" data-cfg data-float step="0.05" min="0.4" max="2.5" value="1.0" />
          </div>
          <div class="field">
            <label>Camera FOV° (0=Auto)</label>
            <input type="number" id="camera_fov_deg" data-cfg data-float step="1" min="0" max="140" value="0" />
          </div>
        </div>
        <label class="check"><input type="checkbox" id="mirror" data-cfg /> Mirror Avatar (Webcam Selfie Mode)</label>
        <label class="check"><input type="checkbox" id="follow_movement" data-cfg checked /> Follow Root Movement (XZ Travel)</label>
        <label class="check"><input type="checkbox" id="camera_tracking_enabled" data-cfg checked /> Dynamic 6-DoF Camera Tracking</label>
      </section>

      <section class="card">
        <h2>Persona Output (VMC + Plugin API)</h2>
        <label class="check"><input type="checkbox" id="vmc_enabled" data-cfg checked /> Send VMC OSC Bundle over UDP</label>
        <label class="check"><input type="checkbox" id="vmc_send_camera" data-cfg checked /> Include VMC Camera (`/VMC/Ext/Cam`)</label>
        <div class="row">
          <div class="field"><label>VMC Host</label><input type="text" id="vmc_host" data-cfg value="127.0.0.1" /></div>
          <div class="field"><label>VMC Port</label><input type="number" id="vmc_port" data-cfg value="39539" /></div>
        </div>
        <hr style="border:0;border-top:1px solid var(--border);margin:10px 0;" />
        <label class="check"><input type="checkbox" id="plugin_api_enabled" data-cfg /> Connect Persona Plugin API (WS)</label>
        <div class="field"><label>Plugin API WebSocket URL</label><input type="text" id="plugin_api_url" data-cfg value="ws://127.0.0.1:25034" /></div>
        <div class="field"><label>Bearer Token (Settings → Developer)</label><input type="password" id="plugin_api_token" data-cfg placeholder="Paste token if enabled" /></div>
        <label class="check"><input type="checkbox" id="plugin_api_sync_camera" data-cfg checked /> Sync Stage Camera (`scene.patch -> vrmCamera`)</label>
        <label class="check"><input type="checkbox" id="plugin_api_inject_params" data-cfg /> Also Inject Live2D Angles (`param.inject`)</label>
        <button id="applySettings" class="primary" style="width:100%;margin-top:6px;" onclick="applyConfig()">Apply Settings</button>
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
          <div class="metric"><div class="k">Cam Dist / FOV</div><div class="v" id="m_cam_dist">—</div></div>
          <div class="metric"><div class="k">Cam Az / El</div><div class="v" id="m_cam_ang">—</div></div>
          <div class="metric"><div class="k">Shot Cuts / Syncs</div><div class="v" id="m_cam_cuts">0 / 0</div></div>
          <div class="metric"><div class="k">VMC Packets Sent</div><div class="v" id="m_vmc">0</div></div>
          <div class="metric"><div class="k">Underruns</div><div class="v" id="m_under">0</div></div>
          <div class="metric"><div class="k">Backend</div><div class="v" id="m_back">—</div></div>
        </div>
      </section>
      <section class="card" style="margin-top:14px;">
        <h2>Motion recording</h2>
        <div class="row" style="flex-wrap:wrap;">
          <button id="recordTake" disabled onclick="changeRecording()">Record take</button>
          <select id="exportFile" aria-label="Export files">
            <optgroup label="Avatar and camera (ZIP)">
              <option value="avatar-vrma-camera.zip">Avatar VRMA + camera VMD</option>
              <option value="avatar-fbx-camera.zip">Avatar FBX + camera VMD</option>
              <option value="avatar-vmd-camera.zip">Avatar VMD + camera VMD</option>
            </optgroup>
            <optgroup label="Individual tracks">
              <option value="avatar.vrma">Avatar · VRMA</option>
              <option value="avatar.fbx">Avatar · FBX</option>
              <option value="avatar.vmd">Avatar · VMD</option>
              <option value="camera.vmd">Camera · VMD</option>
            </optgroup>
          </select>
          <button id="downloadTakeBtn" disabled onclick="downloadTake()">Download</button>
        </div>
        <div id="takeStatus" style="margin-top:8px;color:var(--muted);"></div>
        <div id="takeError" role="alert" style="color:var(--bad);"></div>
      </section>
    </div>
  </main>

  <script>
    let initialized = false;
    let saveTimer = null;
    let saveQueue = Promise.resolve();
    let latestStatus = null;
    let videoStart = null;
    let pickingVideo = false;
    let videoStartError = '';
    let settingsError = '';
    const videoPath = document.getElementById('video_path');
    const autoRecord = document.getElementById('autoRecordVideo');
    const autoRecordKey = 'persona-flowhmr.autoRecordVideo';
    try { autoRecord.checked = localStorage.getItem(autoRecordKey) === 'true'; } catch {}
    autoRecord.addEventListener('change', () => {
      try { localStorage.setItem(autoRecordKey, String(autoRecord.checked)); } catch {}
      if (latestStatus) renderRecording(latestStatus);
    });
    // Config controls carry `data-cfg` (id == EngineConfig field); `data-int` / `data-float` mark numeric fields.
    const cfgFields = () => document.querySelectorAll('[data-cfg]');
    function readField(el) {
      if (el.type === 'checkbox') return el.checked;
      if ('float' in el.dataset) {
        const raw = el.value !== '' ? el.value : (el.defaultValue || '0');
        const v = parseFloat(raw);
        if (!Number.isFinite(v)) return parseFloat(el.defaultValue || '0');
        const lo = el.min !== '' ? parseFloat(el.min) : -Infinity;
        const hi = el.max !== '' ? parseFloat(el.max) : Infinity;
        const clamped = Math.min(hi, Math.max(lo, v));
        el.value = String(clamped);
        return clamped;
      }
      if (el.type === 'number' || 'int' in el.dataset) return parseInt(el.value || el.defaultValue || '0', 10);
      return el.value;
    }
    function writeField(el, value) {
      if (value === undefined || value === null) return;
      if (el.type === 'checkbox') el.checked = value;
      else el.value = String(value);
    }
    function scheduleSaveConfig() {
      if (!initialized) return;
      if (saveTimer) clearTimeout(saveTimer);
      saveTimer = setTimeout(() => { applyConfig(); }, 150);
    }
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
    async function pickVideoFile() {
      const btn = document.getElementById('browseVideoBtn');
      pickingVideo = true;
      btn.disabled = true;
      try {
        const r = await fetch('/api/pick-video', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({current_path: videoPath.value}),
        });
        const d = await r.json();
        if (!r.ok) {
          alert(d.detail || 'Failed to open file dialog');
          return;
        }
        if (d.path) {
          videoPath.value = d.path;
          await videoSelected();
        }
      } finally {
        pickingVideo = false;
        if (latestStatus) renderRecording(latestStatus);
      }
    }
    function gatherConfig() {
      return Object.fromEntries([...cfgFields()].filter(el => !el.disabled).map(el => [el.id, readField(el)]));
    }
    function fillForm(c) {
      const camSel = document.getElementById('camera_index');
      const camVal = String(c.camera_index ?? 0);
      if (![...camSel.options].some(o => o.value === camVal)) {
        const o = document.createElement('option');
        o.value = camVal;
        o.textContent = 'Camera ' + camVal;
        camSel.appendChild(o);
      }
      // The saved token is never sent back; only its presence is (see the placeholder).
      cfgFields().forEach(el => { if (el.id !== 'plugin_api_token') writeField(el, c[el.id]); });
      toggleSourceFields();
    }
    async function postStatus(url, payload) {
      // Config/start/stop all reply with the full status, so render it instead of polling again.
      const init = payload === undefined ? {method: 'POST'} : {
        method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(payload),
      };
      const r = await fetch(url, init);
      const data = await r.json();
      const error = r.ok ? null : (data.detail || 'Request failed.');
      if (url === '/api/config') {
        settingsError = error;
        if (r.status === 409) {
          initialized = false;
          await pollStatus();
        }
      }
      if (r.ok) renderStatus(data);
      return error;
    }
    function applyConfig() {
      clearTimeout(saveTimer);
      const config = gatherConfig();
      saveQueue = saveQueue.catch(() => {}).then(() => postStatus('/api/config', config));
      return saveQueue;
    }
    async function startEngine() {
      if (!await applyConfig()) await postStatus('/api/start');
    }
    async function stopEngine() {
      if (videoStart) {
        if (videoStart.sent) {
          // Let the server acknowledge this start before sending its stop.
          videoStart.cancelled = true;
        } else {
          videoStart = null;
          videoStartError = 'Recording cancelled before starting.';
        }
        renderRecording(latestStatus);
        return;
      }
      await postStatus('/api/stop');
    }
    async function videoSelected() {
      videoStartError = '';
      if (autoRecord.checked && videoPath.value.trim()) {
        await recordFullVideo();
      } else {
        await applyConfig();
      }
    }
    async function recordFullVideo() {
      if (videoStart || !latestStatus || latestStatus.running || latestStatus.recording.full_video) return;
      const settings = gatherConfig();
      const start = {sent: false, cancelled: false};
      videoStart = start;
      videoStartError = '';
      renderRecording(latestStatus);
      try {
        clearTimeout(saveTimer);
        await saveQueue.catch(() => {});
        if (videoStart !== start) return;
        start.sent = true;
        const error = await postStatus('/api/recording/video', settings);
        if (error) throw new Error(error);
        if (start.cancelled && latestStatus.recording.full_video) await postStatus('/api/stop');
      } catch (e) {
        videoStartError = e.message;
      } finally {
        if (videoStart === start) {
          videoStart = null;
          renderRecording(latestStatus);
          await pollStatus();
        }
      }
    }
    let takeActive = false;
    let changingTake = false;
    async function changeRecording() {
      changingTake = true;
      document.getElementById('recordTake').disabled = true;
      const error = document.getElementById('takeError');
      error.textContent = '';
      try {
        const r = await fetch('/api/recording/' + (takeActive ? 'stop' : 'start'), {method: 'POST'});
        const data = await r.json();
        if (!r.ok) throw new Error(data.detail || 'Could not update the recording.');
        renderStatus(data);
      } catch (e) {
        error.textContent = e.message;
      } finally {
        changingTake = false;
        await pollStatus();
      }
    }
    function downloadTake() {
      const filename = document.getElementById('exportFile').value;
      const error = document.getElementById('takeError');
      error.textContent = '';
      // Each format can download independently. Attachments go to the browser's download
      // manager; only an error response navigates the frame and fires its load event.
      const id = 'download-' + filename;
      let frame = document.getElementById(id);
      if (!frame) {
        frame = document.createElement('iframe');
        frame.id = id;
        frame.hidden = true;
        frame.onload = () => {
          try {
            const response = frame.contentDocument;
            if (response.URL === 'about:blank') return;
            error.textContent = JSON.parse(response.body.textContent).detail || 'Could not export the take.';
          } catch {
            error.textContent = 'Could not export the take.';
          }
        };
        document.body.appendChild(frame);
      }
      frame.contentWindow.location.replace('/api/recording/download/' + encodeURIComponent(filename));
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
    function renderRecording(s) {
      const take = s.recording;
      const full = Boolean(videoStart || take.full_video);
      cfgFields().forEach(el => {
        el.disabled = full || (take.active && !s.live_config_fields.includes(el.id));
      });
      if (autoRecord.checked && s.running) videoPath.disabled = true;
      document.getElementById('browseVideoBtn').disabled = videoPath.disabled || pickingVideo;
      document.getElementById('startStream').disabled = full;
      document.getElementById('applySettings').disabled = full;
      const stopButton = document.getElementById('stopStream');
      stopButton.textContent = full ? 'Cancel recording' : 'Stop';
      stopButton.disabled = Boolean(videoStart?.cancelled);
      document.getElementById('recordVideo').disabled = full || s.running || !videoPath.value.trim();
      const frames = take.frames.toLocaleString() + ' frames';
      const cameraEnabled = take.full_video ? s.config.camera_tracking_enabled
        : document.getElementById('camera_tracking_enabled').checked;
      const tracks = cameraEnabled ? 'avatar and camera' : 'avatar';
      let progress = take.full_video && take.frames ? 'Recording ' + tracks + ' · ' + frames : 'Preparing video…';
      if (videoStart?.cancelled) progress = 'Cancelling recording…';
      const results = {complete: 'Video complete', error: 'Recording failed', cancelled: 'Recording cancelled'};
      let videoStatus = 'Records ' + tracks + ' from start to end.';
      if (s.running) videoStatus = 'Stop the stream to record the full video.';
      if (results[take.video_result]) videoStatus = results[take.video_result] + ' · ' + frames;
      if (full) videoStatus = progress;
      document.getElementById('videoRecordStatus').textContent = videoStartError || videoStatus;
      takeActive = take.active;
      const recordButton = document.getElementById('recordTake');
      recordButton.textContent = take.active ? 'Finish take' : (take.frames ? 'Record new take' : 'Record take');
      recordButton.hidden = full;
      recordButton.disabled = changingTake || full || take.finishing || (!take.available && !take.active);
      const ready = !full && !take.active && take.frames > 0;
      const formats = document.getElementById('exportFile');
      for (const option of formats.options) {
        option.disabled = ready && !take.has_camera && option.value.includes('camera');
      }
      if (formats.selectedOptions[0]?.disabled) formats.value = 'avatar.vrma';
      document.getElementById('downloadTakeBtn').disabled = !ready;
      let takeStatus = take.available ? 'Ready to record' : 'Choose a video or start a stream to record a take.';
      if (ready) takeStatus = (take.has_camera ? 'Avatar and camera' : 'Avatar') + ' · ' + frames;
      if (take.active) takeStatus = 'Recording · ' + frames;
      if (take.finishing) takeStatus = 'Finishing take · ' + frames;
      if (full) takeStatus = progress;
      document.getElementById('takeStatus').textContent = takeStatus;
    }
    function renderStatus(s) {
      latestStatus = s;
      if (!initialized) {
        fillForm(s.config);
        initialized = true;
      }
      document.getElementById('plugin_api_token').placeholder = s.config.has_plugin_api_token
        ? 'Token saved (paste to replace)'
        : 'Paste token if enabled';
      renderRecording(s);
      const t = s.telemetry;
      const badge = document.getElementById('stateBadge');
      badge.textContent = t.state.toUpperCase();
      badge.className = 'badge ' + t.state;

      document.getElementById('samAlert').style.display = s.models.sam3d_ready ? 'none' : 'block';
      const errEl = document.getElementById('errAlert');
      errEl.textContent = t.error || settingsError || '';
      errEl.style.display = errEl.textContent ? 'block' : 'none';

      document.getElementById('m_play_fps').textContent = t.playout_fps.toFixed(1);
      document.getElementById('m_cap_fps').textContent = t.capture_fps.toFixed(1);
      document.getElementById('m_lat').textContent = Math.round(t.measured_latency_ms) + ' ms';
      document.getElementById('m_lead').textContent = Math.round(t.buffer_lead_ms) + ' ms';
      document.getElementById('m_yolo').textContent = t.yolox_ms_per_frame.toFixed(1) + ' ms';
      document.getElementById('m_sam').textContent = t.sam_ms_per_frame.toFixed(1) + ' ms';
      document.getElementById('m_mmdit').textContent = t.mmdit_ms_per_window.toFixed(1) + ' ms';
      const showCam = Boolean(s.config.camera_tracking_enabled) && t.camera_distance_m > 0;
      document.getElementById('m_cam_dist').textContent = showCam
        ? t.camera_distance_m.toFixed(2) + 'm · ' + t.camera_fov_deg.toFixed(0) + '°'
        : '—';
      document.getElementById('m_cam_ang').textContent = showCam
        ? (t.camera_azimuth_deg >= 0 ? '+' : '') + t.camera_azimuth_deg.toFixed(1) + '° / ' +
          (t.camera_elevation_deg >= 0 ? '+' : '') + t.camera_elevation_deg.toFixed(1) + '°'
        : '—';
      document.getElementById('m_cam_cuts').textContent = t.camera_cuts_detected + ' / ' + t.plugin_api_camera_syncs;
      document.getElementById('m_vmc').textContent = t.vmc_packets_sent.toLocaleString();
      document.getElementById('m_under').textContent = t.underruns;
      document.getElementById('m_back').textContent = t.active_backend;
    }
    async function pollStatus() {
      try {
        const r = await fetch('/api/status');
        renderStatus(await r.json());
      } catch (_e) {}
    }
    videoPath.addEventListener('input', () => {
      videoStartError = '';
      if (latestStatus) renderRecording(latestStatus);
    });
    cfgFields().forEach(el => el.addEventListener('change', el === videoPath ? videoSelected : scheduleSaveConfig));
    setInterval(pollStatus, 400);
    pollStatus();
  </script>
</body>
</html>
"""


@functools.cache
def _placeholder_jpeg(text: str) -> bytes:
    import cv2
    import numpy as np

    canvas = np.zeros((360, 640, 3), dtype=np.uint8)
    canvas[:] = (18, 22, 30)
    cv2.putText(canvas, text, (180, 185), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (140, 160, 190), 2, cv2.LINE_AA)
    ok, buf = cv2.imencode(".jpg", canvas)
    return buf.tobytes() if ok else b""


def create_app(engine: StreamingMocapEngine | None = None) -> FastAPI:
    eng = engine or StreamingMocapEngine()

    @asynccontextmanager
    async def _lifespan(_app: FastAPI):
        try:
            yield
        finally:
            eng.stop()

    app = FastAPI(title="Persona FlowHMR External Mocap", version=__version__, lifespan=_lifespan)

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

    @app.post("/api/pick-video")
    async def api_pick_video(req: PickVideoRequest) -> dict[str, str]:
        try:
            chosen = await asyncio.to_thread(
                _open_system_video_dialog,
                req.current_path or eng.config.video_path,
            )
            return {"path": chosen}
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post("/api/config")
    def api_config(patch: dict[str, Any]) -> dict[str, Any]:
        try:
            return eng.update_live_settings(patch)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    def control(action: Callable[[], None]) -> dict[str, Any]:
        try:
            action()
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return eng.get_status()

    @app.post("/api/start")
    def api_start() -> dict[str, Any]:
        return control(eng.start)

    @app.post("/api/stop")
    def api_stop() -> dict[str, Any]:
        eng.stop()
        return eng.get_status()

    @app.post("/api/recording/start")
    def start_recording() -> dict[str, Any]:
        return control(eng.begin_recording)

    @app.post("/api/recording/video")
    def record_video(settings: dict[str, Any] | None = None) -> dict[str, Any]:
        return control(lambda: eng.record_video(settings))

    @app.post("/api/recording/stop")
    def finish_recording() -> dict[str, Any]:
        return control(eng.finish_recording)

    @app.get("/api/recording/download/{filename}")
    def download_recording(filename: str) -> FileResponse:
        tracks = _DOWNLOAD_TRACKS.get(filename)
        if tracks is None:
            raise HTTPException(status_code=422, detail="Unsupported motion download.")
        try:
            frames = eng.recording.frames()
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        bundled = len(tracks) > 1
        suffix = ".zip" if bundled else Path(tracks[0]).suffix
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp:
            path = Path(temp.name)
        try:
            if not bundled:
                write_motion(path, frames, camera=tracks[0] == "camera.vmd")
                return _MotionDownload(path, filename=filename, media_type="application/octet-stream")
            with tempfile.TemporaryDirectory() as folder, ZipFile(path, "w", ZIP_DEFLATED, compresslevel=1) as bundle:
                # Writing the camera first reports missing tracking before encoding the skeleton.
                for name in tracks:
                    track = Path(folder) / name
                    write_motion(track, frames, camera=name == "camera.vmd")
                    bundle.write(track, arcname=name)
                    track.unlink()
            return _MotionDownload(path, filename=filename, media_type="application/zip")
        except BaseException as exc:
            path.unlink(missing_ok=True)
            if isinstance(exc, ValueError):
                raise HTTPException(status_code=422, detail=str(exc)) from exc
            raise

    @app.post("/api/setup-models")
    def api_setup_models(req: SetupModelsRequest) -> dict[str, Any]:
        try:
            return download_missing_models(hf_token=req.hf_token)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    def _mjpeg_response(render_fn: Callable[[], bytes | None]) -> StreamingResponse:
        async def frames():
            idle_jpg = _placeholder_jpeg("Stream Idle - Click Start Stream")
            while True:
                jpg = await asyncio.to_thread(render_fn)
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + (jpg or idle_jpg) + b"\r\n"
                await asyncio.sleep(1.0 / 20.0)

        return StreamingResponse(frames(), media_type="multipart/x-mixed-replace; boundary=frame")

    @app.get("/stream/live.mjpg")
    async def stream_live() -> StreamingResponse:
        return _mjpeg_response(eng.render_live_jpeg)

    @app.get("/stream/synced.mjpg")
    async def stream_synced() -> StreamingResponse:
        return _mjpeg_response(eng.render_synced_playout_jpeg)

    return app
