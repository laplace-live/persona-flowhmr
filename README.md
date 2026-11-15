# persona-flowhmr

Standalone external single-camera mocap app for **LAPLACE Persona**, running buffered sliding-window **FlowHMR** (`YOLOX-L` + `SAM 3D Body Lite` + `HunyuanMotionMMDiT`) with smooth **30 / 60 Hz SLERP playout** over **VMC (OSC/UDP)** and **Persona Plugin API (WebSocket)**.

## Features

- **Dual Input Modes (Unified 30.0 Hz Sampler):**
  - **Live Camera (`--source camera`):** Low-latency webcam capture on a dedicated reader thread resampled onto FlowHMR's exact 30.0 Hz clock.
  - **Debug Video (`--source video`):** Plays any `.mp4`/`.mkv`/`.mov` (24/25/30/60 fps) resampled on the fly to 30.0 Hz without offline FFmpeg pre-transcoding, with seamless looping and side-by-side **Live (`t = now`)** vs **Latency-Synced (`t = now - Δ`)** preview feeds.
- **RTX PRO 6000 Blackwell Accelerated (`triton-windows` + `torch.compile`):**
  - Batched BF16 `YOLOX-L` (`416x416` stride-2 with linear bbox interpolation, ~1.3 ms/frame) + loop-aware online `ByteTrack` stabilizer.
  - Compiled BF16 `SAM 3D Body Lite` with GPU ray-conditioning, keypoints-only MHR fast path, configurable decoder depth (`3`/`4`/`6` layers), and `hand_mode` (`off` in-distribution neutral hands ~6.5 ms/frame effective, `fast` 2-layer hands, or `full` 6-layer hands).
  - Compiled TF32 `FlowHMR` 460M MMDiT sliding-window Euler ODE solver (~28–40 ms @ 4–6 steps) with algebraically phase-locked batching (`0` idle cadence stalls across infinite video loops), antipodal SLERP overlap crossfading, and continuous `srv_xz` horizontal root velocity integration.
  - Zero dependency on gated `smplh/neutral/model.npz` or `facebook/VGGT-Omega`.
- **Dual Persona Integration:**
  - **VMC over UDP (`127.0.0.1:39539`):** Atomic OSC 1.0 `#bundle` datagrams containing `/VMC/Ext/Root/Pos`, all **52 SMPL-H body + finger humanoid bones** (`/VMC/Ext/Bone/Pos`), and `/VMC/Ext/OK` at constant 30 or 60 Hz (1 ms Windows timer resolution).
  - **Persona Plugin API (`ws://127.0.0.1:25034`):** Authenticates via Bearer token, runs `session.identify`, auto-creates/enables Persona's `vmc` tracking source (`tracking.addSource`, `pose.setEnabled`), binds the primary VRM avatar (`instance.setTrackingSources`), and optionally injects Live2D `FaceAngleX/Y/Z` & `ParamBodyAngleX/Y/Z` (`param.inject`).

## Latency Presets (RTX PRO 6000 Blackwell · Co-Resident with LAPLACE Persona)

| Preset | Window `[H + S + O + L]` | ODE Steps | SAM Mode | End-to-End Playout Latency |
| --- | --- | --- | --- | --- |
| `ultra_low` | `38 + 6 + 2 + 2 = 48` | `4` | Stride 3 · 3-Layer Body (`hand_mode=off`) | **~0.80 s** |
| `low` *(default)* | `44 + 9 + 3 + 4 = 60` | `5` | Stride 3 · 3-Layer Body (`hand_mode=off`) | **~1.05 s** |
| `balanced` | `48 + 12 + 4 + 6 = 70` | `6` | Stride 3 · 4-Layer Body + 2-Layer Fast Hands | **~1.45 s** |
| `high` | `60 + 15 + 6 + 10 = 91` | `10` | Stride 5 · Full 6-Layer Body + Hands | **~2.15 s** |
| `max` | `60 + 20 + 8 + 28 = 116` | `16` | Stride 5 · Full 6-Layer Body + Hands | **~3.20 s** |

## Setup (Git Bash)

1. Check GPU and installed checkpoints:
   ```bash
   cd /d/Git/persona-flowhmr
   source .venv/Scripts/activate
   persona-flowhmr status
   ```
2. Download any missing checkpoints (automatically pulls `facebook/sam-3d-body-dinov3` from ModelScope with SHA256 verification):
   ```bash
   persona-flowhmr setup
   ```

## Usage (Git Bash)

### 1. External App Control & Preview Web UI
```bash
cd /d/Git/persona-flowhmr
source .venv/Scripts/activate
persona-flowhmr serve --port 7865
```
Open `http://127.0.0.1:7865` to switch between **Live Camera** and **Debug Video**, adjust latency presets (`0.80s`–`3.20s`), configure `Hand Mode` (`auto`/`off`/`fast`/`full`) and `YOLOX` resolution (`416`/`640`), toggle `30 / 60 FPS` and `Mirror`, configure VMC / Persona Plugin API, and view synchronized live + playout HUD previews.

### 2. Headless CLI (Debug Video or Live Camera -> Persona)
```bash
# Low-latency (~1.05s) debug dance video -> VMC UDP 39539 + Persona Plugin API auto-setup
persona-flowhmr run \
  --source video \
  --video "D:/Git/persona/references/dancing girl [2E2okJyQyZI].mp4" \
  --preset low \
  --fps 60 \
  --plugin-api \
  --plugin-token "<YOUR_PERSONA_TOKEN>"

# Live webcam (device 0) with selfie mirror enabled
persona-flowhmr run --source camera --camera 0 --preset low --fps 60 --mirror
```
