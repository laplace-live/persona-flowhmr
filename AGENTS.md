# persona-flowhmr Agent Guide

External single-camera buffered live mocap & dynamic 6-DoF camera bridge for **LAPLACE Persona** (`30 Hz Source -> YOLOX-L + SAM 3D Body (+ ShotCut/VO) -> FlowHMR MMDiT Window -> Stitcher -> 30/60 Hz VMC UDP & Persona Plugin WS`).

## Project Structure

- `pyproject.toml`: `uv` project config (Python 3.10–3.12, `pytorch-cu128`, `ruff`, `pytest`)
- `src/persona_flowhmr/`:
  - `cli.py`: `persona-flowhmr` CLI (`serve`, `run`, `setup`, `status`)
  - `web_ui.py`: FastAPI control dashboard & live/synced MJPEG preview endpoints
  - `config.py`: `EngineConfig` & `LATENCY_PRESETS` (`ultra_low`..`max`)
  - `paths.py`: Repo/vendor bootstrap & git-worktree-shared `models/` resolution
  - `setup_models.py`: Atomic ModelScope checkpoint downloader & status checker
  - `engine.py`: Multi-threaded `StreamingMocapEngine` (capture / GPU / playout)
  - `sources.py`: 30.0 Hz `CameraSource`, `VideoFileSource`, `PreviewRingBuffer`
  - `perception.py`: Batched BF16 `YOLOX-L` + ByteTrack & compiled `SAM 3D Body Lite`
  - `camera_tracker.py`: `ShotCutDetector`, `BackgroundVisualOdometry`, `StreamingCameraSolver`
  - `solver.py`: Compiled TF32 `FlowHMRWindowSolver` (MMDiT Euler ODE)
  - `stitcher.py`: `MotionLatentDecoder`, SLERP overlap crossfade, root & camera commit
  - `playout.py`: `PosePlayoutBuffer` (30/60 Hz pose & cut-aware camera interpolation)
  - `retarget.py`: SMPL-H 52-bone VMC retargeting & Live2D parameter mapping
  - `vmc_sender.py`: Zero-dependency OSC 1.0 `#bundle` UDP sender (`Root`, `Bone`, `Cam`)
  - `persona_api.py`: Async LAPLACE Persona Plugin WS client (VMC setup, `vrmCamera`, `param.inject`)
- `tests/`: Unit, parity, WebUI, camera-tracking, and demo-video pipeline tests
- `vendor/`: Read-only upstream code (`flowhmr`, `sam-3d-body`, `dinov3`, `YOLOX`)
- `models/`: Local or worktree-shared checkpoints (gitignored)
- `cache/`: Runtime `settings.json` persistence (gitignored)

## Verification Rules

Before finishing any code change in `persona-flowhmr`, verify:

1. **Static & Unit Checks**: Run `uv run ruff check src tests` and `uv run pytest` with zero errors, adding regression tests for bug fixes and new behavior. Never modify third-party code under `vendor/`.
2. **Real-Time Latency & Zero-Overhead Gating**: For any capture, perception, solver, stitcher, or playout change, run a real-time `StreamingMocapEngine` check (including `1080×1920` input on `ultra_low`) with optional features tested **both enabled and disabled**:
   - Disabled features must bypass their compute paths (`0 ms` added overhead).
   - Steady-state `measured_latency_ms` must stay near `target_latency_ms` without upward drift or steady-state `underruns` (`< 33.3 ms/frame` budget at 30 Hz).
3. **End-to-End Stage Validation**: Validate tracking/camera changes against real reference videos (including multi-camera hard cuts when touching camera or stitcher logic), and snapshot + restore live LAPLACE Persona scene/camera state in a `finally` block whenever testing against a running instance.
