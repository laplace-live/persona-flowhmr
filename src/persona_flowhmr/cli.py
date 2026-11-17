"""Command-line interface for `persona-flowhmr`."""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import json
import os
import signal
import sys
import tempfile
import time
from pathlib import Path
from typing import get_args

from .config import (
    CONFIG_FIELDS,
    LATENCY_PRESETS,
    EngineConfig,
    HandMode,
    PerceptionBackend,
    PlayoutFps,
    SamStride,
    SourceMode,
    YoloSize,
    YoloStride,
)
from .paths import DEFAULT_DEMO_VIDEO, SETTINGS_FILE
from .setup_models import check_models_status, download_missing_models


def _add_solve_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--preset",
        choices=tuple(LATENCY_PRESETS),
        help="Latency / smoothness sliding-window preset",
    )
    p.add_argument("--mirror", action="store_const", const=True, help="Mirror left/right body & root travel")
    p.add_argument(
        "--no-follow",
        dest="follow_movement",
        action="store_const",
        const=False,
        help="Lock horizontal XZ root travel at origin",
    )
    p.add_argument(
        "--camera-tracking",
        "--camera-track",
        dest="camera_tracking_enabled",
        action="store_const",
        const=True,
        help="Enable dynamic 6-DoF camera tracking",
    )
    p.add_argument(
        "--no-camera-tracking",
        "--no-camera-track",
        dest="camera_tracking_enabled",
        action="store_const",
        const=False,
        help="Disable dynamic 6-DoF camera tracking",
    )
    p.add_argument(
        "--camera-scale",
        dest="camera_avatar_scale",
        type=float,
        help="Avatar-to-performer camera framing scale in [0.4, 2.5] (default: 1.0)",
    )
    p.add_argument(
        "--camera-fov",
        dest="camera_fov_deg",
        type=float,
        help="Calibrated vertical camera FOV in degrees (0 = auto intrinsics + optical zoom tracking)",
    )
    p.add_argument(
        "--backend",
        dest="perception_backend",
        choices=get_args(PerceptionBackend),
        help="Perception backend (`auto`/`sam3d` uses SAM 3D Body; `replay` uses --replay-tokens or bbox probe)",
    )
    p.add_argument(
        "--sam-stride",
        type=int,
        choices=get_args(SamStride),
        help="SAM 3D Body keyframe stride at 30 Hz (0=preset optimal, 3=10Hz, 5=6Hz, 1=every frame)",
    )
    p.add_argument(
        "--hand-mode",
        choices=get_args(HandMode),
        help="SAM 3D hand decoder mode (`auto` uses preset: `off` for ultra_low/low, `fast` for balanced, `full` for high/max)",
    )
    p.add_argument(
        "--yolo-size",
        type=int,
        choices=get_args(YoloSize),
        help="YOLOX-L detector input size (0=preset optimal, 416=fast 3x, 640=full)",
    )
    p.add_argument(
        "--yolo-stride",
        type=int,
        choices=get_args(YoloStride),
        help="YOLOX-L detector frame stride with linear bbox interpolation (default: 2)",
    )
    p.add_argument(
        "--replay-tokens",
        dest="replay_tokens_path",
        metavar="REPLAY_TOKENS",
        help="Optional `.pt` or `.npz` (T, 3072) SAM token file",
    )
    p.add_argument(
        "--no-compile",
        dest="torch_compile",
        action="store_const",
        const=False,
        help="Disable torch.compile Triton acceleration",
    )


def _add_common_engine_args(p: argparse.ArgumentParser) -> None:
    """Engine flags: each `dest` is an `EngineConfig` field; flags left unset keep the base config's value."""
    p.add_argument("--source", dest="source_mode", choices=get_args(SourceMode), help="Input source mode")
    p.add_argument("--video", dest="video_path", metavar="VIDEO", help="Debug video file path")
    p.add_argument(
        "--no-loop",
        dest="loop_video",
        action="store_const",
        const=False,
        help="Play debug video once instead of looping",
    )
    p.add_argument(
        "--camera",
        dest="camera_index",
        metavar="CAMERA",
        type=int,
        help="Camera device index when --source=camera",
    )
    p.add_argument(
        "--fps",
        dest="playout_fps",
        type=int,
        choices=get_args(PlayoutFps),
        help="VMC output playout FPS (30 or 60)",
    )
    _add_solve_args(p)
    p.add_argument("--vmc-host", help="Target VMC receiver UDP host")
    p.add_argument("--vmc-port", type=int, help="Target VMC receiver UDP port")
    p.add_argument("--no-vmc", dest="vmc_enabled", action="store_const", const=False, help="Disable UDP VMC output")
    p.add_argument(
        "--no-vmc-camera",
        dest="vmc_send_camera",
        action="store_const",
        const=False,
        help="Omit `/VMC/Ext/Cam` from UDP VMC bundles",
    )
    p.add_argument(
        "--plugin-api",
        dest="plugin_api_enabled",
        action="store_const",
        const=True,
        help="Connect to LAPLACE Persona Plugin API WebSocket",
    )
    p.add_argument("--plugin-url", dest="plugin_api_url", metavar="PLUGIN_URL", help="Persona Plugin API WebSocket URL")
    p.add_argument(
        "--plugin-token",
        dest="plugin_api_token",
        metavar="PLUGIN_TOKEN",
        default=os.environ.get("PERSONA_PLUGIN_TOKEN") or None,
        help="Persona Plugin API bearer token (or set $PERSONA_PLUGIN_TOKEN)",
    )
    p.add_argument(
        "--inject-params",
        dest="plugin_api_inject_params",
        action="store_const",
        const=True,
        help="Also inject Live2D angles via `param.inject`",
    )
    p.add_argument(
        "--no-plugin-camera",
        dest="plugin_api_sync_camera",
        action="store_const",
        const=False,
        help="Disable syncing stage `vrmCamera` over Persona Plugin API",
    )


def _config_from_args(args: argparse.Namespace, base: EngineConfig) -> EngineConfig:
    """Overlay the engine flags that were given onto `base` (saved settings for `serve`, defaults for `run`)."""
    overrides = {k: v for k, v in vars(args).items() if k in CONFIG_FIELDS and v is not None}
    # Without `--video`, an empty saved path falls back to the demo clip.
    overrides.setdefault("video_path", base.video_path or str(DEFAULT_DEMO_VIDEO))
    return dataclasses.replace(base, **overrides)


def cmd_status(_args: argparse.Namespace) -> int:
    import torch

    gpu = {
        "cuda_available": torch.cuda.is_available(),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    print(json.dumps({"gpu": gpu, "models": check_models_status()}, indent=2))
    return 0


def cmd_setup(args: argparse.Namespace) -> int:
    status = download_missing_models(hf_token=args.hf_token or os.environ.get("HF_TOKEN"))
    print(json.dumps(status, indent=2))
    if status.get("sam3d_download_error"):
        print(f"\n[ACTION REQUIRED] {status['sam3d_download_error']}", file=sys.stderr)
        return 2
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from .engine import StreamingMocapEngine
    from .web_ui import create_app

    cfg = _config_from_args(args, base=EngineConfig.load(SETTINGS_FILE))
    engine = StreamingMocapEngine(cfg, settings_path=SETTINGS_FILE)
    if args.auto_start:
        engine.start(realtime_source=True)
    app = create_app(engine)
    print(f"Persona FlowHMR Control UI: http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    from .engine import StreamingMocapEngine

    cfg = _config_from_args(args, base=EngineConfig())
    engine = StreamingMocapEngine(cfg)

    stopping = False

    def _handle_sig(_sig, _frame) -> None:
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGINT, _handle_sig)
    engine.start(realtime_source=True)
    t0 = time.perf_counter()
    try:
        while not stopping and engine.is_running:
            time.sleep(1.0)
            tel = engine.telemetry
            print(
                f"[{tel.state}] backend={tel.active_backend} cap={tel.capture_fps:.1f}fps "
                f"play={tel.playout_fps:.1f}fps lat={tel.measured_latency_ms:.0f}ms "
                f"lead={tel.buffer_lead_ms:.0f}ms yolo={tel.yolox_ms_per_frame:.1f}ms "
                f"sam={tel.sam_ms_per_frame:.1f}ms mmdit={tel.mmdit_ms_per_window:.1f}ms "
                f"cam={tel.camera_distance_m:.2f}m/{tel.camera_azimuth_deg:+.1f}deg "
                f"cuts={tel.camera_cuts_detected} vmc_pkts={tel.vmc_packets_sent} underruns={tel.underruns}"
            )
            if tel.state == "error":
                print(f"[ERROR] {tel.error}", file=sys.stderr)
                return 1
            if args.duration and (time.perf_counter() - t0) >= args.duration:
                break
    finally:
        engine.stop()
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    """Solve a finite video once and serialize each requested track at the source rate."""
    from .engine import StreamingMocapEngine
    from .motion_export import AVATAR_FORMATS, write_motion

    outputs = [(Path(p), False) for p in args.avatar]
    if args.camera_motion:
        outputs.append((Path(args.camera_motion), True))
    source = Path(args.video_path).resolve()
    try:
        if not source.is_file():
            raise ValueError(f"Video file not found: {source}")
        if not outputs:
            raise ValueError("Specify at least one --avatar or --camera-motion file.")
        seen = {source}
        for dest, camera in outputs:
            formats = ("vmd",) if camera else AVATAR_FORMATS
            if dest.suffix.lower()[1:] not in formats:
                raise ValueError(f"{dest}: expected {', '.join(formats)} extension.")
            if dest.resolve() in seen or dest.is_dir():
                raise ValueError(f"Each output requires a distinct file path: {dest}")
            seen.add(dest.resolve())
        cfg = _config_from_args(args, base=EngineConfig(loop_video=False, vmc_enabled=False))
        if args.camera_motion and not cfg.camera_tracking_enabled:
            raise ValueError("Camera export requires camera tracking.")

        engine = StreamingMocapEngine(cfg)
        try:
            engine.record_video()
            while engine.is_running:
                time.sleep(0.1)
        finally:
            engine.stop()
        if engine.telemetry.state == "error":
            raise RuntimeError(engine.telemetry.error)
        frames = engine.recording.frames()
        with contextlib.ExitStack() as cleanup:
            staged = []
            for dest, camera in outputs:
                dest.parent.mkdir(parents=True, exist_ok=True)
                with tempfile.NamedTemporaryFile(dir=dest.parent, suffix=dest.suffix, delete=False) as fp:
                    tmp = Path(fp.name)
                cleanup.callback(tmp.unlink, missing_ok=True)
                write_motion(tmp, frames, camera=camera)
                staged.append((tmp, dest))
            for tmp, dest in staged:
                tmp.replace(dest)
                print(f"Wrote {dest} ({len(frames)} source frames at 30 Hz)")
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"Export failed: {exc}", file=sys.stderr)
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="persona-flowhmr", description="FlowHMR -> LAPLACE Persona Live Mocap")
    sub = parser.add_subparsers(dest="command")

    p_status = sub.add_parser("status", help="Check GPU and model checkpoint status")
    p_status.set_defaults(func=cmd_status)

    p_setup = sub.add_parser("setup", help="Verify/download public & SAM 3D Body checkpoints")
    p_setup.add_argument("--hf-token", default="", help="Hugging Face access token for gated sam-3d-body-dinov3")
    p_setup.set_defaults(func=cmd_setup)

    p_serve = sub.add_parser("serve", help="Launch the external app Web UI & API server")
    p_serve.add_argument("--host", default="127.0.0.1", help="HTTP bind host")
    p_serve.add_argument("--port", type=int, default=7865, help="HTTP bind port")
    p_serve.add_argument("--auto-start", action="store_true", help="Start streaming immediately on launch")
    _add_common_engine_args(p_serve)
    p_serve.set_defaults(func=cmd_serve)

    p_run = sub.add_parser("run", help="Run headless streaming mocap to VMC / Plugin API")
    p_run.add_argument("--duration", type=float, default=0.0, help="Optional run duration in seconds (0 = until Ctrl+C)")
    _add_common_engine_args(p_run)
    p_run.set_defaults(func=cmd_run)

    p_export = sub.add_parser("export", help="Export animation from one complete video at 30 Hz")
    p_export.add_argument("--video", dest="video_path", required=True, help="Source video file")
    p_export.add_argument(
        "--avatar", action="append", default=[], metavar="FILE", help="Avatar .vrma, .fbx or .vmd (repeatable)"
    )
    p_export.add_argument("--camera-motion", metavar="FILE", help="Camera .vmd output")
    _add_solve_args(p_export)
    p_export.set_defaults(func=cmd_export)

    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return 0
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
