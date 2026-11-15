"""Model checkpoint verification and setup utility (`persona-flowhmr setup`)."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from huggingface_hub import hf_hub_download, snapshot_download
from huggingface_hub.errors import GatedRepoError, HfHubHTTPError

from .paths import (
    FLOWHMR_CFG_PATH,
    FLOWHMR_CKPT_PATH,
    J_REGRESSOR_PATH,
    MODELS_DIR,
    MOTION_STATS_PATH,
    SAM3D_CKPT_PATH,
    SAM3D_CFG_PATH,
    SAM3D_DIR,
    SAM3D_MHR_PATH,
    YOLOX_CKPT_PATH,
    bootstrap_runtime,
)


def check_models_status() -> dict[str, Any]:
    """Return presence and byte sizes of all required public and gated model files."""
    files = {
        "yolox_l": YOLOX_CKPT_PATH,
        "flowhmr_ckpt": FLOWHMR_CKPT_PATH,
        "flowhmr_cfg": FLOWHMR_CFG_PATH,
        "motion_stats": MOTION_STATS_PATH,
        "j_regressor": J_REGRESSOR_PATH,
        "sam3d_ckpt": SAM3D_CKPT_PATH,
        "sam3d_cfg": SAM3D_CFG_PATH,
        "sam3d_mhr": SAM3D_MHR_PATH,
    }
    items = {}
    for key, path in files.items():
        exists = path.is_file()
        items[key] = {
            "path": str(path),
            "exists": exists,
            "size_mb": round(path.stat().st_size / (1024 * 1024), 2) if exists else 0.0,
        }
    public_ready = all(
        items[k]["exists"] for k in ("yolox_l", "flowhmr_ckpt", "flowhmr_cfg", "motion_stats")
    )
    sam3d_ready = all(items[k]["exists"] for k in ("sam3d_ckpt", "sam3d_cfg", "sam3d_mhr"))
    return {
        "models_dir": str(MODELS_DIR),
        "public_ready": public_ready,
        "sam3d_ready": sam3d_ready,
        "ready_for_live_camera": public_ready and sam3d_ready,
        "files": items,
    }


def download_missing_models(hf_token: str | None = None) -> dict[str, Any]:
    """Download any missing public checkpoints and attempt `facebook/sam-3d-body-dinov3`."""
    bootstrap_runtime()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Public YOLOX-L + FlowHMR weights
    if not YOLOX_CKPT_PATH.is_file():
        YOLOX_CKPT_PATH.parent.mkdir(parents=True, exist_ok=True)
        import urllib.request

        urllib.request.urlretrieve(
            "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_l.pth",
            str(YOLOX_CKPT_PATH),
        )

    if not FLOWHMR_CKPT_PATH.is_file() or not FLOWHMR_CFG_PATH.is_file():
        FLOWHMR_CKPT_PATH.parent.mkdir(parents=True, exist_ok=True)
        for rel, dst in (
            ("checkpoints/flowhmr_latest/flowhmr_latest.ckpt", FLOWHMR_CKPT_PATH),
            ("checkpoints/flowhmr_latest/config.yml", FLOWHMR_CFG_PATH),
        ):
            if not dst.is_file():
                cached = hf_hub_download(repo_id="FlowHMR/FlowHMR", filename=rel, token=hf_token)
                Path(dst).write_bytes(Path(cached).read_bytes())

    if not J_REGRESSOR_PATH.is_file():
        cached = hf_hub_download(
            repo_id="FlowHMR/FlowHMR",
            filename="assets/body_models/smpl_neutral_J_regressor.pt",
            token=hf_token,
        )
        J_REGRESSOR_PATH.write_bytes(Path(cached).read_bytes())

    # 2. Gated Meta SAM 3D Body (`facebook/sam-3d-body-dinov3`)
    sam_error: str | None = None
    if not (SAM3D_CKPT_PATH.is_file() and SAM3D_CFG_PATH.is_file() and SAM3D_MHR_PATH.is_file()):
        SAM3D_DIR.mkdir(parents=True, exist_ok=True)
        try:
            snapshot_download(
                repo_id="facebook/sam-3d-body-dinov3",
                local_dir=str(SAM3D_DIR),
                allow_patterns=["model.ckpt", "model_config.yaml", "assets/mhr_model.pt"],
                token=hf_token or True,
            )
        except GatedRepoError as exc:
            sam_error = (
                "Gated Hugging Face repo `facebook/sam-3d-body-dinov3` requires license acceptance "
                "and authentication. Accept the license at https://huggingface.co/facebook/sam-3d-body-dinov3 "
                "then run `.venv\\Scripts\\hf.exe auth login` (or pass `--hf-token`)."
            )
        except HfHubHTTPError as exc:
            if exc.response is not None and exc.response.status_code in (401, 403):
                sam_error = (
                    "Hugging Face authentication required for `facebook/sam-3d-body-dinov3` "
                    f"(HTTP {exc.response.status_code}). Accept the license at "
                    "https://huggingface.co/facebook/sam-3d-body-dinov3 and run "
                    "`.venv\\Scripts\\hf.exe auth login` (or pass `--hf-token`)."
                )
            else:
                sam_error = str(exc)

    status = check_models_status()
    status["sam3d_download_error"] = sam_error
    return status
