"""Project filesystem paths and vendor sys.path bootstrap."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = Path(os.environ.get("PERSONA_FLOWHMR_MODELS", str(PROJECT_ROOT / "models"))).resolve()
VENDOR_DIR = PROJECT_ROOT / "vendor"
CACHE_DIR = PROJECT_ROOT / "cache"
SETTINGS_FILE = CACHE_DIR / "settings.json"
DEFAULT_DEMO_VIDEO = Path(r"D:\Git\persona\references\dancing girl [2E2okJyQyZI].mp4")

YOLOX_CKPT_PATH = MODELS_DIR / "yolox" / "yolox_l.pth"
FLOWHMR_CKPT_PATH = MODELS_DIR / "flowhmr_latest" / "flowhmr_latest.ckpt"
FLOWHMR_CFG_PATH = MODELS_DIR / "flowhmr_latest" / "config.yml"
MOTION_STATS_PATH = MODELS_DIR / "motion_stats.json"
J_REGRESSOR_PATH = MODELS_DIR / "smpl_neutral_J_regressor.pt"
SAM3D_DIR = MODELS_DIR / "sam-3d-body-dinov3"
SAM3D_CKPT_PATH = SAM3D_DIR / "model.ckpt"
SAM3D_CFG_PATH = SAM3D_DIR / "model_config.yaml"
SAM3D_MHR_PATH = SAM3D_DIR / "assets" / "mhr_model.pt"
# Every file the vendored `load_sam_3d_body` needs (it reads `model_config.yaml` next to the checkpoint).
SAM3D_FILES = (SAM3D_CKPT_PATH, SAM3D_CFG_PATH, SAM3D_MHR_PATH)

_BOOTSTRAPPED = False


def is_sam3d_available() -> bool:
    """Whether the `facebook/sam-3d-body-dinov3` checkpoint, config, and MHR model are all installed."""
    return all(p.is_file() for p in SAM3D_FILES)


def bootstrap_runtime() -> None:
    """Configure Windows OS truststore SSL certs and add bundled vendor packages to sys.path."""
    global _BOOTSTRAPPED
    if _BOOTSTRAPPED:
        return

    try:
        import truststore

        truststore.inject_into_ssl()
    except Exception:
        pass

    # Prevent root DEBUG log handlers (e.g. pytest LogCaptureHandler) from activating PyTorch's
    # structured FX trace serializer during `torch.compile`.
    logging.getLogger("torch.__trace").setLevel(logging.WARNING)

    for extra in (
        VENDOR_DIR,
        VENDOR_DIR / "YOLOX",
        VENDOR_DIR / "sam-3d-body",
        VENDOR_DIR / "dinov3",
    ):
        extra_str = str(extra)
        if extra.exists() and extra_str not in sys.path:
            sys.path.insert(0, extra_str)

    _BOOTSTRAPPED = True
