"""Project filesystem paths and vendor sys.path bootstrap."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_models_dir(project_root: Path = PROJECT_ROOT, env_override: str | None = None) -> Path:
    """Resolve `models/`: `$PERSONA_FLOWHMR_MODELS`, local `models/`, or the main git worktree's `models/`.

    In any git worktree (`git worktree add` / Orca workspaces), `<worktree>/.git` is a text file pointing
    to `<main>/.git/worktrees/<id>`, whose `commondir` resolves back to `<main>/.git`. Automatically
    sharing `<main>/models` makes checkpoints available in every worktree without duplicating 5+ GB.
    """
    override = env_override if env_override is not None else os.environ.get("PERSONA_FLOWHMR_MODELS")
    if override:
        return Path(override).expanduser().resolve()

    local_models = project_root / "models"
    if local_models.is_dir() and any(local_models.iterdir()):
        return local_models.resolve()

    dot_git = project_root / ".git"
    if dot_git.is_file():
        try:
            first_line = dot_git.read_text(encoding="utf-8").splitlines()[0].strip()
            if first_line.startswith("gitdir:"):
                raw_gitdir = Path(first_line.split(":", 1)[1].strip())
                gitdir = (raw_gitdir if raw_gitdir.is_absolute() else (project_root / raw_gitdir)).resolve()
                commondir_file = gitdir / "commondir"
                if commondir_file.is_file():
                    raw_common = Path(commondir_file.read_text(encoding="utf-8").strip())
                    common_git_dir = (raw_common if raw_common.is_absolute() else (gitdir / raw_common)).resolve()
                else:
                    common_git_dir = gitdir
                shared_models = common_git_dir.parent / "models"
                if shared_models.is_dir() or not local_models.exists():
                    return shared_models.resolve()
        except (OSError, IndexError):
            pass

    return local_models.resolve()


MODELS_DIR = resolve_models_dir()
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
