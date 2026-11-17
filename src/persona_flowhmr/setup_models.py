"""Model checkpoint verification and setup utility (`persona-flowhmr setup`)."""

from __future__ import annotations

import hashlib
import json
import shutil
import stat
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download, snapshot_download
from huggingface_hub.errors import GatedRepoError, HfHubHTTPError

from . import __version__
from .paths import (
    FLOWHMR_CFG_PATH,
    FLOWHMR_CKPT_PATH,
    J_REGRESSOR_PATH,
    MODELS_DIR,
    MOTION_STATS_PATH,
    SAM3D_CFG_PATH,
    SAM3D_CKPT_PATH,
    SAM3D_DIR,
    SAM3D_FILES,
    SAM3D_MHR_PATH,
    YOLOX_CKPT_PATH,
    bootstrap_runtime,
    is_sam3d_available,
)

_USER_AGENT = f"persona-flowhmr/{__version__}"
_MODELSCOPE_SAM3D_API = "https://modelscope.cn/api/v1/models/facebook/sam-3d-body-dinov3"


def _file_size_mb(path: Path) -> float | None:
    """Size in MB if `path` is a regular file, else None (a single `stat` call)."""
    try:
        st = path.stat()
    except OSError:
        return None
    return round(st.st_size / (1024 * 1024), 2) if stat.S_ISREG(st.st_mode) else None


def check_models_status() -> dict[str, Any]:
    """Return presence and byte sizes of all required public and gated model files."""
    public = {
        "yolox_l": YOLOX_CKPT_PATH,
        "flowhmr_ckpt": FLOWHMR_CKPT_PATH,
        "flowhmr_cfg": FLOWHMR_CFG_PATH,
        "motion_stats": MOTION_STATS_PATH,
    }
    sam3d = {"sam3d_ckpt": SAM3D_CKPT_PATH, "sam3d_cfg": SAM3D_CFG_PATH, "sam3d_mhr": SAM3D_MHR_PATH}
    items = {}
    for key, path in {**public, "j_regressor": J_REGRESSOR_PATH, **sam3d}.items():
        size_mb = _file_size_mb(path)
        items[key] = {"path": str(path), "exists": size_mb is not None, "size_mb": size_mb or 0.0}
    public_ready = all(items[k]["exists"] for k in public)
    sam3d_ready = all(items[k]["exists"] for k in sam3d)
    return {
        "models_dir": str(MODELS_DIR),
        "public_ready": public_ready,
        "sam3d_ready": sam3d_ready,
        "ready_for_live_camera": public_ready and sam3d_ready,
        "files": items,
    }


def _sam3d_repo_files() -> list[str]:
    """Repo-relative paths of the SAM 3D Body files the runtime needs."""
    return [p.relative_to(SAM3D_DIR).as_posix() for p in SAM3D_FILES]


def _download_sam3d_from_modelscope(dest_dir: Path) -> None:
    """Download `facebook/sam-3d-body-dinov3` from ModelScope with size + SHA256 verification."""
    headers = {"User-Agent": _USER_AGENT}
    req = urllib.request.Request(f"{_MODELSCOPE_SAM3D_API}/repo/files?Recursive=true", headers=headers)
    with urllib.request.urlopen(req, timeout=30) as resp:
        payload = json.loads(resp.read().decode("utf-8"))

    wanted = {*_sam3d_repo_files(), "LICENSE", "README.md", "configuration.json"}
    blobs = [
        f
        for f in payload.get("Data", {}).get("Files", [])
        if f.get("Type") == "blob" and f.get("Path") in wanted
    ]
    if not blobs:
        raise RuntimeError("ModelScope file listing returned no matching files for facebook/sam-3d-body-dinov3")

    for meta in blobs:
        rel_path = str(meta["Path"])
        expected_size = int(meta.get("Size") or 0)
        expected_sha = str(meta.get("Sha256") or "").lower()
        dst = dest_dir / rel_path
        if dst.is_file() and expected_size > 0 and dst.stat().st_size == expected_size:
            continue

        dst.parent.mkdir(parents=True, exist_ok=True)
        part = dst.with_suffix(dst.suffix + ".part")
        q_path = urllib.parse.quote(rel_path, safe="")
        dl_req = urllib.request.Request(f"{_MODELSCOPE_SAM3D_API}/repo?Revision=master&FilePath={q_path}", headers=headers)

        hasher = hashlib.sha256()
        downloaded = 0
        t0 = time.perf_counter()
        last_log = t0
        size_mb = expected_size / (1024 * 1024)
        print(f"[ModelScope] Downloading {rel_path} ({size_mb:.2f} MB)...", file=sys.stderr, flush=True)

        with urllib.request.urlopen(dl_req, timeout=120) as r, open(part, "wb") as out:
            while True:
                chunk = r.read(8 * 1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)
                hasher.update(chunk)
                downloaded += len(chunk)
                now = time.perf_counter()
                if expected_size > 10 * 1024 * 1024 and (now - last_log) >= 2.0:
                    pct = (downloaded / expected_size * 100.0) if expected_size else 0.0
                    mb_s = (downloaded / (1024 * 1024)) / max(now - t0, 1e-3)
                    print(
                        f"  {rel_path}: {downloaded / (1024 * 1024):.1f}/{size_mb:.1f} MB "
                        f"({pct:.1f}%, {mb_s:.1f} MB/s)",
                        file=sys.stderr,
                        flush=True,
                    )
                    last_log = now

        if expected_size > 0 and downloaded != expected_size:
            part.unlink(missing_ok=True)
            raise RuntimeError(f"Incomplete download for {rel_path}: got {downloaded}, expected {expected_size}")
        actual_sha = hasher.hexdigest().lower()
        if expected_sha and actual_sha != expected_sha:
            part.unlink(missing_ok=True)
            raise RuntimeError(f"SHA256 mismatch for {rel_path}: got {actual_sha}, expected {expected_sha}")
        part.replace(dst)
        print(f"[ModelScope] Verified {rel_path} ({size_mb:.2f} MB, sha256 ok)", file=sys.stderr, flush=True)


def download_missing_models(hf_token: str | None = None) -> dict[str, Any]:
    """Download missing public checkpoints and `facebook/sam-3d-body-dinov3` (ModelScope, or Hugging Face with a token)."""
    bootstrap_runtime()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Public YOLOX-L + FlowHMR weights
    if not YOLOX_CKPT_PATH.is_file():
        YOLOX_CKPT_PATH.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(
            "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_l.pth",
            str(YOLOX_CKPT_PATH),
        )

    for rel, dst in (
        ("checkpoints/flowhmr_latest/flowhmr_latest.ckpt", FLOWHMR_CKPT_PATH),
        ("checkpoints/flowhmr_latest/config.yml", FLOWHMR_CFG_PATH),
        ("assets/body_models/smpl_neutral_J_regressor.pt", J_REGRESSOR_PATH),
        ("assets/motion_stats.json", MOTION_STATS_PATH),
    ):
        if not dst.is_file():
            dst.parent.mkdir(parents=True, exist_ok=True)
            # Stream the cached file (no multi-GB read into memory) into a `.part` renamed into place, so an
            # interrupted copy never leaves a truncated checkpoint that later runs would accept.
            part = dst.with_suffix(dst.suffix + ".part")
            shutil.copyfile(hf_hub_download(repo_id="FlowHMR/FlowHMR", filename=rel, token=hf_token), part)
            part.replace(dst)

    # 2. Meta SAM 3D Body (`facebook/sam-3d-body-dinov3`)
    sam_error: str | None = None
    if not is_sam3d_available():
        SAM3D_DIR.mkdir(parents=True, exist_ok=True)
        if not hf_token:
            try:
                _download_sam3d_from_modelscope(SAM3D_DIR)
            except Exception as exc:
                sam_error = f"ModelScope download failed: {exc}"
        else:
            try:
                snapshot_download(
                    repo_id="facebook/sam-3d-body-dinov3",
                    local_dir=str(SAM3D_DIR),
                    allow_patterns=_sam3d_repo_files(),
                    token=hf_token,
                )
            except (GatedRepoError, HfHubHTTPError) as exc:
                try:
                    _download_sam3d_from_modelscope(SAM3D_DIR)
                except Exception as ms_exc:
                    sam_error = f"Hugging Face ({exc}) and ModelScope fallback ({ms_exc}) both failed."

    status = check_models_status()
    status["sam3d_download_error"] = sam_error
    return status
