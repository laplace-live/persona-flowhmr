"""Model checkpoint verification and setup utility (`persona-flowhmr setup`)."""

from __future__ import annotations

import hashlib
import json
import sys
import time
import urllib.parse
import urllib.request
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
    SAM3D_CFG_PATH,
    SAM3D_CKPT_PATH,
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


def _download_sam3d_from_modelscope(dest_dir: Path) -> None:
    """Download `facebook/sam-3d-body-dinov3` from ModelScope with size + SHA256 verification."""
    list_url = "https://modelscope.cn/api/v1/models/facebook/sam-3d-body-dinov3/repo/files?Recursive=true"
    req = urllib.request.Request(list_url, headers={"User-Agent": "persona-flowhmr/0.1"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        payload = json.loads(resp.read().decode("utf-8"))

    wanted = {"model_config.yaml", "LICENSE", "README.md", "configuration.json", "assets/mhr_model.pt", "model.ckpt"}
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
        dl_url = f"https://modelscope.cn/api/v1/models/facebook/sam-3d-body-dinov3/repo?Revision=master&FilePath={q_path}"
        dl_req = urllib.request.Request(dl_url, headers={"User-Agent": "persona-flowhmr/0.1"})

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


def download_missing_models(hf_token: str | None = None, prefer_modelscope: bool = True) -> dict[str, Any]:
    """Download missing public checkpoints and `facebook/sam-3d-body-dinov3` (ModelScope or Hugging Face)."""
    bootstrap_runtime()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Public YOLOX-L + FlowHMR weights
    if not YOLOX_CKPT_PATH.is_file():
        YOLOX_CKPT_PATH.parent.mkdir(parents=True, exist_ok=True)
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

    # 2. Meta SAM 3D Body (`facebook/sam-3d-body-dinov3`)
    sam_error: str | None = None
    if not (SAM3D_CKPT_PATH.is_file() and SAM3D_CFG_PATH.is_file() and SAM3D_MHR_PATH.is_file()):
        SAM3D_DIR.mkdir(parents=True, exist_ok=True)
        if prefer_modelscope and not hf_token:
            try:
                _download_sam3d_from_modelscope(SAM3D_DIR)
            except Exception as exc:
                sam_error = f"ModelScope download failed: {exc}"
        else:
            try:
                snapshot_download(
                    repo_id="facebook/sam-3d-body-dinov3",
                    local_dir=str(SAM3D_DIR),
                    allow_patterns=["model.ckpt", "model_config.yaml", "assets/mhr_model.pt"],
                    token=hf_token or True,
                )
            except (GatedRepoError, HfHubHTTPError) as exc:
                try:
                    _download_sam3d_from_modelscope(SAM3D_DIR)
                except Exception as ms_exc:
                    sam_error = f"Hugging Face ({exc}) and ModelScope fallback ({ms_exc}) both failed."

    status = check_models_status()
    status["sam3d_download_error"] = sam_error
    return status
