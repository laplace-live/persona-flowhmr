"""Model setup: checkpoint installs must never leave a truncated file where a later run would accept it."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from persona_flowhmr import setup_models
from persona_flowhmr.paths import resolve_models_dir


def test_interrupted_checkpoint_copy_leaves_nothing_in_place(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    models = tmp_path / "models"
    present = tmp_path / "present.bin"
    present.write_bytes(b"ok")
    ckpt = models / "flowhmr_latest" / "flowhmr_latest.ckpt"
    cached = tmp_path / "hf-cache.ckpt"
    cached.write_bytes(b"x" * 4096)

    monkeypatch.setattr(setup_models, "MODELS_DIR", models)
    for name in ("YOLOX_CKPT_PATH", "FLOWHMR_CFG_PATH", "J_REGRESSOR_PATH"):
        monkeypatch.setattr(setup_models, name, present)
    monkeypatch.setattr(setup_models, "FLOWHMR_CKPT_PATH", ckpt)
    monkeypatch.setattr(setup_models, "hf_hub_download", lambda **_kw: str(cached))

    def copy_until_disk_full(src: str, dst: Path) -> None:
        Path(dst).write_bytes(Path(src).read_bytes()[:1000])
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(setup_models, "shutil", SimpleNamespace(copyfile=copy_until_disk_full))

    with pytest.raises(OSError):
        setup_models.download_missing_models()
    assert not ckpt.exists()  # a retry downloads it again instead of trusting a partial checkpoint


def test_resolve_models_dir_discovers_main_repo_models_from_git_worktree(tmp_path: Path) -> None:
    main_repo = tmp_path / "main-repo"
    main_models = main_repo / "models"
    main_models.mkdir(parents=True)
    (main_models / "motion_stats.json").write_text("{}", encoding="utf-8")

    wt_git_admin = main_repo / ".git" / "worktrees" / "feature-wt"
    wt_git_admin.mkdir(parents=True)
    (wt_git_admin / "commondir").write_text("../..\n", encoding="utf-8")

    worktree = tmp_path / "orca-workspaces" / "feature-wt"
    worktree.mkdir(parents=True)
    (worktree / ".git").write_text(f"gitdir: {wt_git_admin.as_posix()}\n", encoding="utf-8")

    # Fresh worktree with no local `models/` resolves automatically to `main_repo / "models"`.
    assert resolve_models_dir(worktree, env_override="") == main_models.resolve()

    # Explicit environment override still takes precedence when set.
    custom = tmp_path / "custom-models"
    assert resolve_models_dir(worktree, env_override=str(custom)) == custom.resolve()
