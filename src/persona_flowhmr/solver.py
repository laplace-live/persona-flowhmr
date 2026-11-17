"""Compiled TF32 FlowHMR MMDiT sliding-window Euler ODE solver.

Loads `HunyuanMotionMMDiT` directly from `flowhmr_latest.ckpt` + `config.yml` without
instantiating `SMPLSkeleton`/`SMPLMesh` (eliminating `smplh/neutral/model.npz`), and uses
frame-indexed deterministic initial Gaussian noise so overlapping windows remain coherent.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
import yaml

from .config import SAM_TOKEN_DIM
from .paths import FLOWHMR_CFG_PATH, FLOWHMR_CKPT_PATH, bootstrap_runtime


class FlowHMRWindowSolver:
    """Sliding-window FlowHMR MMDiT solver optimized for RTX PRO 6000 Blackwell (TF32 + Triton)."""

    def __init__(
        self,
        ckpt_path: str | Path = FLOWHMR_CKPT_PATH,
        cfg_path: str | Path = FLOWHMR_CFG_PATH,
        device: str | torch.device = "cuda",
        use_compile: bool = True,
        base_seed: int = 42,
    ) -> None:
        bootstrap_runtime()
        ckpt_file = Path(ckpt_path)
        cfg_file = Path(cfg_path)
        if not ckpt_file.is_file():
            raise FileNotFoundError(f"FlowHMR checkpoint missing: {ckpt_file}")
        if not cfg_file.is_file():
            raise FileNotFoundError(f"FlowHMR config missing: {cfg_file}")

        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision("high")

        with open(cfg_file, encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)

        pipe_args = self.cfg.get("train_pipeline_args", {})
        self.motion_style: str = str(pipe_args.get("motion_style", "basic"))
        self.pred_type: str = str(pipe_args.get("pred_type", "x1"))
        self.default_steps: int = int(
            pipe_args.get("infer_noise_scheduler_cfg", {}).get("validation_steps", 20)
        )

        from flowhmr.core.framework.loaders import load_object
        from flowhmr.core.framework.net_utils import select_state_dict_by_prefix

        net_args = dict(self.cfg["network_module_args"])
        net_args["gradient_checkpointing"] = False
        model = load_object(self.cfg["network_module"], net_args)

        raw = torch.load(str(ckpt_file), map_location="cpu", weights_only=False)
        sd = raw.get("model_state_dict", raw.get("state_dict", raw.get("model", raw)))
        mt_sd = (
            select_state_dict_by_prefix(sd, "motion_transformer.")
            or select_state_dict_by_prefix(sd, "module.motion_transformer.")
            or sd
        )

        # Unexpected keys are tolerated: an unprefixed raw checkpoint also carries non-transformer buffers.
        missing, _unexpected = model.load_state_dict(mt_sd, strict=False)
        if missing:
            raise RuntimeError(f"FlowHMR MMDiT missing state_dict keys: {missing[:5]}")

        self.device = torch.device(device)
        model.eval()
        self.eager_model = model.to(self.device, dtype=torch.float32)
        self.latent_dim = int(self.eager_model.motion_input_dim)
        self.vtxt_dim = int(self.eager_model.vtxt_input_dim)
        self.base_seed = int(base_seed)

        self._use_compile = bool(use_compile and self.device.type == "cuda")
        self._compiled_forward = None
        if self._use_compile:
            try:
                self._compiled_forward = torch.compile(
                    self.eager_model.forward,
                    dynamic=False,
                    options={"triton.cudagraphs": False},
                )
            except Exception:
                self._compiled_forward = None

        self._noise_cache: dict[int, torch.Tensor] = {}
        self.last_solve_ms: float = 0.0

    def reset_noise_cache(self) -> None:
        self._noise_cache.clear()

    def _noise_for_frame_indices(self, frame_indices: list[int]) -> torch.Tensor:
        """Build `(1, W, 491)` initial noise where each stream frame index has fixed Gaussian noise."""
        rows: list[torch.Tensor] = []
        for idx in frame_indices:
            vec = self._noise_cache.get(idx)
            if vec is None:
                gen = torch.Generator(device="cpu")
                # Keep seed positive within uint64 range even for negative padded history indices.
                gen.manual_seed((self.base_seed * 1_000_003 + (idx + 1_000_000)) & 0x7FFF_FFFF_FFFF)
                vec = torch.randn((self.latent_dim,), generator=gen, dtype=torch.float32)
                self._noise_cache[idx] = vec
            rows.append(vec)
        # Bound cache size over multi-hour live camera sessions.
        if len(self._noise_cache) > 2048 and frame_indices:
            cutoff = min(frame_indices) - 256
            stale = [k for k in self._noise_cache if k < cutoff]
            for k in stale:
                self._noise_cache.pop(k, None)
        return torch.stack(rows, dim=0).unsqueeze(0).to(self.device, non_blocking=True)

    @torch.no_grad()
    def warmup(self, window_frames: int, steps: int = 2) -> float:
        """Run a dummy forward pass to trigger Triton kernel compilation up front."""
        dummy_tokens = torch.zeros((window_frames, SAM_TOKEN_DIM), dtype=torch.float32)
        t0 = time.perf_counter()
        self.solve_window(dummy_tokens, list(range(window_frames)), ode_steps=max(2, steps))
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        return (time.perf_counter() - t0) * 1000.0

    @torch.no_grad()
    def solve_window(
        self,
        sam_tokens_w_3072: torch.Tensor,
        frame_indices: list[int],
        *,
        ode_steps: int | None = None,
        camera_r_w_9: np.ndarray | torch.Tensor | None = None,
    ) -> np.ndarray:
        """Solve a `(W, 3072)` conditioning window into `(W, 491)` float32 normalized motion latents."""
        w_len = int(sam_tokens_w_3072.shape[0])
        if len(frame_indices) != w_len:
            raise ValueError(f"frame_indices length {len(frame_indices)} != window {w_len}")
        steps = int(ode_steps or self.default_steps)
        if steps < 1:
            raise ValueError(f"ode_steps must be >= 1, got {steps}")

        t_start = time.perf_counter()
        feature = sam_tokens_w_3072.unsqueeze(0).to(self.device, dtype=torch.float32, non_blocking=True)
        if camera_r_w_9 is None:
            cam_r = torch.eye(3, device=self.device, dtype=torch.float32).reshape(1, 1, 9).expand(1, w_len, 9).contiguous()
        else:
            cam_r = torch.as_tensor(camera_r_w_9, dtype=torch.float32).reshape(1, w_len, 9).to(
                self.device, dtype=torch.float32, non_blocking=True
            )
        ctxt_input = {"feature": feature, "camera_R": cam_r}
        vtxt_input = torch.zeros((1, 1, self.vtxt_dim), device=self.device, dtype=torch.float32)
        mask = torch.ones((1, w_len), device=self.device, dtype=torch.bool)

        y0 = self._noise_for_frame_indices(frame_indices)
        x = y0.clone()

        forward_fn = self._compiled_forward or self.eager_model.forward
        ts = torch.linspace(0.0, 1.0, steps + 1, device=self.device, dtype=torch.float32)
        t_eps = 1.0 / float(steps)

        for i in range(steps):
            t_cur = ts[i]
            dt = ts[i + 1] - t_cur
            inputs = dict(
                x=x,
                ctxt_input=ctxt_input,
                vtxt_input=vtxt_input,
                timesteps=t_cur.expand(1),
                x_mask_temporal=mask,
                ctxt_mask_temporal=mask,
            )
            try:
                x_pred = forward_fn(**inputs)
            except Exception:
                # Graceful eager fallback if torch.compile encounters an unexpected shape/backend error.
                self._compiled_forward = None
                forward_fn = self.eager_model.forward
                x_pred = forward_fn(**inputs)

            if self.pred_type == "velocity":
                vel = x_pred
            elif self.pred_type == "x1":
                denom = torch.clamp(1.0 - t_cur, min=t_eps)
                vel = (x_pred - x) / denom
            elif self.pred_type == "x1raw":
                vel = x_pred - y0
            else:
                raise NotImplementedError(f"Unsupported pred_type: {self.pred_type}")

            x = x + dt * vel

        out_np = x[0].detach().cpu().numpy().astype(np.float32)
        self.last_solve_ms = (time.perf_counter() - t_start) * 1000.0
        return out_np
