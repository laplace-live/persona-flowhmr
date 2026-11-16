# Licensing & Third-Party Notices

## 1. Primary Project License (`AGPL-3.0`)

Original code in `persona-flowhmr` (`src/persona_flowhmr/`, `tests/`, packaging, and configuration) is licensed under the **GNU Affero General Public License v3.0 (`AGPL-3.0`)** — see [LICENSE](LICENSE).

### AGPL-3.0 Section 7 Additional Permission (Third-Party Model & Vision Runtime Exception)

As a special exception under **Section 7 of the GNU Affero General Public License v3.0**, the copyright holders of `persona-flowhmr` grant you additional permission to link, import, combine, and distribute the AGPL-3.0-covered portions of `persona-flowhmr` alongside the following third-party components under their respective upstream license terms, and to convey the resulting combined work without applying the terms of the GNU AGPLv3 to those third-party components themselves:

1. **Meta SAM 3D Body (`vendor/sam-3d-body/` and `facebook/sam-3d-body-dinov3` model weights)** — governed solely by the **SAM License** ([vendor/sam-3d-body/LICENSE](vendor/sam-3d-body/LICENSE)).
2. **Meta DINOv3 (`vendor/dinov3/`)** — governed solely by the **DINOv3 License Agreement** ([vendor/dinov3/LICENSE.md](vendor/dinov3/LICENSE.md)).

If you modify `persona-flowhmr`, you may extend this exception to your version of the software, but you are not obligated to do so. If you do not wish to do so, delete this exception statement from your version.

---

## 2. Vendored Third-Party Directories (`vendor/`)

Each subdirectory under `vendor/` remains governed by its own upstream license included in that directory:

| Directory | Upstream Project | License | Local Modifications |
| --- | --- | --- | --- |
| `vendor/flowhmr/` | [`flowhmr/flowhmr`](https://github.com/flowhmr/flowhmr) | Apache License 2.0 ([LICENSE](vendor/flowhmr/LICENSE)) | Modified `utils/lite_sam_infer.py` to support body-only/fast-hand modes (`hand_mode`), configurable decoder layer depth, and keypoints-only MHR execution for real-time streaming. |
| `vendor/YOLOX/` | [`Megvii-BaseDetection/YOLOX`](https://github.com/Megvii-BaseDetection/YOLOX) | Apache License 2.0 ([LICENSE](vendor/YOLOX/LICENSE)) | Minimal inference subset vendored without modification. |
| `vendor/sam-3d-body/` | [`facebookresearch/sam-3d-body`](https://github.com/facebookresearch/sam-3d-body) | Meta SAM License ([LICENSE](vendor/sam-3d-body/LICENSE)) | Modified `promptable_decoder.py` (`_max_layers`), `mhr_head.py` (`_keypoints_only` / `return_vertices`), and `sam3d_body.py` (on-GPU `get_ray_condition` broadcast) for low-latency inference. Distributed under the terms of the SAM License. |
| `vendor/dinov3/` | [`facebookresearch/dinov3`](https://github.com/facebookresearch/dinov3) | Meta DINOv3 License ([LICENSE.md](vendor/dinov3/LICENSE.md)) | Unmodified vision backbone subset distributed under the terms of the DINOv3 License Agreement. |

---

## 3. Downloaded Model Checkpoints (`models/`, not tracked in Git)

Model weights downloaded at runtime via `persona-flowhmr setup` are subject to their respective upstream terms:

- **`flowhmr_latest.ckpt` & `motion_stats.json`** ([`flowhmr/flowhmr`](https://github.com/flowhmr/flowhmr)): Apache License 2.0.
- **`yolox_l.pth`** ([`Megvii-BaseDetection/YOLOX`](https://github.com/Megvii-BaseDetection/YOLOX)): Apache License 2.0.
- **`facebook/sam-3d-body-dinov3` (`model.ckpt` & `mhr_model.pt`)**: Meta SAM License ([vendor/sam-3d-body/LICENSE](vendor/sam-3d-body/LICENSE)).
