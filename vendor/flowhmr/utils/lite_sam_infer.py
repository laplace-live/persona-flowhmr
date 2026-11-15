"""Feature-only SAM scheduling using the unchanged upstream model and weights.

Body inference supplies the hand boxes. Hand inference stops at the decoder
tokens consumed by FlowHMR, before the unused fusion and refinement stages.
"""

from contextlib import nullcontext

import numpy as np
import torch


class _HandTokensReady(Exception):
    """Local control flow: the hand decoder has produced its conditioning token."""


class _BodyTokensReady(Exception):
    """Local control flow: body-only fast mode has produced its conditioning token."""


def _combine_batches(batches, device):
    # Each frame occupies the batch axis, with its own intrinsics. The person
    # axis stays at one even when left and right crops share a forward pass.
    return {
        key: torch.cat([batch[key] for batch in batches], dim=0).to(device)
        for key, value in batches[0].items()
        if isinstance(value, torch.Tensor)
    }


@torch.no_grad()
def extract_tokens(
    estimator,
    images,
    boxes,
    cameras,
    *,
    gpu_preprocess=False,
    merge_hands=False,
    hand_stride=1,
    hand_mode="full",
    neutral_hand_tokens=None,
    body_decoder_layers=None,
    hand_decoder_layers=None,
):
    """Return CPU float32 ``(frames, 3072)`` tokens in body/left/right order.

    ``images`` contains RGB uint8 frames, one selected person per frame.
    ``hand_mode`` supports:
      - ``"full"``: full body + hand crops (subsampled by ``hand_stride``).
      - ``"fast"``: truncated decoder layers + subsampled merged hand crops.
      - ``"off"``: body-only inference stopping immediately after ``model.decoder``
        and filling hand conditioning slots with ``neutral_hand_tokens`` (or zeros).
    """
    if not images or len(images) != len(boxes) or len(images) != len(cameras):
        raise ValueError("Expected equally sized, nonempty images, boxes and cameras")
    if hand_mode not in ("full", "fast", "off"):
        raise ValueError(f"Unsupported hand_mode: {hand_mode}")
    boxes = torch.as_tensor(boxes).detach().cpu().numpy().reshape(-1, 4)
    cameras = torch.as_tensor(cameras).detach().cpu().reshape(-1, 3, 3)
    if len(boxes) != len(images) or len(cameras) != len(images):
        raise ValueError("Expected one box and one camera per frame")
    if not np.isfinite(boxes).all() or not torch.isfinite(cameras).all():
        raise ValueError("Boxes and cameras must be finite")
    for image in images:
        if not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Expected RGB uint8 HWC images")

    from sam_3d_body.data.utils.prepare_batch import prepare_batch

    device = torch.device(estimator.device)
    model = estimator.model
    body_tokens, hand_tokens = [], []
    original_single = (
        len(images) == 1
        and not gpu_preprocess
        and not merge_hands
        and hand_mode == "full"
        and body_decoder_layers is None
        and hand_decoder_layers is None
    )

    def capture_body(_module, _args, output):
        body_tokens.append(output[0].detach() if original_single else output[0][:, 0].detach().clone())
        if hand_mode == "off":
            raise _BodyTokensReady

    def capture_hand(_module, _args, output):
        hand_tokens.append(output[0].detach() if original_single else output[0][:, 0].detach().clone())
        if not original_single or len(hand_tokens) == 2:
            raise _HandTokensReady

    def collected_tokens():
        if len(body_tokens) != 1 or len(hand_tokens) != 2:
            raise RuntimeError("SAM did not produce body, left hand and right hand tokens")
        pieces = [body_tokens[0], *hand_tokens]
        if original_single:
            pieces = [piece[:, 0] for piece in pieces]
        tokens = torch.cat(pieces, dim=-1)
        if tokens.shape != (len(images), 3072):
            raise RuntimeError(f"Unexpected SAM token shape: {tuple(tokens.shape)}")
        return tokens.to(device="cpu", dtype=torch.float32)

    def prepare(image, box, camera, transform):
        if gpu_preprocess:
            from .sam_gpu_preprocess import prepare_gpu_batch
            return prepare_gpu_batch(image, box[None], camera[None], transform)
        return prepare_batch(image, transform, box[None], cam_int=camera[None])

    context = torch.cuda.device(device) if device.type == "cuda" else nullcontext()
    state_names = ("_batch_size", "_max_num_person", "_person_valid", "body_batch_idx", "hand_batch_idx")
    state = {name: getattr(model, name) for name in state_names if hasattr(model, name)}
    prev_skip_body = getattr(model.decoder, "_skip_final_pose_output", None)
    prev_skip_hand = getattr(model.decoder_hand, "_skip_final_pose_output", None)
    prev_max_body = getattr(model.decoder, "_max_layers", None)
    prev_max_hand = getattr(model.decoder_hand, "_max_layers", None)
    prev_kp_body = getattr(model.head_pose, "_keypoints_only", None)
    prev_kp_hand = getattr(model.head_pose_hand, "_keypoints_only", None)
    handles = []
    try:
        with context:
            if not original_single:
                model.decoder._skip_final_pose_output = True
                model.decoder_hand._skip_final_pose_output = True
                model.head_pose._keypoints_only = True
                model.head_pose_hand._keypoints_only = True
                if body_decoder_layers is not None:
                    model.decoder._max_layers = int(body_decoder_layers)
                if hand_decoder_layers is not None:
                    model.decoder_hand._max_layers = int(hand_decoder_layers)
            handles.append(model.decoder.register_forward_hook(capture_body))
            if hand_mode != "off":
                handles.append(model.decoder_hand.register_forward_hook(capture_hand))
            if original_single:
                try:
                    estimator.process_one_image(
                        images[0], bboxes=boxes, cam_int=cameras, inference_type="full",
                    )
                except _HandTokensReady:
                    pass
                return collected_tokens()

            frames = (
                [torch.as_tensor(np.ascontiguousarray(img), device=device) for img in images]
                if gpu_preprocess
                else images
            )
            body_batch = _combine_batches(
                [
                    prepare(image, box, camera, estimator.transform)
                    for image, box, camera in zip(frames, boxes, cameras)
                ],
                device,
            )
            model._initialize_batch(body_batch)

            if hand_mode == "off":
                try:
                    model.forward_step(body_batch, decoder_type="body")
                except _BodyTokensReady:
                    pass
                del body_batch
                if len(body_tokens) != 1:
                    raise RuntimeError("SAM body decoder did not produce conditioning tokens")
                b_cpu = body_tokens[0].to(device="cpu", dtype=torch.float32)
                n_imgs = len(images)
                if neutral_hand_tokens is not None:
                    nh = torch.as_tensor(neutral_hand_tokens, dtype=torch.float32, device="cpu").reshape(1, 2048)
                    h_cpu = nh.expand(n_imgs, 2048)
                else:
                    h_cpu = torch.zeros((n_imgs, 2048), dtype=torch.float32)
                return torch.cat([b_cpu, h_cpu], dim=-1)

            body = model.forward_step(body_batch, decoder_type="body")
            geometry = {"affine_trans": body_batch["affine_trans"].flatten(0, 1).unsqueeze(0)}
            left_boxes, right_boxes = model._get_hand_box(body, geometry)
            del body, body_batch

            def run_hands(indices, sides):
                batches = []
                for side in sides:
                    for index in indices:
                        image = frames[index]
                        box = (left_boxes if side == "left" else right_boxes)[index].copy()
                        if side == "left":
                            box[[0, 2]] = image.shape[1] - box[[2, 0]] - 1
                            image = image.flip(1) if gpu_preprocess else image[:, ::-1]
                        batches.append(prepare(image, box, cameras[index], estimator.transform_hand))
                batch = _combine_batches(batches, device)
                model._initialize_batch(batch)
                try:
                    model.forward_step(batch, decoder_type="hand")
                except _HandTokensReady:
                    pass
                else:
                    raise RuntimeError("SAM hand decoder did not produce conditioning tokens")

            n_imgs = len(images)
            h_stride = max(1, int(hand_stride))
            if h_stride > 1 and n_imgs >= 2:
                m_hands = max(1, (n_imgs + h_stride - 1) // h_stride)
                hand_indices = (
                    [0]
                    if m_hands == 1
                    else np.linspace(0, n_imgs - 1, num=m_hands, dtype=int).tolist()
                )
            else:
                hand_indices = list(range(n_imgs))

            if merge_hands:
                run_hands(hand_indices, ("left", "right"))
                hand_tokens = list(hand_tokens.pop().split(len(hand_indices), dim=0))
            else:
                run_hands(hand_indices, ("left",))
                run_hands(hand_indices, ("right",))

            if len(hand_indices) < n_imgs:
                interp_sides = []
                for side_tok in hand_tokens:
                    if len(hand_indices) == 1:
                        interp_sides.append(side_tok.expand(n_imgs, -1).clone())
                        continue
                    full_side = torch.empty((n_imgs, side_tok.shape[-1]), device=side_tok.device, dtype=side_tok.dtype)
                    seg = 0
                    for idx in range(n_imgs):
                        while seg + 1 < len(hand_indices) - 1 and idx > hand_indices[seg + 1]:
                            seg += 1
                        i0, i1 = hand_indices[seg], hand_indices[seg + 1]
                        if idx <= i0 or i1 == i0:
                            full_side[idx] = side_tok[seg]
                        elif idx >= i1:
                            full_side[idx] = side_tok[seg + 1]
                        else:
                            alpha = float(idx - i0) / float(i1 - i0)
                            full_side[idx] = torch.lerp(side_tok[seg], side_tok[seg + 1], alpha)
                    interp_sides.append(full_side)
                hand_tokens = interp_sides

            return collected_tokens()
    finally:
        for handle in handles:
            handle.remove()

        def _restore_attr(obj, attr, prev):
            if prev is None:
                if hasattr(obj, attr):
                    delattr(obj, attr)
            else:
                setattr(obj, attr, prev)

        _restore_attr(model.decoder, "_skip_final_pose_output", prev_skip_body)
        _restore_attr(model.decoder_hand, "_skip_final_pose_output", prev_skip_hand)
        _restore_attr(model.decoder, "_max_layers", prev_max_body)
        _restore_attr(model.decoder_hand, "_max_layers", prev_max_hand)
        _restore_attr(model.head_pose, "_keypoints_only", prev_kp_body)
        _restore_attr(model.head_pose_hand, "_keypoints_only", prev_kp_hand)
        for name in state_names:
            if name in state:
                setattr(model, name, state[name])
            elif hasattr(model, name):
                delattr(model, name)
