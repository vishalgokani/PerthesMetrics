"""2D segmentation Grad-CAM with explicit window and mirror aggregation."""

from __future__ import annotations

from itertools import combinations, product
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFilter
from torch import nn
from torch.nn import functional as F


def resolve_layer(network: nn.Module, name: str) -> tuple[str, nn.Module]:
    if name == "auto":
        stages = getattr(getattr(network, "decoder", None), "stages", None)
        if stages is None or len(stages) < 3:
            raise ValueError("Cannot select decoder layer automatically; supply --gradcam-layer.")
        name = f"decoder.stages.{len(stages) - 3}"
    modules = dict(network.named_modules())
    if name not in modules:
        raise ValueError(f"Unknown Grad-CAM layer: {name}. Select a name from network.named_modules().")
    return name, modules[name]


class SegGradCAM:
    """Differentiate class scores at a layer without retaining the encoder graph."""

    def __init__(self, network: nn.Module, layer: nn.Module):
        self.network = network
        self.layer = layer
        self.activation = None

    def __enter__(self):
        self.training = [(module, module.training) for module in self.network.modules()]
        self.requires_grad = [parameter.requires_grad for parameter in self.network.parameters()]
        self.network.eval()
        self.network.requires_grad_(False)
        self.handle = self.layer.register_forward_hook(self._capture)
        return self

    def _capture(self, module, inputs, output):
        if not isinstance(output, torch.Tensor) or output.ndim != 4:
            raise ValueError("Grad-CAM requires a layer with a B,C,H,W tensor output.")
        self.activation = output.detach().requires_grad_(True)
        return self.activation

    def __exit__(self, *exc):
        self.handle.remove()
        self.activation = None
        for parameter, requires_grad in zip(self.network.parameters(), self.requires_grad):
            parameter.requires_grad_(requires_grad)
        for module, training in self.training:
            module.training = training

    def compute(self, image: torch.Tensor, targets: dict[int, torch.Tensor]):
        """Targets contain spatial weights for contributions to a global mean logit."""
        if not targets:
            return {}, {}
        with torch.enable_grad():
            logits = self.network(image)
            if not isinstance(logits, torch.Tensor) or logits.ndim != 4:
                raise ValueError("Expected a single 2D segmentation output; disable deep supervision.")
            if self.activation is None:
                raise ValueError("Selected Grad-CAM layer was not executed.")
            maps, norms = {}, {}
            for index, (class_id, weights) in enumerate(targets.items()):
                if class_id <= 0 or class_id >= logits.shape[1]:
                    raise ValueError(f"Invalid foreground class ID: {class_id}")
                if weights.shape != logits.shape[-2:] or not torch.isfinite(weights).all() or (weights < 0).any():
                    raise ValueError("Target weights must be finite, nonnegative, and match the output shape.")
                score = (logits[0, class_id].float() * weights).sum()
                gradients = torch.autograd.grad(
                    score, self.activation, retain_graph=index < len(targets) - 1,
                )[0]
                if not torch.isfinite(gradients).all():
                    raise RuntimeError(f"Nonfinite Grad-CAM gradients for class {class_id}.")
                weights_channel = gradients.float().mean(dim=(2, 3), keepdim=True)
                cam = (weights_channel * self.activation.detach().float()).sum(dim=1, keepdim=True).relu()
                cam = F.interpolate(cam, size=image.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
                if not torch.isfinite(cam).all():
                    raise RuntimeError(f"Nonfinite Grad-CAM map for class {class_id}.")
                maps[class_id] = cam.detach().cpu().numpy()
                norms[class_id] = float(gradients.abs().max().detach().cpu())
            self.activation = None
            return maps, norms


def aggregate_case(network, layer, data, target_labels, classes, patch_size, mirror_axes, device):
    """Return unnormalized maps for one fold in preprocessed C,D,H,W geometry.

    Window target weights include Gaussian/coverage and global class pixel count.
    ReLU CAMs are unflipped, averaged across mirror branches, then Gaussian blended.
    This is an aggregation of branch CAMs, not an exact ensemble attribution.
    """
    from acvl_utils.cropping_and_padding.padding import pad_nd_image
    from nnunetv2.inference.sliding_window_prediction import compute_gaussian, compute_steps_for_sliding_window

    if data.ndim != 4 or data.shape[1] != 1 or target_labels.shape != data.shape[1:]:
        raise ValueError("Expected one 2D radiograph with C,1,H,W data and 1,H,W targets.")
    if len(patch_size) != 2 or any(axis not in (0, 1) for axis in mirror_axes):
        raise ValueError("Grad-CAM supports only 2D patches and mirror axes 0 and 1.")
    padded, undo = pad_nd_image(data[:, 0], patch_size, "constant", {"constant_values": 0}, True)
    labels = pad_nd_image(target_labels[0], patch_size, "constant", {"constant_values": 0})
    steps = compute_steps_for_sliding_window(padded.shape[-2:], patch_size, 0.5)
    windows = [tuple(slice(start, start + size) for start, size in zip(starts, patch_size))
               for starts in product(*steps)]
    # Match nnU-Net's positive Gaussian weights, then accumulate in float32.
    gaussian = compute_gaussian(tuple(patch_size), sigma_scale=1 / 8, value_scaling_factor=10,
                                device=torch.device("cpu")).float().numpy()
    coverage = np.zeros(padded.shape[-2:], dtype=np.float32)
    for window in windows:
        coverage[window] += gaussian
    counts = {c: int(np.count_nonzero(labels == c)) for c in classes}
    maps = {c: np.zeros_like(coverage) for c in classes if counts[c]}
    norms = {c: 0.0 for c in maps}
    flips = [combo for size in range(len(mirror_axes) + 1) for combo in combinations(mirror_axes, size)]
    with SegGradCAM(network, layer) as cam:
        for window in windows:
            targets = {c: torch.from_numpy(
                (labels[window] == c).astype(np.float32) * gaussian / coverage[window] / counts[c]
            ).to(device) for c in maps if np.any(labels[window] == c)}
            if not targets:
                continue
            image = torch.from_numpy(np.ascontiguousarray(padded[(slice(None), *window)][None])).to(device)
            for axes in flips:
                image_axes = tuple(axis + 2 for axis in axes)
                flipped_image = image.flip(image_axes) if axes else image
                flipped_targets = {c: weight.flip(axes) if axes else weight for c, weight in targets.items()}
                branch_maps, branch_norms = cam.compute(flipped_image, flipped_targets)
                for c, branch_map in branch_maps.items():
                    if axes:
                        branch_map = np.flip(branch_map, axes)
                    maps[c][window] += branch_map * gaussian / len(flips)
                    norms[c] = max(norms[c], branch_norms[c])
    return {c: (value / coverage)[undo[1:]][None].copy() for c, value in maps.items()}, norms


def restore_native_map(cam, properties, plans, configuration):
    """Undo resampling/cropping/transposition, then SimpleITK z,y,x to BMP y,x."""
    spacing = [properties["spacing"][i] for i in plans.transpose_forward]
    current_spacing = [spacing[0], *configuration.spacing]
    resized = configuration.resampling_fn_probabilities(
        cam[None], properties["shape_after_cropping_and_before_resampling"], current_spacing, spacing,
    )
    if isinstance(resized, torch.Tensor):
        resized = resized.cpu().numpy()
    restored = np.zeros(properties["shape_before_cropping"], dtype=np.float32)
    restored[tuple(slice(a, b) for a, b in properties["bbox_used_for_cropping"])] = resized[0]
    reader_array = restored.transpose(plans.transpose_backward)
    nifti_array = reader_array.transpose(2, 1, 0)
    if nifti_array.shape[0] != 1:
        raise ValueError(f"Expected singleton first NIfTI axis; got {nifti_array.shape}.")
    return nifti_array[0].copy()


def export_cam(cam: np.ndarray, source: np.ndarray, mask: np.ndarray, folder: Path, case_id: str):
    if cam.shape != source.shape[:2] or mask.shape != cam.shape or not np.isfinite(cam).all():
        raise ValueError("Heatmap, original radiograph, and mask must be finite and spatially aligned.")
    folder.mkdir(parents=True, exist_ok=True)
    np.save(folder / f"{case_id}_raw.npy", cam.astype(np.float32), allow_pickle=False)
    maximum = float(cam.max())
    normalized = np.clip(cam / maximum, 0, 1) if maximum > 0 else np.zeros_like(cam)
    # Fixed black-red-yellow-white palette; opacity goes to zero at zero attribution.
    colors = np.stack([np.minimum(normalized * 3, 1), np.clip(normalized * 3 - 1, 0, 1),
                       np.clip(normalized * 3 - 2, 0, 1)], axis=-1)
    heatmap = np.rint(colors * 255).astype(np.uint8)
    Image.fromarray(heatmap).save(folder / f"{case_id}_heatmap.png")
    alpha = 0.55 * normalized[..., None]
    overlay = np.rint(source * (1 - alpha) + heatmap * alpha).astype(np.uint8)
    mask_image = Image.fromarray((mask * 255).astype(np.uint8))
    eroded = np.asarray(mask_image.filter(ImageFilter.MinFilter(3))) > 0
    overlay[mask & ~eroded] = [0, 255, 255]
    Image.fromarray(overlay).save(folder / f"{case_id}_overlay.png")
    return maximum
