"""VGGT-Omega-compatible RGB preprocessing with exact intrinsics updates."""
from __future__ import annotations

import math

import numpy as np
import torch
from PIL import Image


def _crop_supported(image: Image.Image, K: np.ndarray, min_ratio=0.5, max_ratio=2.0):
    width, height = image.size
    ratio = height / max(width, 1)
    left = top = 0
    if ratio < min_ratio:
        crop_width = min(width, max(1, int(round(height / min_ratio))))
        left = max((width - crop_width) // 2, 0)
        image = image.crop((left, 0, left + crop_width, height))
    elif ratio > max_ratio:
        crop_height = min(height, max(1, int(round(width * max_ratio))))
        top = max((height - crop_height) // 2, 0)
        image = image.crop((0, top, width, top + crop_height))
    K = K.copy()
    K[0, 2] -= left
    K[1, 2] -= top
    return image, K


def _balanced_shape(ratio: float, resolution: int, patch: int) -> tuple[int, int]:
    token_number = (resolution // patch) ** 2
    w_patches = math.sqrt(token_number / ratio)
    h_patches = token_number / w_patches
    return max(1, round(h_patches)) * patch, max(1, round(w_patches)) * patch


def _max_size_shape(ratio: float, resolution: int, patch: int) -> tuple[int, int]:
    round_patch = lambda value: max(patch, round(float(value) / patch) * patch)
    if ratio >= 1.0:
        return resolution, round_patch(resolution / ratio)
    return round_patch(resolution * ratio), resolution


def load_rgb_and_intrinsics(
    path: str,
    intrinsics: np.ndarray,
    mode: str = "balanced",
    image_resolution: int = 512,
    patch_size: int = 16,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return RGB (3,H,W) in [0,1] and K transformed to that image."""
    if mode not in ("balanced", "max_size"):
        raise ValueError(f"unsupported preprocessing mode: {mode}")
    if image_resolution % patch_size:
        raise ValueError("image_resolution must be divisible by patch_size")
    with Image.open(path) as raw:
        if raw.mode == "RGBA":
            background = Image.new("RGBA", raw.size, (255, 255, 255, 255))
            raw = Image.alpha_composite(background, raw)
        image = raw.convert("RGB")
    image, K = _crop_supported(image, np.asarray(intrinsics, dtype=np.float64))
    width, height = image.size
    ratio = height / max(width, 1)
    target_h, target_w = (
        _balanced_shape(ratio, image_resolution, patch_size)
        if mode == "balanced"
        else _max_size_shape(ratio, image_resolution, patch_size)
    )
    image = image.resize((target_w, target_h), Image.Resampling.BICUBIC)
    K[0, :] *= target_w / width
    K[1, :] *= target_h / height
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
    return tensor, torch.from_numpy(K.astype(np.float32))


def pad_packet(images: list[torch.Tensor], intrinsics: list[torch.Tensor]):
    """Center-pad mixed image shapes exactly like the Omega inference loader."""
    max_h = max(image.shape[1] for image in images)
    max_w = max(image.shape[2] for image in images)
    out_images, out_K = [], []
    for image, K0 in zip(images, intrinsics):
        h, w = image.shape[-2:]
        dh, dw = max_h - h, max_w - w
        top, left = dh // 2, dw // 2
        bottom, right = dh - top, dw - left
        if dh or dw:
            image = torch.nn.functional.pad(image, (left, right, top, bottom), value=1.0)
        K = K0.clone()
        K[0, 2] += left
        K[1, 2] += top
        out_images.append(image)
        out_K.append(K)
    return torch.stack(out_images), torch.stack(out_K)
