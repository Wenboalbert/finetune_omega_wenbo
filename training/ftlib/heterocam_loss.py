"""Losses for CCTV-anchored heterogeneous-camera geometry finetuning."""
from __future__ import annotations

import torch
import torch.nn.functional as functional

from .heterocam_geometry import (
    baseline_vectors_in_source_camera,
    fov_degrees,
    pair_indices,
    pair_masks,
    relative_rotations,
    robust_anchor_scale,
    rotation_geodesic_radians,
)


DEFAULT_WEIGHTS = {
    "relative_rotation": 1.0,
    "baseline_direction": 1.0,
    "anchor_baseline": 0.5,
    "cross_baseline": 1.0,
    "fov": 0.02,
}


def _masked_mean(values, mask):
    if bool(mask.any()):
        return values[mask].mean()
    return values.new_zeros(())


def camera_geometry_loss(
    predicted_extrinsics: torch.Tensor,
    predicted_intrinsics: torch.Tensor,
    gt_extrinsics: torch.Tensor,
    gt_intrinsics: torch.Tensor,
    anchor_mask: torch.Tensor,
    image_hw: tuple[int, int],
    weights: dict | None = None,
):
    """No global Sim(3): every term is invariant to one world-coordinate Sim(3)."""
    weights = {**DEFAULT_WEIGHTS, **(weights or {})}
    count = predicted_extrinsics.shape[0]
    first, second = pair_indices(count, predicted_extrinsics.device)
    anchor_pair, cross_pair, _ = pair_masks(anchor_mask, first, second)

    pred_relative = relative_rotations(predicted_extrinsics, first, second)
    gt_relative = relative_rotations(gt_extrinsics, first, second)
    rotation = rotation_geodesic_radians(pred_relative, gt_relative).mean()

    pred_direction, pred_distance = baseline_vectors_in_source_camera(
        predicted_extrinsics, first, second
    )
    gt_direction, gt_distance = baseline_vectors_in_source_camera(gt_extrinsics, first, second)
    direction = (1.0 - (pred_direction * gt_direction).sum(-1).clamp(-1.0, 1.0)).mean()

    scale = robust_anchor_scale(pred_distance, gt_distance, anchor_pair).detach()
    log_ratio = torch.log((pred_distance * scale).clamp_min(1e-8)) - torch.log(
        gt_distance.clamp_min(1e-8)
    )
    distance_penalty = functional.smooth_l1_loss(
        log_ratio, torch.zeros_like(log_ratio), reduction="none", beta=0.05
    )
    anchor_baseline = _masked_mean(distance_penalty, anchor_pair)
    cross_baseline = _masked_mean(distance_penalty, cross_pair)

    height, width = image_hw
    pred_hfov, pred_vfov = fov_degrees(predicted_intrinsics, height, width)
    gt_hfov, gt_vfov = fov_degrees(gt_intrinsics, height, width)
    fov = functional.smooth_l1_loss(
        torch.stack([pred_hfov, pred_vfov], dim=-1),
        torch.stack([gt_hfov, gt_vfov], dim=-1),
        beta=1.0,
    )

    components = {
        "relative_rotation": rotation,
        "baseline_direction": direction,
        "anchor_baseline": anchor_baseline,
        "cross_baseline": cross_baseline,
        "fov": fov,
        "anchor_scale": scale,
    }
    total = sum(weights[name] * components[name] for name in DEFAULT_WEIGHTS)
    components["total"] = total
    return total, components
