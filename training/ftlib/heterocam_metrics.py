"""Auditable packet metrics with no least-squares world alignment."""
from __future__ import annotations

import math
from collections import defaultdict

import torch

from .heterocam_geometry import (
    angular_error_radians,
    baseline_vectors_in_source_camera,
    fov_degrees,
    pair_indices,
    pair_masks,
    relative_rotations,
    robust_anchor_scale,
    rotation_geodesic_radians,
)


@torch.no_grad()
def packet_metrics(pred_E, pred_K, gt_E, gt_K, anchor_mask, image_hw):
    count = pred_E.shape[0]
    first, second = pair_indices(count, pred_E.device)
    anchor_pair, cross_pair, target_pair = pair_masks(anchor_mask, first, second)
    pred_relative = relative_rotations(pred_E, first, second)
    gt_relative = relative_rotations(gt_E, first, second)
    rotation_deg = rotation_geodesic_radians(pred_relative, gt_relative) * (180.0 / math.pi)
    pred_direction, pred_distance = baseline_vectors_in_source_camera(pred_E, first, second)
    gt_direction, gt_distance = baseline_vectors_in_source_camera(gt_E, first, second)
    direction_deg = angular_error_radians(pred_direction, gt_direction) * (180.0 / math.pi)
    scale = robust_anchor_scale(pred_distance, gt_distance, anchor_pair)
    distance_ape = ((pred_distance * scale - gt_distance).abs() / gt_distance.clamp_min(1e-8)) * 100.0
    h, w = image_hw
    pred_hfov, pred_vfov = fov_degrees(pred_K, h, w)
    gt_hfov, gt_vfov = fov_degrees(gt_K, h, w)
    hfov_error = (pred_hfov - gt_hfov).abs()
    vfov_error = (pred_vfov - gt_vfov).abs()
    target_mask = ~anchor_mask

    def mean_or_nan(values, mask):
        return float(values[mask].mean().cpu()) if bool(mask.any()) else float("nan")

    return {
        "relative_rotation_deg": float(rotation_deg.mean().cpu()),
        "anchor_relative_rotation_deg": mean_or_nan(rotation_deg, anchor_pair),
        "cross_relative_rotation_deg": mean_or_nan(rotation_deg, cross_pair),
        "baseline_direction_deg": float(direction_deg.mean().cpu()),
        "anchor_baseline_direction_deg": mean_or_nan(direction_deg, anchor_pair),
        "cross_baseline_direction_deg": mean_or_nan(direction_deg, cross_pair),
        "anchor_baseline_mape": mean_or_nan(distance_ape, anchor_pair),
        "cross_baseline_mape": mean_or_nan(distance_ape, cross_pair),
        "target_baseline_mape": mean_or_nan(distance_ape, target_pair),
        "hfov_mae_deg": float(hfov_error.mean().cpu()),
        "vfov_mae_deg": float(vfov_error.mean().cpu()),
        "anchor_hfov_mae_deg": mean_or_nan(hfov_error, anchor_mask),
        "target_hfov_mae_deg": mean_or_nan(hfov_error, target_mask),
        "anchor_vfov_mae_deg": mean_or_nan(vfov_error, anchor_mask),
        "target_vfov_mae_deg": mean_or_nan(vfov_error, target_mask),
        "anchor_scale": float(scale.cpu()),
    }


def aggregate_metrics(rows: list[dict]) -> dict:
    buckets = defaultdict(list)
    for row in rows:
        for key, value in row.items():
            if key == "frame_id":
                continue
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                buckets[key].append(float(value))
    return {key: sum(values) / len(values) for key, values in sorted(buckets.items()) if values}
