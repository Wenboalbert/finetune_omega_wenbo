"""Similarity-invariant multi-camera geometry primitives."""
from __future__ import annotations

import math

import torch


def as_4x4(extrinsics: torch.Tensor) -> torch.Tensor:
    if extrinsics.shape[-2:] == (4, 4):
        return extrinsics
    if extrinsics.shape[-2:] != (3, 4):
        raise ValueError(f"extrinsics must end in 3x4 or 4x4, got {extrinsics.shape}")
    bottom = torch.zeros(*extrinsics.shape[:-2], 1, 4, device=extrinsics.device, dtype=extrinsics.dtype)
    bottom[..., 0, 3] = 1.0
    return torch.cat([extrinsics, bottom], dim=-2)


def camera_centers(extrinsics: torch.Tensor) -> torch.Tensor:
    E = as_4x4(extrinsics)
    rotation = E[..., :3, :3]
    translation = E[..., :3, 3]
    return -(rotation.transpose(-1, -2) @ translation.unsqueeze(-1)).squeeze(-1)


def pair_indices(count: int, device=None) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.triu_indices(count, count, offset=1, device=device).unbind(0)


def relative_rotations(extrinsics: torch.Tensor, first: torch.Tensor, second: torch.Tensor):
    rotation = as_4x4(extrinsics)[..., :3, :3]
    return rotation[second] @ rotation[first].transpose(-1, -2)


def baseline_vectors_in_source_camera(
    extrinsics: torch.Tensor, first: torch.Tensor, second: torch.Tensor, eps: float = 1e-8
):
    E = as_4x4(extrinsics)
    centers = camera_centers(E)
    world_vector = centers[second] - centers[first]
    camera_vector = (E[first, :3, :3] @ world_vector.unsqueeze(-1)).squeeze(-1)
    distance = torch.linalg.vector_norm(camera_vector, dim=-1)
    direction = camera_vector / distance.clamp_min(eps).unsqueeze(-1)
    return direction, distance


def rotation_geodesic_radians(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    relative = first @ second.transpose(-1, -2)
    cosine = ((relative.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0) / 2.0).clamp(-1.0, 1.0)
    # atan2 avoids the infinite derivative of acos at almost-identical rotations.
    skew = torch.stack(
        [
            relative[..., 2, 1] - relative[..., 1, 2],
            relative[..., 0, 2] - relative[..., 2, 0],
            relative[..., 1, 0] - relative[..., 0, 1],
        ],
        dim=-1,
    )
    sine = 0.5 * torch.sqrt((skew * skew).sum(-1) + 1e-12)
    return torch.atan2(sine, cosine)


def angular_error_radians(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    cosine = (first * second).sum(-1).clamp(-1.0, 1.0)
    return torch.acos(cosine)


def fov_degrees(intrinsics: torch.Tensor, height: int, width: int) -> tuple[torch.Tensor, torch.Tensor]:
    fx = intrinsics[..., 0, 0].clamp_min(1e-8)
    fy = intrinsics[..., 1, 1].clamp_min(1e-8)
    radians_to_degrees = 180.0 / math.pi
    hfov = 2.0 * torch.atan(torch.as_tensor(width, device=fx.device, dtype=fx.dtype) / (2.0 * fx))
    vfov = 2.0 * torch.atan(torch.as_tensor(height, device=fy.device, dtype=fy.dtype) / (2.0 * fy))
    return hfov * radians_to_degrees, vfov * radians_to_degrees


def pair_masks(anchor_mask: torch.Tensor, first: torch.Tensor, second: torch.Tensor):
    anchor_pair = anchor_mask[first] & anchor_mask[second]
    cross_pair = anchor_mask[first] ^ anchor_mask[second]
    target_pair = (~anchor_mask[first]) & (~anchor_mask[second])
    return anchor_pair, cross_pair, target_pair


def robust_anchor_scale(predicted_distance, gt_distance, anchor_pair, eps=1e-8):
    if int(anchor_pair.sum()) < 1:
        raise ValueError("at least one anchor-anchor pair is required")
    ratios = gt_distance[anchor_pair] / predicted_distance[anchor_pair].clamp_min(eps)
    return ratios.median()
