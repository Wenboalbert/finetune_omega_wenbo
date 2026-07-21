"""Unreal Engine camera CSV -> OpenCV camera-from-world conversion.

The input CSV uses UE world axes X-forward, Y-right, Z-up and centimetres.
The output uses a right-handed world basis [UE-Y, -UE-Z, UE-X] and OpenCV
camera axes (x right, y down, z forward).  The conversion is explicit so the
training manifest can be independently audited before any model is loaded.
"""
from __future__ import annotations

import math
from typing import Iterable

import numpy as np


UE_TO_CV_WORLD = np.array(
    [[0.0, 1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]],
    dtype=np.float64,
)


def unreal_rotator_c2w(pitch_deg: float, yaw_deg: float, roll_deg: float) -> np.ndarray:
    """UE local camera-to-UE world rotation; columns are forward/right/up."""
    pitch, yaw, roll = map(math.radians, (pitch_deg, yaw_deg, roll_deg))
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    cr, sr = math.cos(roll), math.sin(roll)
    forward = np.array([cp * cy, cp * sy, sp], dtype=np.float64)
    right = np.array(
        [sr * sp * cy - cr * sy, sr * sp * sy + cr * cy, -sr * cp],
        dtype=np.float64,
    )
    up = np.array(
        [-(cr * sp * cy + sr * sy), sr * cy - cr * sp * sy, cr * cp],
        dtype=np.float64,
    )
    return np.column_stack([forward, right, up])


def ue_pose_to_opencv_extrinsic(
    position_ue: Iterable[float],
    pitch_deg: float,
    yaw_deg: float,
    roll_deg: float,
    position_scale: float = 0.01,
) -> np.ndarray:
    """Convert a UE pose to a 4x4 OpenCV camera-from-world matrix."""
    center_ue = np.asarray(tuple(position_ue), dtype=np.float64) * position_scale
    center_cv = UE_TO_CV_WORLD @ center_ue
    c2w_ue = unreal_rotator_c2w(pitch_deg, yaw_deg, roll_deg)
    c2w_cv = UE_TO_CV_WORLD @ c2w_ue @ UE_TO_CV_WORLD.T
    if not np.allclose(c2w_cv.T @ c2w_cv, np.eye(3), atol=1e-8):
        raise ValueError("converted UE rotation is not orthonormal")
    if not np.isclose(np.linalg.det(c2w_cv), 1.0, atol=1e-8):
        raise ValueError("converted UE rotation determinant is not +1")
    rcw = c2w_cv.T
    tcw = -rcw @ center_cv
    extrinsic = np.eye(4, dtype=np.float64)
    extrinsic[:3, :3] = rcw
    extrinsic[:3, 3] = tcw
    return extrinsic


def intrinsics_from_ue_hfov(width: int, height: int, hfov_deg: float) -> np.ndarray:
    """Centered square-pixel pinhole intrinsics from UE horizontal FOVAngle."""
    if width <= 0 or height <= 0:
        raise ValueError("image width/height must be positive")
    if not 0.0 < hfov_deg < 179.0:
        raise ValueError(f"invalid horizontal FOV: {hfov_deg}")
    focal = (width / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
    return np.array(
        [[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def camera_center_from_extrinsic(extrinsic: np.ndarray) -> np.ndarray:
    rcw = np.asarray(extrinsic, dtype=np.float64)[:3, :3]
    tcw = np.asarray(extrinsic, dtype=np.float64)[:3, 3]
    return -(rcw.T @ tcw)
