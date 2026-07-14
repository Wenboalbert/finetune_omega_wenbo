"""Geometry and rasterization helpers shared by reconstruction renderers."""

from __future__ import annotations

import colorsys
import math

import numpy as np
from PIL import ImageDraw


def normalize(v: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(v))
    if norm < 1e-8:
        return fallback.astype(np.float32)
    return (v / norm).astype(np.float32)


def pose_view_axes(pose: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rotation = pose[:3, :3]
    forward = normalize(rotation[:, 2], np.array([0.0, 0.0, 1.0]))
    right = normalize(rotation[:, 0], np.array([1.0, 0.0, 0.0]))
    up = normalize(-rotation[:, 1], np.array([0.0, 0.0, 1.0]))
    return forward, right, up


def view_basis(eye: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    z_axis = normalize(target - eye, np.array([0.0, 0.0, 1.0]))
    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    x_axis = np.cross(z_axis, world_up)
    if np.linalg.norm(x_axis) < 1e-5:
        x_axis = np.cross(z_axis, np.array([0.0, 1.0, 0.0], dtype=np.float32))
    x_axis = normalize(x_axis, np.array([1.0, 0.0, 0.0]))
    y_axis = normalize(np.cross(x_axis, z_axis), np.array([0.0, 0.0, 1.0]))
    return x_axis, y_axis, z_axis


def project(
    points: np.ndarray,
    eye: np.ndarray,
    target: np.ndarray,
    width: int,
    height: int,
    fov_deg: float,
    view_axes: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_axis, y_axis, z_axis = view_basis(eye, target) if view_axes is None else view_axes
    relative = points - eye[None, :]
    x = relative @ x_axis
    y = relative @ y_axis
    z = relative @ z_axis
    focal_length = 0.5 * height / math.tan(math.radians(fov_deg) * 0.5)
    u = width * 0.5 + focal_length * x / np.maximum(z, 1e-6)
    v = height * 0.5 - focal_length * y / np.maximum(z, 1e-6)
    return u, v, z


def point_offsets(point_size: int) -> list[tuple[int, int]]:
    offsets = [(0, 0)]
    if point_size >= 2:
        offsets += [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if point_size >= 3:
        offsets += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    return offsets


def paint_points(
    image: np.ndarray,
    points: np.ndarray,
    colors: np.ndarray,
    eye: np.ndarray,
    target: np.ndarray,
    width: int,
    height: int,
    fov_deg: float,
    point_size: int,
    view_axes: tuple[np.ndarray, np.ndarray, np.ndarray] | None,
) -> None:
    if points.shape[0] == 0:
        return
    u, v, z = project(points, eye, target, width, height, fov_deg, view_axes)
    mask = (z > 1e-3) & (u >= 0) & (u < width) & (v >= 0) & (v < height)
    if not np.any(mask):
        return
    order = np.argsort(z[mask])[::-1]
    u = u[mask].astype(np.int32)[order]
    v = v[mask].astype(np.int32)[order]
    visible_colors = colors[mask][order]
    for dx, dy in point_offsets(point_size):
        uu = np.clip(u + dx, 0, width - 1)
        vv = np.clip(v + dy, 0, height - 1)
        image[vv, uu] = visible_colors


def frustum_segments(
    pose: np.ndarray,
    scale: float,
    camera_fov_deg: float,
    aspect: float,
) -> list[tuple[np.ndarray, np.ndarray]]:
    center = pose[:3, 3]
    rotation = pose[:3, :3]
    half_height = math.tan(math.radians(camera_fov_deg) * 0.5) * scale
    half_width = aspect * half_height
    corners_camera = np.array(
        [
            [-half_width, -half_height, scale],
            [half_width, -half_height, scale],
            [half_width, half_height, scale],
            [-half_width, half_height, scale],
        ],
        dtype=np.float32,
    )
    corners = center[None, :] + corners_camera @ rotation.T
    segments = [(center, corner) for corner in corners]
    segments.extend((corners[idx], corners[(idx + 1) % 4]) for idx in range(4))
    return segments


def draw_line_3d(
    draw: ImageDraw.ImageDraw,
    p0: np.ndarray,
    p1: np.ndarray,
    eye: np.ndarray,
    target: np.ndarray,
    width: int,
    height: int,
    fov_deg: float,
    color: tuple[int, int, int],
    line_width: int,
    view_axes: tuple[np.ndarray, np.ndarray, np.ndarray] | None,
) -> None:
    u, v, z = project(np.stack([p0, p1], axis=0), eye, target, width, height, fov_deg, view_axes)
    if np.any(z <= 1e-3):
        return
    if np.all((u < -width) | (u > width * 2) | (v < -height) | (v > height * 2)):
        return
    draw.line([(float(u[0]), float(v[0])), (float(u[1]), float(v[1]))], fill=color, width=line_width)


def smooth_camera_color(index: int, total: int) -> tuple[int, int, int]:
    red, green, blue = colorsys.hsv_to_rgb(index / max(total - 1, 1), 0.68, 0.92)
    return tuple(int(round(channel * 255)) for channel in (red, green, blue))


def color_for_white_background(color: tuple[int, int, int]) -> tuple[int, int, int]:
    rgb = np.asarray(color, dtype=np.float32)
    luminance = float(rgb @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32))
    rgb *= 0.58 if luminance > 150.0 else 0.78
    return tuple(int(value) for value in np.clip(rgb, 0, 255))
