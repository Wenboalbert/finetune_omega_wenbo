"""Observation-view strategies for reconstruction-progress rendering."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from utils.render_geometry import (
    color_for_white_background,
    draw_line_3d,
    frustum_segments,
    normalize,
    paint_points,
    point_offsets,
    pose_view_axes,
    smooth_camera_color,
)


_FONT_WARNED = False


def _label_font(name: str, size: int):
    """Pillow's bitmap fallback is tiny and unreadable. Use it if we must, but say so once, otherwise
    every caption silently degrades on a machine without DejaVu installed."""
    global _FONT_WARNED
    try:
        return ImageFont.truetype(name, size)
    except Exception:
        if not _FONT_WARNED:
            _FONT_WARNED = True
            print(f"[render] {name} not found; labels fall back to a tiny bitmap font. "
                  "Install DejaVu, or pass --no-labels.", flush=True)
        return ImageFont.load_default()



FOLLOW_VIEW_SMOOTHING = 0.2


@dataclass
class TopdownCache:
    eye: np.ndarray
    target: np.ndarray
    view_axes: tuple[np.ndarray, np.ndarray, np.ndarray]
    ortho_scale: float
    sorted_u: np.ndarray
    sorted_v: np.ndarray
    sorted_colors: np.ndarray
    sorted_orig_idx: np.ndarray
    last_visible_count: int = -1
    active_u: np.ndarray | None = None
    active_v: np.ndarray | None = None
    active_colors: np.ndarray | None = None
    last_layer_count: int = -1
    last_point_layer: np.ndarray | None = None


def _axis_from_name(name: str) -> np.ndarray:
    return {
        "+x": np.array([1.0, 0.0, 0.0], dtype=np.float32),
        "-x": np.array([-1.0, 0.0, 0.0], dtype=np.float32),
        "+y": np.array([0.0, 1.0, 0.0], dtype=np.float32),
        "-y": np.array([0.0, -1.0, 0.0], dtype=np.float32),
        "+z": np.array([0.0, 0.0, 1.0], dtype=np.float32),
        "-z": np.array([0.0, 0.0, -1.0], dtype=np.float32),
    }[name]


def _overhead_axis(poses: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    if args.top_axis == "camera-up":
        camera_ups = [pose_view_axes(pose)[2] for pose in poses]
        return normalize(np.mean(np.stack(camera_ups, axis=0), axis=0), _axis_from_name("+z"))
    if args.top_axis != "path-normal":
        return _axis_from_name(args.top_axis)

    centers = poses[:, :3, 3]
    _, _, right_vectors = np.linalg.svd(centers - np.mean(centers, axis=0, keepdims=True), full_matrices=False)
    normal = normalize(right_vectors[-1], _axis_from_name("+z"))
    camera_up = _overhead_axis(poses, argparse.Namespace(top_axis="camera-up"))
    return (-normal if float(np.dot(normal, camera_up)) < 0.0 else normal).astype(np.float32)


def _rotate_about_axis(vector: np.ndarray, axis: np.ndarray, degrees: float) -> np.ndarray:
    if abs(degrees) < 1e-8:
        return vector.astype(np.float32)
    theta = math.radians(degrees)
    axis = normalize(axis, _axis_from_name("+z"))
    return (
        vector * math.cos(theta)
        + np.cross(axis, vector) * math.sin(theta)
        + axis * float(np.dot(axis, vector)) * (1.0 - math.cos(theta))
    ).astype(np.float32)


def _principal_in_plane_axis(points: np.ndarray, top_axis: np.ndarray) -> np.ndarray:
    _, _, right_vectors = np.linalg.svd(points - np.mean(points, axis=0, keepdims=True), full_matrices=False)
    for vector in right_vectors:
        candidate = vector - top_axis * float(np.dot(vector, top_axis))
        if np.linalg.norm(candidate) > 1e-5:
            return normalize(candidate, _axis_from_name("+x"))
    fallback = np.cross(_axis_from_name("+z"), top_axis)
    if np.linalg.norm(fallback) < 1e-5:
        fallback = np.cross(_axis_from_name("+x"), top_axis)
    return normalize(fallback, _axis_from_name("+x"))


def _projection_extents(points: np.ndarray, target: np.ndarray, axis: np.ndarray) -> tuple[float, float]:
    values = (points - target[None, :]) @ axis
    return float(np.max(np.abs(values))), float(np.max(values))


def _build_topdown_view(
    points: np.ndarray,
    poses: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray], float]:
    centers = poses[:, :3, 3]
    mins = np.minimum(np.min(points, axis=0), np.min(centers, axis=0))
    maxs = np.maximum(np.max(points, axis=0), np.max(centers, axis=0))
    target = ((mins + maxs) * 0.5).astype(np.float32)

    top_axis = _overhead_axis(poses, args)
    z_axis = -top_axis
    x_axis = _rotate_about_axis(_principal_in_plane_axis(centers, top_axis), top_axis, args.top_yaw_deg)
    x_axis = normalize(x_axis - z_axis * float(np.dot(x_axis, z_axis)), _axis_from_name("+x"))
    y_axis = normalize(np.cross(x_axis, z_axis), _axis_from_name("+z"))
    if args.top_flip_vertical:
        y_axis = -y_axis

    point_half_x, _ = _projection_extents(points, target, x_axis)
    point_half_y, _ = _projection_extents(points, target, y_axis)
    center_half_x, _ = _projection_extents(centers, target, x_axis)
    center_half_y, _ = _projection_extents(centers, target, y_axis)
    margin = max(1.0, float(args.top_margin))
    half_x = max(point_half_x, center_half_x, 1e-3) * margin
    half_y = max(point_half_y, center_half_y, 1e-3) * margin
    top_extent = max(
        float(np.max((points - target[None, :]) @ top_axis)),
        float(np.max((centers - target[None, :]) @ top_axis)),
        0.0,
    )
    distance = top_extent + 1.0
    eye = (target + top_axis * distance).astype(np.float32)
    ortho_scale = min(
        (float(args.width) * 0.5) / max(half_x, 1e-6),
        (float(args.height) * 0.5) / max(half_y, 1e-6),
    ) * max(float(args.top_zoom), 1e-3)
    print(
        "[view] fixed topdown: "
        f"axis={args.top_axis}, eye={eye.round(3).tolist()}, "
        f"target={target.round(3).tolist()}, distance={distance:.3f}, zoom={args.top_zoom:.3f}"
    )
    return eye, target, (x_axis.astype(np.float32), y_axis.astype(np.float32), z_axis.astype(np.float32)), ortho_scale


def _project_topdown(
    points: np.ndarray,
    eye: np.ndarray,
    width: int,
    height: int,
    view_axes: tuple[np.ndarray, np.ndarray, np.ndarray],
    ortho_scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_axis, y_axis, z_axis = view_axes
    relative = points - eye[None, :]
    return (
        width * 0.5 + ortho_scale * (relative @ x_axis),
        height * 0.5 - ortho_scale * (relative @ y_axis),
        relative @ z_axis,
    )


def _get_topdown_cache(points: np.ndarray, colors: np.ndarray, poses: np.ndarray, args: argparse.Namespace) -> TopdownCache:
    cache = getattr(args, "_topdown_cache", None)
    if cache is not None:
        return cache
    eye, target, view_axes, ortho_scale = _build_topdown_view(points, poses, args)
    u, v, z = _project_topdown(
        points, eye, args.width, args.height, view_axes, ortho_scale
    )
    mask = (z > 1e-3) & (u >= 0) & (u < args.width) & (v >= 0) & (v < args.height)
    if not np.any(mask):
        raise RuntimeError("The fitted top-down camera sees no points. Try a different --top-axis.")
    order = np.argsort(z[mask])[::-1]
    cache = TopdownCache(
        eye=eye,
        target=target,
        view_axes=view_axes,
        ortho_scale=ortho_scale,
        sorted_u=u[mask].astype(np.int32)[order],
        sorted_v=v[mask].astype(np.int32)[order],
        sorted_colors=colors[mask][order],
        sorted_orig_idx=np.flatnonzero(mask).astype(np.int32)[order],
    )
    print(f"[view] projected fixed topdown points: {cache.sorted_u.shape[0]} / {points.shape[0]}")
    setattr(args, "_topdown_cache", cache)
    return cache


def _paint_cached_points(image: np.ndarray, visible_count: int, cache: TopdownCache, args: argparse.Namespace) -> None:
    if visible_count <= 0:
        return
    if cache.last_visible_count != visible_count:
        selected = cache.sorted_orig_idx < visible_count
        cache.active_u = cache.sorted_u[selected]
        cache.active_v = cache.sorted_v[selected]
        cache.active_colors = cache.sorted_colors[selected]
        cache.last_visible_count = visible_count
    if cache.active_u is None or cache.active_u.shape[0] == 0:
        return
    for dx, dy in point_offsets(args.point_size):
        uu = np.clip(cache.active_u + dx, 0, args.width - 1)
        vv = np.clip(cache.active_v + dy, 0, args.height - 1)
        image[vv, uu] = cache.active_colors


def chase_view(
    poses: np.ndarray,
    step_idx: int,
    radius: float,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    pose = poses[step_idx]
    center = pose[:3, 3]
    forward, right, up = pose_view_axes(pose)
    eye = (
        center
        - forward * radius * args.chase_back
        + up * radius * args.chase_up
        + right * radius * args.chase_right
    )
    target = eye + forward * radius * args.chase_lookahead
    return eye.astype(np.float32), target.astype(np.float32), (right, up, forward)


def render_follow_frame(
    visible_points: np.ndarray,
    visible_colors: np.ndarray,
    sparse_pose_indices: list[int],
    current_pose_idx: int,
    highlight_pose_idx: int | None,
    points: np.ndarray,
    colors: np.ndarray,
    poses: np.ndarray,
    args: argparse.Namespace,
    smoothed_eye_target: tuple[np.ndarray, np.ndarray] | None,
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    del points, colors
    radius = args.target_radius if args.target_radius > 0 else 10.0
    eye, target, view_axes = chase_view(poses, current_pose_idx, radius, args)
    if smoothed_eye_target is not None:
        previous_eye, previous_target = smoothed_eye_target
        eye = (1.0 - FOLLOW_VIEW_SMOOTHING) * previous_eye + FOLLOW_VIEW_SMOOTHING * eye
        target = (1.0 - FOLLOW_VIEW_SMOOTHING) * previous_target + FOLLOW_VIEW_SMOOTHING * target

    image = np.full((args.height, args.width, 3), 255, dtype=np.uint8)
    paint_points(
        image,
        visible_points,
        visible_colors,
        eye,
        target,
        args.width,
        args.height,
        args.fov_deg,
        args.point_size,
        view_axes,
    )
    pil = Image.fromarray(image)
    draw = ImageDraw.Draw(pil, "RGB")
    highlight_idx = current_pose_idx if highlight_pose_idx is None else highlight_pose_idx
    centers = poses[sparse_pose_indices, :3, 3] if sparse_pose_indices else np.zeros((0, 3), dtype=np.float32)
    for p0, p1 in zip(centers[:-1], centers[1:]):
        draw_line_3d(draw, p0, p1, eye, target, args.width, args.height, args.fov_deg, (85, 90, 100), 2, view_axes)
    for pose_idx in sparse_pose_indices:
        color = color_for_white_background(smooth_camera_color(pose_idx, poses.shape[0]))
        line_width = 4 if pose_idx == highlight_idx else 2
        for p0, p1 in frustum_segments(poses[pose_idx], args.frustum_scale, args.camera_fov_deg, args.camera_aspect):
            draw_line_3d(draw, p0, p1, eye, target, args.width, args.height, args.fov_deg, color, line_width, view_axes)

    _draw_frame_text(draw, args.height, current_pose_idx, poses.shape[0], "frame", args)
    return np.asarray(pil), (eye.astype(np.float32), target.astype(np.float32))


@dataclass
class FrameView:
    eye: np.ndarray
    target: np.ndarray
    view_axes: tuple[np.ndarray, np.ndarray, np.ndarray]
    ortho_scale: float


def _smoothstep(value: float) -> float:
    value = float(np.clip(value, 0.0, 1.0))
    return value * value * value * (value * (value * 6.0 - 15.0) + 10.0)


def _animated_topdown_view(
    cache: TopdownCache,
    current_pose_idx: int,
    pose_count: int,
    args: argparse.Namespace,
) -> FrameView:
    progress = 0.0 if pose_count <= 1 else current_pose_idx / float(pose_count - 1)
    push_ratio = float(np.clip(args.motion_push_ratio, 0.05, 0.9))
    orbit_start = push_ratio * (1.0 - float(np.clip(args.motion_orbit_overlap, 0.0, 0.95)))
    if progress <= push_ratio:
        push = _smoothstep(progress / push_ratio)
        zoom = (1.0 - push) * max(float(args.motion_start_zoom), 1e-3) + push
    else:
        zoom = 1.0

    orbit_t = float(np.clip((progress - orbit_start) / max(1.0 - orbit_start, 1e-6), 0.0, 1.0))
    orbit = float(args.motion_orbit_deg) * (
        _smoothstep(orbit_t / 0.5) if orbit_t <= 0.5 else 1.0 - 2.0 * _smoothstep((orbit_t - 0.5) / 0.5)
    )
    base_x, base_y, base_z = cache.view_axes
    base_top = normalize(-base_z, np.array([0.0, -1.0, 0.0], dtype=np.float32))
    left = normalize(-base_x, np.array([1.0, 0.0, 0.0], dtype=np.float32))
    angle = math.radians(orbit)
    top_axis = normalize(math.cos(angle) * base_top + math.sin(angle) * left, base_top)
    eye = (cache.target + top_axis * float(np.linalg.norm(cache.eye - cache.target))).astype(np.float32)
    z_axis = normalize(cache.target - eye, base_z)
    x_axis = normalize(base_x - z_axis * float(np.dot(base_x, z_axis)), base_x)
    y_axis = normalize(np.cross(x_axis, z_axis), base_y)
    if args.top_flip_vertical:
        y_axis = -y_axis
    return FrameView(eye, cache.target, (x_axis.astype(np.float32), y_axis.astype(np.float32), z_axis.astype(np.float32)), cache.ortho_scale * zoom)


def _paint_points_with_topdown_view(
    image: np.ndarray,
    points: np.ndarray,
    colors: np.ndarray,
    view: FrameView,
    args: argparse.Namespace,
) -> None:
    if points.shape[0] == 0:
        return
    u, v, z = _project_topdown(
        points, view.eye, args.width, args.height, view.view_axes, view.ortho_scale,
    )
    mask = (z > 1e-3) & (u >= 0) & (u < args.width) & (v >= 0) & (v < args.height)
    if not np.any(mask):
        return
    u, v, z, visible_colors = u[mask].astype(np.int32), v[mask].astype(np.int32), z[mask], colors[mask]
    order = np.argsort(z)[::-1]
    u, v, visible_colors = u[order], v[order], visible_colors[order]
    for dx, dy in point_offsets(args.point_size):
        image[np.clip(v + dy, 0, args.height - 1), np.clip(u + dx, 0, args.width - 1)] = visible_colors


def _draw_topdown_line(
    draw: ImageDraw.ImageDraw,
    p0: np.ndarray,
    p1: np.ndarray,
    view: FrameView,
    args: argparse.Namespace,
    color: tuple[int, int, int],
    line_width: int,
) -> None:
    u, v, z = _project_topdown(
        np.stack([p0, p1], axis=0), view.eye, args.width, args.height, view.view_axes, view.ortho_scale,
    )
    if np.any(z <= 1e-3) or np.all((u < -args.width) | (u > args.width * 2) | (v < -args.height) | (v > args.height * 2)):
        return
    draw.line([(float(u[0]), float(v[0])), (float(u[1]), float(v[1]))], fill=color, width=line_width)


def render_topdown_frame(
    visible_points: np.ndarray,
    visible_colors: np.ndarray,
    sparse_pose_indices: list[int],
    current_pose_idx: int,
    highlight_pose_idx: int | None,
    points: np.ndarray,
    colors: np.ndarray,
    poses: np.ndarray,
    args: argparse.Namespace,
    smoothed_eye_target: tuple[np.ndarray, np.ndarray] | None,
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    del smoothed_eye_target
    cache = _get_topdown_cache(points, colors, poses, args)
    if args.view_motion == "cinematic":
        view = _animated_topdown_view(cache, current_pose_idx, poses.shape[0], args)
        image = np.full((args.height, args.width, 3), 255, dtype=np.uint8)
        _paint_points_with_topdown_view(image, visible_points, visible_colors, view, args)
    else:
        view = FrameView(cache.eye, cache.target, cache.view_axes, cache.ortho_scale)
        visible_count = visible_points.shape[0]
        if cache.last_layer_count == visible_count and cache.last_point_layer is not None:
            image = cache.last_point_layer.copy()
        else:
            image = np.full((args.height, args.width, 3), 255, dtype=np.uint8)
            _paint_cached_points(image, visible_count, cache, args)
            cache.last_layer_count = visible_count
            cache.last_point_layer = image.copy()

    pil = Image.fromarray(image)
    draw = ImageDraw.Draw(pil, "RGB")
    highlight_idx = current_pose_idx if highlight_pose_idx is None else highlight_pose_idx
    centers = poses[sparse_pose_indices, :3, 3] if sparse_pose_indices else np.zeros((0, 3), dtype=np.float32)
    for p0, p1 in zip(centers[:-1], centers[1:]):
        _draw_topdown_line(draw, p0, p1, view, args, (85, 90, 100), 2)
    for pose_idx in sparse_pose_indices:
        color = color_for_white_background(smooth_camera_color(pose_idx, poses.shape[0]))
        line_width = 4 if pose_idx == highlight_idx else 2
        for p0, p1 in frustum_segments(poses[pose_idx], args.frustum_scale, args.camera_fov_deg, args.camera_aspect):
            _draw_topdown_line(draw, p0, p1, view, args, color, line_width)

    _draw_frame_text(draw, args.height, current_pose_idx, poses.shape[0], "top view frame", args)
    return np.asarray(pil), (cache.eye, cache.target)


def _draw_frame_text(
    draw: ImageDraw.ImageDraw,
    height: int,
    current_pose_idx: int,
    pose_count: int,
    label: str,
    args: argparse.Namespace,
) -> None:
    if getattr(args, "no_frame_text", False):
        return
    font = _label_font("DejaVuSans.ttf", 20)
    draw.text((24, height - 42), f"{label} {current_pose_idx:04d} / {pose_count - 1:04d}", fill=(25, 28, 32), font=font)
