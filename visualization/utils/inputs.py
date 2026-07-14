"""Shared NPZ/NPY reconstruction input loading for the video renderers."""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass
class PointCloud:
    points: np.ndarray
    colors: np.ndarray


@dataclass
class ReconstructionInput:
    cloud: PointCloud
    poses: np.ndarray
    point_pose_idx: np.ndarray
    camera_aspect: float
    source: str


@dataclass(frozen=True)
class NpyInputPaths:
    root: Path
    intrinsics: Path
    depth: Path
    pose: Path
    rgb_dir: Path


def images_to_uint8(images: np.ndarray) -> np.ndarray:
    if images.ndim != 4:
        raise ValueError(f"Expected images with 4 dims, got shape {images.shape}")
    if images.shape[1] == 3:
        images = np.transpose(images, (0, 2, 3, 1))
    if images.shape[-1] != 3:
        raise ValueError(f"Expected RGB images, got shape {images.shape}")
    if images.dtype == np.uint8:
        return images
    if np.nanmax(images) > 1.5:
        return np.clip(images, 0, 255).astype(np.uint8)
    return (np.clip(images, 0.0, 1.0) * 255).astype(np.uint8)


def extrinsics_to_c2w(extrinsics: np.ndarray) -> np.ndarray:
    extrinsics = np.asarray(extrinsics)
    if extrinsics.ndim != 3 or extrinsics.shape[1:] not in {(3, 4), (4, 4)}:
        raise ValueError(f"Expected extrinsic shape (N,3,4) or (N,4,4), got {extrinsics.shape}")
    world_to_camera = np.tile(np.eye(4, dtype=np.float64), (extrinsics.shape[0], 1, 1))
    world_to_camera[:, : extrinsics.shape[1], : extrinsics.shape[2]] = extrinsics
    return np.linalg.inv(world_to_camera).astype(np.float32)


def load_npz_temporal_cloud(npz_path: Path) -> tuple[PointCloud, np.ndarray, np.ndarray, float]:
    with np.load(npz_path) as data:
        predictions = {key: np.asarray(data[key]) for key in data.files}

    required = ["world_points_from_depth", "images", "extrinsic"]
    missing = [key for key in required if key not in predictions]
    if missing:
        raise KeyError(f"{npz_path} is missing required keys: {', '.join(missing)}")

    points_4d = predictions["world_points_from_depth"].astype(np.float32, copy=False)
    images = images_to_uint8(predictions["images"])
    if points_4d.ndim != 4 or points_4d.shape[-1] != 3:
        raise ValueError(f"Expected world_points_from_depth shape (N,H,W,3), got {points_4d.shape}")
    if images.shape[:3] != points_4d.shape[:3]:
        raise ValueError(f"images shape {images.shape} does not match points {points_4d.shape}")

    flat_points = points_4d.reshape(-1, 3)
    flat_colors = images.reshape(-1, 3)
    frame_ids = np.repeat(
        np.arange(points_4d.shape[0], dtype=np.int32),
        points_4d.shape[1] * points_4d.shape[2],
    )
    mask = np.isfinite(flat_points).all(axis=1)
    points = flat_points[mask].astype(np.float32, copy=False)
    colors = flat_colors[mask].astype(np.uint8, copy=False)
    reveal_pose_idx = frame_ids[mask]
    poses = extrinsics_to_c2w(predictions["extrinsic"])

    print(
        f"[load] NPZ frames={poses.shape[0]}, points={flat_points.shape[0]}, "
        f"finite={points.shape[0]}"
    )
    camera_aspect = images.shape[2] / float(images.shape[1])
    return PointCloud(points, colors), poses, reveal_pose_idx.astype(np.int32), camera_aspect


def _resolve_path(path: Path | None, root: Path, default_name: str) -> Path:
    if path is None:
        return (root / default_name).resolve()
    candidate = path.expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    from_root = (root / candidate).resolve()
    return from_root if from_root.exists() else candidate.resolve()


def _single_default_npy(root: Path, canonical_name: str, pattern: str, option_name: str) -> Path:
    canonical = root / canonical_name
    if canonical.is_file():
        return canonical.resolve()
    matches = sorted(path for path in root.glob(pattern) if path.is_file())
    if len(matches) == 1:
        return matches[0].resolve()
    if len(matches) > 1:
        options = ", ".join(path.name for path in matches)
        raise ValueError(f"Multiple {pattern} files found in {root}: {options}. Set {option_name} explicitly.")
    return canonical.resolve()


def _npy_requested(args: argparse.Namespace, root: Path) -> bool:
    explicit_paths = (args.intrinsics_npy, args.depth_npy, args.pose_npy, args.npy_rgb_dir)
    if args.npy_dir is not None or any(path is not None for path in explicit_paths):
        return True
    return any((root / name).is_file() for name in ("K.npy", "depth.npy", "pose.npy"))


def resolve_npy_input_paths(args: argparse.Namespace) -> NpyInputPaths | None:
    root = (args.npy_dir or args.result_dir).expanduser().resolve()
    if not _npy_requested(args, root):
        return None

    intrinsics = _resolve_path(args.intrinsics_npy, root, "K.npy")
    depth = (
        _resolve_path(args.depth_npy, root, "depth.npy")
        if args.depth_npy is not None
        else _single_default_npy(root, "depth.npy", "depth_*.npy", "--depth-npy")
    )
    pose = (
        _resolve_path(args.pose_npy, root, "pose.npy")
        if args.pose_npy is not None
        else _single_default_npy(root, "pose.npy", "pose_*.npy", "--pose-npy")
    )
    rgb_dir = _resolve_path(args.npy_rgb_dir, root, ".")
    missing = [path for path in (intrinsics, depth, pose) if not path.is_file()]
    if missing:
        message = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(
            "NPY input requires intrinsics, depth, and pose arrays. Missing: "
            f"{message}. Set --npy-dir, --intrinsics-npy, --depth-npy, and --pose-npy as needed."
        )
    return NpyInputPaths(root, intrinsics, depth, pose, rgb_dir)


RGB_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def _sorted_rgb_paths(directory: Path) -> list[Path]:
    """Frames that colour the point cloud. Globbing *.png only used to leave a JPEG clip all-white."""
    if not directory.is_dir():
        return []
    return sorted(
        (path for path in directory.iterdir()
         if path.is_file() and path.suffix.lower() in RGB_SUFFIXES
         and "depth" not in path.name.lower()),
        key=lambda path: [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.name)],
    )


def resize_rgb_to_frame(image: Image.Image, width: int, height: int) -> np.ndarray:
    """Resize RGB proportionally, center-cropping only when aspect ratios differ."""
    image = image.convert("RGB")
    if image.size == (width, height):
        return np.asarray(image, dtype=np.uint8)

    scale = max(width / image.width, height / image.height)
    size = (max(width, int(round(image.width * scale))), max(height, int(round(image.height * scale))))
    resampling = getattr(Image, "Resampling", Image).BILINEAR
    resized = image.resize(size, resampling)
    left = (size[0] - width) // 2
    top = (size[1] - height) // 2
    return np.asarray(resized.crop((left, top, left + width, top + height)), dtype=np.uint8)


def _load_npy_colors(paths: NpyInputPaths, frame_count: int, height: int, width: int) -> list[np.ndarray] | None:
    rgb_paths = _sorted_rgb_paths(paths.rgb_dir)
    if not rgb_paths:
        print(f"[load] NPY RGB: no PNG files under {paths.rgb_dir}; using white points")
        return None
    if len(rgb_paths) < frame_count:
        raise ValueError(
            f"NPY depth has {frame_count} frames but only {len(rgb_paths)} PNG files exist under {paths.rgb_dir}."
        )

    colors = []
    for path in rgb_paths[:frame_count]:
        with Image.open(path) as image:
            colors.append(resize_rgb_to_frame(image, width, height))
    return colors


def unproject_depth_frames(
    intrinsics: np.ndarray,
    depths: np.ndarray,
    poses: np.ndarray,
    *,
    pose_convention: str,
    colors_by_frame: list[np.ndarray] | None = None,
) -> tuple[PointCloud, np.ndarray, np.ndarray]:
    """Unproject depth frames into one world-space cloud.

    depths (N,H,W) float, where <= 0 or non-finite means INVALID. `intrinsics` (3,3) or (N,3,3) must be
    in the DEPTH map's pixel coordinates, not the RGB's. `poses` (N,3,4)|(N,4,4) in `pose_convention`.
    Every valid pixel becomes a point, so a long clip is a lot of memory.
    """
    if depths.ndim != 3:
        raise ValueError(f"Expected depth shape (N,H,W), got {depths.shape}")
    frame_count, height, width = depths.shape
    if intrinsics.shape == (3, 3):
        intrinsics = np.broadcast_to(intrinsics, (frame_count, 3, 3))
    if intrinsics.shape != (frame_count, 3, 3):
        raise ValueError(f"Expected intrinsics shape (N,3,3), got {intrinsics.shape}")
    if poses.ndim != 3 or poses.shape[0] != frame_count or poses.shape[1:] not in {(3, 4), (4, 4)}:
        raise ValueError(f"Expected poses shape (N,3,4) or (N,4,4), got {poses.shape}")
    if pose_convention not in {"w2c", "c2w"}:
        raise ValueError(f"Expected pose convention 'w2c' or 'c2w', got {pose_convention!r}")
    if colors_by_frame is not None and len(colors_by_frame) < frame_count:
        raise ValueError(f"Expected at least {frame_count} RGB frames, got {len(colors_by_frame)}")

    poses_c2w = extrinsics_to_c2w(poses) if pose_convention == "w2c" else _as_homogeneous_poses(poses)
    vv, uu = np.meshgrid(
        np.arange(height, dtype=np.float32),
        np.arange(width, dtype=np.float32),
        indexing="ij",
    )

    all_points = []
    all_colors = []
    all_frame_ids = []
    for frame_idx in range(frame_count):
        depth = depths[frame_idx]
        valid = np.isfinite(depth) & (depth > 0)
        if not np.any(valid):
            continue

        k = intrinsics[frame_idx]
        if abs(float(k[0, 0])) < 1e-8 or abs(float(k[1, 1])) < 1e-8:
            raise ValueError(f"Invalid focal length at frame {frame_idx}")
        x = (uu - k[0, 2]) / k[0, 0] * depth
        y = (vv - k[1, 2]) / k[1, 1] * depth
        points_cam = np.stack([x, y, depth], axis=-1)[valid]
        pose = poses_c2w[frame_idx]
        points_world = points_cam @ pose[:3, :3].T + pose[:3, 3]

        if colors_by_frame is None:
            colors = np.full((points_world.shape[0], 3), 255, dtype=np.uint8)
        else:
            frame_colors = colors_by_frame[frame_idx]
            if frame_colors.shape != (height, width, 3):
                raise ValueError(
                    f"Expected RGB frame {frame_idx} shape {(height, width, 3)}, got {frame_colors.shape}"
                )
            colors = frame_colors[valid]
        all_points.append(points_world.astype(np.float32, copy=False))
        all_colors.append(colors.astype(np.uint8, copy=False))
        all_frame_ids.append(np.full(points_world.shape[0], frame_idx, dtype=np.int32))

    if not all_points:
        raise ValueError("No finite positive depth in any frame -- depth must be > 0 where valid, 0 or NaN where not.")

    points = np.concatenate(all_points, axis=0)
    colors = np.concatenate(all_colors, axis=0)
    reveal_pose_idx = np.concatenate(all_frame_ids, axis=0)
    return PointCloud(points, colors), poses_c2w, reveal_pose_idx


def load_npy_temporal_cloud(paths: NpyInputPaths, args: argparse.Namespace) -> tuple[PointCloud, np.ndarray, np.ndarray, float]:
    intrinsics = np.load(paths.intrinsics).astype(np.float32)
    depths = np.load(paths.depth).astype(np.float32)
    poses = np.load(paths.pose).astype(np.float32)
    if depths.ndim != 3:
        raise ValueError(f"Expected depth shape (N,H,W), got {depths.shape} from {paths.depth}")

    frame_count, height, width = depths.shape
    colors_by_frame = _load_npy_colors(paths, frame_count, height, width)
    cloud, poses_c2w, reveal_pose_idx = unproject_depth_frames(
        intrinsics,
        depths,
        poses,
        pose_convention=args.npy_pose_convention,
        colors_by_frame=colors_by_frame,
    )
    print(
        f"[load] NPY frames={frame_count}, points={cloud.points.shape[0]}, one point per valid pixel, "
        f"pose={args.npy_pose_convention}"
    )
    camera_aspect = width / float(height)
    return cloud, poses_c2w, reveal_pose_idx, camera_aspect


def _as_homogeneous_poses(poses: np.ndarray) -> np.ndarray:
    homogeneous = np.tile(np.eye(4, dtype=np.float32), (poses.shape[0], 1, 1))
    homogeneous[:, : poses.shape[1], : poses.shape[2]] = poses
    return homogeneous


def load_reconstruction_input(args: argparse.Namespace) -> ReconstructionInput:
    """Load the first available source using the fixed NPZ -> NPY priority."""
    npz_path = _resolve_path(args.npz_path, args.result_dir, "predictions.npz")
    if args.npz_path is not None and not npz_path.is_file():
        raise FileNotFoundError(f"Explicit --npz-path does not exist: {npz_path}")
    if npz_path.is_file():
        print(f"[load] source=NPZ path={npz_path}")
        ignored = [n.replace("_", "-") for n in ("intrinsics_npy", "depth_npy", "pose_npy", "npy_dir")
                   if getattr(args, n, None)]
        if ignored:
            print(f"[load] the NPZ takes precedence, so --{', --'.join(ignored)} are IGNORED. "
                  f"Move or rename the NPZ to use the NPY files instead.")
        cloud, poses, point_pose_idx, camera_aspect = load_npz_temporal_cloud(npz_path)
        return ReconstructionInput(cloud, poses, point_pose_idx, camera_aspect, "npz")

    npy_paths = resolve_npy_input_paths(args)
    if npy_paths is not None:
        print(f"[load] source=NPY dir={npy_paths.root}")
        cloud, poses, point_pose_idx, camera_aspect = load_npy_temporal_cloud(npy_paths, args)
        return ReconstructionInput(cloud, poses, point_pose_idx, camera_aspect, "npy")

    raise FileNotFoundError(
        "No usable reconstruction input found. Expected predictions.npz, or NPY arrays "
        "(K.npy, depth*.npy, pose*.npy)."
    )
