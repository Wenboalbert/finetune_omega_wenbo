"""Common argument parsing and frame loop for reconstruction-progress videos."""

from __future__ import annotations

import argparse
import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from utils.inputs import load_reconstruction_input


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def default_output_paths(args: argparse.Namespace, renderer_name: str) -> tuple[Path, Path]:
    if args.output is not None:
        mp4_path = args.output
    else:
        scene_name = args.result_dir.name.replace("-", "")
        vis_dir = Path(__file__).resolve().parents[1]
        mp4_path = vis_dir / "demo" / f"{scene_name}_reconstruction_{renderer_name}.mp4"
    return mp4_path, mp4_path.with_suffix(".gif")


def natural_key(path: Path) -> list:
    """Count the way a human does: rgb_2 before rgb_10. A plain lexicographic sort silently attaches
    colours to the wrong frames as soon as a clip is longer than 9 frames."""
    return [int(p) if p.isdigit() else p.lower() for p in re.split(r"(\d+)", path.name)]


def sorted_input_images(image_dir: Path | None) -> list[Path]:
    if image_dir is None or not image_dir.exists():
        return []
    return sorted(
        (p for p in image_dir.iterdir()
         if p.suffix.lower() in IMAGE_SUFFIXES and "depth" not in p.name.lower()),
        key=natural_key,
    )


def load_input_panel(
    image_paths: list[Path], pose_idx: int, n_poses: int, width: int, height: int
) -> np.ndarray | None:
    if not image_paths:
        return None
    image_idx = 0 if len(image_paths) <= 1 or n_poses <= 1 else round(pose_idx * (len(image_paths) - 1) / (n_poses - 1))
    with Image.open(image_paths[int(np.clip(image_idx, 0, len(image_paths) - 1))]) as image:
        image = image.convert("RGB")
        scale = min(width / image.width, height / image.height)
        size = (max(1, int(round(image.width * scale))), max(1, int(round(image.height * scale))))
        resampling = getattr(Image, "Resampling", Image).BILINEAR
        canvas = Image.new("RGB", (width, height), (255, 255, 255))
        canvas.paste(image.resize(size, resampling), ((width - size[0]) // 2, (height - size[1]) // 2))
        return np.asarray(canvas, dtype=np.uint8)


def compose_frame(input_panel: np.ndarray | None, reconstruction_panel: np.ndarray) -> np.ndarray:
    if input_panel is None:
        return reconstruction_panel
    if input_panel.shape[0] != reconstruction_panel.shape[0]:
        raise ValueError("Input and reconstruction panels must have the same height.")
    return np.concatenate([input_panel, reconstruction_panel], axis=1)


def normalize_scene(points: np.ndarray, poses: np.ndarray, target_radius: float) -> tuple[np.ndarray, np.ndarray, float]:
    centers = poses[:, :3, 3]
    center = np.mean(points, axis=0) if points.size else np.mean(centers, axis=0)
    normalized_points = points - center[None, :]
    normalized_poses = poses.copy()
    normalized_poses[:, :3, 3] -= center[None, :]
    all_positions = np.concatenate([normalized_points, normalized_poses[:, :3, 3]], axis=0)
    radius = max(float(np.linalg.norm(all_positions.max(axis=0) - all_positions.min(axis=0)) * 0.5), 1e-9)
    scale = target_radius / radius if target_radius > 0 else 1.0
    normalized_points *= scale
    normalized_poses[:, :3, 3] *= scale
    return normalized_points.astype(np.float32), normalized_poses.astype(np.float32), scale


@dataclass(frozen=True)
class ReconstructionProgress:
    sparse_pose_indices: list[int]
    pose_idx: int
    highlight_pose_idx: int
    reveal_until: int
    step: int | None


def sparse_reconstruction_pose_indices(poses: np.ndarray, args: argparse.Namespace) -> list[int]:
    indices = list(range(0, poses.shape[0], max(1, args.stride)))
    if indices[-1] != poses.shape[0] - 1:
        indices.append(poses.shape[0] - 1)
    if args.max_steps is not None:
        indices = indices[: args.max_steps]
    return indices


def reconstruction_progress_frames(
    poses: np.ndarray,
    args: argparse.Namespace,
    has_input_panel: bool,
) -> Iterator[ReconstructionProgress]:
    """Yield exactly the reconstruction reveal schedule used by the video renderer."""
    sparse_pose_indices = sparse_reconstruction_pose_indices(poses, args)
    if has_input_panel:
        render_pose_count = poses.shape[0]
        if args.max_steps is not None:
            render_pose_count = min(render_pose_count, max(1, args.max_steps * max(1, args.stride)))
        sparse_arr = np.asarray(sparse_pose_indices, dtype=np.int32)
        for pose_idx in range(render_pose_count):
            reveal_until = min(
                ((pose_idx // max(1, args.stride)) + 1) * max(1, args.stride) - 1,
                render_pose_count - 1,
                poses.shape[0] - 1,
            )
            active_count = max(1, int(np.searchsorted(sparse_arr, pose_idx, side="right")))
            yield ReconstructionProgress(
                sparse_pose_indices[:active_count],
                pose_idx,
                sparse_pose_indices[active_count - 1],
                reveal_until,
                None,
            )
        return

    for step, pose_idx in enumerate(sparse_pose_indices):
        reveal_until = min(pose_idx + max(1, args.stride) - 1, poses.shape[0] - 1)
        for _ in range(max(1, args.frames_per_step)):
            yield ReconstructionProgress(sparse_pose_indices[: step + 1], pose_idx, pose_idx, reveal_until, step)


def run_reconstruction_video(
    args: argparse.Namespace,
    output_paths: tuple[Path, Path],
    make_writer: Callable[[Path, int, int, int], tuple[object, Path]],
    render_frame: Callable[..., tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]],
) -> None:
    args.result_dir = args.result_dir.expanduser().resolve()
    mp4_path, _ = output_paths
    image_dir = None
    image_paths: list[Path] = []
    if not args.no_input_panel:
        image_dir = args.image_dir.expanduser().resolve() if args.image_dir else None
        image_paths = sorted_input_images(image_dir)
        print(f"[load] input images: {len(image_paths)} from {image_dir}" if image_paths else "[load] input images: none found; rendering reconstruction only")

    loaded = load_reconstruction_input(args)
    args.camera_aspect = loaded.camera_aspect
    points, poses, scale = normalize_scene(loaded.cloud.points, loaded.poses, args.target_radius)
    colors = loaded.cloud.colors
    point_pose_idx = loaded.point_pose_idx
    source = {"npz": "NPZ frames", "npy": "NPY depth frames"}[loaded.source]
    print(f"[assign] {points.shape[0]} points -> {source}, camera-aspect={args.camera_aspect:.6g}")
    order = np.argsort(point_pose_idx)
    points, colors, point_pose_idx = points[order], colors[order], point_pose_idx[order]

    sparse_pose_indices = sparse_reconstruction_pose_indices(poses, args)

    input_panel_width = args.width if image_paths else 0
    output_width = args.width + input_panel_width
    try:
        writer, actual_path = make_writer(mp4_path, args.fps, output_width, args.height)
    except Exception as exc:
        raise RuntimeError("Could not create video writer.") from exc

    print(
        f"[render] steps={len(sparse_pose_indices)}, stride={args.stride}, poses={poses.shape[0]}, "
        f"scale={scale:.6g}, size={output_width}x{args.height}, output={actual_path}"
    )

    smoothed = None
    frame_counter = 0
    try:
        for progress in reconstruction_progress_frames(poses, args, bool(image_paths)):
            point_end = int(np.searchsorted(point_pose_idx, progress.reveal_until, side="right"))
            input_panel = load_input_panel(image_paths, progress.pose_idx, poses.shape[0], input_panel_width, args.height)
            reconstruction, smoothed = render_frame(
                points[:point_end],
                colors[:point_end],
                progress.sparse_pose_indices,
                progress.pose_idx,
                progress.highlight_pose_idx,
                points,
                colors,
                poses,
                args,
                smoothed,
            )
            image = compose_frame(input_panel, reconstruction)
            writer.append_data(image)
            frame_counter += 1
            if progress.step is not None:
                print(
                    f"[render] {progress.step + 1:03d}/{len(sparse_pose_indices)} "
                    f"pose={progress.pose_idx:04d} reveal<={progress.reveal_until:04d} points={point_end}"
                )
    finally:
        writer.close()

    print(f"[done] wrote video: {actual_path}")
