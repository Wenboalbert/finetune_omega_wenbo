#!/usr/bin/env python3
"""Render synchronized GT, VGGT-Omega, and Ours reconstruction videos."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from utils.inputs import resize_rgb_to_frame, unproject_depth_frames
from utils.renderers import render_follow_frame, render_topdown_frame
from utils.video import load_input_panel, reconstruction_progress_frames, sorted_input_images
from utils.video_writers import make_ffmpeg_first_writer


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



MODEL_SPECS = (
    ("gt", "GT", None),
    ("original", "VGGT-Omega", "scale_original_to_gt"),
    ("ours", "Ours", "scale_ours_to_gt"),
)

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SAMPLES_ROOT = SCRIPT_DIR / "samples"
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "outputs"


@dataclass
class ModelScene:
    key: str
    label: str
    points: np.ndarray
    colors: np.ndarray
    poses: np.ndarray
    point_pose_idx: np.ndarray
    camera_aspect: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--samples-root",
        type=Path,
        default=DEFAULT_SAMPLES_ROOT,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )
    parser.add_argument("--sample", action="append", help="Render only this sample name. Can be repeated.")
    parser.add_argument("--output", type=Path, default=None, help="Output path when rendering one sample.")
    parser.add_argument("--image-dir", type=Path, default=None, help="Optional source-image directory for the left panel.")
    parser.add_argument("--no-input-panel", action="store_true")
    parser.add_argument("--pose-convention", choices=("w2c", "c2w"), default="w2c",
                        help="convention of pose_*.npy. Feeding c2w as w2c renders a plausible but "
                             "WRONG cloud, with no error.")
    parser.add_argument("--no-labels", action="store_true",
                        help="do not draw the Input / GT / method captions on the panels")
    parser.add_argument("--skip-existing", action="store_true")

    progress_args = parser.add_argument_group("Reconstruction Progress")
    progress_args.add_argument("--stride", type=int, default=20)
    progress_args.add_argument("--max-steps", type=int, default=None)
    progress_args.add_argument("--frames-per-step", type=int, default=4)
    progress_args.add_argument("--target-radius", type=float, default=10.0)

    render_args = parser.add_argument_group("Rendering")
    render_args.add_argument("--width", type=int, default=512)
    render_args.add_argument("--height", type=int, default=512)
    render_args.add_argument("--fps", type=int, default=12)
    render_args.add_argument("--point-size", type=int, default=2)
    render_args.add_argument("--frustum-scale", type=float, default=0.65)
    render_args.add_argument("--camera-fov-deg", type=float, default=55.0)
    render_args.add_argument("--view", choices=("follow", "topdown"), default="follow")

    follow_args = parser.add_argument_group("Follow View")
    follow_args.add_argument("--chase-back", type=float, default=0.05)
    follow_args.add_argument("--chase-right", type=float, default=0.03)
    follow_args.add_argument("--chase-up", type=float, default=0.03)
    follow_args.add_argument("--chase-lookahead", type=float, default=1.0)
    follow_args.add_argument("--fov-deg", type=float, default=50.0)

    topdown_args = parser.add_argument_group("Top-Down View")
    topdown_args.add_argument(
        "--top-axis",
        choices=("path-normal", "camera-up", "+x", "-x", "+y", "-y", "+z", "-z"),
        default="-y",
    )
    topdown_args.add_argument("--top-margin", type=float, default=1.14)
    topdown_args.add_argument("--top-zoom", type=float, default=1.3)
    topdown_args.add_argument("--top-yaw-deg", type=float, default=0.0)
    topdown_args.add_argument("--top-flip-vertical", action="store_true")
    topdown_args.add_argument("--view-motion", choices=("cinematic", "fixed"), default="cinematic")
    topdown_args.add_argument("--motion-start-zoom", type=float, default=0.62)
    topdown_args.add_argument("--motion-push-ratio", type=float, default=0.45)
    topdown_args.add_argument("--motion-orbit-deg", type=float, default=30.0)
    topdown_args.add_argument("--motion-orbit-overlap", type=float, default=0.35)
    return parser.parse_args()


def load_rgb_images(sample_dir: Path) -> list[Image.Image]:
    paths = sorted_input_images(sample_dir)
    if not paths:
        raise FileNotFoundError(f"No source images found under {sample_dir}")
    images = []
    for path in paths:
        with Image.open(path) as image:
            images.append(image.convert("RGB"))
    return images


def scale_pose_translations(poses_w2c: np.ndarray, scales: np.ndarray) -> np.ndarray:
    scaled = poses_w2c.astype(np.float32, copy=True)
    scaled[:, :3, 3] *= scales[:, None]
    return scaled


def load_model_scene(
    sample_dir: Path,
    key: str,
    label: str,
    scale_key: str | None,
    metadata: dict,
    rgb_images: list[Image.Image],
    pose_convention: str = "w2c",
) -> ModelScene:
    intrinsics = np.load(sample_dir / "K.npy").astype(np.float32)
    depths = np.load(sample_dir / f"depth_{key}.npy").astype(np.float32)
    poses_w2c = np.load(sample_dir / f"pose_{key}.npy").astype(np.float32)
    if depths.ndim != 3:
        raise ValueError(f"{sample_dir.name} {key} depth must have shape (N,H,W), got {depths.shape}")
    frame_count, height, width = depths.shape
    if len(rgb_images) < frame_count:
        raise ValueError(f"{sample_dir.name} has {len(rgb_images)} RGB frames but {key} has {frame_count} depth frames")

    if scale_key is not None:
        # Optional per-frame factor putting this method on the GT scale. Absent -> 1.0, which is
        # what you want when comparing two of your own checkpoints that already share a scale.
        scales = np.asarray(metadata.get(scale_key, np.ones(len(depths), np.float32)),
                            dtype=np.float32)
        if scales.shape != (frame_count,):
            raise ValueError(f"{scale_key} must have shape {(frame_count,)}, got {scales.shape}")
        depths *= scales[:, None, None]
        poses_w2c = scale_pose_translations(poses_w2c, scales)

    colors_by_frame = [resize_rgb_to_frame(image, width, height) for image in rgb_images[:frame_count]]
    cloud, poses_c2w, point_pose_idx = unproject_depth_frames(
        intrinsics,
        depths,
        poses_w2c,
        pose_convention=pose_convention,
        colors_by_frame=colors_by_frame,
    )

    # Put every method's first camera at the same origin before common scaling.
    alignment = np.linalg.inv(poses_c2w[0])
    points = cloud.points @ alignment[:3, :3].T + alignment[:3, 3]
    poses = alignment @ poses_c2w
    order = np.argsort(point_pose_idx)
    return ModelScene(
        key,
        label,
        points[order].astype(np.float32, copy=False),
        cloud.colors[order].astype(np.uint8, copy=False),
        poses.astype(np.float32, copy=False),
        point_pose_idx[order],
        width / float(height),
    )


def normalize_scenes(scenes: list[ModelScene], target_radius: float) -> float:
    point_chunks = [scene.points for scene in scenes if scene.points.size]
    pose_chunks = [scene.poses[:, :3, 3] for scene in scenes]
    all_positions = np.concatenate([*point_chunks, *pose_chunks], axis=0)
    center = np.mean(all_positions, axis=0, keepdims=True)
    centered = all_positions - center
    radius = max(float(np.linalg.norm(centered.max(axis=0) - centered.min(axis=0)) * 0.5), 1e-9)
    scale = target_radius / radius if target_radius > 0 else 1.0
    for scene in scenes:
        scene.points = ((scene.points - center) * scale).astype(np.float32)
        scene.poses = scene.poses.copy()
        scene.poses[:, :3, 3] = ((scene.poses[:, :3, 3] - center) * scale).astype(np.float32)
    return scale


def renderer_args(args: argparse.Namespace, camera_aspect: float) -> SimpleNamespace:
    render_args = SimpleNamespace(**vars(args))
    render_args.camera_aspect = camera_aspect
    render_args.no_frame_text = True
    return render_args


def draw_label(image: np.ndarray, label: str) -> np.ndarray:
    panel = Image.fromarray(image)
    draw = ImageDraw.Draw(panel, "RGB")
    font = _label_font("DejaVuSans-Bold.ttf", 18)
    bbox = draw.textbbox((16, 16), label, font=font)
    pad = 6
    draw.rounded_rectangle((bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad), radius=4, fill=(255, 255, 255))
    draw.text((16, 16), label, fill=(25, 28, 32), font=font)
    return np.asarray(panel, dtype=np.uint8)


def output_path(sample_dir: Path, output_root: Path, args: argparse.Namespace) -> Path:
    if args.output is not None:
        return args.output
    suffix = "follow" if args.view == "follow" else f"topdown_{args.view_motion}"
    return output_root / sample_dir.name / f"{sample_dir.name}_comparison_{suffix}.mp4"


def render_sample(sample_dir: Path, output_root: Path, args: argparse.Namespace) -> Path:
    with (sample_dir / "meta.json").open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    rgb_images = load_rgb_images(sample_dir)
    scenes = [
        load_model_scene(sample_dir, key, label, scale_key, metadata, rgb_images,
                         args.pose_convention)
        for key, label, scale_key in MODEL_SPECS
    ]
    pose_count = scenes[0].poses.shape[0]
    if any(scene.poses.shape[0] != pose_count for scene in scenes[1:]):
        raise ValueError(f"{sample_dir.name} methods have different pose counts")
    scale = normalize_scenes(scenes, args.target_radius)

    image_dir = (args.image_dir or sample_dir).expanduser().resolve()
    image_paths = [] if args.no_input_panel else sorted_input_images(image_dir)
    if not args.no_input_panel and not image_paths:
        print(f"[load] input images: none found under {image_dir}; rendering method panels only")
    elif image_paths:
        print(f"[load] input images: {len(image_paths)} from {image_dir}")

    panel_width = args.width if image_paths else 0
    frame_width = args.width * len(scenes) + panel_width
    path = output_path(sample_dir, output_root, args)
    if path.exists() and args.skip_existing:
        print(f"[skip] {path}")
        return path
    writer, actual_path = make_ffmpeg_first_writer(path, args.fps, frame_width, args.height)
    render_fn = render_follow_frame if args.view == "follow" else render_topdown_frame
    per_scene_args = {scene.key: renderer_args(args, scene.camera_aspect) for scene in scenes}
    smoothed = {scene.key: None for scene in scenes}
    point_counts = ", ".join(f"{scene.label}={scene.points.shape[0]}" for scene in scenes)
    print(
        f"[render] sample={sample_dir.name}, methods={point_counts}, poses={pose_count}, "
        f"scale={scale:.6g}, size={frame_width}x{args.height}, output={actual_path}"
    )

    written = 0
    try:
        for progress in reconstruction_progress_frames(scenes[0].poses, args, bool(image_paths)):
            panels = []
            if image_paths:
                source_panel = load_input_panel(image_paths, progress.pose_idx, pose_count, args.width, args.height)
                if source_panel is not None:
                    panels.append(source_panel if args.no_labels else draw_label(source_panel, "Input"))
            for scene in scenes:
                point_end = int(np.searchsorted(scene.point_pose_idx, progress.reveal_until, side="right"))
                panel, smoothed[scene.key] = render_fn(
                    scene.points[:point_end],
                    scene.colors[:point_end],
                    progress.sparse_pose_indices,
                    progress.pose_idx,
                    progress.highlight_pose_idx,
                    scene.points,
                    scene.colors,
                    scene.poses,
                    per_scene_args[scene.key],
                    smoothed[scene.key],
                )
                panels.append(panel if args.no_labels else draw_label(panel, scene.label))
            writer.append_data(np.concatenate(panels, axis=1))
            written += 1
    finally:
        writer.close()

    print(f"[done] wrote comparison video: {actual_path} ({written} frames)")
    return actual_path


def selected_samples(samples_root: Path, names: list[str] | None) -> list[Path]:
    if not samples_root.is_dir():
        raise SystemExit(
            f"--samples-root does not exist: {samples_root}\n"
            "This repo ships no reconstruction data -- you supply it. See visualization/README.md ('Inputs') for the two accepted layouts: a per-method sample directory (K.npy / depth_*.npy / pose_*.npy / rgb_*.png / meta.json), or a single predictions.npz.")
    samples = sorted(path for path in samples_root.iterdir() if path.is_dir())
    if names is None:
        return samples
    requested = set(names)
    selected = [path for path in samples if path.name in requested]
    missing = sorted(requested - {path.name for path in selected})
    if missing:
        raise FileNotFoundError(f"Missing requested samples: {', '.join(missing)}")
    return selected


def main() -> None:
    args = parse_args()
    args.samples_root = args.samples_root.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    args.width = max(2, args.width + args.width % 2)
    args.height = max(2, args.height + args.height % 2)
    args.stride = max(1, args.stride)
    args.frames_per_step = max(1, args.frames_per_step)
    samples = selected_samples(args.samples_root, args.sample)
    if not samples:
        raise FileNotFoundError(f"No samples found under {args.samples_root}")
    if args.output is not None and len(samples) != 1:
        raise ValueError("--output requires exactly one selected sample")
    for sample_dir in samples:
        render_sample(sample_dir, args.output_root, args)


if __name__ == "__main__":
    main()
