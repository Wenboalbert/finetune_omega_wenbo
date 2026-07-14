#!/usr/bin/env python3
"""Render reconstruction progress from a follow or overhead observation view."""

from __future__ import annotations

import argparse
from pathlib import Path

from utils.renderers import render_follow_frame, render_topdown_frame
from utils.video import default_output_paths, run_reconstruction_video
from utils.video_writers import make_ffmpeg_first_writer


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SAMPLE_DIR = SCRIPT_DIR / "samples" / "example"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)

    # -------------------------------------------------------------------------
    # Reconstruction input: NPZ or NPY depth/intrinsics/pose data.
    # -------------------------------------------------------------------------
    input_args = parser.add_argument_group("Reconstruction Input")
    input_args.add_argument(
        "--result-dir",
        type=Path,
        default=DEFAULT_SAMPLE_DIR,
    )
    input_args.add_argument("--npz-path", type=Path, default=None)
    input_args.add_argument("--npy-dir", type=Path, default=None)
    input_args.add_argument("--intrinsics-npy", type=Path, default=None)
    input_args.add_argument("--depth-npy", type=Path, default=None)
    input_args.add_argument("--pose-npy", type=Path, default=None)
    input_args.add_argument("--npy-rgb-dir", type=Path, default=None)
    input_args.add_argument("--npy-pose-convention", choices=("w2c", "c2w"), default="w2c")

    # -------------------------------------------------------------------------
    # Output video and optional source-image panel.
    # -------------------------------------------------------------------------
    output_args = parser.add_argument_group("Output and Source Panel")
    output_args.add_argument("--output", type=Path, default=None)
    output_args.add_argument("--image-dir", type=Path, default=None)
    output_args.add_argument("--no-input-panel", action="store_true")
    output_args.add_argument("--no-labels", action="store_true",
                             help="hide all on-frame text overlays (the frame counter)")

    # -------------------------------------------------------------------------
    # Shared reconstruction-progress rendering options.
    # -------------------------------------------------------------------------
    render_args = parser.add_argument_group("Shared Rendering")
    render_args.add_argument("--stride", type=int, default=20)
    render_args.add_argument("--target-radius", type=float, default=10.0)
    render_args.add_argument("--width", type=int, default=1280)
    render_args.add_argument("--height", type=int, default=720)
    render_args.add_argument("--fps", type=int, default=60)
    render_args.add_argument("--frames-per-step", type=int, default=4)
    render_args.add_argument("--point-size", type=int, default=2)
    render_args.add_argument("--frustum-scale", type=float, default=0.65)
    render_args.add_argument("--camera-fov-deg", type=float, default=55.0)
    render_args.add_argument("--max-steps", type=int, default=None)

    # -------------------------------------------------------------------------
    # Virtual observation-view selection.
    # -------------------------------------------------------------------------
    view_args = parser.add_argument_group("Observation View")
    view_args.add_argument("--view", choices=("follow", "topdown"), default="follow")

    # -------------------------------------------------------------------------
    # Follow-view only: ignored when --view topdown is selected.
    # -------------------------------------------------------------------------
    follow_args = parser.add_argument_group("Follow View Only")
    follow_args.add_argument("--chase-back", type=float, default=0.05)
    follow_args.add_argument("--chase-right", type=float, default=0.03)
    follow_args.add_argument("--chase-up", type=float, default=0.03)
    follow_args.add_argument("--chase-lookahead", type=float, default=1.0)
    follow_args.add_argument("--fov-deg", type=float, default=50.0)

    # -------------------------------------------------------------------------
    # Global top-down view: camera direction and framing.
    # -------------------------------------------------------------------------
    topdown_args = parser.add_argument_group("Top-Down View Only")
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

    # -------------------------------------------------------------------------
    # Animated top-down view: ignored when --view-motion fixed is selected.
    # -------------------------------------------------------------------------
    motion_args = parser.add_argument_group("Cinematic Top-Down Only")
    motion_args.add_argument("--motion-start-zoom", type=float, default=0.62)
    motion_args.add_argument("--motion-push-ratio", type=float, default=0.45)
    motion_args.add_argument("--motion-orbit-deg", type=float, default=30.0)
    motion_args.add_argument("--motion-orbit-overlap", type=float, default=0.35)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.width = max(2, args.width + args.width % 2)      # libx264 + yuv420p needs even dimensions
    args.height = max(2, args.height + args.height % 2)
    root = args.npy_dir or args.result_dir
    if args.npz_path is None and root is not None and not Path(root).is_dir():
        raise SystemExit(
            f"--result-dir does not exist: {root}\n"
            "This repo ships no reconstruction data -- you supply it. See visualization/README.md ('Inputs') for the two accepted layouts: a per-method sample directory (K.npy / depth_*.npy / pose_*.npy / rgb_*.png / meta.json), or a single predictions.npz.")
    args.no_frame_text = args.no_labels          # --no-labels -> suppress the frame-counter overlay
    if args.view == "follow":
        renderer = render_follow_frame
        output_name = "follow"
    else:
        renderer = render_topdown_frame
        output_name = f"topdown_{args.view_motion}"
    run_reconstruction_video(
        args,
        default_output_paths(args, output_name),
        make_ffmpeg_first_writer,
        renderer,
    )


if __name__ == "__main__":
    main()
