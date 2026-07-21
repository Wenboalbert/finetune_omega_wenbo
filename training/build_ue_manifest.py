#!/usr/bin/env python3
"""Build synchronized-camera JSONL from one or more UE scene directories."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from PIL import Image

from ftlib.ue_camera import intrinsics_from_ue_hfov, ue_pose_to_opencv_extrinsic


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-root", action="append", required=True, help="repeat for multiple scenes")
    parser.add_argument("--output", required=True)
    parser.add_argument("--anchors", default="CCTV_01,CCTV_02,CCTV_03,CCTV_04")
    parser.add_argument("--cameras", default="", help="optional comma-separated allow-list")
    parser.add_argument("--frame-start", type=int)
    parser.add_argument("--frame-end", type=int)
    parser.add_argument("--position-scale", type=float, default=0.01, help="CSV position units to metres")
    return parser.parse_args()


def camera_dirs(scene_root: Path) -> dict[str, Path]:
    return {
        path.name.casefold(): path
        for path in scene_root.iterdir()
        if path.is_dir() and path.name.casefold() != "posedata"
    }


def resolve_image(scene_root: Path, dirs: dict[str, Path], camera: str, image_name: str) -> Path:
    camera_dir = dirs.get(camera.casefold())
    if camera_dir is None:
        raise FileNotFoundError(f"camera directory {camera!r} absent under {scene_root}")
    candidate = camera_dir / image_name
    if candidate.is_file():
        return candidate
    matches = [p for p in camera_dir.iterdir() if p.is_file() and p.name.casefold() == image_name.casefold()]
    if len(matches) == 1:
        return matches[0]
    raise FileNotFoundError(f"image {image_name!r} absent under {camera_dir}")


def scene_records(scene_root, anchors, allow_cameras, frame_start, frame_end, position_scale):
    csv_path = scene_root / "PoseData" / "camera_parameters.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)
    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    required = {
        "frame_id", "camera_name", "image_name", "pos_x", "pos_y", "pos_z",
        "pitch", "yaw", "roll", "fov",
    }
    if not rows or required - set(rows[0]):
        raise ValueError(f"invalid camera CSV {csv_path}; missing={sorted(required - set(rows[0] if rows else []))}")
    grouped = defaultdict(list)
    for row in rows:
        camera = row["camera_name"].strip()
        frame = int(row["frame_id"].strip())
        if allow_cameras and camera not in allow_cameras:
            continue
        if frame_start is not None and frame < frame_start:
            continue
        if frame_end is not None and frame > frame_end:
            continue
        grouped[frame].append(row)

    dirs = camera_dirs(scene_root)
    anchor_set = set(anchors)
    records = []
    for frame in sorted(grouped):
        views = []
        for row in sorted(grouped[frame], key=lambda item: item["camera_name"].strip()):
            camera = row["camera_name"].strip()
            image_name = row["image_name"].strip()
            image_path = resolve_image(scene_root, dirs, camera, image_name)
            with Image.open(image_path) as image:
                width, height = image.size
            hfov = float(row["fov"])
            extrinsic = ue_pose_to_opencv_extrinsic(
                (float(row["pos_x"]), float(row["pos_y"]), float(row["pos_z"])),
                float(row["pitch"]), float(row["yaw"]), float(row["roll"]),
                position_scale,
            )
            intrinsics = intrinsics_from_ue_hfov(width, height, hfov)
            views.append({
                "camera": camera,
                "role": "anchor" if camera in anchor_set else "target",
                "rgb": str(image_path.resolve()),
                "image_name": image_name,
                "original_size_wh": [width, height],
                "hfov_deg": hfov,
                "intrinsics": intrinsics.tolist(),
                "extrinsics": extrinsic.tolist(),
            })
        present = {view["camera"] for view in views}
        missing = anchor_set - present
        if missing:
            raise ValueError(f"{scene_root.name} frame {frame} missing anchors {sorted(missing)}")
        if not (present - anchor_set):
            raise ValueError(f"{scene_root.name} frame {frame} has no target cameras")
        records.append({
            "schema_version": 1,
            "scene_id": scene_root.name,
            "scene_root": str(scene_root.resolve()),
            "frame_id": frame,
            "anchors": anchors,
            "views": views,
        })
    if not records:
        raise ValueError(f"no records built from {scene_root}")
    return records


def main():
    args = parse_args()
    anchors = [part.strip() for part in args.anchors.split(",") if part.strip()]
    if len(anchors) < 3:
        raise ValueError("at least three anchors are required; four are recommended")
    allow = {part.strip() for part in args.cameras.split(",") if part.strip()} or None
    if allow and not set(anchors).issubset(allow):
        raise ValueError("--cameras must include all anchors")
    roots = [Path(text).expanduser().resolve() for text in args.scene_root]
    if len({root.name for root in roots}) != len(roots):
        raise ValueError("scene folder basenames must be unique")
    records = []
    for root in roots:
        records.extend(scene_records(
            root, anchors, allow, args.frame_start, args.frame_end, args.position_scale
        ))
    output = Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")
    print(json.dumps({
        "output": str(output.resolve()),
        "scenes": sorted({record["scene_id"] for record in records}),
        "records": len(records),
        "anchors": anchors,
        "cameras": sorted({view["camera"] for record in records for view in record["views"]}),
        "frame_range": [min(r["frame_id"] for r in records), max(r["frame_id"] for r in records)],
    }, indent=2))


if __name__ == "__main__":
    main()
