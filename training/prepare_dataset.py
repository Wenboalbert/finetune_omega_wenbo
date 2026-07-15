"""Prepare supported datasets for finetune-omega manifests.

One command handles the local benchmark datasets used here:

  python training/prepare_dataset.py eth3d <eth3d_root> --out-dir <eth3d_prepared> --undistort
  python training/prepare_dataset.py 7scenes <7Scenes_root> --out-dir <7scenes_prepared>
  python training/prepare_dataset.py nrgbd <nrgbd_root> --out-dir <nrgbd_prepared>
  python training/prepare_dataset.py tum-dynamic <tum_root> --out-dir <tum_dynamic_prepared>

Each subcommand writes scene-level JSONL manifests with the training schema:

    {"frames": [
      {"rgb": "...", "depth": "...npy", "intrinsics": [[...]], "extrinsics": [[...]]},
      ...
    ]}

Depth arrays are float .npy maps in meters with 0 meaning invalid. Extrinsics are cam-from-world in
OpenCV convention. ETH3D can also undistort its THIN_PRISM_FISHEYE RGB/depth into a pinhole camera.
"""

import argparse
import bisect
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image


DEFAULT_ETH3D_TEST = ("relief", "relief_2", "terrains")
DEFAULT_7SCENES_TEST = ("heads", "stairs")
DEFAULT_NRGBD_TEST = ("staircase", "thin_geometry", "whiteroom")
DEFAULT_TUM_DYNAMIC_TEST = (
    "rgbd_dataset_freiburg3_walking_static",
    "rgbd_dataset_freiburg3_walking_xyz",
)

OPENGL_TO_OPENCV = np.diag([1.0, -1.0, -1.0, 1.0])


def natural_key(text):
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", str(text))]


def parse_names(text):
    if not text:
        return None
    return tuple(name.strip() for name in text.split(",") if name.strip())


def write_manifest(path, scenes):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        for scene in scenes:
            handle.write(json.dumps({"frames": scene["frames"]}, separators=(",", ":")) + "\n")


def invert_pose(pose):
    pose = np.asarray(pose, np.float64)
    inv = np.eye(4, dtype=np.float64)
    R = pose[:3, :3]
    t = pose[:3, 3]
    inv[:3, :3] = R.T
    inv[:3, 3] = -R.T @ t
    return inv


def c2w_opencv_to_w2c(c2w):
    return invert_pose(c2w)


def c2w_opengl_to_w2c_opencv(c2w_gl):
    c2w_cv = np.asarray(c2w_gl, np.float64) @ OPENGL_TO_OPENCV
    return invert_pose(c2w_cv)


def load_matrix(path):
    mat = np.loadtxt(path, dtype=np.float64)
    mat = mat.reshape(4, 4)
    if not np.isfinite(mat).all():
        raise ValueError(f"non-finite pose in {path}")
    return mat


def load_pose_stack(path):
    values = np.loadtxt(path, dtype=np.float64)
    if values.size % 16 != 0:
        raise ValueError(f"{path} does not contain 4x4 pose blocks")
    poses = values.reshape(-1, 4, 4)
    return poses


def save_depth_png_as_npy(png_path, npy_path, scale, invalid_max=None, overwrite=False):
    if not overwrite and npy_path.is_file():
        try:
            return np.load(npy_path, mmap_mode="r")
        except Exception:
            pass

    raw = np.asarray(Image.open(png_path))
    if raw.ndim != 2:
        raise ValueError(f"depth must be single-channel: {png_path} has shape {raw.shape}")
    depth = raw.astype(np.float32) / float(scale)
    invalid = raw <= 0
    if invalid_max is not None:
        invalid |= raw >= invalid_max
    depth[invalid | ~np.isfinite(depth)] = 0.0
    npy_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = npy_path.with_suffix(npy_path.suffix + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()
    with open(tmp_path, "wb") as handle:
        np.save(handle, depth)
    tmp_path.replace(npy_path)
    return depth


def depth_coverage(depth):
    return float((np.asarray(depth) > 0).sum() / np.asarray(depth).size)


def image_size(path):
    with Image.open(path) as im:
        return im.size


def build_scene_record(name, frames, coverages):
    coverage = 100.0 * float(np.mean(coverages)) if coverages else 0.0
    return {"name": name, "frames": frames, "coverage": coverage}


def split_scenes(prepared, train_names, test_names):
    test_set = set(test_names or ())
    if train_names is None:
        train_set = {scene["name"] for scene in prepared if scene["name"] not in test_set}
    else:
        train_set = set(train_names)
    train = [scene for scene in prepared if scene["name"] in train_set]
    test = [scene for scene in prepared if scene["name"] in test_set]
    all_names = {scene["name"] for scene in prepared}
    missing = sorted((train_set | test_set) - all_names)
    return train, test, missing


def maybe_resolve_7scenes_root(root):
    root = root.resolve()
    if (root / "7Scenes").is_dir():
        return root / "7Scenes"
    if (root / "7scenes" / "7Scenes").is_dir():
        return root / "7scenes" / "7Scenes"
    return root


def qvec_to_rotmat(qvec):
    qvec = np.asarray(qvec, np.float64)
    qvec = qvec / np.linalg.norm(qvec)
    w, x, y, z = qvec
    return np.array([
        [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * z * x + 2 * y * w],
        [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
        [2 * z * x - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y],
    ], dtype=np.float64)


def read_cameras(path):
    cameras = {}
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split()
            camera_id = int(fields[0])
            model = fields[1]
            width, height = int(fields[2]), int(fields[3])
            params = [float(value) for value in fields[4:]]
            if model == "SIMPLE_PINHOLE":
                f, cx, cy = params[:3]
                fx, fy = f, f
            elif model in ("PINHOLE", "OPENCV", "OPENCV_FISHEYE", "THIN_PRISM_FISHEYE"):
                fx, fy, cx, cy = params[:4]
            elif model == "SIMPLE_RADIAL":
                f, cx, cy = params[:3]
                fx, fy = f, f
            else:
                raise ValueError(f"unsupported COLMAP camera model {model!r} in {path}")
            cameras[camera_id] = {
                "model": model,
                "width": width,
                "height": height,
                "params": params,
                "K": [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
            }
    return cameras


def read_images(path):
    images = {}
    with open(path) as handle:
        lines = [line.strip() for line in handle if line.strip() and not line.startswith("#")]

    index = 0
    while index < len(lines):
        fields = lines[index].split()
        if len(fields) < 10:
            raise ValueError(f"malformed image header in {path}: {lines[index]}")
        image_id = int(fields[0])
        qvec = [float(value) for value in fields[1:5]]
        tvec = [float(value) for value in fields[5:8]]
        camera_id = int(fields[8])
        name = " ".join(fields[9:])
        rot = qvec_to_rotmat(qvec)
        extrinsic = np.eye(4, dtype=np.float64)
        extrinsic[:3, :3] = rot
        extrinsic[:3, 3] = np.asarray(tvec, np.float64)
        images[name] = {
            "image_id": image_id,
            "camera_id": camera_id,
            "extrinsics": extrinsic.tolist(),
        }
        index += 2
    return images


def resize_depth(depth, max_side):
    if max_side <= 0 or max(depth.shape) <= max_side:
        return depth
    height, width = depth.shape
    scale = max_side / max(height, width)
    new_width = max(1, round(width * scale))
    new_height = max(1, round(height * scale))
    return np.asarray(Image.fromarray(depth).resize((new_width, new_height), Image.NEAREST), np.float32)


def read_raw_depth(path, height, width):
    depth = np.fromfile(path, dtype=np.float32)
    expected = height * width
    if depth.size != expected:
        raise ValueError(f"{path} has {depth.size} float32 values; expected {expected} for {width}x{height}")
    depth = depth.reshape(height, width)
    depth = depth.astype(np.float32, copy=False)
    depth[~np.isfinite(depth) | (depth <= 0)] = 0.0
    return depth


def target_camera(camera, max_side):
    src_width, src_height = camera["width"], camera["height"]
    scale = 1.0 if max_side <= 0 else min(1.0, max_side / max(src_width, src_height))
    width = max(1, round(src_width * scale))
    height = max(1, round(src_height * scale))
    K = np.asarray(camera["K"], np.float64).copy()
    K[0] *= width / src_width
    K[1] *= height / src_height
    return width, height, K


def thin_prism_fisheye_map(camera, max_side):
    width, height, K_dst = target_camera(camera, max_side)
    params = camera["params"]
    fx, fy, cx, cy = params[:4]
    k1, k2, p1, p2, k3, k4, sx1, sy1 = params[4:12]

    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    u = (xx - K_dst[0, 2]) / K_dst[0, 0]
    v = (yy - K_dst[1, 2]) / K_dst[1, 1]
    radius = np.sqrt(u * u + v * v)
    theta = np.arctan(radius)
    scale = np.ones_like(radius, dtype=np.float32)
    np.divide(theta, radius, out=scale, where=radius > 1e-12)
    uu = u * scale
    vv = v * scale

    u2 = uu * uu
    uv = uu * vv
    v2 = vv * vv
    r2 = u2 + v2
    r4 = r2 * r2
    r6 = r4 * r2
    r8 = r6 * r2
    radial = k1 * r2 + k2 * r4 + k3 * r6 + k4 * r8
    du = uu * radial + 2 * p1 * uv + p2 * (r2 + 2 * u2) + sx1 * r2
    dv = vv * radial + 2 * p2 * uv + p1 * (r2 + 2 * v2) + sy1 * r2
    map_x = (fx * (uu + du) + cx).astype(np.float32)
    map_y = (fy * (vv + dv) + cy).astype(np.float32)
    return map_x, map_y, K_dst


def remap_nearest(src, map_x, map_y, fill=0):
    height, width = src.shape[:2]
    xi = np.rint(map_x).astype(np.int64)
    yi = np.rint(map_y).astype(np.int64)
    valid = (xi >= 0) & (xi < width) & (yi >= 0) & (yi < height)
    out = np.full(map_x.shape + src.shape[2:], fill, dtype=src.dtype)
    out[valid] = src[yi[valid], xi[valid]]
    return out


def remap_bilinear(src, map_x, map_y, fill=0):
    src = src.astype(np.float32, copy=False)
    height, width = src.shape[:2]
    x0 = np.floor(map_x).astype(np.int64)
    y0 = np.floor(map_y).astype(np.int64)
    x1 = x0 + 1
    y1 = y0 + 1
    valid = (x0 >= 0) & (x1 < width) & (y0 >= 0) & (y1 < height)

    x0c = np.clip(x0, 0, width - 1)
    x1c = np.clip(x1, 0, width - 1)
    y0c = np.clip(y0, 0, height - 1)
    y1c = np.clip(y1, 0, height - 1)
    wx = (map_x - x0)[..., None]
    wy = (map_y - y0)[..., None]
    top = src[y0c, x0c] * (1 - wx) + src[y0c, x1c] * wx
    bottom = src[y1c, x0c] * (1 - wx) + src[y1c, x1c] * wx
    out = top * (1 - wy) + bottom * wy
    out[~valid] = fill
    return np.clip(out, 0, 255).astype(np.uint8)


def convert_frame(rgb_path, raw_depth_path, camera, image_out, depth_out, overwrite, depth_max_side, undistort):
    if not overwrite and image_out is not None and image_out.is_file() and depth_out.is_file():
        _, _, K_dst = target_camera(camera, depth_max_side)
        return np.load(depth_out, mmap_mode="r"), K_dst
    if not overwrite and image_out is None and depth_out.is_file():
        return np.load(depth_out, mmap_mode="r"), np.asarray(camera["K"], np.float64)

    raw_depth = read_raw_depth(raw_depth_path, camera["height"], camera["width"])
    if not undistort:
        depth = resize_depth(raw_depth, depth_max_side)
        np.save(depth_out, depth)
        return depth, np.asarray(camera["K"], np.float64)

    map_x, map_y, K_dst = thin_prism_fisheye_map(camera, depth_max_side)
    image = np.asarray(Image.open(rgb_path).convert("RGB"), np.uint8)
    rgb = remap_bilinear(image, map_x, map_y)
    depth = remap_nearest(raw_depth, map_x, map_y, 0.0).astype(np.float32, copy=False)
    Image.fromarray(rgb).save(image_out, quality=95)
    np.save(depth_out, depth)
    return depth, K_dst


def build_eth3d_scene(scene_dir, image_out_dir, depth_out_dir, overwrite, depth_max_side, undistort):
    calibration_dir = scene_dir / "dslr_calibration_jpg"
    image_dir = scene_dir / "images"
    raw_depth_dir = scene_dir / "ground_truth_depth"
    cameras = read_cameras(calibration_dir / "cameras.txt")
    images = read_images(calibration_dir / "images.txt")
    camera_models = sorted({camera["model"] for camera in cameras.values()})

    frames = []
    valid_counts = []
    for name in sorted(images, key=natural_key):
        rgb = image_dir / name
        raw_depth = raw_depth_dir / name
        if not rgb.is_file() or not raw_depth.is_file():
            continue
        image_meta = images[name]
        camera = cameras[image_meta["camera_id"]]
        depth_out = depth_out_dir / scene_dir.name / (Path(name).stem + ".npy")
        depth_out.parent.mkdir(parents=True, exist_ok=True)
        image_out = None
        manifest_rgb = rgb
        if undistort:
            image_out = image_out_dir / scene_dir.name / Path(name).name
            image_out.parent.mkdir(parents=True, exist_ok=True)
            manifest_rgb = image_out
        depth, K = convert_frame(
            rgb, raw_depth, camera, image_out, depth_out, overwrite, depth_max_side, undistort
        )
        valid_counts.append(float((depth > 0).sum() / depth.size))
        frames.append({
            "rgb": str(manifest_rgb.resolve()),
            "depth": str(depth_out.resolve()),
            "intrinsics": K.tolist(),
            "extrinsics": image_meta["extrinsics"],
        })

    return frames, valid_counts, camera_models


def prepare_eth3d(args):
    eth3d_root = Path(args.root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    image_out_dir = out_dir / "images_undistorted"
    depth_out_dir = out_dir / "depth_npy"
    manifest_dir = out_dir / "manifests"

    requested_train = parse_names(args.train_scenes)
    requested_test = parse_names(args.test_scenes) or ()
    requested_test_set = set(requested_test)

    scene_dirs = sorted(
        [path for path in eth3d_root.iterdir() if (path / "dslr_calibration_jpg" / "images.txt").is_file()],
        key=lambda path: natural_key(path.name),
    )
    assert scene_dirs, f"no ETH3D scenes found under {eth3d_root}"
    if requested_train is not None:
        selected_names = set(requested_train) | requested_test_set
        scene_dirs = [path for path in scene_dirs if path.name in selected_names]
        assert scene_dirs, "none of the requested train/test scenes exist"

    prepared = []
    print(f"[eth3d] root={eth3d_root}")
    print(f"[eth3d] out={out_dir}")
    print(f"[eth3d] mode={'undistort' if args.undistort else 'direct'}")
    for scene_dir in scene_dirs:
        frames, valid_counts, camera_models = build_eth3d_scene(
            scene_dir, image_out_dir, depth_out_dir, args.overwrite, args.depth_max_side, args.undistort
        )
        if len(frames) < args.min_frames:
            print(f"[eth3d] skip {scene_dir.name}: {len(frames)} usable frames")
            continue
        coverage = 100.0 * float(np.mean(valid_counts)) if valid_counts else 0.0
        prepared.append({"name": scene_dir.name, "frames": frames, "coverage": coverage})
        print(f"[eth3d] {scene_dir.name:14} frames={len(frames):3d} depth_valid={coverage:5.1f}% "
              f"camera={','.join(camera_models)}")

    if requested_train is None:
        train_names = {scene["name"] for scene in prepared if scene["name"] not in requested_test_set}
    else:
        train_names = set(requested_train)

    train = [scene for scene in prepared if scene["name"] in train_names]
    test = [scene for scene in prepared if scene["name"] in requested_test_set]
    all_named = {scene["name"] for scene in prepared}
    missing = sorted((train_names | requested_test_set) - all_named)
    if missing:
        print(f"[eth3d] warning: requested scene(s) not found or dropped: {', '.join(missing)}")
    assert train, "no train scenes selected"
    assert test, "no test scenes selected"

    write_manifest(manifest_dir / "eth3d_all.jsonl", prepared)
    write_manifest(manifest_dir / "eth3d_train.jsonl", train)
    write_manifest(manifest_dir / "eth3d_test.jsonl", test)

    train_frames = sum(len(scene["frames"]) for scene in train)
    test_frames = sum(len(scene["frames"]) for scene in test)
    print(f"[eth3d] wrote {manifest_dir / 'eth3d_train.jsonl'} ({len(train)} scenes, {train_frames} frames)")
    print(f"[eth3d] wrote {manifest_dir / 'eth3d_test.jsonl'} ({len(test)} scenes, {test_frames} frames)")
    print(f"[eth3d] converted max side: {args.depth_max_side or 'full'}")
    if args.undistort:
        print("[eth3d] undistorted RGB/depth with COLMAP THIN_PRISM_FISHEYE projection.")
    else:
        print("[eth3d] note: THIN_PRISM_FISHEYE distortion is approximated by fx/fy/cx/cy only.")


def prepare_7scenes(args):
    root = maybe_resolve_7scenes_root(Path(args.root).expanduser())
    out_dir = Path(args.out_dir).expanduser().resolve()
    depth_out_dir = out_dir / "depth_npy"
    manifest_dir = out_dir / "manifests"

    requested_train = parse_names(args.train_scenes)
    requested_test = parse_names(args.test_scenes) or ()

    scene_dirs = [
        path for path in root.iterdir()
        if path.is_dir() and path.name != "meshes" and list(path.glob("seq-*"))
    ]
    scene_dirs = sorted(scene_dirs, key=lambda path: natural_key(path.name))
    if requested_train is not None:
        selected = set(requested_train) | set(requested_test)
        scene_dirs = [path for path in scene_dirs if path.name in selected]
    assert scene_dirs, f"no 7Scenes scenes found under {root}"

    K = [[585.0, 0.0, 320.0], [0.0, 585.0, 240.0], [0.0, 0.0, 1.0]]
    prepared = []
    print(f"[7scenes] root={root}")
    print(f"[7scenes] out={out_dir}")
    for scene_dir in scene_dirs:
        frames = []
        coverages = []
        seq_dirs = sorted(scene_dir.glob("seq-*"), key=lambda path: natural_key(path.name))
        for seq_dir in seq_dirs:
            for rgb in sorted(seq_dir.glob("frame-*.color.png"), key=lambda path: natural_key(path.name)):
                stem = rgb.name.replace(".color.png", "")
                depth_png = seq_dir / f"{stem}.depth.png"
                pose_txt = seq_dir / f"{stem}.pose.txt"
                if not depth_png.is_file() or not pose_txt.is_file():
                    continue
                try:
                    c2w = load_matrix(pose_txt)
                except ValueError as exc:
                    print(f"[7scenes] skip {pose_txt}: {exc}")
                    continue
                depth_npy = depth_out_dir / scene_dir.name / seq_dir.name / f"{stem}.npy"
                depth = save_depth_png_as_npy(
                    depth_png, depth_npy, scale=1000.0, invalid_max=65535, overwrite=args.overwrite
                )
                coverages.append(depth_coverage(depth))
                frames.append({
                    "rgb": str(rgb.resolve()),
                    "depth": str(depth_npy.resolve()),
                    "intrinsics": K,
                    "extrinsics": c2w_opencv_to_w2c(c2w).tolist(),
                })
        if len(frames) < args.min_frames:
            print(f"[7scenes] skip {scene_dir.name}: {len(frames)} usable frames")
            continue
        record = build_scene_record(scene_dir.name, frames, coverages)
        prepared.append(record)
        print(f"[7scenes] {scene_dir.name:12} frames={len(frames):5d} depth_valid={record['coverage']:5.1f}%")

    train, test, missing = split_scenes(prepared, requested_train, requested_test)
    if missing:
        print(f"[7scenes] warning: requested scene(s) not found or dropped: {', '.join(missing)}")
    assert train, "no train scenes selected"
    assert test, "no test scenes selected"

    write_manifest(manifest_dir / "7scenes_all.jsonl", prepared)
    write_manifest(manifest_dir / "7scenes_train.jsonl", train)
    write_manifest(manifest_dir / "7scenes_test.jsonl", test)
    print(f"[7scenes] wrote {manifest_dir / '7scenes_train.jsonl'} ({len(train)} scenes)")
    print(f"[7scenes] wrote {manifest_dir / '7scenes_test.jsonl'} ({len(test)} scenes)")


def prepare_nrgbd(args):
    root = Path(args.root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    depth_out_dir = out_dir / "depth_npy"
    manifest_dir = out_dir / "manifests"

    requested_train = parse_names(args.train_scenes)
    requested_test = parse_names(args.test_scenes) or ()

    scene_dirs = sorted(
        [path for path in root.iterdir() if (path / "images").is_dir() and (path / "poses.txt").is_file()],
        key=lambda path: natural_key(path.name),
    )
    if requested_train is not None:
        selected = set(requested_train) | set(requested_test)
        scene_dirs = [path for path in scene_dirs if path.name in selected]
    assert scene_dirs, f"no NRGBD scenes found under {root}"

    prepared = []
    print(f"[nrgbd] root={root}")
    print(f"[nrgbd] out={out_dir}")
    print(f"[nrgbd] depth-dir={args.depth_dir_name}")
    for scene_dir in scene_dirs:
        image_dir = scene_dir / "images"
        depth_dir = scene_dir / args.depth_dir_name
        if not depth_dir.is_dir():
            depth_dir = scene_dir / "depth"
        poses = load_pose_stack(scene_dir / "poses.txt")
        f = float((scene_dir / "focal.txt").read_text().strip())
        rgb_paths = sorted(image_dir.glob("img*.png"), key=lambda path: natural_key(path.name))

        frames = []
        coverages = []
        for idx, rgb in enumerate(rgb_paths):
            if idx >= len(poses):
                break
            depth_png = depth_dir / f"depth{idx}.png"
            if not depth_png.is_file():
                continue
            width, height = image_size(rgb)
            K = [[f, 0.0, width / 2.0], [0.0, f, height / 2.0], [0.0, 0.0, 1.0]]
            depth_npy = depth_out_dir / scene_dir.name / f"depth{idx}.npy"
            depth = save_depth_png_as_npy(depth_png, depth_npy, scale=1000.0, overwrite=args.overwrite)
            coverages.append(depth_coverage(depth))
            frames.append({
                "rgb": str(rgb.resolve()),
                "depth": str(depth_npy.resolve()),
                "intrinsics": K,
                "extrinsics": c2w_opengl_to_w2c_opencv(poses[idx]).tolist(),
            })
        if len(frames) < args.min_frames:
            print(f"[nrgbd] skip {scene_dir.name}: {len(frames)} usable frames")
            continue
        record = build_scene_record(scene_dir.name, frames, coverages)
        prepared.append(record)
        print(f"[nrgbd] {scene_dir.name:18} frames={len(frames):5d} depth_valid={record['coverage']:5.1f}%")

    train, test, missing = split_scenes(prepared, requested_train, requested_test)
    if missing:
        print(f"[nrgbd] warning: requested scene(s) not found or dropped: {', '.join(missing)}")
    assert train, "no train scenes selected"
    assert test, "no test scenes selected"

    write_manifest(manifest_dir / "nrgbd_all.jsonl", prepared)
    write_manifest(manifest_dir / "nrgbd_train.jsonl", train)
    write_manifest(manifest_dir / "nrgbd_test.jsonl", test)
    print(f"[nrgbd] wrote {manifest_dir / 'nrgbd_train.jsonl'} ({len(train)} scenes)")
    print(f"[nrgbd] wrote {manifest_dir / 'nrgbd_test.jsonl'} ({len(test)} scenes)")


def read_tum_list(path):
    out = []
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split()
            out.append((float(fields[0]), fields[1:]))
    return out


def nearest_entry(entries, timestamp, max_delta):
    times = [item[0] for item in entries]
    idx = bisect.bisect_left(times, timestamp)
    best = None
    for cand in (idx - 1, idx):
        if 0 <= cand < len(entries):
            delta = abs(entries[cand][0] - timestamp)
            if best is None or delta < best[0]:
                best = (delta, entries[cand])
    if best is None or best[0] > max_delta:
        return None
    return best[1]


def quat_xyzw_to_rot(qx, qy, qz, qw):
    q = np.asarray([qw, qx, qy, qz], np.float64)
    q = q / max(np.linalg.norm(q), 1e-12)
    w, x, y, z = q
    return np.array([
        [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
        [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
        [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y],
    ], dtype=np.float64)


def tum_pose_from_fields(fields):
    tx, ty, tz, qx, qy, qz, qw = [float(value) for value in fields]
    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, :3] = quat_xyzw_to_rot(qx, qy, qz, qw)
    c2w[:3, 3] = [tx, ty, tz]
    return c2w


def tum_intrinsics(sequence_name):
    if "freiburg1" in sequence_name:
        return [[517.3, 0.0, 318.6], [0.0, 516.5, 255.3], [0.0, 0.0, 1.0]]
    if "freiburg2" in sequence_name:
        return [[520.9, 0.0, 325.1], [0.0, 521.0, 249.7], [0.0, 0.0, 1.0]]
    if "freiburg3" in sequence_name:
        return [[535.4, 0.0, 320.1], [0.0, 539.2, 247.6], [0.0, 0.0, 1.0]]
    raise ValueError(f"cannot infer TUM intrinsics for {sequence_name}")


def prepare_tum_dynamic(args):
    root = Path(args.root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    depth_out_dir = out_dir / "depth_npy"
    manifest_dir = out_dir / "manifests"

    requested_train = parse_names(args.train_scenes)
    requested_test = parse_names(args.test_scenes) or ()

    sequence_dirs = sorted(
        [path for path in root.iterdir() if (path / "rgb.txt").is_file() and (path / "depth.txt").is_file()],
        key=lambda path: natural_key(path.name),
    )
    if requested_train is not None:
        selected = set(requested_train) | set(requested_test)
        sequence_dirs = [path for path in sequence_dirs if path.name in selected]
    assert sequence_dirs, f"no TUM-Dynamic sequences found under {root}"

    prepared = []
    print(f"[tum-dynamic] root={root}")
    print(f"[tum-dynamic] out={out_dir}")
    print(f"[tum-dynamic] max-delta={args.max_delta}")
    for seq_dir in sequence_dirs:
        rgb_entries = read_tum_list(seq_dir / "rgb.txt")
        depth_entries = read_tum_list(seq_dir / "depth.txt")
        gt_entries = read_tum_list(seq_dir / "groundtruth.txt")
        K = tum_intrinsics(seq_dir.name)

        frames = []
        coverages = []
        for rgb_ts, rgb_fields in rgb_entries:
            depth_match = nearest_entry(depth_entries, rgb_ts, args.max_delta)
            gt_match = nearest_entry(gt_entries, rgb_ts, args.max_delta)
            if depth_match is None or gt_match is None:
                continue
            rgb = seq_dir / rgb_fields[0]
            depth_png = seq_dir / depth_match[1][0]
            if not rgb.is_file() or not depth_png.is_file():
                continue
            c2w = tum_pose_from_fields(gt_match[1])
            depth_npy = depth_out_dir / seq_dir.name / (Path(depth_png).stem + ".npy")
            depth = save_depth_png_as_npy(depth_png, depth_npy, scale=5000.0, overwrite=args.overwrite)
            coverages.append(depth_coverage(depth))
            frames.append({
                "rgb": str(rgb.resolve()),
                "depth": str(depth_npy.resolve()),
                "intrinsics": K,
                "extrinsics": c2w_opencv_to_w2c(c2w).tolist(),
            })
        if len(frames) < args.min_frames:
            print(f"[tum-dynamic] skip {seq_dir.name}: {len(frames)} usable frames")
            continue
        record = build_scene_record(seq_dir.name, frames, coverages)
        prepared.append(record)
        print(f"[tum-dynamic] {seq_dir.name:42} frames={len(frames):5d} depth_valid={record['coverage']:5.1f}%")

    train, test, missing = split_scenes(prepared, requested_train, requested_test)
    if missing:
        print(f"[tum-dynamic] warning: requested scene(s) not found or dropped: {', '.join(missing)}")
    assert train, "no train scenes selected"
    assert test, "no test scenes selected"

    write_manifest(manifest_dir / "tum_dynamic_all.jsonl", prepared)
    write_manifest(manifest_dir / "tum_dynamic_train.jsonl", train)
    write_manifest(manifest_dir / "tum_dynamic_test.jsonl", test)
    print(f"[tum-dynamic] wrote {manifest_dir / 'tum_dynamic_train.jsonl'} ({len(train)} scenes)")
    print(f"[tum-dynamic] wrote {manifest_dir / 'tum_dynamic_test.jsonl'} ({len(test)} scenes)")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="dataset", required=True)

    pe = sub.add_parser("eth3d", help="prepare ETH3D")
    pe.add_argument("root", help="path to the extracted ETH3D root containing scene folders")
    pe.add_argument("--out-dir", default="eth3d_prepared", help="directory for .npy depth + manifests")
    pe.add_argument("--train-scenes", default="", help="comma-separated scene names; default = all non-test scenes")
    pe.add_argument("--test-scenes", default=",".join(DEFAULT_ETH3D_TEST),
                    help="comma-separated scene names kept out of training")
    pe.add_argument("--min-frames", type=int, default=4, help="drop scenes with fewer usable RGB-D frames")
    pe.add_argument("--depth-max-side", type=int, default=2048,
                    help="converted depth/RGB longest side; 0 keeps full ETH3D resolution")
    pe.add_argument("--undistort", action="store_true", help="write undistorted RGB/depth and pinhole K")
    pe.add_argument("--overwrite", action="store_true", help="rewrite existing converted depth files")
    pe.set_defaults(func=prepare_eth3d)

    p7 = sub.add_parser("7scenes", help="prepare 7Scenes")
    p7.add_argument("root", help="path to 7Scenes root")
    p7.add_argument("--out-dir", default="7scenes_prepared")
    p7.add_argument("--train-scenes", default="", help="comma-separated scene names; default = all non-test scenes")
    p7.add_argument("--test-scenes", default=",".join(DEFAULT_7SCENES_TEST))
    p7.add_argument("--min-frames", type=int, default=4)
    p7.add_argument("--overwrite", action="store_true")
    p7.set_defaults(func=prepare_7scenes)

    pn = sub.add_parser("nrgbd", help="prepare Neural RGB-D")
    pn.add_argument("root", help="path to NRGBD root")
    pn.add_argument("--out-dir", default="nrgbd_prepared")
    pn.add_argument("--train-scenes", default="", help="comma-separated scene names; default = all non-test scenes")
    pn.add_argument("--test-scenes", default=",".join(DEFAULT_NRGBD_TEST))
    pn.add_argument("--depth-dir-name", default="depth_filtered", help="depth folder to use; falls back to depth")
    pn.add_argument("--min-frames", type=int, default=4)
    pn.add_argument("--overwrite", action="store_true")
    pn.set_defaults(func=prepare_nrgbd)

    pt = sub.add_parser("tum-dynamic", help="prepare TUM RGB-D Dynamic")
    pt.add_argument("root", help="path to TUM-Dynamic root")
    pt.add_argument("--out-dir", default="tum_dynamic_prepared")
    pt.add_argument("--train-scenes", default="", help="comma-separated sequence names; default = all non-test scenes")
    pt.add_argument("--test-scenes", default=",".join(DEFAULT_TUM_DYNAMIC_TEST))
    pt.add_argument("--max-delta", type=float, default=0.02, help="max timestamp delta for RGB/depth/GT association")
    pt.add_argument("--min-frames", type=int, default=4)
    pt.add_argument("--overwrite", action="store_true")
    pt.set_defaults(func=prepare_tum_dynamic)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

