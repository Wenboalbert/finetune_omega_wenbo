# Visualization: 3D reconstruction reveal videos

Turn a VGGT-Ω reconstruction into a video where the **colored point cloud, camera trajectory, and
view frustums build up frame by frame** as the input clip is revealed. It renders a single method, or
a synchronized **`Input | GT | VGGT-Omega | Ours`** side-by-side comparison.

Pure **NumPy + Pillow** software rasterizer: no GPU, no Open3D, no OpenGL, **and no PyTorch or
VGGT-Ω**: this tool is standalone and will render any reconstruction, whatever produced it. MP4 is
written with whichever backend is on the machine (ffmpeg preferred; OpenCV / imageio fallback; install
ffmpeg for real H.264).

> See the repo [root README](../README.md) for how the two tools fit together.

Two entry points:

| script | what it renders |
|---|---|
| [`render_reconstruction_video.py`](render_reconstruction_video.py) | one reconstruction (GT, frozen, or finetuned) |
| [`render_omega_samples_comparison.py`](render_omega_samples_comparison.py) | all methods side by side, scale-aligned & reveal-synchronized |

```bash
cd visualization
PYTHON=/path/to/python bash run.sh          # ready-to-edit example commands
```

## Inputs: bring your own reconstruction

You supply the reconstruction data; nothing is bundled. Two formats are accepted.

**A. Per-method sample directory** (drives the comparison renderer). One directory per clip of `S`
frames, holding a depth map + camera pose for each of **GT / ours (finetuned) / original (frozen)**,
plus RGB and intrinsics so depth can be unprojected:

| file | shape | meaning |
|---|---|---|
| `rgb_*.png` | H×W×3 | one RGB image per frame, used to colour the points. Frames are **natural-sorted**, so `rgb_2` comes before `rgb_10`. |
| `K.npy` | (S,3,3) or (3,3) | pinhole intrinsics **in the depth map's pixel coordinates**, not the RGB's. The RGB is resized/cropped onto the depth grid, so if the two differ it is `K` at the *depth* resolution that is correct. |
| `depth_gt.npy` / `depth_ours.npy` / `depth_original.npy` | (S,H,W) float | depth per method. **`<= 0` or non-finite = invalid** and is dropped. |
| `pose_gt.npy` / `pose_ours.npy` / `pose_original.npy` | (S,4,4) or (S,3,4) | camera pose, **cam-from-world** (w2c, OpenCV). The comparison renderer assumes w2c and has no flag, so invert first if yours is c2w. |
| `meta.json` | | **optional.** Only two keys are read: `scale_original_to_gt` and `scale_ours_to_gt`, each a list of `S` floats putting that method on the GT scale. Omit them (or omit the file's keys) and they default to `1.0`, which is what you want when your methods already share a scale. |

The method names, labels and file names above are **fixed**: the comparison renderer always renders
`Input | GT | VGGT-Omega | Ours` from exactly these files.

**B. A single-method NPZ.** `predictions.npz`, accepted directly by `render_reconstruction_video.py`:

| key | shape | note |
|---|---|---|
| `world_points_from_depth` | (N,H,W,3) float | already-unprojected world points |
| `images` | (N,3,H,W) or (N,H,W,3) | uint8, or float in `[0,1]` |
| `extrinsic` | (N,3,4) or (N,4,4) | **cam-from-world (w2c)**; `--npy-pose-convention` does *not* apply to this path |

Note: if a `predictions.npz` exists in `--result-dir` it **takes priority** over the NPY files and the
`--*-npy` flags are ignored.

## `render_reconstruction_video.py`: one reconstruction

Reads `predictions.npz` if present, otherwise NPY (`K.npy`, `depth*.npy`, `pose*.npy`); unprojects
depth into a colored cloud and reveals it over time.

```bash
python render_reconstruction_video.py --view follow \
    --result-dir samples/<name> --depth-npy depth_ours.npy --pose-npy pose_ours.npy \
    --image-dir samples/<name>
```

- Pick the method with `--depth-npy` / `--pose-npy` (`depth_ours.npy`, `depth_original.npy`, `depth_gt.npy`).
- Use `--npy-pose-convention c2w` if the camera flies away (default `w2c`).
- Point directly at an NPZ with `--npz-path`, or an NPY dir with `--npy-dir`.

## `render_omega_samples_comparison.py`: side-by-side comparison

Renders the panels together, aligning each method to GT scale via `meta.json` and placing every
method's first camera at a common origin so the reveal stays synchronized.

```bash
# one clip:
python render_omega_samples_comparison.py --sample <name> --view follow \
    --samples-root samples --output demo/<name>_comparison_follow.mp4

# batch every clip under --samples-root:
python render_omega_samples_comparison.py --view topdown --view-motion fixed \
    --samples-root samples --output-root outputs --skip-existing
```

## Views

- **follow**: third-person chase behind the current camera (`--chase-back/-right/-up/-lookahead`, `--fov-deg`).
- **topdown fixed**: fixed orthographic bird's-eye (`--top-axis`, `--top-margin`, `--top-zoom`, `--top-yaw-deg`).
- **topdown cinematic**: push-in then orbit (`--motion-start-zoom`, `--motion-push-ratio`, `--motion-orbit-deg`).

## Common options

`--width` / `--height` / `--fps`, `--stride` (frustum sparsity + reveal granularity), `--point-size`
(saturates at 3), `--frustum-scale`, `--camera-fov-deg`, `--target-radius`, `--max-steps`, `--output`,
and `--no-labels` (text-free frames; it drops the panel captions and the frame counter, handy for figures).

**Memory.** Every valid pixel of every frame becomes a point: a 100-frame 512x512 clip is ~26M points (~300 MB), and the comparison renderer holds three scenes at once. Shorten the clip or downsample the depth if you run out of RAM.

`--frames-per-step` interpolates the reveal, but **only when there is no input panel**; with
`--image-dir` set (as in every example above) the schedule is one video frame per pose, so raise
`--fps` instead. Outputs default to `demo/` for the single renderer and `outputs/` for the comparison
renderer.
