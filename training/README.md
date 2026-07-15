# Training: an unofficial VGGT-Ω finetuning implementation

Finetune a pretrained [VGGT-Ω](https://github.com/facebookresearch/vggt-omega) on **your** dataset of
posed RGB-D frames. It jointly adapts the depth and camera heads (plus the last few backbone blocks)
under a held-out validation gate, and writes out a plain `VGGTOmega` checkpoint you can load with the
ordinary loader.

- Depth is the supervised target; camera extrinsics drive a pose objective.
- The backbone stays frozen except its last blocks, which train at a flat `0.15×` micro-LR. You adapt
  the top of the network without rewriting what it already knows.
- The **best** checkpoint is chosen on held-out scenes and saved **only when depth (AbsRel) *and*
  trajectory (ATE) both improve** and δ<1.25 does not regress (early-stop). Never on train loss.

> See the repo [root README](../README.md) for how the two tools fit together and how to obtain VGGT-Ω.

## Prerequisites

- A Python env with **PyTorch (CUDA)**, plus `pyyaml`, `numpy`, `pillow`.
- The official **VGGT-Ω** package importable. Put its checkout on `PYTHONPATH`, or set:
  ```bash
  export VGGT_OMEGA_PATH=/path/to/vggt-omega
  ```
- A pretrained VGGT-Ω checkpoint (e.g. `vggt_omega_1b_512.pt`) and a dataset in the manifest format below.

## Quickstart

```bash
# 1) Fill the two placeholders in configs/finetune.yaml:
#      init:          <VGGT_OMEGA_CKPT>   -> pretrained VGGT-Ω checkpoint
#      data.manifest: <TRAIN_MANIFEST>    -> your JSONL manifest (schema below)

cd training
export VGGT_OMEGA_PATH=/path/to/vggt-omega
bash run_train.sh configs/finetune.yaml
```

Outputs land in the config's `out:` directory (default `runs/finetune/`):

| file | what it is |
|---|---|
| `best.pt` | a **bare `VGGTOmega` `state_dict`**, the both-improve-selected checkpoint. Load it exactly like the pretrained weights. |
| `decision.json` | frozen-baseline vs. best metrics (depth AbsRel / δ<1.25, pose ATE / RPE-rot) and whether the gate passed. |

Prefer trajectory error (ATE)? Use `configs/finetune_cons.yaml`, which turns on the reprojection
depth↔pose consistency term (`w_cons`).

## Dataset presets

The supported benchmark manifests are prepared through one public entrypoint:

```bash
python training/prepare_dataset.py eth3d <eth3d_root> --out-dir <eth3d_prepared> --undistort
python training/prepare_dataset.py 7scenes <7scenes_root> --out-dir <7scenes_prepared>
python training/prepare_dataset.py nrgbd <nrgbd_root> --out-dir <nrgbd_prepared>
python training/prepare_dataset.py tum-dynamic <tum_root> --out-dir <tum_dynamic_prepared>
```

The preparation code for four datasets lives in `prepare_dataset.py`.

The trainer can consume either a regular YAML config path or one of these dataset preset names:

```bash
bash training/run_train.sh eth3d
bash training/run_train.sh 7scenes
bash training/run_train.sh nrgbd
bash training/run_train.sh tum-dynamic
```

Edit common hyperparameters in `configs/base.yaml`. Edit dataset paths and the few dataset-specific
knobs (`manifest`, `out`, `val_frac`, `max_per_scene`, `w_grad`) in `configs/datasets.yaml`.

## Data format: the only thing tying this to your dataset

One **JSONL manifest**, one JSON object per line describing a *scene* (a temporally ordered clip):

```json
{"frames": [
  {"rgb":        "<path/to/frame.jpg>",
   "depth":      "<path/to/depth.npy>",
   "intrinsics": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
   "extrinsics": [[r,r,r,tx], [r,r,r,ty], [r,r,r,tz], [0,0,0,1]]},
  ...
]}
```

| field | meaning |
|---|---|
| `rgb` | path to an RGB image, any resolution (it is resized to `img_size`). |
| `depth` | `HxW` float `.npy` depth map, the sole GT. **0 = invalid**, so a *sparse* map is fine (see below). Metric units are not required; the default loss is scale-invariant. |
| `intrinsics` | 3×3 pinhole `K` **in the RGB's native resolution** (rescaled internally to `img_size`). |
| `extrinsics` | 4×4 (or 3×4) **cam-from-world**, OpenCV convention. Feeds the pose loss. |

### Sparse depth is a first-class input

The depth loss and every metric mask on `depth > 0`, so you can supervise straight from a **COLMAP
point cloud** (reproject `points3D` into each frame) or from **LiDAR returns**, with no densification and no
depth completion. Coverage of ~1% of the pixels is enough. Two of the benchmarks in the root README's
results table are supervised exactly this way.

Two loss weights must change when the depth is sparse, or they will feed the model noise:

| key | dense | sparse | why |
|---|---|---|---|
| `w_grad` | `0.5` | **`0`** | Edge-aware gradient matching compares *adjacent* valid pixels. At ~1% coverage almost no neighbouring pair is both-valid, so the term contributes nothing but cost. |
| `w_cons` | `1.0` (in `finetune_cons.yaml`) | **`0`** | Reprojection consistency samples GT depth on a dense stride-32 grid. On a sparse map that grid is almost entirely invalid, and the term would supervise against zeros. |

**To plug in a new dataset, write a manifest. Nothing else changes.** Non-overlapping windows of
length `seq_len` are cut from each scene's frame list (at most `max_per_scene` per scene); validation
scenes are carved off whole (`val_frac`), so val is scene-disjoint. Keep any final test split in a
*separate* manifest and score it with [`eval.py`](eval.py). The full loader is one small file:
[`ftlib/data.py`](ftlib/data.py).

Things the loader will not tell you but will punish you for:

| | |
|---|---|
| **Paths** | resolved relative to the **process CWD** (i.e. `training/`), not to the manifest. Absolute paths are safest. |
| **Invalid depth** | anything `<= 0` or non-finite. Use `0` (or NaN), never a sentinel like `-1` or `1e9`. |
| **Depth shape** | exactly `(H, W)`. An `(H, W, 1)` array crashes in the resize. |
| **Images** | fed to the model as raw `[0,1]`. Do **not** ImageNet-normalize; the aggregator does its own normalization. |
| **Resolution** | every frame is resized to a **square** `img_size × img_size`, anisotropically. `K` is rescaled to match, so the geometry stays self-consistent, but a 16:9 clip *is* squashed. |
| **Scene count** | you need enough scenes that `val_frac` still leaves a train split. The run asserts this up front. |
| **`seq_len`** | must be ≥ 3, or the trajectory metrics are `nan` and the gate can never pass. Asserted. |

## Framing and intrinsics: start here

Every dataset arrives with a different resolution, aspect ratio, depth density and amount of camera
motion, and each of those maps onto a specific knob. **Point the inspector at your manifest and it
will read your data and print the config it needs:**

```bash
cd training
python inspect_manifest.py <your_manifest.jsonl>
```

On a wide-angle dataset it prints, among other things:

```
--- frames
    1408x376   aspect 3.74:1
--- intrinsics (as given, in the RGB's own pixels)
    fx 552.6  fy 552.6   cx 682.0  cy 238.8      implied FOV 104° x 38°
--- depth
    valid pixels 100.0%   ->  DENSE

RECOMMENDED CONFIG
data.aspect: pad
    aspect 3.74:1 is extreme. `square` would stretch every frame 3.7x, far from what the
    backbone was pretrained on. `pad` keeps the geometry and masks the padding out of the loss.
val.ate_stat: median
    your trajectories are near-collinear (straightness 1.00) ...
!! fewer than 2 val scenes ...
```

Everything below is what those recommendations mean.

### `data.aspect`: how a non-square frame reaches the square canvas

The backbone wants a square input. Your camera almost certainly does not produce one. You choose how
to bridge that:

| mode | what it does | when |
|---|---|---|
| **`square`** *(default)* | Stretches the frame to `img_size × img_size` and rescales `K` anisotropically to match. The pinhole model stays **exact**: the camera simply acquires non-square pixels. Nothing is cropped, nothing is invented. | Your frames are near 1:1, or you want to reproduce the runs in the results table. **Every number in the results table was trained this way, including sources as wide as 3.7:1.** |
| **`pad`** | Fits the long side and pads the short one. Padded pixels get depth 0, so they are masked out of the loss automatically. No distortion; you pay with unused canvas. | Your aspect ratio is extreme (≳ 2.2:1). A 3.7:1 frame stretched to a square is a long way from the aspect the backbone was pretrained on. |
| **`crop`** | Fits the short side and centre-crops the long one. No distortion, no padding, but you lose field of view. | You have plenty of frames and would rather throw away the periphery than waste canvas. |

#### Measure this before you train anything

`square` is the config default because it is what the runs in the results table used. **It is very
probably not the best choice for your data.** On a held-out split of a 1.78:1 source, changing nothing
but the framing (same clips, same GT depth pixels, the **frozen** model, zero training):

| aspect | AbsRel ↓ | δ<1.25 ↑ | |
|---|---|---|---|
| `square` | 0.1492 | 0.8080 | the published default |
| **`pad`** | **0.0876** | **0.9180** | **−41% AbsRel, for free** |
| `crop` | 0.0754 | 0.9345 | −49%, but it also drops field of view, so the pixels are not the same |

Read that again: **not distorting the input bought more depth accuracy than our entire finetune did**
on that dataset (0.149 → 0.088 by reframing, versus 0.149 → 0.101 by finetuning at `square`). The
backbone was pretrained on undistorted images, and a 1.78:1 frame stretched to a square is already far
enough outside that to cost it dearly. A wider source can only be worse.

So: **run `inspect_manifest.py`, take its framing recommendation, and only then train.** The numbers
above are one dataset and one frozen model. They are a warning, not a law, but they are cheap to
reproduce on yours (`eval.py --aspect square|pad|crop`), and it costs a few minutes to find out whether
your baseline is being handicapped by nothing more than an image resize.

### `intrinsics`: the one thing only you can get right

The loader will take whatever `K` you hand it and transform it **exactly**, for every framing mode:
`square` rescales `fx, cx` and `fy, cy` independently; `pad` and `crop` scale uniformly and shift the
principal point by the pad/crop offset. The geometry that reaches the model is correct **for the `K`
you supplied**.

What it cannot do is tell whether that `K` is *your camera*. A wrong `K` is a silent, systematic
error: nothing crashes, no metric screams, the model just learns a slightly wrong world.

So:

- Give it the `K` of the camera that took the RGB, **in that RGB's own pixel coordinates**: not
  resized, not cropped, not normalized. The loader does all of that itself.
- If your frames are rectified/undistorted, use the rectified `K`. This repo has no distortion model:
  `k1, k2, p1, p2` are not read, so undistort before you build the manifest, or accept the error.
- `inspect_manifest.py` prints your implied field of view and flags a principal point far from centre.
  Neither is *wrong* (rectified cameras often do put the principal point well off centre), but both
  are also exactly what a mistaken `K` looks like, so both are worth a second look.
- **The defaults will get you the table above. Your own `K` is what gets you past it.**

### `img_size` and `patch`

`img_size` should match the resolution your pretrained checkpoint was trained at (`vggt_omega_1b_512`
→ 512). `patch` is the backbone's patch size; `img_size` is rounded **down** to a multiple of it, so
with `patch: 16` a request for 518 silently becomes 512. If you switch to a backbone with a different
patch size, set `patch` and the rounding follows.

## The recipe used here

| Knob | Value |
|---|---|
| Trainable | `dense_head` + `camera_head` + the last `last_n` blocks of **each** aggregator stack. On the 1B model (24 frame + 24 inter-frame blocks) `last_n: 8` unfreezes **16** blocks, not 8. Read the `[wrap] trainable=` line the run prints. |
| LR | heads `5e-6`; every unfrozen backbone block shares one flat `× 0.15 ≈ 7.5e-7` (no layer-wise decay) |
| Pose loss | frame-0-relative, per-clip translation scale-normalized, geodesic rotation, rotation-emphasis (`w_rot=2`, `w_trans=1`) |
| Depth loss | scale-invariant, confidence-weighted, multi-scale gradient matching |
| Optional | `w_cons`: point-cloud consistency. It unprojects the predicted and GT depth into the frame-0 camera and compares the two clouds, tying depth to pose (helps ATE where parallax allows) |
| Selection | held-out validation; saves only when AbsRel ↓ **and** ATE ↓ **and** δ<1.25 has not dropped by >0.02. RPE-rot is reported but **not** gated. |

Why these choices matter:

- **A correct pose objective.** Frame-0-relative poses + per-clip translation scale-normalization +
  geodesic rotation handle VGGT-Ω's metric-vs-up-to-scale mismatch and the quaternion double-cover
  correctly, a numerically stable formulation. See [`ftlib/losses_pose.py`](ftlib/losses_pose.py).
- **Validation is the control loop.** [`ftlib/val.py`](ftlib/val.py) scores held-out depth and pose;
  [`ftlib/train.py`](ftlib/train.py) saves `best.pt` only when *both* beat the frozen baseline.

## Config reference (`configs/finetune.yaml`)

Hardware: a single CUDA GPU (no CPU / MPS / multi-GPU path). The default recipe
(`img_size 512`, `seq_len 4`, `batch_size 3`, `last_n 8`) peaks around **25 GB** of VRAM; `batch_size 12`
peaks around **67 GB**. Autocast is bf16, so Ampere or newer.

| group | key | note |
|---|---|---|
| (root) | `seed` | **required**. Seeds torch and the train/val scene split |
| (root) | `init` | pretrained VGGT-Ω checkpoint (placeholder) |
| (root) | `out` | output dir for `best.pt` + `decision.json` |
| `model` | `freeze` / `last_n` | `heads` \| `lastN` \| `full`. `lastN` keeps the last `last_n` blocks of **each** aggregator stack trainable (+ both heads). |
| `data` | `manifest` | your JSONL manifest (placeholder) |
| `data` | `img_size` | training resolution (patch-aligned); **match it to your inference resolution** |
| `data` | `seq_len` | frames per clip; **≥ 4** for a meaningful pose ATE |
| `data` | `val_frac` / `workers` | val split fraction; dataloader workers (keep low to bound host RAM) |
| `data` | `max_per_scene` | cap on clips taken per scene (default 40). A 400-frame scene at `seq_len 4` yields 100 windows, so the default would use only the first 40. |
| `optim` | `lr` / `backbone_lr_mult` | head LR; unfrozen-backbone LR = `lr × backbone_lr_mult` |
| `optim` | `batch_size` / `steps` / `warmup_frac` / `grad_clip` | standard optimization knobs |
| `loss` | `w_depth`/`w_pose`/`w_rot`/`w_trans`/`w_grad`/`w_conf` | loss term weights |
| `loss` | `w_cons` | reprojection depth↔pose consistency (0 here; on in `finetune_cons.yaml`) |
| `loss` | `metric` | `false` = scale-invariant depth (matches the up-to-scale released head) |
| `val` | `every` / `max_clips` / `patience` | validate every N steps; clips per eval; early-stop patience |
| `val` | `ate_stat` | `mean` (the default, and how our own runs selected their checkpoint) or `median`. Per-clip ATE is heavy-tailed on near-collinear trajectories, so `median` is the more robust gate. |
| (root) | `seed` | required; seeds torch and the train/val scene split |

## What to expect

The both-improve gate only saves when held-out geometry actually beats the frozen model. In practice
depth (scale-invariant AbsRel) and rotation (RPE-rot) improve the most; trajectory error (ATE)
improves where sequences have enough parallax. **The gate declining to save is informative**: it
means your data or GT isn't giving the model a signal it can beat its own strong prior with.

## Honest limitations

- **Up-to-scale.** VGGT-Ω's depth and translation are scene-normalized; metrics align scale (median
  for depth, Sim3-Umeyama for the trajectory) before measuring, so finetuning improves *geometry*, not
  metric scale.
- **ATE is bounded by parallax.** On wide-baseline / low-parallax clips, camera translation is only
  weakly observable, so ATE barely moves even as rotation improves. That is a property of the data, not a bug.
- **Your GT is the ceiling.** VGGT-Ω is already strong, so an *inaccurate* target degrades it.
  Sparse is fine; wrong is not. Prefer few trustworthy depth pixels over dense interpolated ones.
- **Framing is a real choice, not a formality.** `square` distorts wide frames and `pad` wastes canvas;
  neither is free. See [Framing and intrinsics](#framing-and-intrinsics-start-here).
- **Scope.** A focused engineering tool, not a large benchmarked research release.
