#!/usr/bin/env bash
# Example commands for the reconstruction-visualization scripts. Set SAMPLE to one of your sample
# directories under samples/ (each holds K.npy + depth_*.npy + pose_*.npy + RGB frames + meta.json).
set -euo pipefail

VIS_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$VIS_DIR"

PYTHON="${PYTHON:-python3}"
SAMPLES_ROOT="${SAMPLES_ROOT:-samples}"
SAMPLE="${SAMPLE:-example}"                 # a subdirectory of $SAMPLES_ROOT
OUT_ROOT="${OUT_ROOT:-outputs}"
S="$SAMPLES_ROOT/$SAMPLE"

# This repo ships no reconstruction data -- point SAMPLES_ROOT/SAMPLE at your own.
# See README.md ("Inputs") for the two accepted layouts.
[ -d "$S" ] || { echo "no such sample dir: $S  (set SAMPLES_ROOT / SAMPLE, see README.md)" >&2; exit 1; }

# ---------------------------------------------------------------------------
# 1) Three-method comparison: Input | GT | VGGT-Omega | Ours, synchronized reveal.
#    Swap --view follow for --view topdown (+ the top-down flags below) for a bird's-eye version.
# ---------------------------------------------------------------------------
"$PYTHON" render_omega_samples_comparison.py \
  --samples-root "$SAMPLES_ROOT" --sample "$SAMPLE" --output-root "$OUT_ROOT" --image-dir "$S" \
  --view follow --stride 20 --target-radius 10.0 --width 512 --height 512 --fps 12 \
  --frames-per-step 4 --point-size 2 --frustum-scale 0.65 --camera-fov-deg 55.0 \
  --chase-back 0.05 --chase-right 0.03 --chase-up 0.03 --chase-lookahead 1.0 --fov-deg 50.0

# ---------------------------------------------------------------------------
# 2) Single-method reconstruction video (pass --depth-npy/--pose-npy to pick GT / original / ours).
# ---------------------------------------------------------------------------
# follow (third-person chase)
"$PYTHON" render_reconstruction_video.py \
  --view follow --result-dir "$S" --depth-npy depth_ours.npy --pose-npy pose_ours.npy --image-dir "$S" \
  --stride 20 --target-radius 10.0 --width 1280 --height 720 --fps 60 --frames-per-step 4 \
  --point-size 2 --frustum-scale 0.65 --camera-fov-deg 55.0 \
  --chase-back 0.05 --chase-right 0.03 --chase-up 0.03 --chase-lookahead 1.0 --fov-deg 50.0

# fixed top-down (orthographic)
"$PYTHON" render_reconstruction_video.py \
  --view topdown --view-motion fixed --result-dir "$S" --depth-npy depth_ours.npy --pose-npy pose_ours.npy \
  --image-dir "$S" --stride 20 --target-radius 10.0 --width 1280 --height 720 --fps 60 --frames-per-step 4 \
  --point-size 2 --frustum-scale 0.65 --camera-fov-deg 55.0 \
  --top-axis=-y --top-margin 1.14 --top-zoom 1.3 --top-yaw-deg 0.0

# cinematic top-down (push-in then orbit)
"$PYTHON" render_reconstruction_video.py \
  --view topdown --view-motion cinematic --result-dir "$S" --depth-npy depth_ours.npy --pose-npy pose_ours.npy \
  --image-dir "$S" --stride 20 --target-radius 10.0 --width 1280 --height 720 --fps 60 --frames-per-step 4 \
  --point-size 2 --frustum-scale 0.65 --camera-fov-deg 55.0 \
  --top-axis=-y --top-margin 1.14 --top-zoom 1.3 --top-yaw-deg 0.0 \
  --motion-start-zoom 0.62 --motion-push-ratio 0.45 --motion-orbit-deg 30.0 --motion-orbit-overlap 0.35
