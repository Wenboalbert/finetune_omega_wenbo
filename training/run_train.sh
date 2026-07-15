#!/bin/bash
# run_train.sh <config.yaml|dataset> -- launch a joint depth+pose finetune.
#
# Prereqs:
#   * a Python env with torch (CUDA), pyyaml, numpy, pillow
#   * the pretrained VGGT-Ω package importable: either on PYTHONPATH or via VGGT_OMEGA_PATH
#   * for local presets, edit configs/base.yaml and configs/datasets.yaml
#
# Example:
#   export VGGT_OMEGA_PATH=<path/to/vggt-omega>
#   bash run_train.sh configs/finetune.yaml
#   bash run_train.sh nrgbd
set -uo pipefail
cd "$(dirname "$0")"                         # ftlib and the default out: dir are relative
CFG="${1:?usage: run_train.sh <config.yaml|dataset>}"

export PYTHONUNBUFFERED=1
# keep host RAM/CPU bounded (raise if you have headroom)
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
# reduce CUDA VRAM fragmentation OOMs
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "[omega-forge] gpu=${CUDA_VISIBLE_DEVICES:-?} cfg=$CFG"
command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1

exec python -m ftlib.train --config "$CFG"
