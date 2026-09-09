#!/usr/bin/env bash
# CPU-only import/path check, not training or inference.
set -euo pipefail
SCENE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1
unset PYTHONPATH
unset VGGT_OMEGA_PATH
exec /home/n12388815/phd/vggt_omega_project/env_finetune/bin/python "$SCENE_ROOT/scripts/check_environment.py" "$@"
