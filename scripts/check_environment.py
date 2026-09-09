#!/usr/bin/env python3
"""Check QUT scene-adaptation infrastructure without loading model tensors."""
import argparse
import importlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys


def git(path, *args):
    return subprocess.check_output(["git", "-C", str(path), *args], text=True).strip()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=root / "configs/qut_environment.json")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    require(config["schema_version"] == 1, "unsupported config schema")
    require(git(root, "branch", "--show-current") == config["branch"], "wrong experiment branch")
    require(Path(sys.prefix).resolve() == Path(config["environment"]).resolve(), "wrong Python environment")
    model = Path(config["model_root"]).resolve(strict=True)
    require(git(model, "branch", "--show-current") == config["model_branch"], "wrong model branch")
    require(git(model, "rev-parse", "HEAD") == config["model_commit"], "wrong model commit")
    require(not git(model, "status", "--porcelain", "--untracked-files=all"), "model worktree is dirty")
    checkpoint = Path(config["checkpoint"]).resolve(strict=True)
    require(checkpoint.is_file(), "checkpoint is not a file")
    require(checkpoint.stat().st_size == config["checkpoint_size_bytes"], "checkpoint size differs")
    os.environ["VGGT_OMEGA_PATH"] = str(model)
    sys.path.insert(0, str(model))
    package = importlib.import_module("vggt_omega")
    require(model in Path(package.__file__).resolve().parents, "wrong package import")
    from vggt_omega.models import VGGTOmega
    class_file = Path(inspect.getfile(VGGTOmega)).resolve()
    require(model in class_file.parents, "wrong model class import")
    import torch
    report = {
        "status": "PASS", "scope": "paths_and_imports_only",
        "experiment_root": str(root), "branch": config["branch"],
        "experiment_commit": git(root, "rev-parse", "HEAD"),
        "working_tree": git(root, "status", "--short"),
        "environment": str(Path(sys.prefix).resolve()),
        "python": sys.version.split()[0], "torch": torch.__version__,
        "model_commit": config["model_commit"], "model_package": package.__file__,
        "model_class": str(class_file), "checkpoint": str(checkpoint),
        "checkpoint_size_bytes": checkpoint.stat().st_size,
        "checkpoint_sha256_checked_by_this_script": False,
        "checkpoint_loaded": False, "model_instantiated": False,
        "forward_backward_run": False, "job_submitted": False,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    try:
        main()
    except (KeyError, OSError, RuntimeError, ValueError, subprocess.CalledProcessError) as exc:
        print(json.dumps({"status": "STOP", "error": str(exc)}), file=sys.stderr)
        raise SystemExit(1)
