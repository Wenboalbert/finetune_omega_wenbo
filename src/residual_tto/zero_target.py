"""Group-3 metadata and invariants; never changes numerical optimizer behavior."""
from copy import deepcopy
import hashlib
from pathlib import Path
import subprocess
from .io import read_json, sha256
from .study import assert_study_frozen

GROUP = "v001_group3_zero_target"
REFERENCE_GROUP = "v001_group2_budget_ladder"
REFERENCE_CODE = "1d4f0e24d9e58948d6531a987ea76b1b270ac281"
BUDGETS = (.003,.01,.02,.04)
SUFFIXES = ("003","010","020","040")
CORE_FILES = tuple("src/residual_tto/"+name+".py" for name in (
    "solver","trust_policy","constrained_gn","intervention","ladder_optimizer",
    "specs","preprocess","runner"))

def reference_config(root, budget):
    suffix = SUFFIXES[BUDGETS.index(budget)]
    rel = "configs/ladder_frame40_b"+suffix+".json"
    original = subprocess.check_output(["git","show",REFERENCE_CODE+":"+rel],cwd=root)
    if (Path(root)/rel).read_bytes() != original:
        raise ValueError("historical configuration changed: "+rel)
    return read_json(Path(root)/rel)

def validate_config(config, reference):
    expected = deepcopy(reference)
    if expected["experiment_group"] != REFERENCE_GROUP or expected["optimizer"]["target_mean_focal_relative_error"] != .15:
        raise ValueError("reference is not the frozen group-2 protocol")
    expected["experiment_group"] = GROUP
    expected["optimizer"]["target_mean_focal_relative_error"] = 0.0
    if config != expected:
        raise ValueError("group 3 may change only target threshold and group metadata")
    return True

def verify_core(root):
    hashes = {}
    for rel in CORE_FILES:
        expected = hashlib.sha256(subprocess.check_output(
            ["git","show",REFERENCE_CODE+":"+rel],cwd=root)).hexdigest()
        if sha256(Path(root)/rel) != expected:
            raise ValueError("frozen numerical core changed: "+rel)
        hashes[rel] = expected
    return hashes

def reference_study(root, first):
    first = Path(first).resolve()
    if first.parent != (Path(root)/"runs").resolve():
        raise ValueError("reference must belong to this worktree")
    frozen = assert_study_frozen(first)
    study = read_json(first/"study_manifest.json")
    if study["code_commit"] != REFERENCE_CODE or [
            x["cumulative_budget_relative"] for x in study["runs"]] != list(BUDGETS):
        raise ValueError("wrong group-2 reference study")
    for item in study["runs"]:
        p=Path(item["path"])
        config=read_json(p/"resolved_config.json")
        if config != reference_config(root,item["cumulative_budget_relative"]):
            raise ValueError("reference run protocol differs from pinned source")
    return dict(first_run=str(first),study_manifest_sha256=sha256(first/"study_manifest.json"),
                study_id=study["study_id"],code_commit=study["code_commit"],
                runs=study["runs"],frozen_endpoints=frozen["endpoints"])

def assert_reference_unchanged(root, reference):
    if reference_study(root,reference["first_run"]) != reference:
        raise ValueError("group-2 reference changed")
