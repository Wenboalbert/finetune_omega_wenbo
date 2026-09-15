"""Group 5: fixed three-member study, focal-only dispatch; no geometry GT."""
from copy import deepcopy
from pathlib import Path
import json
import subprocess
import numpy as np
from .io import read_json, sha256
from .extended import (check_endpoint, endpoint_focal, input_identity,
                       verify_frozen as verify_group4, OPT)
from .extended_validity import audit_validity as audit_group4, close
from .trust_policy import TERMINATIONS

GROUP = "v001_group5_budget_expansion"
REFERENCE_CODE = "5a3f75967dd608046ef75d2afe5552d46feadccf"
BUDGETS = (.07, .08, .09)
NEAR_ZERO = .001
SCOPE = "frame40 development; independent zero starts; 7 then conditional 8 then conditional 9 percent; all executed endpoints frozen before geometry"
DECISION = "logs/group5_decision.json"
CLOSURE = "logs/group5_closure.json"
POLICY = "valid non-near-zero endpoints continue regardless of normal termination; hard failures stop"
PROTOCOL = "docs/GROUP5_V001.md"

def reference_config(root):
    rel = "configs/extended_frame40_b050.json"
    original = subprocess.check_output(["git", "show", REFERENCE_CODE + ":" + rel], cwd=root)
    if (Path(root) / rel).read_bytes() != original:
        raise ValueError("historical group-4 config changed")
    return json.loads(original)

def validate_config(config, reference):
    if reference.get("experiment_group") != "v001_group4_extended":
        raise ValueError("requires frozen group-4 reference")
    if config.get("cumulative_budget_relative") not in BUDGETS:
        raise ValueError("group 5 allows only 7, 8, 9 percent")
    expected = deepcopy(reference)
    expected.update(experiment_group=GROUP, scope=SCOPE,
                    cumulative_budget_relative=config["cumulative_budget_relative"])
    if config != expected:
        raise ValueError("only cumulative budget and group/scope metadata may change")
    return True

def verify_frozen(root):
    root = Path(root)
    hashes = verify_group4(root)  # Eight core modules and execution bodies unchanged.
    # Preserve every pre-existing config/script/module except the two routing headers.
    paths = subprocess.check_output(
        ["git", "ls-tree", "-r", "--name-only", REFERENCE_CODE, "--",
         "configs", "scripts", "src/residual_tto"], cwd=root, text=True).splitlines()
    for rel in paths:
        if rel in ("src/residual_tto/ladder_runner.py", "src/residual_tto/evaluate.py"):
            continue
        old = subprocess.check_output(["git", "show", REFERENCE_CODE + ":" + rel], cwd=root)
        if (root / rel).read_bytes() != old:
            raise ValueError("historical file changed: " + rel)
        hashes[rel] = sha256(root / rel)
    # The evaluator's helper formulas before main must also stay unchanged.
    rel = "src/residual_tto/evaluate.py"
    old = subprocess.check_output(["git", "show", REFERENCE_CODE + ":" + rel], cwd=root, text=True)
    if (root / rel).read_text().split("def main():")[0] != old.split("def main():")[0]:
        raise ValueError("evaluation helpers changed")
    return hashes

def reference_record(root, run):
    root, run = Path(root).resolve(), Path(run).resolve()
    if run.parent != root / "runs":
        raise ValueError("reference outside V001")
    study = read_json(run / "study_manifest.json")
    if study.get("experiment_group") != "v001_group4_extended" or study["code_commit"] != REFERENCE_CODE:
        raise ValueError("requires original group-4 reference")
    if read_json(run / "resolved_config.json") != reference_config(root):
        raise ValueError("requires group-4 5-percent reference")
    done = check_endpoint(run)
    pre = read_json(run / "numerical_preflight.json")
    if pre["status"] != "PASS" or read_json(run / OPT)["geometry_gt_used"]:
        raise ValueError("invalid reference preflight/supervision")
    rels = list(done["files"]) + [
        "logs/ladder_completion.json", "logs/ladder_provenance.json",
        "logs/preflight_jacobian.npz", "numerical_preflight.json",
        "resolved_config.json", "environment_config.json",
        "study_manifest.json", "inputs/focal_manifest.json"]
    return dict(path=str(run), code_commit=REFERENCE_CODE,
                files={p: sha256(run / p) for p in rels},
                input_identity=input_identity(run))

def check_study(root, run):
    root, run = Path(root).resolve(), Path(run).resolve()
    study = read_json(run / "study_manifest.json")
    paths = [Path(x["path"]).resolve() for x in study["runs"]]
    if study.get("experiment_group") != GROUP or len(paths) != 3 or len(set(paths)) != 3:
        raise ValueError("invalid group-5 membership")
    if run not in paths or any(p.parent != root / "runs" for p in paths):
        raise ValueError("invalid group-5 location")
    if [x["cumulative_budget_relative"] for x in study["runs"]] != list(BUDGETS):
        raise ValueError("wrong budget order")
    if study["near_zero_emax"] != NEAR_ZERO or study["continuation_policy"] != POLICY:
        raise ValueError("branch policy changed")
    if study["geometry_selection_allowed"] is not False or study["evaluation_override"] != {
            "invalid_source_depth_values_cm": [65504.]}:
        raise ValueError("GT policy changed")
    if study["protocol_sha256"] != sha256(root / PROTOCOL):
        raise ValueError("protocol changed")
    if study["frozen_core_hashes"] != verify_frozen(root):
        raise ValueError("frozen code changed")
    if reference_record(root, study["reference"]["path"]) != study["reference"]:
        raise ValueError("historical reference changed")
    ref = reference_config(root)
    for item, path in zip(study["runs"], paths):
        if read_json(path / "study_manifest.json") != study:
            raise ValueError("member manifest changed")
        if sha256(path / "resolved_config.json") != item["config_sha256"]:
            raise ValueError("member config changed")
        validate_config(read_json(path / "resolved_config.json"), ref)
        if sha256(path / "environment_config.json") != study["reference"]["files"]["environment_config.json"]:
            raise ValueError("environment changed")
        if input_identity(path) != study["input_identity"] or study["input_identity"] != study["reference"]["input_identity"]:
            raise ValueError("input differs from frozen reference")
    return study, paths

def audit_validity(run):
    result = audit_group4(run)  # No optimizer or existing audit changes.
    opt = read_json(Path(run) / OPT)
    if any("forward_numerical_error" in e for e in opt["attempts"]):
        raise ValueError("recorded nonfinite forward: stop group 5, do not expand")
    if not np.isfinite([opt["baseline_loss"], opt["final_loss"], opt["h0_rms"],
                        opt["epsilon_repeat"], opt["cumulative_relative"]]).all():
        raise ValueError("nonfinite endpoint")
    return result

def compare_validity(recorded, recomputed):
    # Only this derived summary can vary in low bits across CPU/BLAS platforms.
    # audit_validity has ALREADY repeated strict FP32 state/step budget checks.
    a, b = dict(recorded), dict(recomputed)
    close(a.pop("max_step_relative"), b.pop("max_step_relative"), "derived max step")
    if a != b:
        raise ValueError("validity record changed")

def decide(validity, focal, budget):
    if validity.get("status") != "PASS" or validity.get("termination") not in TERMINATIONS - {"SOLVER_FAILURE"}:
        raise ValueError("invalid endpoint cannot branch")
    errors = np.asarray(focal["relative_error_xy"], dtype=np.float64)
    if errors.shape != (2,) or not np.isfinite(errors).all() or np.any(errors < 0):
        raise ValueError("invalid focal errors")
    if focal["emax"] != float(errors.max()):
        raise ValueError("inconsistent focal maximum")
    index = BUDGETS.index(budget)
    success = focal["emax"] <= NEAR_ZERO
    action = "STOP_NEAR_ZERO" if success else ("CONTINUE" if index < 2 else "STOP_MAX_BUDGET")
    return dict(action=action, focal_status="NEAR_ZERO" if success else "NOT_NEAR_ZERO",
                next_budget=BUDGETS[index + 1] if action == "CONTINUE" else None,
                termination=validity["termination"], validity="PASS",
                near_zero_emax=NEAR_ZERO, focal=focal, budget=budget,
                geometry_gt_read=False)

def branch_decision(run, validity):
    run = Path(run)
    done = check_endpoint(run)
    if validity.get("completion_sha256") != sha256(run / "logs/ladder_completion.json"):
        raise ValueError("validity belongs to another endpoint")
    if validity.get("termination") != done["termination_reason"]:
        raise ValueError("termination mismatch")
    result = decide(validity, endpoint_focal(run),
                    read_json(run / "resolved_config.json")["cumulative_budget_relative"])
    result["source_completion_sha256"] = sha256(run / "logs/ladder_completion.json")
    return result

def initialization_diagnostic(previous, current):
    previous, current = Path(previous), Path(current)
    from importlib.util import spec_from_file_location, module_from_spec
    # Reuse the frozen group-4 comparison, not a new trajectory tolerance.
    source = Path(__file__).resolve().parents[2] / "scripts/run_extended.py"
    spec = spec_from_file_location("_frozen_group4_driver", source)
    module = module_from_spec(spec); spec.loader.exec_module(module)
    result = module.initialization(previous, current)
    result.update(reference=str(previous), prefix_status="NOT_APPLICABLE",
                  reason="different budgets; initial J diagnostic only",
                  files={str(p / rel): sha256(p / rel)
                         for p in (previous, current)
                         for rel in ("baseline/predictions.npz", "logs/preflight_jacobian.npz")})
    return result

def assert_ready(run):
    run = Path(run).resolve()
    study, paths = check_study(run.parents[1], run)
    if (paths[0] / "logs/study_failure.json").exists():
        raise ValueError("failed study cannot release geometry")
    endpoints, skipped = [], []
    stopped = None
    for path in paths:
        if stopped is not None:
            expected = dict(status="SKIPPED_NEAR_ZERO", trigger_run=str(stopped),
                            decision_sha256=sha256(stopped / DECISION))
            if read_json(path / "logs/skip.json") != expected:
                raise ValueError("invalid skip record")
            if any((path / p).exists() for p in (
                    "logs/ladder_provenance.json", "logs/ladder_completion.json",
                    "adapted/post14_D02_registers/optimization.json", DECISION,
                    "logs/optimizer_process.log", "logs/geometry_release.json",
                    "comparison/geometry_metrics.json")):
                raise ValueError("skipped member was executed")
            skipped.append(dict(run=str(path), skip_sha256=sha256(path / "logs/skip.json")))
            continue
        check_endpoint(path)
        valid = read_json(path / "logs/optimization_validity.json")
        compare_validity(valid, audit_validity(path))
        decision = branch_decision(path, valid)
        if read_json(path / DECISION) != decision:
            raise ValueError("branch decision changed")
        diag = read_json(path / "logs/initialization_diagnostic.json")
        expected_reference = study["reference"]["path"] if not endpoints else str(paths[0])
        if diag["reference"] != expected_reference or diag["prefix_status"] != "NOT_APPLICABLE":
            raise ValueError("initialization reference changed")
        if any(diag.get(key) is not True for key in ("baseline_exact", "initial_r_exact", "initial_h0_exact")):
            raise ValueError("zero-state consistency failed")
        expected_files = {str(p / rel): sha256(p / rel)
                          for p in (Path(expected_reference), path)
                          for rel in ("baseline/predictions.npz", "logs/preflight_jacobian.npz")}
        if diag["files"] != expected_files:
            raise ValueError("initialization evidence changed")
        endpoints.append(dict(run=str(path),
                              completion_sha256=sha256(path / "logs/ladder_completion.json"),
                              validity_sha256=sha256(path / "logs/optimization_validity.json"),
                              decision_sha256=sha256(path / DECISION),
                              initialization_sha256=sha256(path / "logs/initialization_diagnostic.json")))
        if decision["action"] == "STOP_NEAR_ZERO":
            stopped = path
        elif decision["action"] == "STOP_MAX_BUDGET" and path != paths[-1]:
            raise ValueError("premature budget stop")
    return dict(study_id=study["study_id"], experiment_group=GROUP,
                study_sha256=sha256(paths[0] / "study_manifest.json"),
                endpoints=endpoints, skipped=skipped,
                search_stop_reason="NEAR_ZERO" if stopped else "MAX_BUDGET",
                all_executed_endpoints_frozen_before_gt=True,
                prefix_status="NOT_APPLICABLE_DIFFERENT_BUDGETS",
                invalid_source_depth_values_cm=[65504.],
                geometry_gt_used_for_selection=False)

def assert_released(run):
    run = Path(run).resolve()
    barrier = assert_ready(run)
    first = Path(read_json(run / "study_manifest.json")["runs"][0]["path"])
    if read_json(first / CLOSURE) != barrier:
        raise ValueError("study has not closed")
    if str(run) not in [x["run"] for x in barrier["endpoints"]]:
        raise ValueError("skipped member has no geometry release")
    if read_json(run / "logs/geometry_release.json") != barrier:
        raise ValueError("geometry release changed")
    return barrier

def evaluation_barrier(run):
    if read_json(Path(run) / "study_manifest.json").get("experiment_group") == GROUP:
        return assert_released(run)
    from .extended import evaluation_barrier as old_barrier
    return old_barrier(run)
