"""Group-4 contracts and endpoint barriers. No geometry GT is read."""
from copy import deepcopy
from pathlib import Path
import hashlib
import json
import subprocess
import numpy as np
from .io import read_json, sha256
from .study import FROZEN_FILES, assert_study_frozen
from .zero_target import verify_core
from .trust_policy import TERMINATIONS

GROUP = "v001_group4_extended"
REFERENCE_CODE = "f502ba4823d1ec774be5a2a1c02060a6890d22a3"
NEAR_ZERO = .001
SCOPE = "frame40 development; independent zero starts; 4 percent extended then conditional 5 percent; all executed endpoints frozen before geometry"
OPT = "adapted/post14_D02_registers/optimization.json"
DECISION = "logs/branch_decision.json"

def reference_config(root):
    rel = "configs/zero_target_frame40_b040.json"
    original = subprocess.check_output(["git","show",REFERENCE_CODE+":"+rel],cwd=root)
    if (Path(root)/rel).read_bytes()!=original:
        raise ValueError("historical group-3 configuration changed")
    return json.loads(original)

def validate_config(config, reference):
    if reference["experiment_group"]!="v001_group3_zero_target" or reference["cumulative_budget_relative"]!=.04:
        raise ValueError("requires group-3 4-percent reference")
    if config.get("cumulative_budget_relative") not in (.04,.05):
        raise ValueError("group 4 permits only 4 and conditional 5 percent")
    expected=deepcopy(reference)
    expected.update(experiment_group=GROUP,scope=SCOPE,cumulative_budget_relative=config["cumulative_budget_relative"])
    expected["optimizer"].update(max_accepted_steps=80,max_jacobians=80,max_candidate_forwards=320)
    if config!=expected: raise ValueError("unapproved group-4 config change")
    return True

def verify_frozen(root):
    root=Path(root);hashes=verify_core(root)
    for rel in ("src/residual_tto/study.py","src/residual_tto/prefix_audit.py",
                "src/residual_tto/zero_target.py","scripts/run_budget_ladder.py"):
        old=subprocess.check_output(["git","show",REFERENCE_CODE+":"+rel],cwd=root)
        if (root/rel).read_bytes()!=old: raise ValueError("historical code changed: "+rel)
        hashes[rel]=sha256(root/rel)
    for rel,marker in (("src/residual_tto/ladder_runner.py",'    if config["supervised_views"]'),
                       ("src/residual_tto/evaluate.py",'    gt_manifest=read_json')):
        old=subprocess.check_output(["git","show",REFERENCE_CODE+":"+rel],cwd=root,text=True)
        new=(root/rel).read_text()
        if old[old.index(marker):]!=new[new.index(marker):]:
            raise ValueError("frozen execution body changed: "+rel)
        hashes[rel+":frozen_body"]=hashlib.sha256(new[new.index(marker):].encode()).hexdigest()
    return hashes

def input_identity(run):
    m=read_json(Path(run)/"inputs/focal_manifest.json")
    if sha256(m["packet_file"])!=m["packet_sha256"]: raise ValueError("packet hash changed")
    with np.load(m["packet_file"],allow_pickle=False) as z:
        digest=hashlib.sha256(z["images"].tobytes()+z["valid_pixels"].tobytes()).hexdigest()
    return dict(views=m["views"],focal_xy=m["focal_xy"],array_sha256=digest)

def reference_record(root, run):
    root=Path(root).resolve();run=Path(run).resolve()
    if run.parent!=root/"runs": raise ValueError("reference outside worktree")
    study=read_json(run/"study_manifest.json")
    if study.get("experiment_group")!="v001_group3_zero_target" or study["code_commit"]!=REFERENCE_CODE:
        raise ValueError("reference must be original group 3")
    if read_json(run/"resolved_config.json")!=reference_config(root): raise ValueError("wrong reference budget")
    barrier=assert_study_frozen(run)
    if any((Path(x["path"])/"logs/geometry_release.json").exists() or
           (Path(x["path"])/"comparison/geometry_metrics.json").exists() for x in study["runs"]):
        raise ValueError("group 3 must remain unevaluated")
    files=["study_manifest.json","resolved_config.json","environment_config.json","inputs/focal_manifest.json"]
    files += [str(p.relative_to(run)) for p in sorted((run/"logs").glob("accepted_*.npz"))]
    files += [str(p.relative_to(run)) for p in sorted((run/"logs").glob("jacobian_*.npz"))]
    files += ["logs/preflight_jacobian.npz"]
    return dict(path=str(run),code_commit=REFERENCE_CODE,files={p:sha256(run/p) for p in files},
                frozen_group3_endpoints=barrier["endpoints"],input_identity=input_identity(run))

def check_study(root,run):
    root=Path(root).resolve();run=Path(run).resolve();study=read_json(run/"study_manifest.json")
    if study.get("experiment_group")!=GROUP: raise ValueError("not a group-4 study")
    paths=[Path(x["path"]).resolve() for x in study["runs"]]
    if len(paths)!=2 or len(set(paths))!=2 or run not in paths or any(p.parent!=root/"runs" for p in paths):
        raise ValueError("invalid group-4 members")
    if [x["cumulative_budget_relative"] for x in study["runs"]]!=[.04,.05]:
        raise ValueError("wrong conditional budget order")
    if study["near_zero_emax"]!=NEAR_ZERO or study["evaluation_override"]!={"invalid_source_depth_values_cm":[65504.]}:
        raise ValueError("branch/evaluation policy changed")
    if study["frozen_core_hashes"]!=verify_frozen(root): raise ValueError("frozen code mismatch")
    if reference_record(root,study["reference"]["path"])!=study["reference"]: raise ValueError("old group 3 changed")
    for item,path in zip(study["runs"],paths):
        if read_json(path/"study_manifest.json")!=study or sha256(path/"resolved_config.json")!=item["config_sha256"]:
            raise ValueError("member manifest/config changed")
        validate_config(read_json(path/"resolved_config.json"),reference_config(root))
        if sha256(path/"environment_config.json")!=study["reference"]["files"]["environment_config.json"]:
            raise ValueError("environment config changed")
        if input_identity(path)!=study["input_identity"] or study["input_identity"]!=study["reference"]["input_identity"]:
            raise ValueError("input identity changed")
    return study,paths

def check_endpoint(run):
    run=Path(run);done=read_json(run/"logs/ladder_completion.json")
    if done["status"]!="FROZEN" or done["termination_reason"] not in TERMINATIONS: raise ValueError("endpoint not frozen")
    if done["config_sha256"]!=sha256(run/"resolved_config.json") or done["study_sha256"]!=sha256(run/"study_manifest.json"):
        raise ValueError("endpoint metadata changed")
    if set(done["files"])!=set(FROZEN_FILES): raise ValueError("incomplete frozen files")
    for rel,digest in done["files"].items():
        if sha256(run/rel)!=digest: raise ValueError("frozen file changed: "+rel)
    return done

def endpoint_focal(run):
    run=Path(run);opt=read_json(run/OPT);m=read_json(run/"inputs/focal_manifest.json")
    i=m["camera_order"].index("Drone_02")
    xy=np.asarray(opt["final_focal"]["focal_xy"],dtype=np.float64)[i]
    target=np.asarray(m["focal_xy"],dtype=np.float64)[i]
    if not np.isfinite(xy).all() or not np.isfinite(target).all() or np.any(xy<=0) or np.any(target<=0):
        raise ValueError("invalid focal")
    error=np.abs(xy/target-1)
    return dict(relative_error_xy=error.tolist(),emax=float(error.max()),mean=float(error.mean()))

def branch_decision(run,validity):
    if validity.get("status")!="PASS": raise ValueError("invalid optimization cannot trigger a budget expansion")
    check_endpoint(run);focal=endpoint_focal(run)
    return dict(status="SKIPPED_NEAR_ZERO" if focal["emax"]<=NEAR_ZERO else "RUN_5_PERCENT",
                near_zero_emax=NEAR_ZERO,focal=focal,geometry_gt_read=False,
                source_completion_sha256=sha256(Path(run)/"logs/ladder_completion.json"))

def assert_ready(run):
    run=Path(run).resolve();root=run.parents[1];study,paths=check_study(root,run)
    decision=read_json(paths[0]/DECISION)
    validity=read_json(paths[0]/"logs/optimization_validity.json")
    if decision!=branch_decision(paths[0],validity): raise ValueError("focal branch decision changed")
    executed=paths if decision["status"]=="RUN_5_PERCENT" else paths[:1]
    if len(executed)==1:
        skip=read_json(paths[1]/"logs/skip.json")
        if skip!={"status":"SKIPPED_NEAR_ZERO","decision_sha256":sha256(paths[0]/DECISION)}:
            raise ValueError("skip record mismatch")
        if (paths[1]/"logs/ladder_provenance.json").exists() or (paths[1]/"logs/ladder_completion.json").exists():
            raise ValueError("skipped run was executed")
    endpoints=[]
    for path in executed:
        check_endpoint(path)
        v=read_json(path/"logs/optimization_validity.json")
        if v["status"]!="PASS" or v["completion_sha256"]!=sha256(path/"logs/ladder_completion.json"):
            raise ValueError("optimization validity failed")
        # Independent audit is repeated before GT release, not trusted by label alone.
        from .extended_validity import audit_validity
        if audit_validity(path)!=v: raise ValueError("validity audit changed")
        endpoints.append(dict(run=str(path),completion_sha256=sha256(path/"logs/ladder_completion.json"),
                              validity_sha256=sha256(path/"logs/optimization_validity.json")))
    audit=read_json(paths[0]/"logs/group3_prefix_diagnostic.json")
    if audit["rtol"]!=1e-6 or audit["atol"]!=1e-8: raise ValueError("prefix tolerance changed")
    # PASS is deliberately NOT required. Prefix reproducibility is diagnostic.
    return dict(study_id=study["study_id"],experiment_group=GROUP,endpoints=endpoints,
                all_executed_endpoints_frozen_before_gt=True,decision_sha256=sha256(paths[0]/DECISION),
                prefix_status=audit["status"],prefix_diagnostic_sha256=sha256(paths[0]/"logs/group3_prefix_diagnostic.json"),
                prefix_is_geometry_gate=False,invalid_source_depth_values_cm=[65504.])

def assert_released(run):
    run=Path(run).resolve();barrier=assert_ready(run)
    if str(run) not in [x["run"] for x in barrier["endpoints"]]: raise ValueError("skipped arm has no evaluation")
    if read_json(run/"logs/geometry_release.json")!=barrier: raise ValueError("geometry release invalid")
    return barrier

def evaluation_barrier(run):
    """Route metadata outside the evaluator's GT-reading stage."""
    study=read_json(Path(run)/"study_manifest.json")
    if study.get("experiment_group")==GROUP:
        return assert_released(run)
    from .study import assert_study_frozen as legacy_barrier
    return legacy_barrier(run)
