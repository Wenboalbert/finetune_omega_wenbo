#!/usr/bin/env python3
"""Group-4 PBS driver. Geometry GT is inaccessible until the conditional study closes."""
import argparse,json,os,subprocess,sys,traceback
from pathlib import Path
import numpy as np
from residual_tto.io import read_json,write_json,sha256,git
from residual_tto.extended import check_study,branch_decision,assert_ready,DECISION
from residual_tto.extended_validity import audit_validity
from residual_tto.extended_prefix import audit_prefix

def initialization(previous,current):
    result=dict(geometry_gt_read=False)
    with np.load(previous/"baseline/predictions.npz",allow_pickle=False) as p,np.load(current/"baseline/predictions.npz",allow_pickle=False) as n:
        same=set(p.files)==set(n.files) and all(np.array_equal(p[k],n[k],equal_nan=True) for k in p.files)
    if not same: raise ValueError("baseline parity failed")
    result["baseline_exact"]=True
    with np.load(previous/"logs/preflight_jacobian.npz",allow_pickle=False) as p,np.load(current/"logs/preflight_jacobian.npz",allow_pickle=False) as n:
        for key in ("r","h0"):
            if not np.array_equal(p[key],n[key]): raise ValueError("different zero-state "+key)
        result.update(initial_r_exact=True,initial_h0_exact=True,
            initial_J_max_abs=float(np.max(np.abs(p["J"]-n["J"]))),
            initial_J_relative_fro=float(np.linalg.norm(p["J"]-n["J"])/np.linalg.norm(p["J"])))
    return result

def execute(path):
    with (path/"logs/optimizer_process.log").open("x") as log:
        subprocess.run([sys.executable,"-m","residual_tto.ladder_runner","--run",str(path)],
                       stdout=log,stderr=subprocess.STDOUT,check=True)
    valid=audit_validity(path)
    write_json(path/"logs/optimization_validity.json",valid)
    print(json.dumps(dict(event="OPTIMIZATION_VALID",run=str(path),**valid)),flush=True)
    return valid

def main():
    p=argparse.ArgumentParser();p.add_argument("--study-run",required=True);args=p.parse_args()
    if not os.environ.get("PBS_JOBID"): raise RuntimeError("PBS allocation required")
    root=Path(__file__).resolve().parents[1];run=Path(args.study_run).resolve()
    study,paths=check_study(root,run)
    if run!=paths[0]: raise ValueError("start from first arm")
    if study["code_commit"]!=git(root,"rev-parse","HEAD") or git(root,"status","--porcelain"):
        raise ValueError("unreviewed implementation")
    for path in paths:
        for rel in ("logs/ladder_provenance.json",DECISION,"logs/geometry_release.json","logs/study_started.json"):
            if (path/rel).exists(): raise FileExistsError("no resume or overwrite: "+str(path/rel))
    write_json(run/"logs/study_started.json",dict(job_id=os.environ["PBS_JOBID"],code_commit=study["code_commit"]))
    try:
        valid=execute(paths[0])
        write_json(paths[0]/"logs/initialization_diagnostic.json",initialization(Path(study["reference"]["path"]),paths[0]))
        decision=branch_decision(paths[0],valid)
        write_json(paths[0]/DECISION,decision)
        print(json.dumps(dict(event="FOCAL_BRANCH",**decision)),flush=True)
        if decision["status"]=="RUN_5_PERCENT":
            execute(paths[1])
            init=initialization(paths[0],paths[1])
            init.update(prefix_status="NOT_APPLICABLE",reason="different cumulative budgets; trajectories need not match")
            write_json(paths[1]/"logs/initialization_diagnostic.json",init)
            executed=paths
        else:
            write_json(paths[1]/"logs/skip.json",dict(status="SKIPPED_NEAR_ZERO",decision_sha256=sha256(paths[0]/DECISION)))
            executed=paths[:1]
        prefix=audit_prefix(study["reference"]["path"],paths[0])
        write_json(paths[0]/"logs/group3_prefix_diagnostic.json",prefix)
        print(json.dumps(dict(event="PREFIX_DIAGNOSTIC",status=prefix["status"],
                             numeric_mismatches=len(prefix["numeric_mismatches"]),
                             decision_mismatches=len(prefix["decision_mismatches"]))),flush=True)
        barrier=assert_ready(paths[0])
        for path in executed: write_json(path/"logs/geometry_release.json",barrier)
        for path in executed:
            with (path/"logs/evaluator_process.log").open("x") as log:
                subprocess.run([sys.executable,"-m","residual_tto.evaluate","--run",str(path),
                                "--invalid-source-depth","65504"],stdout=log,stderr=subprocess.STDOUT,check=True)
            print("EVALUATED "+str(path),flush=True)
        check_study(root,run) # Old group 3 remains unchanged and unevaluated.
        write_json(run/"logs/study_completion.json",dict(status="COMPLETE",optimization_validity="PASS",
            prefix_reproducibility=prefix["status"],geometry_evaluation="COMPLETE",
            geometry_metrics={str(p):sha256(p/"comparison/geometry_metrics.json") for p in executed},
            branch_decision_sha256=sha256(run/DECISION),geometry_gt_not_used_for_selection=True))
    except BaseException as error:
        write_json(run/"logs/study_failure.json",dict(status="STOP",error_type=type(error).__name__,
                   error=str(error),traceback=traceback.format_exc()))
        raise
if __name__=="__main__": main()
