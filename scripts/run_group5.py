#!/usr/bin/env python3
"""Group-5 driver: fixed optional 7/8/9 members; all decisions precede GT."""
import argparse,json,os,subprocess,sys,traceback
from pathlib import Path
from residual_tto.io import read_json,write_json,sha256,git
from residual_tto.group5 import (check_study,branch_decision,assert_ready,audit_validity,
                                initialization_diagnostic,DECISION,CLOSURE)

def execute(path):
    with (path/"logs/optimizer_process.log").open("x") as log:
        subprocess.run([sys.executable,"-m","residual_tto.ladder_runner","--run",str(path)],
                       stdout=log,stderr=subprocess.STDOUT,check=True)
    valid=audit_validity(path)
    write_json(path/"logs/optimization_validity.json",valid)
    print(json.dumps(dict(event="OPTIMIZATION_VALID",run=str(path),
                          termination=valid["termination"],focal=valid["focal"])),flush=True)
    return valid

def main():
    parser=argparse.ArgumentParser();parser.add_argument("--study-run",required=True)
    args=parser.parse_args()
    if not os.environ.get("PBS_JOBID"): raise RuntimeError("PBS allocation required")
    root=Path(__file__).resolve().parents[1];run=Path(args.study_run).resolve()
    study,paths=check_study(root,run)
    if run!=paths[0]: raise ValueError("start with the 7-percent member")
    if study["code_commit"]!=git(root,"rev-parse","HEAD") or git(root,"status","--porcelain"):
        raise ValueError("unreviewed or dirty implementation")
    for path in paths:
        for rel in ("logs/ladder_provenance.json","logs/optimizer_process.log",
                    DECISION,CLOSURE,"logs/geometry_release.json","logs/study_started.json",
                    "logs/study_failure.json","logs/skip.json"):
            if (path/rel).exists(): raise FileExistsError("no resume/overwrite: "+str(path/rel))
    write_json(run/"logs/study_started.json",dict(job_id=os.environ["PBS_JOBID"],
                                                code_commit=study["code_commit"]))
    executed=[];active=None
    try:
        for index,path in enumerate(paths):
            active=path
            valid=execute(path)
            previous=Path(study["reference"]["path"]) if index==0 else paths[0]
            write_json(path/"logs/initialization_diagnostic.json",
                       initialization_diagnostic(previous,path))
            decision=branch_decision(path,valid)
            write_json(path/DECISION,decision);executed.append(path)
            print(json.dumps(dict(event="FOCAL_BRANCH",run=str(path),**decision)),flush=True)
            if decision["action"]=="STOP_NEAR_ZERO":
                for skipped in paths[index+1:]:
                    write_json(skipped/"logs/skip.json",dict(status="SKIPPED_NEAR_ZERO",
                        trigger_run=str(path),decision_sha256=sha256(path/DECISION)))
                break
        barrier=assert_ready(paths[0])
        write_json(paths[0]/CLOSURE,barrier)
        for path in executed: write_json(path/"logs/geometry_release.json",barrier)
        for path in executed:
            with (path/"logs/evaluator_process.log").open("x") as log:
                subprocess.run([sys.executable,"-m","residual_tto.evaluate","--run",str(path),
                                "--invalid-source-depth","65504"],
                               stdout=log,stderr=subprocess.STDOUT,check=True)
            print("EVALUATED "+str(path),flush=True)
        check_study(root,paths[0])
        write_json(paths[0]/"logs/study_completion.json",dict(
            status="COMPLETE",optimization_validity="PASS",geometry_evaluation="COMPLETE",
            search_stop_reason=barrier["search_stop_reason"],closure_sha256=sha256(paths[0]/CLOSURE),
            prefix_reproducibility="NOT_APPLICABLE_DIFFERENT_BUDGETS",
            geometry_metrics={str(p):sha256(p/"comparison/geometry_metrics.json") for p in executed},
            geometry_gt_not_used_for_selection=True))
    except BaseException as error:
        write_json(run/"logs/study_failure.json",dict(status="INCOMPLETE",
            active_run=str(active) if active else None,executed=[str(p) for p in executed],
            error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc()))
        raise

if __name__=="__main__": main()
