#!/usr/bin/env python3
"""PBS driver: every optimizer exits and freezes before the first GT evaluator."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
from residual_tto.io import read_json,write_json,sha256,git
from residual_tto.study import assert_study_frozen

def main():
    p=argparse.ArgumentParser();p.add_argument("--study-run",required=True);args=p.parse_args()
    if not os.environ.get("PBS_JOBID"): raise RuntimeError("PBS allocation required")
    root=Path(__file__).resolve().parents[1];run=Path(args.study_run).resolve()
    study=read_json(run/"study_manifest.json")
    if study["code_commit"]!=git(root,"rev-parse","HEAD"): raise RuntimeError("study commit changed")
    paths=[Path(x["path"]).resolve() for x in study["runs"]]
    if len(paths)!=4 or len(set(paths))!=4 or any(p.parent!=root/"runs" for p in paths):
        raise ValueError("invalid study paths")
    if [x["cumulative_budget_relative"] for x in study["runs"]]!=[.003,.01,.02,.04]:
        raise ValueError("budget ladder changed")
    reference=None
    for path,item in zip(paths,study["runs"]):
        if read_json(path/"study_manifest.json")!=study or sha256(path/"resolved_config.json")!=item["config_sha256"]:
            raise ValueError("study member changed")
        cfg=read_json(path/"resolved_config.json");cfg.pop("cumulative_budget_relative")
        if reference is None: reference=cfg
        elif cfg!=reference: raise ValueError("more than cumulative budget varies")
        if (path/"logs/ladder_provenance.json").exists(): raise FileExistsError(path)
    for path in paths:
        with (path/"logs/optimizer_process.log").open("x") as log:
            subprocess.run([sys.executable,"-m","residual_tto.ladder_runner","--run",str(path)],
                           stdout=log,stderr=subprocess.STDOUT,check=True)
        print("FROZEN "+str(path),flush=True)
    barrier=assert_study_frozen(paths[0])
    # Same pixels/weights/precision must reproduce the same paired baseline.
    with np.load(paths[0]/"baseline/predictions.npz",allow_pickle=False) as first:
        reference_arrays={k:first[k].copy() for k in first.files}
    for path in paths[1:]:
        with np.load(path/"baseline/predictions.npz",allow_pickle=False) as other:
            if set(other.files)!=set(reference_arrays) or any(
                    not np.array_equal(other[k],v,equal_nan=True) for k,v in reference_arrays.items()):
                raise RuntimeError("independent-run baseline parity failed; no geometry release")
    barrier["baseline_arrays_exactly_equal"]=True
    for path in paths: write_json(path/"logs/geometry_release.json",barrier)
    for path in paths:
        with (path/"logs/evaluator_process.log").open("x") as log:
            subprocess.run([sys.executable,"-m","residual_tto.evaluate","--run",str(path)],
                           stdout=log,stderr=subprocess.STDOUT,check=True)
        print("EVALUATED "+str(path),flush=True)

if __name__=="__main__": main()
