#!/usr/bin/env python3
"""Prepare four independent paired runs, snapshot one shared private study manifest."""
import argparse
import datetime
import hashlib
import json
import subprocess
import sys
import uuid
from pathlib import Path
import numpy as np
from residual_tto.io import read_json,write_json,sha256,git

def main():
    p=argparse.ArgumentParser();p.add_argument("--source-root",required=True)
    p.add_argument("--variant",choices=("group2","group3"),default="group2")
    p.add_argument("--reference-study-run")
    args=p.parse_args()
    root=Path(__file__).resolve().parents[1]
    if git(root,"status","--porcelain"): raise RuntimeError("commit reviewed implementation before preparation")
    commit=git(root,"rev-parse","HEAD")
    reference=None;core=None
    if args.variant=="group3":
        from residual_tto.zero_target import reference_study,verify_core,validate_config,reference_config
        if not args.reference_study_run: raise ValueError("group 3 requires the group-2 reference")
        reference=reference_study(root,args.reference_study_run);core=verify_core(root)
    elif args.reference_study_run: raise ValueError("reference is only valid for group 3")
    prefix="zero_target" if args.variant=="group3" else "ladder"
    configs=["configs/"+prefix+"_frame40_b"+x+".json" for x in ("003","010","020","040")]
    if reference:
        for name in configs:
            cfg=read_json(root/name)
            validate_config(cfg,reference_config(root,cfg["cumulative_budget_relative"]))
    records=[];identity=None
    for config in configs:
        output=subprocess.check_output([sys.executable,str(root/"scripts/prepare_frame40.py"),
            "--source-root",args.source_root,"--config",config],text=True)
        prepared=json.loads(output.strip().splitlines()[-1]);run=Path(prepared["run"])
        if prepared["pixel_error"]!=0 or not all(prepared["depth_files_present"]):
            raise RuntimeError("data preparation failed; preserve partial runs and do not submit")
        m=read_json(run/"inputs/focal_manifest.json")
        with np.load(m["packet_file"],allow_pickle=False) as data:
            arrays=hashlib.sha256(data["images"].tobytes()+data["valid_pixels"].tobytes()).hexdigest()
        this=dict(views=m["views"],focal_xy=m["focal_xy"],array_sha256=arrays)
        if identity is None: identity=this
        elif this!=identity: raise RuntimeError("independent runs do not have identical input")
        records.append(dict(path=str(run),config_sha256=sha256(run/"resolved_config.json"),
            cumulative_budget_relative=read_json(run/"resolved_config.json")["cumulative_budget_relative"]))
    study=dict(schema_version=1,study_id="budget_ladder_"+datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")+"_"+uuid.uuid4().hex[:8],
        code_commit=commit,runs=records,input_identity=identity,
        protocol="four independent zero starts; target 15 percent; all endpoints frozen before offline GT",
        geometry_selection_allowed=False)
    if reference:
        if identity!=read_json(Path(reference["first_run"])/"study_manifest.json")["input_identity"]:
            raise ValueError("group-3 pixels/focal differ from group 2; preserve partial runs")
        study.update(experiment_group="v001_group3_zero_target",reference_study=reference,
            frozen_core_hashes=core,target_mean_focal_relative_error=0.0,
            protocol="remove 15 percent target early-stop only; zero does not promise exact focal; all endpoints frozen before geometry",
            evaluation_override={"invalid_source_depth_values_cm":[65504.0]})
    for record in records: write_json(Path(record["path"])/"study_manifest.json",study)
    print(json.dumps(study),flush=True)

if __name__=="__main__": main()
