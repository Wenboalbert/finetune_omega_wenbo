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
    p=argparse.ArgumentParser();p.add_argument("--source-root",required=True);args=p.parse_args()
    root=Path(__file__).resolve().parents[1]
    if git(root,"status","--porcelain"): raise RuntimeError("commit reviewed implementation before preparation")
    commit=git(root,"rev-parse","HEAD")
    configs=["configs/ladder_frame40_b"+x+".json" for x in ("003","010","020","040")]
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
    for record in records: write_json(Path(record["path"])/"study_manifest.json",study)
    print(json.dumps(study),flush=True)

if __name__=="__main__": main()
