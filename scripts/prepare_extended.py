#!/usr/bin/env python3
"""Prepare independent 4% and conditional 5% arms; no model execution."""
import argparse,datetime,json,subprocess,sys,uuid
from pathlib import Path
from residual_tto.io import read_json,write_json,sha256,git
from residual_tto.extended import (GROUP,NEAR_ZERO,reference_record,reference_config,
                                  validate_config,verify_frozen,input_identity,check_study)
def main():
    p=argparse.ArgumentParser();p.add_argument("--source-root",required=True)
    p.add_argument("--reference-run",required=True);args=p.parse_args()
    root=Path(__file__).resolve().parents[1]
    if git(root,"status","--porcelain"): raise RuntimeError("commit reviewed code before preparation")
    reference=reference_record(root,args.reference_run);core=verify_frozen(root);records=[]
    for suffix,budget in (("040",.04),("050",.05)):
        rel="configs/extended_frame40_b"+suffix+".json"
        validate_config(read_json(root/rel),reference_config(root))
        output=subprocess.check_output([sys.executable,str(root/"scripts/prepare_frame40.py"),
                "--source-root",args.source_root,"--config",rel],text=True)
        prepared=json.loads(output.strip().splitlines()[-1]);run=Path(prepared["run"])
        if prepared["pixel_error"]!=0 or not all(prepared["depth_files_present"]):
            raise ValueError("preparation failed; preserve partial run")
        if input_identity(run)!=reference["input_identity"]: raise ValueError("different source inputs")
        records.append(dict(path=str(run),config_sha256=sha256(run/"resolved_config.json"),
                            cumulative_budget_relative=budget))
    study=dict(schema_version=1,experiment_group=GROUP,
        study_id="extended_"+datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")+"_"+uuid.uuid4().hex[:8],
        code_commit=git(root,"rev-parse","HEAD"),runs=records,reference=reference,
        input_identity=reference["input_identity"],frozen_core_hashes=core,near_zero_emax=NEAR_ZERO,
        evaluation_override={"invalid_source_depth_values_cm":[65504.]},
        geometry_selection_allowed=False,initialization="independent zeros; no resume",
        protocol="4% freezes; focal-only branch; 5% freezes or SKIPPED_NEAR_ZERO; then unified offline geometry",
        prefix_policy="group3 4% reference only; diagnostic, not geometry gate")
    for item in records: write_json(Path(item["path"])/"study_manifest.json",study)
    check_study(root,records[0]["path"])
    print(json.dumps(study),flush=True)
if __name__=="__main__": main()
