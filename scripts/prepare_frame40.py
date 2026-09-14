#!/usr/bin/env python3
"""CPU-only input audit. Does not instantiate Omega or submit a job."""
import argparse
import csv
import datetime
import json
import sys
import uuid
from pathlib import Path
import numpy as np
import torch
from residual_tto.io import read_json,write_json,sha256,git,verify_provider
from residual_tto.preprocess import preprocess

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--source-root",required=True)
    parser.add_argument("--config",default="configs/smoke_frame40.json")
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    config=read_json(root/args.config);env=read_json(root/"configs/qut_environment.json")
    verify_provider(env["model_root"],env["model_commit"])
    source=Path(args.source_root).resolve()
    csv_path=source/"PoseData/camera_parameters.csv"
    with csv_path.open(encoding="utf-8-sig",newline="") as f:
        rows=[r for r in csv.DictReader(f) if int(r["frame_id"])==config["frame"]]
    names=config["camera_order"]
    selected=[]
    for name in names:
        matches=[r for r in rows if r["camera_name"].strip()==name]
        if len(matches)!=1: raise ValueError("expected one row for "+name)
        selected.append(matches[0])
    paths=[source/n/r["image_name"].strip() for n,r in zip(names,selected)]
    for p in paths:
        if not p.is_file() or p.name.startswith("._"): raise FileNotFoundError(p)
    hfovs=[float(r["fov"]) for r in selected]
    images,masks,records=preprocess(paths,hfovs,**config["preprocessing"])
    from vggt_omega.utils.load_fn import load_and_preprocess_images
    pp=config["preprocessing"]
    official=load_and_preprocess_images([str(p) for p in paths],pp["mode"],pp["resolution"],pp["patch"])
    error=float((images-official).abs().max())
    if error!=0: raise RuntimeError("provider pixel parity failed: "+str(error))
    run_id=datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")+"_frame40_"+uuid.uuid4().hex[:8]
    run=root/"runs"/run_id
    run.mkdir(parents=True,exist_ok=False)
    for name in ["baseline","adapted/post14_D02_registers","comparison","logs","inputs","private_evaluation"]:
        (run/name).mkdir(parents=True)
    array_path=run/"inputs/rgb_packet.npz"
    np.savez_compressed(array_path,images=images.numpy(),valid_pixels=masks.numpy())
    view_records=[]
    for name,path,record in zip(names,paths,records):
        view_records.append(dict(camera=name,frame=config["frame"],rgb=str(path),rgb_sha256=sha256(path),transform=record))
    focal_manifest=dict(schema_version=1,camera_order=names,frame=config["frame"],
        packet_file=str(array_path),packet_sha256=sha256(array_path),
        views=view_records,focal_xy=[[r["fx"],r["fy"]] for r in records],
        semantics_source=config["semantics_source"])
    write_json(run/"inputs/focal_manifest.json",focal_manifest)
    write_json(run/"camera_order.json",dict(camera_order=names,name_to_index={n:i for i,n in enumerate(names)}))
    write_json(run/"resolved_config.json",config)
    write_json(run/"environment_config.json",env)
    write_json(run/"preprocessing_audit.json",dict(status="PASS",pixel_max_abs_error=error,
        exact_pixel_equality=True,shape=list(images.shape),views=view_records))
    # This object is exclusively for the separate evaluator. Never pass it to the GN runner.
    evaluation=[]
    for name,row in zip(names,selected):
        depth=source/name/"depth"/(name+"_depth_"+str(config["frame"]).zfill(6)+".exr")
        evaluation.append(dict(camera=name,ue_pose={k:float(row[k]) for k in
            ["pos_x","pos_y","pos_z","pitch","yaw","roll"]},depth_path=str(depth),
            depth_exists=depth.is_file(),depth_sha256=sha256(depth) if depth.is_file() else None))
    write_json(run/"private_evaluation/manifest.json",dict(csv_path=str(csv_path),
        csv_sha256=sha256(csv_path),views=evaluation,protocol=config["evaluation"],
        semantics_source=config["semantics_source"]))
    write_json(run/"run_manifest.json",dict(run_id=run_id,status="PREPROCESSING_PASS",
        base_code_commit=git(root,"rev-parse","HEAD"),provider_commit=env["model_commit"],
        focal_manifest_sha256=sha256(run/"inputs/focal_manifest.json"),
        geometry_gt_access="separate evaluator only after optimization is frozen"))
    print(json.dumps(dict(run=str(run),shape=list(images.shape),pixel_error=error,
        focal_xy=focal_manifest["focal_xy"],depth_files_present=[r["depth_exists"] for r in evaluation])))
if __name__=="__main__":
    torch.set_num_threads(1)
    main()
