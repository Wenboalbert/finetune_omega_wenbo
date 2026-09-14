"""Offline evaluator. Imported only after optimization has ended; GT never enters runner."""
import argparse
import itertools
import json
import math
import os
from pathlib import Path
import numpy as np
from PIL import Image
from .io import read_json,write_json,sha256

def ue_extrinsic(row, scale=.01):
    # Formula audited against legacy ue_camera.py at 9b4bac03a7fe.
    p,y,r=map(math.radians,[row["pitch"],row["yaw"],row["roll"]])
    cp,sp,cy,sy,cr,sr=math.cos(p),math.sin(p),math.cos(y),math.sin(y),math.cos(r),math.sin(r)
    axes=np.column_stack(([cp*cy,cp*sy,sp],
        [sr*sp*cy-cr*sy,sr*sp*sy+cr*cy,-sr*cp],
        [-(cr*sp*cy+sr*sy),sr*cy-cr*sp*sy,cr*cp]))
    basis=np.array([[0.,1,0],[0,0,-1],[1,0,0]])
    center=basis@np.array([row["pos_x"],row["pos_y"],row["pos_z"]])*scale
    rotation=(basis@axes@basis.T).T
    if not np.allclose(rotation@rotation.T,np.eye(3),atol=1e-8) or not np.isclose(np.linalg.det(rotation),1):
        raise ValueError("invalid UE rotation conversion")
    return np.column_stack((rotation,-rotation@center))

def centers(extrinsics):
    return -np.einsum("sji,sj->si",extrinsics[:,:,:3],extrinsics[:,:,3])

def angle(a,b):
    cosine=np.clip((np.trace(a@b.T)-1)/2,-1.,1.)
    return float(np.degrees(np.arccos(cosine)))

def sim3(src,dst,condition_min=.001):
    xs,ys=src-src.mean(0),dst-dst.mean(0)
    for points in [xs,ys]:
        singular=np.linalg.svd(points,compute_uv=False)
        if singular[0]<1e-12 or singular[1]/singular[0]<condition_min:
            raise ValueError("degenerate support; do not change support or alignment protocol")
    u,d,vt=np.linalg.svd(ys.T@xs/len(src))
    sign=np.eye(3);sign[-1,-1]=1 if np.linalg.det(u@vt)>0 else -1
    q=u@sign@vt
    s=float(np.sum(d*np.diag(sign))/np.mean(np.sum(xs*xs,axis=1)))
    b=dst.mean(0)-s*q@src.mean(0)
    if not math.isfinite(s) or s<=0 or not np.allclose(q@q.T,np.eye(3),atol=1e-8):
        raise ValueError("invalid Sim3")
    return s,q,b

def depth_gt(path, transform, unit_scale):
    os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR","1")
    import cv2
    raw=cv2.imread(str(path),cv2.IMREAD_UNCHANGED)
    if raw is None: raise ValueError("unable to decode depth")
    channel="single"
    if raw.ndim==3:
        rgb=raw[...,:3]
        if all(np.array_equal(rgb[...,0],rgb[...,i],equal_nan=True) for i in range(1,rgb.shape[-1])):
            raw=rgb[...,0];channel="identical_rgb"
        else:
            active=[i for i in range(rgb.shape[-1]) if np.any(np.isfinite(rgb[...,i]) & (rgb[...,i]!=0))]
            if len(active)!=1: raise ValueError("ambiguous EXR channels; explicit exporter metadata required")
            raw=rgb[...,active[0]];channel="only_nonzero_channel_"+str(active[0])
    if list(raw.shape)!=transform["raw_hw"]: raise ValueError("RGB/depth source size mismatch")
    raw=raw.astype(np.float32)*unit_scale
    raw_valid=np.isfinite(raw)&(raw>0)
    l,t,r,b=transform["crop_ltrb"];rh,rw=transform["resize_hw"]
    # Nearest resampling preserves depth values, using the same image-space crop and padding.
    resized=np.array(Image.fromarray(raw[t:b,l:r]).resize((rw,rh),Image.Resampling.NEAREST))
    valid=np.array(Image.fromarray(raw_valid[t:b,l:r]).resize((rw,rh),Image.Resampling.NEAREST))
    pl,pt,pr,pb=transform["pad_ltrb"]
    return np.pad(resized,((pt,pb),(pl,pr))),np.pad(valid,((pt,pb),(pl,pr))),channel

def main():
    parser=argparse.ArgumentParser();parser.add_argument("--run",required=True);args=parser.parse_args()
    run=Path(args.run).resolve()
    if (run/"study_manifest.json").exists():
        # This barrier runs BEFORE the first read of any geometry GT.
        from .study import assert_study_frozen
        assert_study_frozen(run)
        if not (run/"logs/geometry_release.json").exists():
            raise RuntimeError("study driver has not released offline evaluation")
    elif read_json(run/"logs/smoke_completion.json")["status"]!="PASS": raise RuntimeError("optimizer not frozen")
    optimization=read_json(run/"adapted/post14_D02_registers/optimization.json")
    residual_path=run/"adapted/post14_D02_registers/residual.pt"
    if sha256(residual_path)!=optimization["residual_sha256"]: raise ValueError("residual changed")
    gt_manifest=read_json(run/"private_evaluation/manifest.json")
    focal=read_json(run/"inputs/focal_manifest.json")
    protocol=gt_manifest["protocol"];names=focal["camera_order"]
    if [v["camera"] for v in gt_manifest["views"]]!=names: raise ValueError("GT camera order mismatch")
    gt_e=np.stack([ue_extrinsic(v["ue_pose"],protocol["pose_position_to_m"]) for v in gt_manifest["views"]])
    gt_c=centers(gt_e)
    support=[names.index(n) for n in protocol["support_views"]]
    target=names.index(protocol["heldout_view"])
    if target in support: raise ValueError("held-out target included in support")
    gt_depths=[]
    for item,view in zip(gt_manifest["views"],focal["views"]):
        if not item["depth_exists"] or sha256(item["depth_path"])!=item["depth_sha256"]:
            raise ValueError("depth source missing or changed")
        gt_depths.append(depth_gt(item["depth_path"],view["transform"],protocol["depth_to_m"]))
    reports={}
    for arm,path in [("baseline",run/"baseline/predictions.npz"),
                     ("adapted",run/"adapted/post14_D02_registers/predictions.npz")]:
        with np.load(path,allow_pickle=False) as f:
            pred_e=f["extrinsic"][0].astype(np.float64)
            pred_k=f["intrinsic"][0].astype(np.float64)
            pred_depth=f["depth"][0,...,0].astype(np.float64)
        if not np.isfinite(pred_e).all(): raise ValueError("nonfinite predicted extrinsics")
        for e in pred_e:
            if not np.allclose(e[:,:3]@e[:,:3].T,np.eye(3),atol=1e-4) or abs(np.linalg.det(e[:,:3])-1)>1e-4:
                raise ValueError("invalid predicted rotation")
        pred_c=centers(pred_e)
        s,q,b=sim3(pred_c[support],gt_c[support],protocol["support_sigma2_over_sigma1_min"])
        aligned=s*(pred_c@q.T)+b
        rows=[]
        for i,name in enumerate(names):
            gt_d,valid,channel=gt_depths[i];depth=s*pred_depth[i]
            pred_valid=np.isfinite(depth)&(depth>0);common=valid&pred_valid
            invalid_count=int((valid&~pred_valid).sum())
            absrel=float(np.mean(np.abs(depth[common]-gt_d[common])/gt_d[common])) if common.any() else None
            transform=focal["views"][i]["transform"]
            rows.append(dict(camera=name,alignment_role="support_in_sample" if i in support else "heldout_from_alignment",
                center_error_m=float(np.linalg.norm(aligned[i]-gt_c[i])),
                rotation_error_deg=angle(pred_e[i,:,:3]@q.T,gt_e[i,:,:3]),
                focal_relative_error_xy=[float(abs(pred_k[i,0,0]/transform["fx"]-1)),float(abs(pred_k[i,1,1]/transform["fy"]-1))],
                depth_absrel=absrel if invalid_count==0 else None,depth_absrel_valid_predictions_only=absrel,
                depth_gt_valid_count=int(valid.sum()),depth_prediction_invalid_count=invalid_count,
                depth_gt_coverage=float(valid.mean()),depth_channel_policy=channel,
                depth_gt_m_min=float(gt_d[valid].min()) if valid.any() else None,
                depth_gt_m_max=float(gt_d[valid].max()) if valid.any() else None))
        pairs=[]
        for i,j in itertools.combinations(range(len(names)),2):
            gt_vector=gt_e[i,:,:3]@(gt_c[j]-gt_c[i])
            pred_vector=pred_e[i,:,:3]@(pred_c[j]-pred_c[i])
            gt_length,pred_length=np.linalg.norm(gt_vector),np.linalg.norm(pred_vector)
            direction=None if min(gt_length,pred_length)<1e-10 else float(np.degrees(np.arccos(np.clip(
                np.dot(gt_vector,pred_vector)/(gt_length*pred_length),-1,1))))
            pairs.append(dict(first=names[i],second=names[j],
                relative_rotation_deg=angle(pred_e[j,:,:3]@pred_e[i,:,:3].T,gt_e[j,:,:3]@gt_e[i,:,:3].T),
                baseline_direction_deg=direction,
                baseline_length_relative_error=float(abs(s*pred_length/gt_length-1)) if gt_length>1e-10 else None))
        reports[arm]=dict(scale=s,Q=q.tolist(),b=b.tolist(),views=rows,pairs=pairs,
            support_rms_m=float(np.sqrt(np.mean(np.sum((aligned[support]-gt_c[support])**2,axis=1)))),
            predictions_sha256=sha256(path))
    write_json(run/"comparison/geometry_metrics.json",dict(protocol=protocol,arms=reports,
        interpretation="one-packet post-adaptation diagnostic; D02 is not an unseen view",
        gt_used_only_after_optimizer_frozen=True,optimization_status=optimization["status"],
        evaluator_source_sha256=sha256(__file__),gt_manifest_sha256=sha256(run/"private_evaluation/manifest.json"),
        numpy_version=np.__version__))
    summary={arm:{"scale":r["scale"],"D02":r["views"][target]} for arm,r in reports.items()}
    print(json.dumps(summary),flush=True)
if __name__=="__main__": main()
