"""Read-only focal/activation regression audit; no geometry GT is loaded."""
from pathlib import Path
import numpy as np
from .io import read_json, sha256
from .zero_target import validate_config

RTOL = 1e-6
ATOL = 1e-8

def audit_prefix(previous, current):
    previous,current = Path(previous),Path(current)
    validate_config(read_json(current/"resolved_config.json"),read_json(previous/"resolved_config.json"))
    old=read_json(previous/"adapted/post14_D02_registers/optimization.json")
    new=read_json(current/"adapted/post14_D02_registers/optimization.json")
    ostate=read_json(previous/"logs/ladder_completion.json")
    nstate=read_json(current/"logs/ladder_completion.json")
    if ostate["status"]!="FROZEN" or nstate["status"]!="FROZEN":
        raise RuntimeError("audit requires both endpoints frozen")
    exact=True;checks=0;max_abs=0.;sources={}
    def compare(a,b,label):
        nonlocal exact,checks,max_abs
        x,y=np.asarray(a,dtype=np.float64),np.asarray(b,dtype=np.float64)
        if x.shape!=y.shape or not np.isfinite(x).all() or not np.isfinite(y).all():
            raise RuntimeError("invalid prefix arrays: "+label)
        error=float(np.max(np.abs(x-y))) if x.size else 0.
        max_abs=max(max_abs,error);exact=exact and bool(np.array_equal(x,y));checks+=1
        if not np.allclose(x,y,rtol=RTOL,atol=ATOL):
            raise RuntimeError("trajectory prefix diverged: "+label+" max_abs="+str(error))
    def npz_pair(rel,keys):
        for p in (previous/rel,current/rel): sources[str(p)]=sha256(p)
        with np.load(previous/rel,allow_pickle=False) as a,np.load(current/rel,allow_pickle=False) as b:
            for key in keys: compare(a[key],b[key],rel+":"+key)
    npz_pair("baseline/predictions.npz",("pose_enc","depth","intrinsic","extrinsic"))
    npz_pair("logs/preflight_jacobian.npz",("J","r","h0"))
    compare(old["h0_rms"],new["h0_rms"],"h0_rms")
    compare(old["epsilon_repeat"],new["epsilon_repeat"],"epsilon_repeat")
    old_steps=old["counters"]["accepted_steps"]
    if new["counters"]["accepted_steps"]<old_steps:
        raise RuntimeError("new run ended before old accepted prefix")
    for step in range(1,old_steps+1):
        npz_pair("logs/accepted_%03d.npz"%step,("a","r","pose_enc"))
    for index in range(1,old["counters"]["jacobians"]+1):
        npz_pair("logs/jacobian_%03d.npz"%index,("J","r","a"))
    if len(new["attempts"])<len(old["attempts"]):
        raise RuntimeError("new run has fewer candidate events than reference")
    integer_keys=("candidate","jacobian","local_candidate","accepted","accepted_step","radius_action","growth_streak")
    numeric_keys=("radius_relative","current_loss","tau","current_cumulative_relative",
        "step_relative","cumulative_relative","predicted_decrease","actual_decrease",
        "rho","candidate_loss","next_radius_relative","lambda_step","mu_cumulative")
    for i,(a,b) in enumerate(zip(old["attempts"],new["attempts"])):
        for key in integer_keys:
            if a.get(key)!=b.get(key): raise RuntimeError("candidate decision changed: "+str((i,key)))
        for key in numeric_keys:
            if a.get(key) is None or b.get(key) is None:
                if a.get(key)!=b.get(key): raise RuntimeError("candidate field changed: "+key)
            else: compare(a[key],b[key],str(i)+":"+key)
        if "diagnostics" in a:
            for key in ("focal_xy","relative_error_xy","supervised_mean_relative_error"):
                compare(a["diagnostics"][key],b["diagnostics"][key],str(i)+":"+key)
    if old["termination_reason"]!="TARGET_REACHED":
        if new["termination_reason"]!=old["termination_reason"] or new["counters"]!=old["counters"]:
            raise RuntimeError("non-target reference should reproduce its stop and counts")
    return dict(status="PASS",reference_run=str(previous),current_run=str(current),
        rtol=RTOL,atol=ATOL,all_compared_values_exact=exact,max_absolute_difference=max_abs,
        numeric_comparisons=checks,reference_accepted_steps=old_steps,
        reference_jacobians=old["counters"]["jacobians"],reference_candidates=len(old["attempts"]),
        reference_termination=old["termination_reason"],new_termination=new["termination_reason"],
        new_accepted_steps=new["counters"]["accepted_steps"],geometry_gt_read=False,
        source_hashes=sources)
