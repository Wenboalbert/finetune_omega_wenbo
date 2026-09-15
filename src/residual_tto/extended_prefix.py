"""Non-blocking group-4 prefix diagnostic. Tolerances remain unchanged."""
from pathlib import Path
import numpy as np
from .io import read_json,sha256
from .extended import OPT

RTOL=1e-6
ATOL=1e-8

def audit_prefix(previous,current):
    previous,current=Path(previous),Path(current)
    old=read_json(previous/OPT);new=read_json(current/OPT)
    numeric=[];decisions=[];missing=[];checks=0;exact=True;sources={}
    def compare(a,b,label):
        nonlocal checks,exact
        x,y=np.asarray(a,dtype=np.float64),np.asarray(b,dtype=np.float64)
        checks+=1
        if x.shape!=y.shape or not np.isfinite(x).all() or not np.isfinite(y).all():
            numeric.append(dict(label=label,issue="shape_or_nonfinite"));exact=False;return
        eq=bool(np.array_equal(x,y));exact=exact and eq
        if not np.allclose(x,y,rtol=RTOL,atol=ATOL):
            numeric.append(dict(label=label,max_absolute_difference=float(np.max(np.abs(x-y))),
                                relative_l2_difference=float(np.linalg.norm(x-y)/max(np.linalg.norm(x),1e-300))))
    def pair(rel,keys):
        pp,cp=previous/rel,current/rel
        if not cp.exists(): missing.append(rel);return
        sources[str(pp)]=sha256(pp);sources[str(cp)]=sha256(cp)
        with np.load(pp,allow_pickle=False) as p,np.load(cp,allow_pickle=False) as n:
            for k in keys: compare(p[k],n[k],rel+":"+k)
    pair("baseline/predictions.npz",("pose_enc","depth","intrinsic","extrinsic"))
    pair("logs/preflight_jacobian.npz",("J","r","h0"))
    for k in ("h0_rms","epsilon_repeat"): compare(old[k],new[k],k)
    for i in range(1,old["counters"]["accepted_steps"]+1):
        pair("logs/accepted_%03d.npz"%i,("a","r","pose_enc"))
    for i in range(1,old["counters"]["jacobians"]+1):
        pair("logs/jacobian_%03d.npz"%i,("J","r","a"))
    integer=("candidate","jacobian","local_candidate","accepted","accepted_step","radius_action","growth_streak")
    numeric_keys=("radius_relative","current_loss","tau","current_cumulative_relative","step_relative",
                  "cumulative_relative","predicted_decrease","actual_decrease","rho","candidate_loss",
                  "next_radius_relative","lambda_step","mu_cumulative")
    for idx,a in enumerate(old["attempts"]):
        if idx>=len(new["attempts"]): missing.append("candidate_%03d"%(idx+1));continue
        b=new["attempts"][idx]
        for key in integer:
            if a.get(key)!=b.get(key): decisions.append(dict(candidate=idx+1,field=key,old=a.get(key),new=b.get(key)))
        for key in numeric_keys:
            if a.get(key) is None or b.get(key) is None:
                if a.get(key)!=b.get(key): numeric.append(dict(label=str(idx+1)+":"+key,issue="missing_field"))
            else: compare(a[key],b[key],str(idx+1)+":"+key)
        if "diagnostics" in a and "diagnostics" in b:
            for key in ("focal_xy","relative_error_xy","supervised_mean_relative_error"):
                compare(a["diagnostics"][key],b["diagnostics"][key],str(idx+1)+":"+key)
    # Termination and final counters are NOT compared: 40 -> 80 is intended.
    status="INCOMPLETE" if missing else ("FAIL" if numeric or decisions else "PASS")
    return dict(status=status,rtol=RTOL,atol=ATOL,all_compared_values_exact=exact,
                reference=str(previous),current=str(current),numeric_comparisons=checks,
                numeric_mismatches=numeric,decision_mismatches=decisions,missing_prefix=missing,
                reference_counters=old["counters"],new_counters=new["counters"],
                reference_termination=old["termination_reason"],new_termination=new["termination_reason"],
                expected_protocol_difference="old compute stop may be extended; final termination/counters not required to match",
                alignment_note="after any decision divergence, equal candidate/accepted indices need not represent the same state",
                geometry_gt_read=False,is_geometry_gate=False,source_hashes=sources)
