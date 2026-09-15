"""Independent CPU audit of saved states; never loads geometry ground truth."""
from pathlib import Path
import numpy as np
from .io import read_json,sha256
from .trust_policy import loss,noise_floor,accepted,next_radius
from .extended import check_endpoint,OPT,endpoint_focal

def close(a,b,label):
    if not np.allclose(a,b,rtol=1e-9,atol=1e-12,equal_nan=False):
        raise ValueError("numerical audit mismatch: "+label)

def audit_validity(run):
    run=Path(run);done=check_endpoint(run);cfg=read_json(run/"resolved_config.json")
    opt=read_json(run/OPT);options=cfg["optimizer"];pre=read_json(run/"numerical_preflight.json")
    provenance=read_json(run/"logs/ladder_provenance.json")
    if pre["status"]!="PASS" or opt["geometry_gt_used"] or provenance["geometry_gt_used"]:
        raise ValueError("preflight/GT isolation failed")
    if done["hook_handles_remaining"]!=0 or done["cleanup_max_abs"]>1e-6:
        raise ValueError("hooks were not restored")
    if opt["termination_reason"]=="SOLVER_FAILURE": raise ValueError("solver failure is not valid optimization")
    if sha256(run/"adapted/post14_D02_registers/residual.pt")!=opt["residual_sha256"]:
        raise ValueError("residual checksum mismatch")
    if done["counters"]!=opt["counters"]: raise ValueError("counter mismatch")
    with np.load(run/"logs/preflight_jacobian.npz",allow_pickle=False) as z:
        h0=z["h0"].astype(np.float64);initial_r=z["r"].copy()
    scale=float(np.linalg.norm(h0));a=np.zeros(h0.size,dtype=np.float32)
    close(scale,pre["norm_scale"],"norm scale");close(scale/np.sqrt(h0.size),opt["h0_rms"],"h0 rms")
    radius=options["initial_step_relative"];streak=0;steps=0;forwards=0;per_j={}
    max_step=0.;seen=set();sources={}
    for idx,e in enumerate(opt["attempts"],1):
        if e["candidate"]!=idx: raise ValueError("candidate order changed")
        jp=run/("logs/jacobian_%03d.npz"%e["jacobian"]);sources[str(jp.relative_to(run))]=sha256(jp)
        with np.load(jp,allow_pickle=False) as z: J=z["J"];r=z["r"];ja=z["a"]
        if not np.array_equal(ja,a): raise ValueError("Jacobian is not at current accepted state")
        seen.add(e["jacobian"]);per_j[e["jacobian"]]=per_j.get(e["jacobian"],0)+1
        if e["local_candidate"]!=per_j[e["jacobian"]] or per_j[e["jacobian"]]>options["max_candidates_per_jacobian"]:
            raise ValueError("per-J candidate limit/order failed")
        L=loss(r);tau=noise_floor(L,opt["epsilon_repeat"])
        close(e["current_loss"],L,"current loss");close(e["tau"],tau,"tau")
        close(e["radius_relative"],radius,"radius")
        close(e["current_cumulative_relative"],np.linalg.norm(a.astype(np.float64))/scale,"current budget")
        if not e.get("solver",{}).get("kkt",{}).get("passed",False):
            raise ValueError("subproblem certificate failed")
        pred=e["predicted_decrease"];actual=e.get("actual_decrease");rho=e.get("rho")
        legal=e["casting"]["legal"]
        expected=actual is not None and accepted(pred,actual,rho,tau,legal)
        if bool(e["accepted"])!=bool(expected): raise ValueError("acceptance rule mismatch")
        if actual is not None:
            forwards+=1
            close(actual,L-e["candidate_loss"],"actual decrease")
            close(rho,actual/pred,"rho")
        elif "forward_numerical_error" in e: forwards+=1
        if e["accepted"]:
            steps+=1
            if e["accepted_step"]!=steps: raise ValueError("accepted step order")
            ap=run/("logs/accepted_%03d.npz"%steps);sources[str(ap.relative_to(run))]=sha256(ap)
            with np.load(ap,allow_pickle=False) as z:
                new=z["a"];rr=z["r"];pose=z["pose_enc"]
            if new.dtype!=np.float32 or not np.isfinite(new).all(): raise ValueError("invalid state storage")
            delta=new.astype(np.float64)-a.astype(np.float64)
            sn=float(np.linalg.norm(delta));cn=float(np.linalg.norm(new.astype(np.float64)))
            if sn>radius*scale or cn>cfg["cumulative_budget_relative"]*scale:
                raise ValueError("actual FP32 state violates a budget")
            close(pred,L-loss(r+J@delta),"effective predicted decrease")
            close(actual,L-loss(rr),"effective actual decrease")
            close(e["step_relative"],sn/scale,"effective step")
            close(e["cumulative_relative"],cn/scale,"effective cumulative")
            max_step=max(max_step,sn/scale);a=new.copy()
        if "next_radius_relative" in e:
            utilization=e["casting"]["step_l2"]/(scale*radius)
            nr,ns,action=next_radius(radius,options["minimum_step_relative"],options["maximum_step_relative"],
                                     bool(e["accepted"]),rho if rho is not None else -1.,utilization,streak)
            close(e["next_radius_relative"],nr,"next radius")
            if e["growth_streak"]!=ns or e["radius_action"]!=action: raise ValueError("radius action mismatch")
            radius,streak=nr,ns
    counters=opt["counters"]
    if counters["accepted_steps"]!=steps or counters["candidate_forwards"]!=forwards or counters["candidates_generated"]!=len(opt["attempts"]):
        raise ValueError("execution counters incorrect")
    if seen and seen!=set(range(1,counters["jacobians"]+1)): raise ValueError("Jacobian sequence incomplete")
    if steps>options["max_accepted_steps"] or counters["jacobians"]>options["max_jacobians"] or forwards>options["max_candidate_forwards"]:
        raise ValueError("compute cap exceeded")
    close(opt["cumulative_relative"],np.linalg.norm(a.astype(np.float64))/scale,"endpoint budget")
    close(opt["final_radius_relative"],radius,"endpoint radius")
    # Stored residual is checked against actual accepted FP32 state, not just a hash.
    import torch
    state=torch.load(run/"adapted/post14_D02_registers/residual.pt",map_location="cpu",weights_only=True)
    actual_state=np.concatenate([v.numpy().reshape(-1) for v in state.values()])
    if not np.array_equal(actual_state,a): raise ValueError("saved residual differs from accepted endpoint")
    if steps:
        with np.load(run/"adapted/post14_D02_registers/predictions.npz",allow_pickle=False) as z:
            if np.max(np.abs(z["pose_enc"]-pose))>1e-6: raise ValueError("endpoint prediction replay differs")
    return dict(status="PASS",completion_sha256=sha256(run/"logs/ladder_completion.json"),
                preflight_sha256=sha256(run/"numerical_preflight.json"),sources=sources,
                counters=counters,max_step_relative=max_step,cumulative_relative=opt["cumulative_relative"],
                termination=opt["termination_reason"],termination_detail=opt["termination_detail"],
                focal=endpoint_focal(run),geometry_gt_read=False)
