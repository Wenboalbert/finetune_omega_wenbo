"""GPU-only numerical preflight and one accepted GN step. No geometry GT imports."""
import argparse
from dataclasses import asdict
import gc
import json
import os
from pathlib import Path
import platform
import time
import numpy as np
import torch
from .io import read_json, write_json, sha256, git, verify_provider
from .specs import resolve_specs
from .intervention import ResidualBank, HookManager
from .solver import focal_xy, focal_residual, explicit_jacobian, solve_direction, budget_stats

def code_hashes(root):
    paths = sorted((root/"src/residual_tto").glob("*.py"))
    paths += sorted((root/"scripts").glob("*.pbs"))
    return {str(p.relative_to(root)):sha256(p) for p in paths}

def save_prediction(path, pose, depth, confidence, hw):
    from vggt_omega.utils.pose_enc import encoding_to_camera
    extrinsic,intrinsic=encoding_to_camera(pose,hw)
    arrays=dict(pose_enc=pose.detach().cpu().numpy(),extrinsic=extrinsic.detach().cpu().numpy(),
        intrinsic=intrinsic.detach().cpu().numpy(),depth=depth.detach().cpu().numpy(),
        depth_conf=confidence.detach().cpu().numpy())
    if Path(path).exists(): raise FileExistsError(path)
    np.savez_compressed(path,**arrays)
    return arrays

def diff_stats(new, old):
    delta=new.detach().float().cpu()-old
    rms=float(delta.square().mean().sqrt())
    return dict(rms=rms,max_abs=float(delta.abs().max()),
                relative_rms=rms/max(float(old.square().mean().sqrt()),1e-8))

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--run",required=True)
    parser.add_argument("--stage",choices=["preflight","smoke"],required=True)
    args=parser.parse_args()
    if not os.environ.get("PBS_JOBID") or not torch.cuda.is_available():
        raise RuntimeError("real Omega forward requires a PBS GPU allocation")
    run=Path(args.run).resolve();root=Path(__file__).resolve().parents[2]
    if run.parent != root/"runs": raise ValueError("run must belong to this V001 checkout")
    config=read_json(run/"resolved_config.json");env=read_json(run/"environment_config.json")
    manifest=read_json(run/"inputs/focal_manifest.json")
    allowed={"schema_version","camera_order","frame","packet_file","packet_sha256","views","focal_xy","semantics_source"}
    if set(manifest)!=allowed: raise ValueError("unexpected focal-only manifest fields")
    if sha256(manifest["packet_file"])!=manifest["packet_sha256"]:
        raise ValueError("packet checksum changed")
    if manifest["camera_order"]!=config["camera_order"]:
        raise ValueError("camera order mismatch")
    verify_provider(env["model_root"],env["model_commit"])
    stage=args.stage
    frozen=read_json(run/"numerical_preflight.json") if stage=="smoke" else None
    if frozen and (frozen["status"]!="PASS" or frozen["code_hashes"]!=code_hashes(root)
                   or frozen["config_sha256"]!=sha256(run/"resolved_config.json")):
        raise RuntimeError("preflight configuration or implementation changed")
    started=time.time()
    torch.set_num_threads(4)
    torch.manual_seed(config["seed"]);np.random.seed(config["seed"])
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    checkpoint_hash=sha256(env["checkpoint"])
    if checkpoint_hash!=env["checkpoint_expected_sha256"]:
        raise ValueError("checkpoint SHA256 mismatch")
    from vggt_omega.models.vggt_omega import VGGTOmega
    model=VGGTOmega().float().eval()
    state=torch.load(env["checkpoint"],map_location="cpu",weights_only=True)
    model.load_state_dict(state,strict=True);del state
    model.requires_grad_(False);model=model.to("cuda")
    if any(p.requires_grad for p in model.parameters()): raise RuntimeError("unfrozen model")
    with np.load(manifest["packet_file"],allow_pickle=False) as packet:
        images=torch.from_numpy(packet["images"]).unsqueeze(0).to("cuda",dtype=torch.float32)
    if images.shape[:2]!=(config["B"],config["S"]): raise ValueError("packet shape mismatch")
    names=config["camera_order"];views=len(names);hw=tuple(images.shape[-2:])
    target=torch.tensor(manifest["focal_xy"],device="cuda",dtype=torch.float32)
    supervised=tuple(names.index(n) for n in config["supervised_views"])
    specs=resolve_specs(config["specs"],names)
    bank=ResidualBank(specs,"cuda",1024)
    torch.cuda.reset_peak_memory_stats()
    hooks=HookManager(model,bank,views)

    def forward(values=None, grad=False, dense=False, capture=False):
        if values is not None: hooks.reset(values,capture)
        with torch.set_grad_enabled(grad),torch.autocast("cuda",enabled=False):
            caches,start=model.aggregator(images)
            if start!=17 or any(c is not None and c.dtype!=torch.float32 for c in caches):
                raise RuntimeError("wrong token layout or non-FP32 aggregator path")
            pose=model.camera_head(caches,patch_token_start=start)
            r=focal_residual(pose,hw,target,supervised)
            d,c=model.dense_head(caches,images=images,patch_token_start=start) if dense else (None,None)
        if values is not None: hooks.assert_hits()
        return pose,r,d,c,caches

    base_pose,base_r,base_depth,base_conf,base_caches=forward(dense=True)
    base_cache_cpu={i:c.detach().cpu() for i,c in enumerate(base_caches) if c is not None}
    del base_caches
    base_loss=float(base_r.square().sum())
    base_focal=focal_xy(base_pose,hw)[0].tolist()
    if stage=="preflight":
        save_prediction(run/"baseline/predictions.npz",base_pose,base_depth,base_conf,hw)
    elif not np.allclose(np.load(run/"baseline/predictions.npz")["pose_enc"],base_pose.cpu().numpy(),atol=1e-6,rtol=1e-6):
        raise RuntimeError("paired FP32 baseline changed since preflight")

    provenance=dict(stage=stage,git_commit=git(root,"rev-parse","HEAD"),
        git_status=git(root,"status","--porcelain"),code_hashes=code_hashes(root),
        provider_commit=env["model_commit"],checkpoint_sha256=checkpoint_hash,
        config_sha256=sha256(run/"resolved_config.json"),
        focal_manifest_sha256=sha256(run/"inputs/focal_manifest.json"),
        python=platform.python_version(),torch=torch.__version__,cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(),job_id=os.environ["PBS_JOBID"],host=platform.node(),
        autocast=False,tf32=False,parameter_dtype="float32",activation_dtype="float32",
        trainable_pretrained_parameters=0,active_residual_variables=bank.flatten(bank.current).numel(),
        specs=[asdict(s) for s in specs],supervised_views=list(supervised),
        jacobian_rows=[names[v]+"_"+axis for v in supervised for axis in ["logfx","logfy"]],
        jacobian_column_groups=bank.column_groups())
    write_json(run/("logs/"+stage+"_provenance.json"),provenance)
    print(json.dumps(dict(stage=stage,event="model_loaded",gpu=provenance["gpu"],base_focal=base_focal)),flush=True)

    with hooks:
        pose0,r0,d0,c0,caches0=forward(bank.current,dense=True,capture=True)
        noop={name:float((a-b).abs().max()) for name,a,b in
            [("pose",pose0,base_pose),("depth",d0,base_depth),("confidence",c0,base_conf)]}
        if any(value>1e-6 for value in noop.values()): raise RuntimeError("no-op mismatch: "+str(noop))
        del pose0,r0,d0,c0,caches0
        if stage=="preflight":
            hooks.audit_enabled=True
            routed={k:torch.full_like(v,1e-4) for k,v in bank.current.items()}
            routing_outputs=forward(routed)
            del routing_outputs
            hooks.audit_enabled=False
        probes=bank.probes()
        pose,r,_,_,caches=forward(bank.effective(probes),grad=True)
        jac=explicit_jacobian(r,probes,specs)
        r=r.detach()
        del pose,caches,probes
        hooks.reset(bank.current)
        gc.collect();torch.cuda.empty_cache()
        gram=jac@jac.T
        eig=torch.linalg.eigvalsh(gram)
        if not float(eig[-1])>0: raise RuntimeError("zero focal controllability")
        print(json.dumps(dict(stage=stage,event="jacobian_ready",shape=list(jac.shape),eigenvalues=eig.tolist())),flush=True)

        if stage=="preflight":
            numerical=config["numerics"]
            fd=[]
            generator=torch.Generator(device="cuda").manual_seed(config["seed"])
            row=jac[torch.argmax(torch.linalg.vector_norm(jac,dim=1))]
            directions=[row/torch.linalg.vector_norm(row).clamp_min(1e-20)]
            random=torch.randn(jac.shape[1],generator=generator,device="cuda")
            directions.append(random/torch.linalg.vector_norm(random))
            for idx,v in enumerate(directions):
                expected=jac@v
                for epsilon in numerical["fd_epsilons"]:
                    rp=forward(bank.unflatten(epsilon*v))[1]
                    rm=forward(bank.unflatten(-epsilon*v))[1]
                    measured=(rp-rm)/(2*epsilon)
                    relative=float(torch.linalg.vector_norm(measured-expected)/torch.linalg.vector_norm(expected).clamp_min(1e-12))
                    cosine=float(torch.nn.functional.cosine_similarity(measured,expected,dim=0,eps=1e-12))
                    fd.append(dict(direction=idx,kind="jacobian_row" if idx==0 else "random",
                        epsilon=epsilon,relative_error=relative,cosine=cosine,
                        predicted=expected.tolist(),measured=measured.tolist(),
                        passed=relative<=numerical["fd_relative_tolerance"] and cosine>=numerical["fd_cosine_min"]))
            fd_pass=all(any(row["passed"] for row in fd if row["direction"]==idx) for idx in range(len(directions)))
            mean_diag=max(float(gram.trace()/r.numel()),1e-12)
            trials=[]
            chosen=None
            for ratio in numerical["damping_relative_grid"]:
                damping=ratio*mean_diag
                step=solve_direction(jac,r,damping,config["jacobian_mode"],supervised,bank)
                stats=budget_stats(bank.unflatten(step),hooks.base,specs,
                    numerical["budget_absolute_rms_cap"],numerical["budget_relative_rms_cap"])
                linear_loss=float((r+jac@step).square().sum())
                passed=all(s["passed"] for s in stats) and linear_loss<base_loss
                trials.append(dict(damping=damping,ratio=ratio,budgets=stats,linear_loss=linear_loss,passed=passed))
                if chosen is None and passed: chosen=damping
            passed=fd_pass and chosen is not None
            preflight=dict(status="PASS" if passed else "STOP",code_hashes=code_hashes(root),
                config_sha256=sha256(run/"resolved_config.json"),noop=noop,routing=hooks.routing_audit,finite_difference=fd,
                gram_eigenvalues=eig.tolist(),initial_damping=chosen,algebraic_damping_trials=trials,
                budget_absolute_rms=numerical["budget_absolute_rms_cap"],
                budget_relative_rms=numerical["budget_relative_rms_cap"],
                budget_status="conservative development safety caps; not geometry-calibrated",
                selection_inputs="zero-residual activations, focal residual, Jacobian only",
                base_loss=base_loss,base_focal=base_focal)
            write_json(run/"numerical_preflight.json",preflight)
            np.savez_compressed(run/"logs/preflight_jacobian.npz",J=jac.cpu().numpy(),r=r.cpu().numpy())
        else:
            damping=frozen["initial_damping"]
            attempts=[];accepted=False
            final_pose,final_depth,final_conf=base_pose,base_depth,base_conf
            final_focal=base_focal;propagation={}
            for attempt in range(config["max_attempts"]):
                step=solve_direction(jac,r,damping,config["jacobian_mode"],supervised,bank)
                candidate=bank.flatten(bank.current)+step
                values=bank.unflatten(candidate)
                stats=budget_stats(values,hooks.base,specs,frozen["budget_absolute_rms"],frozen["budget_relative_rms"])
                valid=all(s["passed"] for s in stats)
                candidate_loss=None
                if valid:
                    cp,cr,_,_,ccaches=forward(values)
                    candidate_loss=float(cr.square().sum())
                    valid=candidate_loss<base_loss-config["numerics"]["minimum_loss_improvement"]
                    del ccaches
                attempts.append(dict(attempt=attempt,damping=damping,budgets=stats,
                    focal_loss=candidate_loss,accepted=valid))
                if valid:
                    bank.accept(candidate);accepted=True
                    final_pose,fr,final_depth,final_conf,fcaches=forward(bank.current,dense=True)
                    final_focal=focal_xy(final_pose,hw)[0].tolist()
                    for layer,base in base_cache_cpu.items():
                        for half,sl in [("frame",slice(0,1024)),("inter",slice(1024,2048))]:
                            propagation[str(layer)+"_"+half+"_patch"]=diff_stats(fcaches[layer][:,:,17:,sl],base[:,:,17:,sl])
                    for label,sl in [("camera",slice(0,1)),("registers",slice(1,17))]:
                        propagation["final_"+label]=diff_stats(fcaches[23][:,:,sl],base_cache_cpu[23][:,:,sl])
                    propagation["depth"]=diff_stats(final_depth,base_depth.detach().cpu())
                    del fcaches
                    break
                damping*=config["damping_retry_factor"]
            out=run/"adapted/post14_D02_registers"
            save_prediction(out/"predictions.npz",final_pose,final_depth,final_conf,hw)
            torch.save({k:v.cpu() for k,v in bank.current.items()},out/"residual.pt")
            write_json(out/"optimization.json",dict(status="ACCEPTED" if accepted else "STOP_NO_ACCEPT",
                accepted_steps=int(accepted),attempts=attempts,baseline_focal=base_focal,final_focal=final_focal,
                baseline_loss=base_loss,propagation=propagation,
                geometry_gt_used=False,selection_rule=config["acceptance_rule"],
                residual_sha256=sha256(out/"residual.pt")))
    cleanup_pose,_,cleanup_depth,_,cleanup_caches=forward(dense=True)
    cleanup=max(float((cleanup_pose-base_pose).abs().max()),float((cleanup_depth-base_depth).abs().max()))
    if cleanup>1e-6 or hooks.handles: raise RuntimeError("hook cleanup failed")
    write_json(run/("logs/"+stage+"_completion.json"),dict(status="PASS" if stage=="smoke" or passed else "STOP_NUMERICAL",cleanup_max_abs=cleanup,
        hook_handles_remaining=len(hooks.handles),seconds=time.time()-started,
        max_gpu_memory_bytes=torch.cuda.max_memory_allocated(),code_hashes=code_hashes(root)))
    print(json.dumps(dict(stage=stage,event="completed",cleanup_max_abs=cleanup)),flush=True)
    if stage=="preflight" and not passed: raise SystemExit("STOP: numerical preflight failed")

if __name__=="__main__":
    try:
        main()
    except BaseException as error:
        print(json.dumps(dict(event="STOP",error_type=type(error).__name__,message=str(error))),flush=True)
        raise
