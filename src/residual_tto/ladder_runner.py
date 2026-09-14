"""PBS-only V001 group 2: external register residual, focal-only optimization."""
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
from .io import read_json,write_json,sha256,git,verify_provider
from .specs import resolve_specs
from .intervention import ResidualBank,HookManager
from .solver import focal_xy,focal_residual,explicit_jacobian
from .runner import code_hashes,save_prediction,diff_stats
from .ladder_optimizer import optimize
from .trust_policy import loss,focal_diagnostics
from .study import freeze_endpoint


def np64(x):
    return x.detach().cpu().numpy().astype(np.float64)


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--run",required=True);args=parser.parse_args()
    if not os.environ.get("PBS_JOBID") or not torch.cuda.is_available():
        raise RuntimeError("Omega execution requires PBS GPU allocation")
    root=Path(__file__).resolve().parents[2];run=Path(args.run).resolve()
    if run.parent!=root/"runs": raise ValueError("run must belong to V001")
    config=read_json(run/"resolved_config.json");env=read_json(run/"environment_config.json")
    manifest=read_json(run/"inputs/focal_manifest.json");study=read_json(run/"study_manifest.json")
    allowed={"schema_version","camera_order","frame","packet_file","packet_sha256","views","focal_xy","semantics_source"}
    if set(manifest)!=allowed: raise ValueError("unexpected focal-only fields")
    if config["experiment_group"] not in ("v001_group2_budget_ladder","v001_group3_zero_target") or config["frame"]!=40:
        raise ValueError("wrong experiment group")
    if config["experiment_group"]=="v001_group3_zero_target":
        from .zero_target import validate_config,reference_config,verify_core
        validate_config(config,reference_config(root,config["cumulative_budget_relative"]))
        if verify_core(root)!=study["frozen_core_hashes"]: raise ValueError("frozen core hash mismatch")
    if config["supervised_views"]!=["Drone_02"] or config["camera_order"]!=[
            "CCTV_01","CCTV_02","CCTV_03","CCTV_04","Drone_02"]:
        raise ValueError("frozen focal supervision/order changed")
    if manifest["camera_order"]!=config["camera_order"] or sha256(manifest["packet_file"])!=manifest["packet_sha256"]:
        raise ValueError("input changed")
    member=[x for x in study["runs"] if Path(x["path"]).resolve()==run]
    if len(member)!=1 or member[0]["config_sha256"]!=sha256(run/"resolved_config.json"):
        raise ValueError("study membership/config changed")
    if study["code_commit"]!=git(root,"rev-parse","HEAD") or git(root,"status","--porcelain"):
        raise RuntimeError("unreviewed or dirty implementation")
    verify_provider(env["model_root"],env["model_commit"])
    started=time.time();options=config["optimizer"]
    torch.set_num_threads(4);torch.manual_seed(config["seed"]);np.random.seed(config["seed"])
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    checkpoint_hash=sha256(env["checkpoint"])
    if checkpoint_hash!=env["checkpoint_expected_sha256"]: raise ValueError("weight hash changed")
    from vggt_omega.models.vggt_omega import VGGTOmega
    model=VGGTOmega().float().eval()
    state=torch.load(env["checkpoint"],map_location="cpu",weights_only=True)
    model.load_state_dict(state,strict=True);del state
    model.requires_grad_(False);model=model.to("cuda")
    with np.load(manifest["packet_file"],allow_pickle=False) as packet:
        images=torch.from_numpy(packet["images"]).unsqueeze(0).to("cuda",dtype=torch.float32)
    if images.shape[:2]!=(1,5): raise ValueError("wrong packet shape")
    names=config["camera_order"];hw=tuple(images.shape[-2:])
    target=torch.tensor(manifest["focal_xy"],device="cuda",dtype=torch.float32)
    specs=resolve_specs(config["specs"],names)
    if len(specs)!=1 or specs[0].site!="post_frame" or specs[0].layer!=14 or specs[0].view_ids!=(4,) or tuple(specs[0].slots())!=tuple(range(1,17)):
        raise ValueError("intervention differs from post14 D02 registers")
    bank=ResidualBank(specs,"cuda",1024);hooks=HookManager(model,bank,5)
    n=bank.flatten(bank.current).numel()
    if n!=16384 or any(p.requires_grad for p in model.parameters()): raise RuntimeError("invalid trainable scope")
    torch.cuda.reset_peak_memory_stats()
    raw_forward_count=0

    def forward(values=None,grad=False,dense=False,capture=False):
        nonlocal raw_forward_count
        raw_forward_count+=1
        if values is not None: hooks.reset(values,capture)
        with torch.set_grad_enabled(grad),torch.autocast("cuda",enabled=False):
            caches,start=model.aggregator(images)
            if start!=17 or any(c is not None and c.dtype!=torch.float32 for c in caches):
                raise RuntimeError("wrong token/precision path")
            pose=model.camera_head(caches,patch_token_start=start)
            r=focal_residual(pose,hw,target,(4,))
            depth,conf=model.dense_head(caches,images=images,patch_token_start=start) if dense else (None,None)
        if values is not None: hooks.assert_hits()
        return pose,r,depth,conf,caches

    def data_of(pose,r):
        return dict(r=np64(r),pose_enc=pose.detach().cpu().tolist(),
                    diagnostics=focal_diagnostics(np64(focal_xy(pose,hw)[0]),manifest["focal_xy"]))

    def values_of(a):
        return bank.unflatten(torch.from_numpy(np.asarray(a,dtype=np.float32)).to("cuda"))

    def candidate_forward(a):
        try:
            p,r,_,_,caches=forward(values_of(a))
            data=data_of(p,r)
            del p,r,caches
            return data
        except ValueError as error:
            if "focal" in str(error).lower() or "finite" in str(error).lower():
                raise FloatingPointError(str(error)) from error
            raise

    def jacobian(a):
        bank.accept(torch.from_numpy(a).to("cuda"))
        probes=bank.probes()
        p,r,_,_,caches=forward(bank.effective(probes),grad=True)
        J=explicit_jacobian(r,probes,specs)
        data=data_of(p,r);result=np64(J)
        del p,r,caches,probes,J
        hooks.reset(bank.current)
        gc.collect();torch.cuda.empty_cache()
        return result,data

    base_pose,base_r,base_depth,base_conf,caches=forward(dense=True)
    # Baseline depth/caches are diagnostic only, not an optimization signal.
    base_cache_cpu={i:c.detach().cpu() for i,c in enumerate(caches) if c is not None}
    del caches
    baseline_data=data_of(base_pose,base_r)
    save_prediction(run/"baseline/predictions.npz",base_pose,base_depth,base_conf,hw)
    provenance=dict(git_commit=git(root,"rev-parse","HEAD"),code_hashes=code_hashes(root),
        provider_commit=env["model_commit"],checkpoint_sha256=checkpoint_hash,
        config_sha256=sha256(run/"resolved_config.json"),study_sha256=sha256(run/"study_manifest.json"),
        focal_manifest_sha256=sha256(run/"inputs/focal_manifest.json"),
        python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,
        cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),job_id=os.environ["PBS_JOBID"],
        host=platform.node(),autocast=False,tf32=False,parameter_dtype="float32",
        state_dtype="float32",solver_dtype="float64",active_variables=n,
        trainable_native_parameters=0,specs=[asdict(s) for s in specs],
        baseline_focal=baseline_data["diagnostics"],geometry_gt_used=False)
    write_json(run/"logs/ladder_provenance.json",provenance)
    print(json.dumps(dict(event="model_loaded",budget=config["cumulative_budget_relative"],
                         baseline=baseline_data["diagnostics"])),flush=True)
    with hooks:
        p,r,d,c,cache=forward(bank.current,dense=True,capture=True)
        noop={name:float((x-y).abs().max()) for name,x,y in [
            ("pose",p,base_pose),("depth",d,base_depth),("confidence",c,base_conf)]}
        if any(v>1e-6 for v in noop.values()): raise RuntimeError("zero residual no-op failed")
        repeat=[data_of(p,r)];del p,r,d,c,cache
        h0=bank.flatten(hooks.base).detach().cpu().numpy().astype(np.float64)
        h0_rms=float(np.sqrt(np.mean(h0*h0)));norm_scale=h0_rms*np.sqrt(n)
        if not np.isfinite(norm_scale) or norm_scale<=0: raise RuntimeError("invalid reference activation")
        for _ in range(options["repeat_forwards"]-1):
            repeat.append(candidate_forward(np.zeros(n,dtype=np.float32)))
        repeat_losses=[loss(x["r"]) for x in repeat]
        epsilon_repeat=max(repeat_losses)-min(repeat_losses)
        hooks.audit_enabled=True
        outputs=forward({k:torch.full_like(v,1e-4) for k,v in bank.current.items()})
        del outputs
        hooks.audit_enabled=False
        J,initial=jacobian(np.zeros(n,dtype=np.float32))
        # Also include no-grad/grad replay disagreement in the fixed repeat floor.
        epsilon_repeat=max(epsilon_repeat,abs(loss(initial["r"])-loss(repeat[0]["r"])))
        np.savez_compressed(run/"logs/preflight_jacobian.npz",J=J,r=initial["r"],h0=h0)
        numerics=config["numerics"];fd=[]
        generator=torch.Generator(device="cuda").manual_seed(config["seed"])
        jtorch=torch.tensor(J,device="cuda",dtype=torch.float32)
        row=jtorch[torch.argmax(torch.linalg.vector_norm(jtorch,dim=1))]
        directions=[row/torch.linalg.vector_norm(row).clamp_min(1e-20)]
        random=torch.randn(n,generator=generator,device="cuda")
        directions.append(random/torch.linalg.vector_norm(random))
        for index,v in enumerate(directions):
            expected=jtorch@v
            for epsilon in numerics["fd_epsilons"]:
                rp=forward(bank.unflatten(epsilon*v))[1]
                rm=forward(bank.unflatten(-epsilon*v))[1]
                measured=(rp-rm)/(2*epsilon)
                relative=float(torch.linalg.vector_norm(measured-expected)/torch.linalg.vector_norm(expected).clamp_min(1e-12))
                cosine=float(torch.nn.functional.cosine_similarity(measured,expected,dim=0,eps=1e-12))
                fd.append(dict(direction=index,epsilon=epsilon,relative_error=relative,cosine=cosine,
                    passed=relative<=numerics["fd_relative_tolerance"] and cosine>=numerics["fd_cosine_min"],
                    predicted=expected.tolist(),measured=measured.tolist()))
        passed=all(any(x["passed"] for x in fd if x["direction"]==i) for i in (0,1))
        write_json(run/"numerical_preflight.json",dict(status="PASS" if passed else "STOP",
            noop=noop,routing=hooks.routing_audit,finite_difference=fd,
            repeat_losses=repeat_losses,epsilon_repeat=epsilon_repeat,h0_rms=h0_rms,norm_scale=norm_scale,
            cumulative_budget_relative=config["cumulative_budget_relative"],
            cumulative_budget_absolute_rms=config["cumulative_budget_relative"]*h0_rms,
            independent_absolute_cap=None,initial_jacobian_reused=True,
            code_hashes=code_hashes(root),config_sha256=sha256(run/"resolved_config.json")))
        if not passed: raise RuntimeError("numerical preflight STOP; no optimizer or geometry")
        del directions,jtorch,row,random,rp,rm
        print(json.dumps(dict(event="preflight_pass",h0_rms=h0_rms,epsilon_repeat=epsilon_repeat)),flush=True)
        def event(entry):
            write_json(run/("logs/candidate_%03d.json"%entry["candidate"]),entry)
            print(json.dumps(dict(event="candidate",**{k:entry.get(k) for k in
                ("candidate","jacobian","accepted","step_relative","cumulative_relative","rho","termination_reason")})),flush=True)
        def j_event(index,J,r,a,spec):
            np.savez_compressed(run/("logs/jacobian_%03d.npz"%index),J=J,r=r,a=a)
            write_json(run/("logs/jacobian_%03d.json"%index),dict(index=index,**spec))
        def a_event(index,a,data):
            np.savez_compressed(run/("logs/accepted_%03d.npz"%index),
                a=a,r=data["r"],pose_enc=np.asarray(data["pose_enc"],dtype=np.float32))
        result=optimize(initial,J,norm_scale,config["cumulative_budget_relative"],options,
            epsilon_repeat,candidate_forward,jacobian,event,j_event,a_event)
        bank.accept(torch.from_numpy(result["a"]).to("cuda"))
        final_pose,fr,final_depth,final_conf,fcaches=forward(bank.current,dense=True)
        final_data=data_of(final_pose,fr)
        if np.max(np.abs(final_data["r"]-result["data"]["r"]))>1e-6:
            raise RuntimeError("frozen endpoint replay changed")
        propagation={}
        for layer,base in base_cache_cpu.items():
            for half,sl in [("frame",slice(0,1024)),("inter",slice(1024,2048))]:
                propagation[str(layer)+"_"+half+"_patch"]=diff_stats(fcaches[layer][:,:,17:,sl],base[:,:,17:,sl])
        propagation["depth"]=diff_stats(final_depth,base_depth.detach().cpu())
        del fcaches
        out=run/"adapted/post14_D02_registers"
        save_prediction(out/"predictions.npz",final_pose,final_depth,final_conf,hw)
        torch.save({k:v.cpu() for k,v in bank.current.items()},out/"residual.pt")
        summary=dict(status="FROZEN",termination_reason=result["termination_reason"],
            termination_detail=result["termination_detail"],counters=result["counters"],
            accepted_steps=result["counters"]["accepted_steps"],attempts=result["history"],
            baseline_loss=loss(initial["r"]),final_loss=loss(final_data["r"]),
            baseline_focal=baseline_data["diagnostics"],final_focal=final_data["diagnostics"],
            cumulative_relative=result["cumulative_relative"],h0_rms=h0_rms,
            propagation=propagation,geometry_gt_used=False,
            residual_sha256=sha256(out/"residual.pt"),epsilon_repeat=epsilon_repeat,
            final_radius_relative=result["final_radius_relative"])
        write_json(out/"optimization.json",summary)
    p,_,d,_,cache=forward(dense=True)
    cleanup=max(float((p-base_pose).abs().max()),float((d-base_depth).abs().max()))
    if cleanup>1e-6 or hooks.handles: raise RuntimeError("hook cleanup failed")
    freeze_endpoint(run,result["termination_reason"],result["counters"],dict(
        cleanup_max_abs=cleanup,hook_handles_remaining=len(hooks.handles),
        seconds=time.time()-started,max_gpu_memory_bytes=torch.cuda.max_memory_allocated(),
        total_model_forwards=raw_forward_count,code_hashes=code_hashes(root)))
    print(json.dumps(dict(event="endpoint_frozen",termination=result["termination_reason"],
        focal=final_data["diagnostics"],cumulative_relative=result["cumulative_relative"],
        counters=result["counters"])),flush=True)

if __name__=="__main__":
    try: main()
    except BaseException as error:
        print(json.dumps(dict(event="STOP",error_type=type(error).__name__,message=str(error))),flush=True)
        raise
