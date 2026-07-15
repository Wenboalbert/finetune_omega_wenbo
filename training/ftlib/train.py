"""Joint depth+pose finetuner with configurable freeze. VAL (depth AbsRel + pose ATE/RPE) is the
control loop: best-by-combined-score + early-stop; decision gate PASSES only if BOTH depth AND pose
improve over the frozen baseline. Saves a bare VGGTOmega best.pt.

  python -m ftlib.train --config configs/finetune.yaml
  python -m ftlib.train --config nrgbd
"""
import argparse, copy, os, json, math, random
from pathlib import Path
import numpy as np
import yaml, torch
from torch.utils.data import DataLoader

from .model_wrap import build, forward_joint, save_bare
from .losses import depth_loss
from .losses_pose import pose_loss, reproj_consistency
from .val import evaluate
from . import data as D


def deep_merge(base, override):
    merged = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_config(config_ref):
    config_path = Path(config_ref)
    if config_path.is_file():
        with open(config_path) as handle:
            return yaml.safe_load(handle), config_path.name

    config_dir = Path(__file__).resolve().parents[1] / "configs"
    with open(config_dir / "base.yaml") as handle:
        base = yaml.safe_load(handle)
    with open(config_dir / "datasets.yaml") as handle:
        datasets = yaml.safe_load(handle)["datasets"]

    aliases = {}
    for name, spec in datasets.items():
        aliases[name] = name
        for alias in spec.get("aliases", []) or []:
            aliases[alias] = name

    if config_ref not in aliases:
        known = ", ".join(sorted(aliases))
        raise FileNotFoundError(f"{config_ref!r} is neither a config file nor a dataset preset ({known})")

    name = aliases[config_ref]
    return deep_merge(base, datasets[name].get("config", {})), name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="config YAML path or dataset preset from configs/datasets.yaml")
    a = ap.parse_args()
    c, config_label = load_config(a.config)
    os.makedirs(c["out"], exist_ok=True)
    torch.manual_seed(c["seed"])
    torch.cuda.manual_seed_all(c["seed"])
    random.seed(c["seed"])
    np.random.seed(c["seed"])
    o, ls, vv = c["optim"], c["loss"], c["val"]

    m, groups, train_bb, decode = build(c["init"], c["model"]["freeze"], c["model"].get("last_n", 2),
                                        o["lr"], o.get("backbone_lr_mult", 0.05))
    m.eval()
    fwd = lambda mm, im: forward_joint(mm, im, train_bb)
    base_lrs = [g["lr"] for g in groups]

    assert c["data"]["seq_len"] >= 3, ("seq_len must be >= 3: the trajectory metrics need 3+ frames. "
                                       "Below that ATE is nan and the gate can never pass, silently.")
    tr, va = D.make_clips(c["data"]["manifest"], c["data"]["seq_len"],
                          c["data"].get("stride", c["data"]["seq_len"]),
                          c["data"]["val_frac"], c["seed"], c["data"].get("max_per_scene", 40))
    assert len(tr) >= o["batch_size"], (
        f"only {len(tr)} train clips for batch_size {o['batch_size']}. The manifest needs enough "
        f"scenes that val_frac={c['data']['val_frac']} still leaves a usable train split.")
    dl = dict(batch_size=o["batch_size"], num_workers=c["data"]["workers"])
    ds = lambda cl: D.ClipDataset(cl, c["data"]["img_size"], c["data"].get("patch", 16),
                                  c["data"].get("aspect", "square"))
    g = torch.Generator().manual_seed(c["seed"])
    trl = DataLoader(ds(tr), shuffle=True, drop_last=True, generator=g, **dl)
    val = DataLoader(ds(va[:vv["max_clips"]]), **dl)
    print(f"[data] train_clips={len(tr)} val_clips={len(va)}", flush=True)

    opt = torch.optim.AdamW([{"params": g["params"], "lr": g["lr"]} for g in groups],
                            weight_decay=o.get("weight_decay", 0.0))
    total, warm = o["steps"], int(o.get("warmup_frac", 0.05) * o["steps"])

    metric = ls.get("metric", False)
    b = evaluate(m, val, fwd, decode, metric)
    print(f"[val] BASELINE absrel={b['absrel']:.4f} d1={b['d1']:.4f} ate={b['ate']:.3f} rpe_rot={b['rpe_rot']:.3f}", flush=True)

    best_score, best = 1e9, None
    bad, it, step = 0, iter(trl), 0
    w_cons = ls.get("w_cons", 0.0)
    Sf = c["data"]["seq_len"]
    while step < total:
        try:
            images, dep, extr, intr = next(it)
        except StopIteration:
            it = iter(trl); images, dep, extr, intr = next(it)
        f = (step / max(1, warm)) if step < warm else 0.5 * (1 + math.cos(math.pi * min(1.0, (step - warm) / max(1, total - warm))))
        for g, bl in zip(opt.param_groups, base_lrs):
            g["lr"] = bl * f
        images = images.cuda()
        depth, conf, pe = fwd(m, images)
        dl_, ar = depth_loss(depth, conf, dep.reshape(-1, *dep.shape[-2:]).cuda(),
                             ls["w_grad"], ls["w_conf"], ls.get("metric", False))
        pe3 = pe if pe.dim() == 3 else pe[None]
        ep, _ = decode(pe3, images.shape[-2:])                # (B,S,3,4) with grad
        pls, rots, cons = [], [], []
        for bi in range(ep.shape[0]):
            pl, rd, _ = pose_loss(ep[bi], extr[bi, :, :3, :].cuda(), ls["w_rot"], ls["w_trans"])
            pls.append(pl); rots.append(rd)
            if w_cons > 0:
                cons.append(reproj_consistency(depth[bi * Sf:(bi + 1) * Sf], dep[bi].cuda(),
                                               intr[bi].cuda(), ep[bi], extr[bi].cuda()))
        if dl_ is None:                      # every image in the batch fell below the valid-depth
            step += 1; continue              # floor (e.g. the model went non-finite) -> skip the step
        pl_ = torch.stack(pls).mean()
        cons_ = torch.stack(cons).mean() if cons else dl_.new_zeros(())
        loss = ls["w_depth"] * dl_ + ls["w_pose"] * pl_ + w_cons * cons_
        if not torch.isfinite(loss):
            print(f"[train] step {step} non-finite loss, skip", flush=True)
            opt.zero_grad(); step += 1; continue
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g["params"]], o.get("grad_clip", 1.0))
        opt.step()
        if step % 20 == 0:
            print(f"[train] step {step} loss {loss.item():.3f} dep {dl_.item():.3f} pose {pl_.item():.3f} "
                  f"(rot {torch.stack(rots).mean():.1f}deg) absrel {ar.item():.4f} lr {opt.param_groups[0]['lr']:.2e}", flush=True)
        if step > 0 and step % vv["every"] == 0:
            v = evaluate(m, val, fwd, decode, metric)
            # combined score (lower better); both-improved gate.
            # ate_stat picks which ATE the gate keys on. The MEAN is outlier-dominated on
            # near-collinear trajectories (see val.py); "median" is the robust choice.
            ak = "ate" if vv.get("ate_stat", "mean") == "mean" else "ate_med"
            score = v["absrel"] / max(b["absrel"], 1e-6) + v[ak] / max(b[ak], 1e-6)
            both = (v["absrel"] < b["absrel"]) and (v[ak] < b[ak]) and (v["d1"] >= b["d1"] - 0.02)
            tag = ""
            if score < best_score - 1e-4 and both:
                best_score, best, bad = score, dict(v), 0
                save_bare(m, os.path.join(c["out"], "best.pt")); tag = " <- best (both improved, saved)"
            else:
                bad += 1
            print(f"[val] step {step} absrel {v['absrel']:.4f} d1 {v['d1']:.4f} ate {v['ate']:.3f} "
                  f"rpe_rot {v['rpe_rot']:.3f} | score {score:.3f} both={both}{tag}", flush=True)
            if bad >= vv["patience"]:
                print("[val] early stop", flush=True); break
        step += 1

    dec = {"config": config_label, "freeze": c["model"]["freeze"],
           "baseline": b, "best": best, "best_score": best_score if best else None,
           "depth_rel_drop": (b["absrel"] - best["absrel"]) / b["absrel"] if best else 0.0,
           "pose_ate_rel_drop": (b["ate"] - best["ate"]) / b["ate"] if best else 0.0,
           "PASSED": bool(best is not None)}
    json.dump(dec, open(os.path.join(c["out"], "decision.json"), "w"), indent=2)
    print("[decision] " + json.dumps(dec), flush=True)


if __name__ == "__main__":
    main()
