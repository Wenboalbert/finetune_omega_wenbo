"""Non-invasive wrapper over vggt_omega.VGGTOmega for JOINT depth+pose finetuning with a configurable
freeze level. We never edit vggt_omega; we call its public submodules and save bare VGGTOmega dicts.

freeze modes:
  heads  -> freeze aggregator; train dense_head + camera_head (lowest-risk)
  lastN  -> unfreeze the last `last_n` blocks of EACH aggregator stack (frame / inter-frame / global),
            so last_n=8 unfreezes 16 blocks on the 1B model, not 8. Read the `[wrap] trainable=` line.
  full   -> unfreeze the whole aggregator
Unfrozen backbone params share ONE flat lr = base_lr * backbone_lr_mult. There is no layer-wise decay.

The pretrained VGGT-Ω package must be importable. Either put it on PYTHONPATH, or point the
VGGT_OMEGA_PATH environment variable at your local checkout, e.g.

  export VGGT_OMEGA_PATH=<path/to/vggt-omega>
"""
import os, sys
import torch

_KEEP = ("aggregator", "camera_head", "dense_head", "point_head", "track_head", "text_alignment")


def _add_vggt_omega_to_path():
    p = os.environ.get("VGGT_OMEGA_PATH")
    if p and p not in sys.path:
        sys.path.insert(0, p)


def _clean_sd(sd):
    if isinstance(sd, dict) and not any(str(k).startswith(("aggregator", "net.")) for k in sd):
        for k in ("model", "state_dict", "ema", "module"):
            if k in sd and hasattr(sd[k], "keys"):
                sd = sd[k]; break
    out = {}
    for k, v in sd.items():
        for pref in ("module.", "net.", "model."):
            if k.startswith(pref):
                k = k[len(pref):]
        if k.startswith(_KEEP):
            out[k] = v
    return out


def _last_blocks(agg, n):
    outs = []
    for name in ("frame_blocks", "inter_frame_blocks", "global_blocks"):
        bl = getattr(agg, name, None)
        if bl is not None and len(bl):
            outs += list(bl)[-n:]
    return outs


def build(init_ckpt, freeze="heads", last_n=2, base_lr=5e-6, backbone_lr_mult=0.05, device="cuda"):
    _add_vggt_omega_to_path()
    from vggt_omega.models import VGGTOmega
    from vggt_omega.utils.pose_enc import encoding_to_camera
    m = VGGTOmega()
    sd = _clean_sd(torch.load(init_ckpt, map_location="cpu", weights_only=True))
    miss, unexp = m.load_state_dict(sd, strict=False)
    assert len(miss) < 50, f"init incompatible: {len(miss)} missing, {len(unexp)} unexpected"

    for p in m.parameters():
        p.requires_grad_(False)
    head_params = []
    for h in (m.dense_head, m.camera_head):
        for p in h.parameters():
            p.requires_grad_(True); head_params.append(p)
    bb_params = []
    if freeze == "full":
        for p in m.aggregator.parameters():
            p.requires_grad_(True); bb_params.append(p)
    elif freeze == "lastN":
        for b in _last_blocks(m.aggregator, last_n):
            for p in b.parameters():
                p.requires_grad_(True); bb_params.append(p)
    elif freeze != "heads":
        raise ValueError(f"freeze must be heads|lastN|full, got {freeze!r} "
                         "(a typo used to fall through to heads-only, silently)")
    groups = [{"params": head_params, "lr": base_lr, "name": "heads"}]
    if bb_params:
        groups.append({"params": bb_params, "lr": base_lr * backbone_lr_mult, "name": "backbone"})
    train_bb = freeze in ("full", "lastN")
    ntr = sum(p.numel() for g in groups for p in g["params"])
    print(f"[wrap] freeze={freeze} last_n={last_n} trainable={ntr/1e6:.1f}M "
          f"(heads={sum(p.numel() for p in head_params)/1e6:.1f}M bb={sum(p.numel() for p in bb_params)/1e6:.1f}M) "
          f"train_backbone={train_bb} missing={len(miss)}", flush=True)
    return m.to(device), groups, train_bb, encoding_to_camera


def forward_joint(m, images, train_backbone, amp=torch.bfloat16):
    # amp: bf16 needs Ampere or newer. Pass torch.float16 on older cards, or None to disable.
    """Returns depth (N,H,W), conf (N,H,W), pose_enc (B,S,9). Aggregator under no_grad when frozen."""
    if images.dim() == 4:
        images = images.unsqueeze(0)
    B, S = images.shape[:2]
    with torch.autocast("cuda", dtype=amp):
        if train_backbone:
            tokens, ps = m.aggregator(images)
        else:
            with torch.no_grad():
                tokens, ps = m.aggregator(images)
            tokens = [t.detach() if torch.is_tensor(t) else t for t in tokens]
    with torch.autocast("cuda", enabled=False):
        pose_enc = m.camera_head(tokens, patch_token_start=ps)
        if isinstance(pose_enc, (list, tuple)):
            pose_enc = pose_enc[-1]
        depth, conf = m.dense_head(tokens, images=images, patch_token_start=ps)
    d = depth[..., 0] if depth.dim() == 5 else depth
    cf = conf[..., 0] if conf.dim() == 5 else conf          # same trailing-channel form as depth
    return (d.reshape(B * S, *d.shape[-2:]).float(),
            cf.reshape(B * S, *cf.shape[-2:]).float(),
            pose_enc)


def save_bare(m, path):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    tmp = path + ".tmp"
    torch.save(m.state_dict(), tmp)
    os.replace(tmp, path)
