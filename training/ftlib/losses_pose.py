"""Pose objective for VGGT-Omega finetuning, and why a naive one collapsed the camera head in our runs.

The model predicts poses FRAME-0-RELATIVE and UP-TO-SCALE, while your GT lives in world units. A naive
translation L1 is therefore dominated by that scale mismatch rather than by the pose error. In our
experiments that was enough to collapse the camera head. What we settled on:
  (1) frame-0-relativize BOTH pred and GT (frame 0 -> identity),
  (2) per-clip SCALE-NORMALIZE camera translations (each by its own mean camera-center norm),
  (3) rotation via GEODESIC angle (not raw quaternion -> no double-cover sign bug),
  (4) optional point-reproj consistency ties depth to pose.
Depth and pose thus live in one clip-consistent scale.
"""
import torch


def _to44(E):                                              # (S,3,4)|(S,4,4) -> (S,4,4)
    if E.shape[-2:] == (4, 4):
        return E
    S = E.shape[0]
    bottom = torch.tensor([0, 0, 0, 1.], device=E.device, dtype=E.dtype).view(1, 1, 4).repeat(S, 1, 1)
    return torch.cat([E, bottom], dim=1)


def _rigid_inv(M):                                          # (B,4,4) -> inverse via transpose (never singular)
    R, t = M[:, :3, :3], M[:, :3, 3]
    Ri = R.transpose(-1, -2)
    out = torch.zeros_like(M)
    out[:, :3, :3] = Ri
    out[:, :3, 3] = -torch.einsum("bij,bj->bi", Ri, t)
    out[:, 3, 3] = 1.0
    return out


def _centers(E):                                            # world->cam (S,4,4) -> camera centers (S,3)
    R, t = E[:, :3, :3], E[:, :3, 3]
    return torch.einsum("sij,sj->si", R.transpose(-1, -2), -t)


def _geodesic(Ra, Rb):                                      # (S,3,3) -> (S,) radians
    R = Ra @ Rb.transpose(-1, -2)
    tr = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    return torch.arccos(torch.clamp((tr - 1) * 0.5, -1 + 1e-6, 1 - 1e-6))


def pose_loss(pred_extr, gt_extr, w_rot=1.0, w_trans=1.0):
    """pred_extr, gt_extr: (S,3,4) or (S,4,4) world->cam. Returns (loss, rot_deg, trans_norm)."""
    Pp, Pg = _to44(pred_extr.float()), _to44(gt_extr.float())
    Pp = Pp @ _rigid_inv(Pp[0:1])                           # frame-0 relative (rigid inverse, never singular)
    Pg = Pg @ _rigid_inv(Pg[0:1])
    rot = _geodesic(Pp[:, :3, :3], Pg[:, :3, :3])           # (S,)
    Cp, Cg = _centers(Pp), _centers(Pg)
    sg = Cg.norm(dim=1).mean()
    if sg < 1e-8:
        # The GT camera did not move: its translation DIRECTION is unobservable, and dividing by it
        # would hand the camera head a gradient made of nothing but noise. Rotation still trains.
        # The real fix is upstream -- drop near-static clips when you build the manifest.
        trans = torch.zeros_like(rot)
    else:
        sp = Cp.norm(dim=1).mean().clamp_min(1e-8)
        trans = ((Cp / sp) - (Cg / sg)).norm(dim=1)         # scale-normalized (S,)
    loss = w_rot * rot.mean() + w_trans * trans.mean()
    return loss, torch.rad2deg(rot.mean()).detach(), trans.mean().detach()


def reproj_consistency(depth_p, depth_g, K, extr_p, extr_g, stride=32):
    """Couple depth<->pose: unproject pred & GT depth (same pixels, same K) into the FRAME-0 camera
    frame using each side's frame-0-relative poses, per-clip scale-normalize, and compare the two point
    clouds pixel-wise (Huber). Pose-TRANSLATION error mislocates pred points vs GT -> gives translation
    a gradient sourced from the (already improving) depth. depth_*: (S,H,W); K,extr_*: (S,3/4,*)."""
    S, H, W = depth_p.shape
    dev = depth_p.device
    ys = torch.arange(0, H, stride, device=dev)
    xs = torch.arange(0, W, stride, device=dev)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    u = gx.reshape(-1).float(); v = gy.reshape(-1).float()

    def cloud(depth, extr):
        Er = _to44(extr.float())
        Er = Er @ _rigid_inv(Er[0:1])                       # frame-0 relative
        outs = []
        for s in range(S):
            d = depth[s][gy, gx].reshape(-1).clamp(1e-3, 1e3)   # bound early garbage depth
            fx, fy = K[s, 0, 0], K[s, 1, 1]
            cx, cy = K[s, 0, 2], K[s, 1, 2]
            cam = torch.stack([(u - cx) / fx * d, (v - cy) / fy * d, d], dim=-1)   # (N,3) cam frame
            R, t = Er[s, :3, :3], Er[s, :3, 3]
            outs.append((cam - t) @ R)                      # -> frame-0 camera frame
        P = torch.nan_to_num(torch.cat(outs, 0), 0.0, 0.0, 0.0)
        return P / P.norm(dim=-1).mean().clamp_min(1e-6)     # per-clip scale-norm
    loss = torch.nn.functional.huber_loss(cloud(depth_p, extr_p), cloud(depth_g, extr_g), delta=1.0)
    return torch.nan_to_num(loss, 0.0, 0.0, 0.0)
