"""Held-out validation = the control loop. Reports BOTH depth (SI AbsRel / delta<1.25 vs the dense GT)
and pose (Sim3-aligned ATE / geodesic RPE-rot vs GT extrinsics). Selection/early-stop key on these,
never on train loss (train loss stays low while test geometry silently degrades).

`traj_metrics` is self-contained here (Umeyama Sim3 for ATE, geodesic angle for RPE-rot) so this file
has no external dependency beyond numpy/torch and the sibling `losses` module."""
import numpy as np, torch

from .losses import _valid


def _centers(extr):
    """Camera centers in world coords from cam-from-world extrinsics (S,3,4) or (S,4,4): C = -R^T t."""
    R, t = extr[:, :3, :3], extr[:, :3, 3]
    return -np.einsum("sij,sj->si", np.transpose(R, (0, 2, 1)), t)


def _umeyama(src, dst):
    """Similarity (scale s, rotation R, translation T) mapping src->dst, least-squares (Umeyama 1991)."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    s0, d0 = src - mu_s, dst - mu_d
    C = d0.T @ s0 / src.shape[0]
    U, D, Vt = np.linalg.svd(C)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    var = (s0 ** 2).sum() / src.shape[0]
    s = (D * np.diag(S)).sum() / max(var, 1e-12)
    T = mu_d - s * R @ mu_s
    return s, R, T


def _geodesic_deg(Ra, Rb):
    """Angle (deg) of the relative rotation Ra^T Rb."""
    Rd = np.transpose(Ra, (0, 2, 1)) @ Rb
    tr = np.clip((np.trace(Rd, axis1=1, axis2=2) - 1) / 2, -1.0, 1.0)
    return np.degrees(np.arccos(tr))


def traj_metrics(pred, gt):
    """pred, gt: (S,3,4) or (S,4,4) cam-from-world. Returns {ate_rmse, rpe_rot_deg} or None if degenerate.
    ATE = RMSE of Sim3-aligned camera centers; RPE-rot = mean geodesic angle of per-step relative rotation
    error. Needs >=3 frames for a non-trivial Sim3 fit."""
    pred, gt = np.asarray(pred, np.float64), np.asarray(gt, np.float64)
    if pred.shape[0] < 3:
        return None
    Cp, Cg = _centers(pred), _centers(gt)
    if not (np.isfinite(Cp).all() and np.isfinite(Cg).all()):
        return None
    try:
        s, R, T = _umeyama(Cp, Cg)
    except np.linalg.LinAlgError:
        return None
    Cp_a = (s * (R @ Cp.T).T + T)
    ate = float(np.sqrt(((Cp_a - Cg) ** 2).sum(1).mean()))
    Rp, Rg = pred[:, :3, :3], gt[:, :3, :3]
    rel_p = np.transpose(Rp[:-1], (0, 2, 1)) @ Rp[1:]      # per-step relative rotation
    rel_g = np.transpose(Rg[:-1], (0, 2, 1)) @ Rg[1:]
    rpe = float(_geodesic_deg(rel_p, rel_g).mean()) if len(rel_p) else float("nan")
    return {"ate_rmse": ate, "rpe_rot_deg": rpe}


@torch.no_grad()
def evaluate(m, loader, forward_fn, decode, metric=False):
    ars, d1s, ates, rots = [], [], [], []
    for images, dep, extr, intr in loader:
        images = images.cuda()
        depth, conf, pose_enc = forward_fn(m, images)
        gt = dep.reshape(-1, *dep.shape[-2:]).cuda()
        for i in range(depth.shape[0]):
            p, g = depth[i], gt[i]; k = _valid(g, p)
            if k.sum() < 50:
                continue
            pc = p.clamp_min(1e-3)
            if not metric:                       # scale-invariant: align the medians first
                pc = pc * (g[k].median() / pc[k].median())
            ars.append((((pc[k] - g[k]).abs()) / g[k]).mean().item())
            r = torch.maximum(pc[k] / g[k], g[k] / pc[k])
            d1s.append((r < 1.25).float().mean().item())
        pe = pose_enc if pose_enc.dim() == 3 else pose_enc[None]
        B = pe.shape[0]
        ep, _ = decode(pe, images.shape[-2:])                 # (B,S,3,4)
        ep = ep.float().cpu().numpy()
        eg = extr.numpy()                                     # (B,S,4,4)
        for b in range(B):
            tr = traj_metrics(ep[b], eg[b][:, :3, :])
            if tr:
                ates.append(tr["ate_rmse"]); rots.append(tr["rpe_rot_deg"])
    f = lambda x: float(np.mean(x)) if x else float("nan")
    # Per-clip ATE is HEAVY-TAILED, so report the median beside the mean. The Sim3 fit to a clip's few
    # camera centres is ill-conditioned whenever they are near-collinear -- any camera translating along
    # a straight line -- and the handful of clips that then blow up hijack the mean, which can end up
    # several times the median with a per-clip std larger than itself. `val.ate_stat` decides which of
    # the two gates checkpoint selection; the median is the one to REPORT.
    med = lambda x: float(np.median(x)) if x else float("nan")
    return dict(absrel=f(ars), d1=f(d1s), ate=f(ates), rpe_rot=f(rots),
                ate_med=med(ates), rpe_rot_med=med(rots),
                ate_std=float(np.std(ates)) if ates else float("nan"),
                n_depth=len(ars), n_clip=len(ates))
