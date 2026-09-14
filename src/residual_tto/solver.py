import math
import torch

def focal_xy(pose, hw):
    h, w = hw
    angles = pose[..., 7:9]
    if not torch.isfinite(angles).all() or not ((angles > 0) & (angles < math.pi)).all():
        raise ValueError("invalid FoV; do not clamp an invalid prediction into the loss")
    fy = (h / 2) / torch.tan(angles[..., 0] / 2)
    fx = (w / 2) / torch.tan(angles[..., 1] / 2)
    return torch.stack((fx, fy), dim=-1)

def focal_residual(pose, hw, target_xy, supervised):
    if pose.shape[0] != 1 or target_xy.shape != (pose.shape[1], 2):
        raise ValueError("focal shape mismatch")
    if not torch.isfinite(target_xy).all() or not (target_xy > 0).all():
        raise ValueError("invalid focal GT")
    pred = focal_xy(pose, hw)[0, list(supervised)]
    return (pred.log() - target_xy[list(supervised)].log()).reshape(-1)

def explicit_jacobian(residual, probes, specs):
    ordered = [probes[s.name] for s in specs]
    rows = []
    for i in range(residual.numel()):
        grads = torch.autograd.grad(residual[i], ordered,
            retain_graph=i < residual.numel()-1, create_graph=False, allow_unused=False)
        rows.append(torch.cat([g.reshape(-1) for g in grads]))
    result = torch.stack(rows).detach()
    if not torch.isfinite(result).all():
        raise ValueError("nonfinite Jacobian")
    return result

def damped_dual_gn(jacobian, residual, damping):
    if damping <= 0 or not math.isfinite(float(damping)):
        raise ValueError("positive finite damping required")
    if not torch.isfinite(jacobian).all() or not torch.isfinite(residual).all():
        raise ValueError("nonfinite solve input")
    identity = torch.eye(residual.numel(), device=jacobian.device, dtype=jacobian.dtype)
    return -jacobian.T @ torch.linalg.solve(jacobian @ jacobian.T + damping*identity,
                                            residual.detach())

def solve_direction(jacobian, residual, damping, mode, supervised, bank):
    if mode == "joint":
        return damped_dual_gn(jacobian, residual, damping)
    if mode != "per_view":
        raise ValueError("unknown Jacobian mode")
    groups = bank.column_groups()
    if set(groups) != set(supervised):
        raise ValueError("per_view requires the same editable and supervised view set")
    delta = torch.zeros(jacobian.shape[1], device=jacobian.device, dtype=jacobian.dtype)
    for row, view in enumerate(supervised):
        cols = torch.tensor(groups[view], device=jacobian.device)
        own = jacobian[2*row:2*row+2].index_select(1, cols)
        part = damped_dual_gn(own, residual[2*row:2*row+2], damping)
        delta = delta.index_copy(0, cols, part)
    return delta

def budget_stats(values, base, specs, abs_limit, rel_limit, floor=1e-8):
    rows = []
    for spec in specs:
        for local, view in enumerate(spec.view_ids):
            rms = float(values[spec.name][local].square().mean().sqrt())
            ref = float(base[spec.name][local].square().mean().sqrt())
            rel = rms / max(ref, floor)
            rows.append(dict(spec=spec.name, view=view, rms=rms, base_rms=ref,
                relative_rms=rel, passed=math.isfinite(rms) and rms <= abs_limit and rel <= rel_limit))
    return rows
