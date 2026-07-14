"""Depth objective: scale-invariant (per-image median align), confidence-weighted, with multi-scale
gradient matching on the log residual.
GT = the dense metric depth supplied by the data loader (manifest field 'depth')."""
import torch


def _valid(gt, pred):
    return torch.isfinite(gt) & torch.isfinite(pred) & (gt > 1e-3)


def _grad_match(res, m, scales=4):
    loss, mm = res.new_zeros(()), m.float()
    for s in range(scales):
        r = res[::2 ** s, ::2 ** s]; k = mm[::2 ** s, ::2 ** s]
        gx = (r[:, 1:] - r[:, :-1]).abs() * k[:, 1:] * k[:, :-1]
        gy = (r[1:, :] - r[:-1, :]).abs() * k[1:, :] * k[:-1, :]
        denom = (k[:, 1:] * k[:, :-1]).sum() + (k[1:, :] * k[:-1, :]).sum()
        loss = loss + (gx.sum() + gy.sum()) / denom.clamp_min(1.0)
    return loss / scales


def depth_loss(pred, conf, gt, w_grad=0.5, w_conf=0.05, metric=False):
    losses, absrels = [], []
    for i in range(pred.shape[0]):
        p, c, g = pred[i], conf[i].clamp_min(1e-3), gt[i]
        m = _valid(g, p)
        if m.sum() < 50:                 # too few valid pixels to median-align against: the scale
            continue                     # factor would be noise, so skip the image entirely
        pc = p.clamp_min(1e-3)
        if not metric:
            pc = pc * (g[m].median() / pc[m].median()).detach()
        res = torch.log(pc) - torch.log(g.clamp_min(1e-3))
        data = (res.abs()[m] * c[m] - w_conf * torch.log(c[m])).mean()
        losses.append(data + w_grad * _grad_match(res * m.float(), m))
        absrels.append(((pc[m] - g[m]).abs() / g[m]).mean().detach())
    if not losses:
        return None, None
    return torch.stack(losses).mean(), torch.stack(absrels).mean()
