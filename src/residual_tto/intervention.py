import torch

class ResidualBank:
    def __init__(self, specs, device, dim=1024):
        self.specs = specs
        self.current = {s.name: torch.zeros(len(s.view_ids), len(s.slots()), dim,
                        device=device, dtype=torch.float32) for s in specs}
        self.dim = dim

    def probes(self):
        return {k: torch.zeros_like(v, requires_grad=True) for k, v in self.current.items()}

    def flatten(self, values):
        return torch.cat([values[s.name].reshape(-1) for s in self.specs])

    def unflatten(self, vector):
        result, offset = {}, 0
        for spec in self.specs:
            ref = self.current[spec.name]
            result[spec.name] = vector[offset:offset + ref.numel()].reshape(ref.shape)
            offset += ref.numel()
        if offset != vector.numel():
            raise ValueError("residual dimension mismatch")
        return result

    def effective(self, probes):
        return {k: v + probes[k] for k, v in self.current.items()}

    def accept(self, vector):
        self.current = {k: v.detach().clone() for k, v in self.unflatten(vector).items()}

    def column_groups(self):
        result, offset = {}, 0
        for spec in self.specs:
            width = len(spec.slots()) * self.dim
            for j, view in enumerate(spec.view_ids):
                result.setdefault(view, []).extend(range(offset+j*width, offset+(j+1)*width))
            offset += self.current[spec.name].numel()
        return result

def inject_selected(x, spec, delta, views):
    if x.ndim != 3 or x.shape[0] != views or x.shape[1] < 17:
        raise ValueError("V001 requires B=1 and [S,T,D] at the hook")
    if x.dtype != torch.float32 or delta.dtype != torch.float32 or x.device != delta.device:
        raise ValueError("FP32 injection requires matching FP32 device/dtype")
    expected = (len(spec.view_ids), len(spec.slots()), x.shape[-1])
    if tuple(delta.shape) != expected:
        raise ValueError("active residual shape mismatch")
    # index_add is out-of-place and preserves the probe's autograd connection.
    indices = torch.tensor([v*x.shape[1]+t for v in spec.view_ids for t in spec.slots()],
                           device=x.device)
    addition = torch.zeros_like(x).reshape(-1, x.shape[-1]).index_add(
        0, indices, delta.reshape(-1, x.shape[-1]))
    return (x + addition.reshape_as(x)).contiguous()

class HookManager:
    def __init__(self, model, bank, views):
        self.model, self.bank, self.views = model, bank, views
        self.values = bank.current
        self.handles, self.counts, self.base = [], {}, {}
        self.capture_base = False
        self.audit_enabled = False
        self.routing_audit = []

    def reset(self, values, capture_base=False):
        self.values = values
        self.counts = {s.name: 0 for s in self.bank.specs}
        self.capture_base = capture_base

    def apply(self, x, spec):
        self.counts[spec.name] += 1
        if self.capture_base:
            self.base[spec.name] = torch.stack([
                x[v, list(spec.slots())].detach().clone() for v in spec.view_ids])
        output = inject_selected(x, spec, self.values[spec.name], self.views)
        if self.audit_enabled:
            with torch.no_grad():
                changed = output - x
                mask = torch.zeros(x.shape[:2], dtype=torch.bool, device=x.device)
                for view in spec.view_ids:
                    mask[view, list(spec.slots())] = True
                outside = float(changed[~mask].abs().max())
                selected = float(changed[mask].abs().max())
                self.routing_audit.append(dict(spec=spec.name, outside_max_abs=outside,
                    selected_max_abs=selected, shape=list(x.shape), dtype=str(x.dtype)))
                if outside != 0 or selected == 0:
                    raise RuntimeError("nonzero routing check failed")
        return output

    def assert_hits(self):
        if any(v != 1 for v in self.counts.values()):
            raise RuntimeError("hook hit counts must equal one: " + str(self.counts))

    def __enter__(self):
        try:
            for spec in self.bank.specs:
                spec.validate(self.views, len(self.model.aggregator.frame_blocks))
                block = self.model.aggregator.frame_blocks[spec.layer]
                if spec.site == "pre_frame":
                    def pre(module, args, s=spec):
                        return (self.apply(args[0], s),) + args[1:]
                    handle = block.register_forward_pre_hook(pre)
                else:
                    def post(module, args, output, s=spec):
                        if not isinstance(output, torch.Tensor):
                            raise TypeError("expected Tensor block result")
                        return self.apply(output, s)
                    handle = block.register_forward_hook(post)
                self.handles.append(handle)
        except BaseException:
            self.close()
            raise
        return self

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def __exit__(self, *args):
        self.close()
