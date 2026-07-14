"""Score a checkpoint on a held-out manifest — the frozen model, or one you finetuned.

Same metrics and the same protocol as the training control loop, but run over every scene in the
manifest, so it can be pointed at a test split that training never touched:

  depth   AbsRel and delta<1.25, scale-invariant (per-image median alignment)
  pose    ATE (Sim3-aligned trajectory RMSE) and RPE-rot (geodesic per-step rotation error, degrees)

Both pose metrics are reported as a mean AND a median. Prefer the median: a Sim3 fit to the few camera
centres of one clip is ill-conditioned when they are near-collinear (a camera moving in a straight
line), so a handful of clips blow up and dominate the mean.

  python eval.py --ckpt runs/finetune/best.pt --manifest test.jsonl --out results.json
"""
import argparse
import json
import os

import torch
from torch.utils.data import DataLoader

from ftlib import data as D
from ftlib.model_wrap import build, forward_joint
from ftlib.val import evaluate


def clips_of(manifest, seq_len, max_per_scene):
    """Non-overlapping windows, exactly as training cuts them."""
    out = []
    for line in open(manifest):
        if not line.strip():
            continue
        frames = json.loads(line)["frames"]
        taken = 0
        for start in range(0, max(1, len(frames) - seq_len + 1), seq_len):
            window = frames[start:start + seq_len]
            if len(window) == seq_len:
                out.append(window)
                taken += 1
            if taken >= max_per_scene:
                break
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True, help="a VGGT-Omega checkpoint: the pretrained one, or best.pt")
    ap.add_argument("--manifest", required=True, help="JSONL, same schema as training (see README)")
    ap.add_argument("--out", default="", help="write the metrics here as JSON")
    ap.add_argument("--tag", default="", help="label carried into the JSON")
    ap.add_argument("--img-size", type=int, default=512, help="must match how the model was trained")
    ap.add_argument("--patch", type=int, default=16)
    ap.add_argument("--aspect", choices=("square", "pad", "crop"), default="square",
                    help="must match how the model was trained (see ftlib/data.py)")
    ap.add_argument("--seq-len", type=int, default=4)
    ap.add_argument("--max-per-scene", type=int, default=80)
    ap.add_argument("--batch-size", type=int, default=3)
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()

    assert a.seq_len >= 3, "seq_len must be >= 3 or the trajectory metrics are undefined"
    clips = clips_of(a.manifest, a.seq_len, a.max_per_scene)
    assert clips, f"no clips of length {a.seq_len} in {a.manifest}"
    print(f"[eval] {len(clips)} clips from {a.manifest}", flush=True)

    m, _, _, decode = build(a.ckpt, "heads", 0)          # freeze mode is irrelevant under no_grad
    m.eval()
    loader = DataLoader(D.ClipDataset(clips, a.img_size, a.patch, a.aspect),
                        batch_size=a.batch_size, num_workers=a.workers)
    v = evaluate(m, loader, lambda mm, im: forward_joint(mm, im, False), decode)

    print(f"[eval] {a.tag or os.path.basename(a.ckpt)}\n"
          f"       depth  AbsRel {v['absrel']:.4f}   delta<1.25 {v['d1']:.4f}\n"
          f"       pose   ATE    {v['ate_med']:.4f} (median)  {v['ate']:.4f} (mean)\n"
          f"              RPE-rot {v['rpe_rot_med']:.3f} deg (median)  {v['rpe_rot']:.3f} deg (mean)",
          flush=True)

    if a.out:
        v["tag"], v["ckpt"], v["manifest"] = a.tag, a.ckpt, a.manifest
        d = os.path.dirname(os.path.abspath(a.out))
        os.makedirs(d, exist_ok=True)
        json.dump(v, open(a.out, "w"), indent=2)
        print(f"[eval] wrote {a.out}", flush=True)


if __name__ == "__main__":
    main()
