"""Look at your data BEFORE you train, and get the config settings it actually needs.

Every dataset arrives with a different resolution, aspect ratio, depth density and amount of camera
motion, and each of those maps onto a specific knob in the config. This script reads a manifest,
reports what the loader will see, and turns it into concrete recommendations.

    python inspect_manifest.py <manifest.jsonl>

It reads a sample of frames, so it is fast even on a large manifest.
"""
import argparse
import json
import random

import numpy as np
from PIL import Image


def centers(E):
    E = np.asarray(E, np.float64)
    if E.shape[-2:] == (3, 4):
        E = np.concatenate([E, np.tile([[0, 0, 0, 1.0]], (len(E), 1, 1))], 1)
    R, t = E[:, :3, :3], E[:, :3, 3]
    return -np.einsum("sij,sj->si", np.transpose(R, (0, 2, 1)), t)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("manifest")
    ap.add_argument("--sample", type=int, default=60, help="frames to open per statistic")
    ap.add_argument("--seq-len", type=int, default=4)
    ap.add_argument("--img-size", type=int, default=512)
    a = ap.parse_args()

    scenes = [json.loads(l) for l in open(a.manifest) if l.strip()]
    assert scenes, "empty manifest"
    nfr = [len(s["frames"]) for s in scenes]
    flat = [fr for s in scenes for fr in s["frames"]]
    rng = random.Random(0)
    probe = rng.sample(flat, min(a.sample, len(flat)))

    print(f"\n=== {a.manifest}")
    print(f"scenes {len(scenes)}   frames {sum(nfr)}   frames/scene "
          f"{min(nfr)}–{max(nfr)} (median {int(np.median(nfr))})")

    # ---------- geometry of the source frames ----------
    sizes, ars = [], []
    for fr in probe:
        with Image.open(fr["rgb"]) as im:
            sizes.append(im.size)
    uniq = sorted({s: sizes.count(s) for s in sizes}.items(), key=lambda kv: -kv[1])
    ars = [w / h for w, h in sizes]
    ar = float(np.median(ars))
    print("\n--- frames")
    for (w, h), c in uniq[:4]:
        print(f"    {w}x{h}   aspect {w/h:.2f}:1   ({c}/{len(sizes)} sampled)")
    if len(uniq) > 4:
        print(f"    ... {len(uniq)-4} more sizes")

    # ---------- intrinsics ----------
    Ks = [np.asarray(fr["intrinsics"], float) for fr in probe]
    fx = np.median([K[0, 0] for K in Ks]); fy = np.median([K[1, 1] for K in Ks])
    cx = np.median([K[0, 2] for K in Ks]); cy = np.median([K[1, 2] for K in Ks])
    W, H = float(np.median([s[0] for s in sizes])), float(np.median([s[1] for s in sizes]))
    fov_x = 2 * np.degrees(np.arctan(W / (2 * fx)))
    fov_y = 2 * np.degrees(np.arctan(H / (2 * fy)))
    print("\n--- intrinsics (as given, in the RGB's own pixels)")
    print(f"    fx {fx:.1f}  fy {fy:.1f}  (fx/fy {fx/fy:.3f})   cx {cx:.1f}  cy {cy:.1f}")
    print(f"    implied FOV  {fov_x:.0f}° x {fov_y:.0f}°")
    off = max(abs(cx / W - 0.5), abs(cy / H - 0.5))
    if off > 0.05:
        print(f"    !! the principal point sits {off*100:.0f}% off centre. That is legal, but it is also "
              f"what a WRONG K looks like -- check it.")
    if abs(fx / fy - 1) > 0.05:
        print(f"    !! fx and fy differ by {abs(fx/fy-1)*100:.0f}%: non-square pixels, or a wrong K.")

    # ---------- depth ----------
    cov, vals = [], []
    for fr in probe[:min(30, len(probe))]:
        d = np.load(fr["depth"])
        v = d[np.isfinite(d) & (d > 0)]
        cov.append(v.size / d.size)
        if v.size:
            vals.append(np.percentile(v, [1, 50, 99]))
    covm = float(np.mean(cov)) * 100
    q = np.median(np.array(vals), 0) if vals else [0, 0, 0]
    sparse = covm < 25
    print("\n--- depth")
    print(f"    valid pixels {covm:.1f}%   range p1 {q[0]:.2f}  median {q[1]:.2f}  p99 {q[2]:.2f}")
    print(f"    -> {'SPARSE' if sparse else 'DENSE'}")

    # ---------- camera motion ----------
    paths, straight = [], []
    for s in scenes:
        f = s["frames"]
        for st in range(0, max(1, len(f) - a.seq_len + 1), a.seq_len):
            w = f[st:st + a.seq_len]
            if len(w) < a.seq_len:
                continue
            C = centers([x["extrinsics"] for x in w])
            step = float(np.linalg.norm(np.diff(C, axis=0), axis=1).sum())
            paths.append(step)
            straight.append(float(np.linalg.norm(C[-1] - C[0]) / max(step, 1e-9)))
    paths = np.array(paths)
    nwin = len(paths)
    near_static = float((paths < 0.05 * np.median(paths)).mean())
    print("\n--- camera motion (per clip, in your GT's world units)")
    print(f"    {nwin} windows of {a.seq_len} frames   path p10 {np.percentile(paths,10):.3f}  "
          f"median {np.median(paths):.3f}  p90 {np.percentile(paths,90):.3f}")
    print(f"    straightness (median) {np.median(straight):.2f}   "
          f"({near_static*100:.0f}% of clips move < 5% of the median)")

    # ---------- what to put in the config ----------
    print("\n" + "=" * 78)
    print("RECOMMENDED CONFIG")
    print("=" * 78)

    if 0.9 <= ar <= 1.12:
        asp, why = "square", f"your frames are already ~1:1 ({ar:.2f}), so `square` distorts nothing"
    else:
        asp, why = ("pad", f"aspect {ar:.2f}:1 — `square` would stretch every frame {max(ar,1/ar):.1f}x. "
                           f"On a 1.78:1 source that costs the FROZEN model 41% of its depth accuracy "
                           f"(AbsRel 0.149 -> 0.088) before any training happens; a wider source can only "
                           f"be worse. `pad` keeps the geometry and masks the padding out of the loss. "
                           f"`square` remains the config default because it is what the published "
                           f"checkpoints used -- it is not the best choice for your data.")
    print(f"\ndata.aspect: {asp}")
    print(f"    {why}")
    print(f"\ndata.img_size: {a.img_size}  (keep it at whatever your pretrained checkpoint used)")
    print(f"\ndata.max_per_scene: {max(40, int(np.median(nfr) // a.seq_len))}")
    print(f"    your scenes hold ~{int(np.median(nfr)//a.seq_len)} windows each; a lower cap silently "
          f"uses only the first few")
    print(f"\ndata.val_frac: 0.15   ->  {max(1, int(len(scenes)*0.15))} val scene(s) of {len(scenes)}")
    if max(1, int(len(scenes) * 0.15)) < 2:
        print("    !! fewer than 2 val scenes. The gate that decides whether to save your checkpoint "
              "would rest on a single scene. Get more scenes, or raise val_frac.")

    if sparse:
        print(f"\nloss.w_grad: 0.0   loss.w_cons: 0.0")
        print(f"    depth is sparse ({covm:.1f}%). Gradient matching needs ADJACENT valid pixels and "
              f"consistency samples a dense grid; at this density both feed the model noise.")
    else:
        print(f"\nloss.w_grad: 0.5   loss.w_cons: 0.0  (1.0 in finetune_cons.yaml if ATE is your priority)")
        print(f"    depth is dense ({covm:.1f}%), so both terms have something to work with.")

    print(f"\nloss.metric: false")
    print(f"    unless you specifically want to teach it absolute scale — the released head is up-to-scale.")

    if np.median(straight) > 0.9:
        print(f"\nval.ate_stat: median")
        print(f"    your trajectories are near-collinear (straightness {np.median(straight):.2f}). A Sim3 "
              f"fit to near-collinear camera centres is ill-conditioned, so a few clips blow up and "
              f"hijack the MEAN ATE. Gate on the median.")

    if near_static > 0.1:
        print(f"\n!! {near_static*100:.0f}% of your clips barely move the camera.")
        print(f"    A static clip carries NO translation information — it hands the camera head an "
              f"arbitrary gradient. Filter them out of the manifest (the loss guards against the "
              f"degenerate case, but it cannot invent a signal that is not there).")

    print(f"\nAnd the one thing this script cannot check for you: whether `intrinsics` is actually YOUR")
    print(f"camera. The loader rescales whatever K you hand it, correctly, for every framing mode — but")
    print(f"it has no way to know if the numbers are right. A wrong K is a silent, systematic error.")
    print()


if __name__ == "__main__":
    main()
