"""Generic clip data loader. Reads a JSONL manifest -- one JSON object per line describing a *scene*:

    {"frames": [
        {"rgb": "<path/to/frame.jpg>",        # RGB image (any size)
         "depth": "<path/to/depth.npy>",       # HxW float depth; <=0 or non-finite = INVALID (masked)
         "intrinsics": [[fx,0,cx],[0,fy,cy],[0,0,1]],   # 3x3, in the RGB's native resolution
         "extrinsics": [[...],[...],[...],[...]]},       # 4x4 (or 3x4) cam-from-world, OpenCV
        ...
    ]}

Depth is the sole supervised GT and it may be sparse (0 = invalid). Extrinsics feed the pose loss.
Windows of length `seq_len` are cut from each scene's frame list; validation scenes are carved off
whole (keep any held-out test split in a separate manifest). No dataset-specific logic lives here.

FRAMING (`aspect`). Every frame ends up on a square img_size x img_size canvas, but you choose how:

  square  Resize to img_size x img_size, ignoring the source aspect ratio. K is rescaled
          anisotropically to match, so the pinhole model stays exact -- the camera simply acquires
          non-square pixels. Nothing is lost and nothing is invented, and this is what the released
          checkpoints were trained with. But a 3.3:1 frame really is stretched 3.3x vertically, which
          is a long way from the aspect the backbone was pretrained on.
  pad     Keep the aspect ratio: fit the LONG side to img_size and pad the short side. Padded pixels
          carry depth 0, so they are masked out of the loss automatically. Nothing is distorted; you
          pay with wasted canvas.
  crop    Keep the aspect ratio: fit the SHORT side to img_size and centre-crop the long one. No
          distortion and no padding, but you lose field of view.

K is transformed to stay exact for whichever frame the model actually sees.
"""
import json
import random

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def _place(arr, size, ox, oy, fill):
    """Put `arr` (h,w[,c]) on a size x size canvas at offset (ox, oy). A negative offset crops."""
    h, w = arr.shape[:2]
    out = np.full((size, size) + arr.shape[2:], fill, arr.dtype)
    dx, dy = max(0, ox), max(0, oy)                   # where it lands on the canvas
    sx, sy = max(0, -ox), max(0, -oy)                 # where it is taken from
    cw, ch = min(w - sx, size - dx), min(h - sy, size - dy)
    if cw > 0 and ch > 0:
        out[dy:dy + ch, dx:dx + cw] = arr[sy:sy + ch, sx:sx + cw]
    return out


def _resize_depth(d, h, w):
    """Nearest: sparse depth keeps its exact values, and invalid pixels stay invalid."""
    return F.interpolate(torch.from_numpy(d)[None, None], size=(h, w), mode="nearest")[0, 0].numpy()


class ClipDataset(torch.utils.data.Dataset):
    def __init__(self, clips, img_size=512, patch=16, aspect="square", pad_value=0.0):
        assert aspect in ("square", "pad", "crop"), f"aspect must be square|pad|crop, got {aspect!r}"
        self.clips = clips
        self.S = (img_size // patch) * patch          # the backbone wants a whole number of patches
        self.aspect = aspect
        self.pad_value = pad_value

    def __len__(self):
        return len(self.clips)

    def _frame(self, fr):
        S = self.S
        im0 = Image.open(fr["rgb"]).convert("RGB")
        Wn, Hn = im0.size

        g = np.load(fr["depth"]).astype(np.float32)
        assert g.ndim == 2, f"depth must be (H,W); got {g.shape} from {fr['depth']}"

        K = np.asarray(fr["intrinsics"], np.float64).copy()

        if self.aspect == "square":
            rgb = np.asarray(im0.resize((S, S), Image.BILINEAR), np.float32) / 255.0
            dep = _resize_depth(g, S, S)
            K[0] *= S / Wn
            K[1] *= S / Hn
        else:
            s = S / (max(Wn, Hn) if self.aspect == "pad" else min(Wn, Hn))
            w, h = max(1, round(Wn * s)), max(1, round(Hn * s))
            rgb = np.asarray(im0.resize((w, h), Image.BILINEAR), np.float32) / 255.0
            dep = _resize_depth(g, h, w)
            ox, oy = (S - w) // 2, (S - h) // 2       # >= 0 pads, < 0 crops
            rgb = _place(rgb, S, ox, oy, self.pad_value)
            dep = _place(dep, S, ox, oy, 0.0)         # 0 = invalid, so padding never enters the loss
            K[:2] *= s
            K[0, 2] += ox
            K[1, 2] += oy

        E = np.asarray(fr["extrinsics"], np.float32)
        if E.shape == (3, 4):
            E = np.vstack([E, [0, 0, 0, 1]]).astype(np.float32)

        return (torch.from_numpy(rgb).permute(2, 0, 1).contiguous(),
                torch.from_numpy(dep),
                torch.from_numpy(E),
                torch.from_numpy(K.astype(np.float32)))

    def __getitem__(self, i):
        frames = [self._frame(fr) for fr in self.clips[i]]
        return tuple(torch.stack(x) for x in zip(*frames))


def make_clips(manifest, seq_len, stride, val_frac, seed, max_per_scene=40):
    scenes = [json.loads(l) for l in open(manifest) if l.strip()]
    assert scenes, f"empty manifest: {manifest}"
    random.Random(seed).shuffle(scenes)
    nval = max(1, int(len(scenes) * val_frac))

    def clips_of(sl):
        cs = []
        for sc in sl:
            f = sc["frames"]
            n = 0
            for st in range(0, max(1, len(f) - seq_len + 1), stride):
                w = f[st:st + seq_len]
                if len(w) == seq_len:                 # UNIFORM length only (mixed -> collate crash)
                    cs.append(w)
                    n += 1
                if n >= max_per_scene:
                    break
        return cs

    tr, va = clips_of(scenes[nval:]), clips_of(scenes[:nval])
    # Val clips come out grouped scene-by-scene and the trainer then truncates to `val.max_clips`.
    # Without this shuffle, a max_clips smaller than one scene's worth of windows would silently
    # validate -- and gate the whole run -- on a SINGLE scene.
    random.Random(seed + 1).shuffle(va)
    return tr, va
