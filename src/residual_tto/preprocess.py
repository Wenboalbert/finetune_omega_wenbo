"""Auditable counterpart of provider 39a0cb8 load_fn; provider remains unchanged."""
import math
import numpy as np
import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor

def preprocess(paths, hfovs, mode="balanced", resolution=512, patch=16):
    if not paths or len(paths) != len(hfovs) or resolution <= 0 or resolution % patch:
        raise ValueError("invalid image/preprocessing configuration")
    tensors, records = [], []
    for path, hfov in zip(paths, hfovs):
        if not 0 < hfov < 179:
            raise ValueError("HFOV must be in degrees, in (0,179)")
        with Image.open(path) as raw:
            if raw.mode == "RGBA":
                raw = Image.alpha_composite(Image.new("RGBA", raw.size, (255,255,255,255)), raw)
            image = raw.convert("RGB")
        w, h = image.size
        f = (w/2) / math.tan(math.radians(hfov)/2)
        raw_k = np.array([[f,0,w/2],[0,f,h/2],[0,0,1.]], dtype=np.float64)
        left = top = 0
        cw, ch = w, h
        if h/w < .5:
            cw = min(w, max(1, int(round(h/.5))))
            left = (w-cw)//2
        elif h/w > 2:
            ch = min(h, max(1, int(round(w*2))))
            top = (h-ch)//2
        image = image.crop((left, top, left+cw, top+ch))
        ratio = ch/cw
        if mode == "balanced":
            count = (resolution//patch)**2
            wp = np.sqrt(count/ratio)
            hp = count/wp
            rw, rh = max(1,int(np.round(wp)))*patch, max(1,int(np.round(hp)))*patch
        elif mode == "max_size":
            rounded = lambda v: max(patch, int(np.round(v/patch))*patch)
            rh, rw = (resolution, rounded(resolution/ratio)) if ratio >= 1 else (rounded(resolution*ratio), resolution)
        else:
            raise ValueError("unknown resize mode")
        tensors.append(to_tensor(image.resize((rw,rh), Image.Resampling.BICUBIC)))
        sx, sy = rw/cw, rh/ch
        affine = np.array([[sx,0,-sx*left],[0,sy,-sy*top],[0,0,1.]])
        records.append(dict(raw_hw=[h,w], raw_hfov_deg=hfov, raw_K=raw_k.tolist(),
            crop_ltrb=[left,top,left+cw,top+ch], cropped_hw=[ch,cw],
            resize_hw=[rh,rw], sx=sx, sy=sy, raw_to_final=affine.tolist(),
            coordinate_convention="continuous edge-origin; K center=W/2,H/2",
            exif_orientation="not applied, matching pinned provider"))
    height = max(t.shape[1] for t in tensors)
    width = max(t.shape[2] for t in tensors)
    result, masks = [], []
    for tensor, record in zip(tensors, records):
        rh,rw = record["resize_hw"]
        pt,pl = (height-rh)//2,(width-rw)//2
        pb,pr = height-rh-pt,width-rw-pl
        result.append(torch.nn.functional.pad(tensor,(pl,pr,pt,pb),value=1.))
        masks.append(torch.nn.functional.pad(torch.ones(rh,rw,dtype=torch.bool),(pl,pr,pt,pb)))
        affine = np.array(record["raw_to_final"])
        affine[0,2] += pl
        affine[1,2] += pt
        k = affine @ np.array(record["raw_K"])
        record.update(pad_ltrb=[pl,pt,pr,pb], final_hw=[height,width],
            raw_to_final=affine.tolist(), final_K=k.tolist(),
            fx=float(k[0,0]),fy=float(k[1,1]),cx=float(k[0,2]),cy=float(k[1,2]))
        # Algebraic ray check does not depend on depth/pose GT.
        raw_points = np.array([[0., record["raw_hw"][1]/2, record["raw_hw"][1]-1],
                               [0., record["raw_hw"][0]/2, record["raw_hw"][0]-1],[1,1,1]])
        before = np.linalg.solve(np.array(record["raw_K"]),raw_points)
        after = np.linalg.solve(k,affine@raw_points)
        error = float(np.max(np.abs(before-after)))
        if error > 1e-10:
            raise RuntimeError("intrinsics/ray transform mismatch")
        record["ray_max_abs_error"] = error
    return torch.stack(result), torch.stack(masks), records
