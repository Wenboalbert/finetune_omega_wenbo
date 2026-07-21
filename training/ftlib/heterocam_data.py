"""Synchronized CCTV-anchor + target-camera packets for Omega finetuning."""
from __future__ import annotations

import itertools
import json
import random
from pathlib import Path

import numpy as np
import torch

from .heterocam_preprocess import load_rgb_and_intrinsics, pad_packet


def read_manifest(path: str) -> list[dict]:
    records = [json.loads(line) for line in Path(path).open(encoding="utf-8") if line.strip()]
    if not records:
        raise ValueError(f"empty manifest: {path}")
    return records


class HeteroCameraPacketDataset(torch.utils.data.Dataset):
    """Enumerate same-time packets with anchors first and selected targets last.

    packets_per_record=0 enumerates every target combination. For the recommended
    targets_per_packet=1 this means one packet per held-out camera per timestamp.
    """

    def __init__(
        self,
        manifest: str,
        anchors: list[str],
        targets_per_packet: int = 1,
        packets_per_record: int = 0,
        image_resolution: int = 512,
        patch_size: int = 16,
        preprocess_mode: str = "balanced",
        seed: int = 42,
    ):
        self.records = read_manifest(manifest)
        self.anchors = tuple(anchors)
        self.image_resolution = image_resolution
        self.patch_size = patch_size
        self.preprocess_mode = preprocess_mode
        self.index: list[tuple[int, tuple[str, ...]]] = []
        rng = random.Random(seed)
        for record_index, record in enumerate(self.records):
            by_name = {view["camera"]: view for view in record["views"]}
            missing = set(self.anchors) - set(by_name)
            if missing:
                raise ValueError(
                    f"{record['scene_id']} frame {record['frame_id']} missing anchors {sorted(missing)}"
                )
            targets = sorted(set(by_name) - set(self.anchors))
            if len(targets) < targets_per_packet:
                raise ValueError(
                    f"{record['scene_id']} frame {record['frame_id']} has too few targets"
                )
            combinations = list(itertools.combinations(targets, targets_per_packet))
            if packets_per_record > 0 and len(combinations) > packets_per_record:
                combinations = rng.sample(combinations, packets_per_record)
            self.index.extend((record_index, combo) for combo in combinations)
        if not self.index:
            raise ValueError("manifest produced no camera packets")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index: int) -> dict:
        record_index, targets = self.index[index]
        record = self.records[record_index]
        by_name = {view["camera"]: view for view in record["views"]}
        names = list(self.anchors) + list(targets)
        views = [by_name[name] for name in names]
        images, intrinsics = [], []
        for view in views:
            image, K = load_rgb_and_intrinsics(
                view["rgb"],
                np.asarray(view["intrinsics"], dtype=np.float64),
                mode=self.preprocess_mode,
                image_resolution=self.image_resolution,
                patch_size=self.patch_size,
            )
            images.append(image)
            intrinsics.append(K)
        image_tensor, K_tensor = pad_packet(images, intrinsics)
        extrinsics = torch.tensor(
            np.asarray([view["extrinsics"] for view in views], dtype=np.float32)
        )
        return {
            "images": image_tensor,
            "extrinsics": extrinsics,
            "intrinsics": K_tensor,
            "anchor_mask": torch.tensor([name in self.anchors for name in names], dtype=torch.bool),
            "camera_names": names,
            "target_names": list(targets),
            "scene_id": record["scene_id"],
            "frame_id": int(record["frame_id"]),
        }


def scene_ids(manifest: str) -> set[str]:
    return {record["scene_id"] for record in read_manifest(manifest)}
