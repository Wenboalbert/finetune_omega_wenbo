#!/usr/bin/env python3
"""Fail-fast checks before allocating an expensive GPU training job."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

from ftlib.heterocam_config import load_config
from ftlib.heterocam_data import HeteroCameraPacketDataset, read_manifest, scene_ids


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--load-model", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    errors = []
    model_config, data_config = config["model"], config["data"]
    for label, path in (
        ("vggt_omega_root", model_config["vggt_omega_root"]),
        ("init_checkpoint", model_config["init_checkpoint"]),
        ("train_manifest", data_config["train_manifest"]),
        ("val_manifest", data_config["val_manifest"]),
    ):
        if not Path(path).exists():
            errors.append(f"missing {label}: {path}")
    if errors:
        raise SystemExit("\n".join(errors))

    overlap = scene_ids(data_config["train_manifest"]) & scene_ids(data_config["val_manifest"])
    if overlap:
        errors.append("train/validation scene leakage: " + ", ".join(sorted(overlap)))
    anchors = set(data_config["anchors"])
    for manifest_name in ("train_manifest", "val_manifest"):
        for record in read_manifest(data_config[manifest_name]):
            names = {view["camera"] for view in record["views"]}
            if anchors - names:
                errors.append(
                    f"{manifest_name} {record['scene_id']}:{record['frame_id']} missing anchors "
                    f"{sorted(anchors - names)}"
                )
                break
    if errors:
        raise SystemExit("\n".join(errors))

    dataset = HeteroCameraPacketDataset(
        data_config["train_manifest"],
        data_config["anchors"],
        int(data_config.get("targets_per_packet", 1)),
        min(int(data_config.get("packets_per_record", 0)) or 1, 1),
        int(data_config.get("image_resolution", 512)),
        int(data_config.get("patch_size", 16)),
        data_config.get("preprocess_mode", "balanced"),
        int(config.get("seed", 42)),
    )
    packet = dataset[0]
    if packet["images"].shape[0] < 5:
        raise SystemExit("recommended packet requires four anchors plus at least one target")
    if not torch.isfinite(packet["images"]).all():
        raise SystemExit("non-finite image tensor")
    if not torch.isfinite(packet["extrinsics"]).all() or not torch.isfinite(packet["intrinsics"]).all():
        raise SystemExit("non-finite camera ground truth")
    omega_root = Path(model_config["vggt_omega_root"]).resolve()
    sys.path.insert(0, str(omega_root))
    os.environ["VGGT_OMEGA_PATH"] = str(omega_root)
    import vggt_omega
    imported = Path(vggt_omega.__file__).resolve()
    if omega_root not in imported.parents:
        raise SystemExit(f"wrong vggt_omega imported: {imported}")
    report = {
        "status": "PASS",
        "vggt_omega_import": str(imported),
        "train_scenes": sorted(scene_ids(data_config["train_manifest"])),
        "val_scenes": sorted(scene_ids(data_config["val_manifest"])),
        "first_packet": {
            "scene": packet["scene_id"],
            "frame": packet["frame_id"],
            "cameras": packet["camera_names"],
            "image_shape": list(packet["images"].shape),
        },
    }
    if args.load_model:
        from ftlib.model_wrap import build_camera_geometry

        built = build_camera_geometry(
            model_config["init_checkpoint"],
            train_mode=model_config.get("train_mode", "camera_only"),
            last_n=int(model_config.get("last_n", 2)),
            head_lr=float(model_config.get("head_lr", 5e-6)),
            backbone_lr_mult=float(model_config.get("backbone_lr_mult", 0.05)),
            device="cpu",
            expected_omega_root=str(omega_root),
        )
        report["trainable_parameters"] = sum(
            parameter.numel() for parameter in built["model"].parameters() if parameter.requires_grad
        )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
