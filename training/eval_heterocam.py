#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

import torch

from ftlib.heterocam_config import load_config, save_json
from ftlib.heterocam_train import amp_dtype, build_dataset, build_loader
from ftlib.heterocam_val import evaluate_packets
from ftlib.model_wrap import build_camera_geometry, _clean_sd


def main():
    parser = argparse.ArgumentParser(description="Evaluate a heterocamera Omega checkpoint")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", help="Bare VGGTOmega state dict; default is initial checkpoint")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    device = torch.device(config.get("device", "cuda"))
    built = build_camera_geometry(
        config["model"]["init_checkpoint"],
        train_mode=config["model"].get("train_mode", "camera_only"),
        last_n=int(config["model"].get("last_n", 2)),
        head_lr=float(config["model"].get("head_lr", 5e-6)),
        backbone_lr_mult=float(config["model"].get("backbone_lr_mult", 0.05)),
        device=device,
        expected_omega_root=config["model"]["vggt_omega_root"],
    )
    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        built["model"].load_state_dict(_clean_sd(state), strict=False)
    dataset = build_dataset(config, config["data"]["val_manifest"])
    loader = build_loader(dataset, config, False)
    result = evaluate_packets(
        built["model"],
        loader,
        built["decode_camera"],
        device,
        amp_dtype(config.get("amp", "bfloat16")),
        int(config.get("validation", {}).get("max_packets", 0)),
    )
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    save_json(output / "evaluation.json", result)
    if result["by_target_camera"]:
        fields = ["target_camera"] + sorted(next(iter(result["by_target_camera"].values())))
        with (output / "by_target_camera.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for camera, values in result["by_target_camera"].items():
                writer.writerow({"target_camera": camera, **values})
    print(result["summary"])


if __name__ == "__main__":
    main()
