#!/usr/bin/env python3
"""One GPU forward/backward packet; does not update or overwrite a checkpoint."""
import argparse

import torch

from ftlib.heterocam_config import load_config
from ftlib.heterocam_loss import camera_geometry_loss
from ftlib.heterocam_metrics import packet_metrics
from ftlib.heterocam_train import amp_dtype, build_dataset
from ftlib.model_wrap import build_camera_geometry, forward_camera


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", help="defaults to config data.train_manifest")
    args = parser.parse_args()
    config = load_config(args.config)
    if not torch.cuda.is_available():
        raise RuntimeError("GPU smoke test requires a CUDA PBS allocation")
    manifest = args.manifest or config["data"]["train_manifest"]
    packet = build_dataset(config, manifest)[0]
    device = torch.device("cuda")
    model_config = config["model"]
    built = build_camera_geometry(
        model_config["init_checkpoint"],
        train_mode=model_config.get("train_mode", "camera_only"),
        last_n=int(model_config.get("last_n", 2)),
        head_lr=float(model_config.get("head_lr", 5e-6)),
        backbone_lr_mult=float(model_config.get("backbone_lr_mult", 0.05)),
        device=device,
        expected_omega_root=model_config["vggt_omega_root"],
    )
    for key in ("images", "extrinsics", "intrinsics", "anchor_mask"):
        packet[key] = packet[key].to(device)
    prediction = forward_camera(
        built["model"], packet["images"], built["train_backbone"],
        built["decode_camera"], amp_dtype(config.get("amp", "bfloat16"))
    )
    loss, components = camera_geometry_loss(
        prediction["extrinsics"].squeeze(0),
        prediction["intrinsics"].squeeze(0),
        packet["extrinsics"], packet["intrinsics"], packet["anchor_mask"],
        tuple(packet["images"].shape[-2:]), config.get("loss_weights"),
    )
    loss.backward()
    finite_gradients = all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in built["model"].parameters() if parameter.requires_grad
    )
    if not finite_gradients:
        raise RuntimeError("non-finite gradients in smoke test")
    print("SMOKE PASS")
    print("cameras:", packet["camera_names"])
    print("image shape:", tuple(packet["images"].shape))
    print("losses:", {key: float(value.detach().cpu()) for key, value in components.items()})
    print(
        "metrics:",
        packet_metrics(
            prediction["extrinsics"].squeeze(0),
            prediction["intrinsics"].squeeze(0),
            packet["extrinsics"],
            packet["intrinsics"],
            packet["anchor_mask"],
            tuple(packet["images"].shape[-2:]),
        ),
    )


if __name__ == "__main__":
    main()
