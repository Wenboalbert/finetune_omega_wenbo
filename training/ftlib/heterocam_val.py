"""Validation for camera geometry finetuning."""
from __future__ import annotations

from collections import defaultdict

import torch

from .heterocam_metrics import aggregate_metrics, packet_metrics
from .model_wrap import forward_camera


def _move(packet, device):
    moved = dict(packet)
    for key in ("images", "extrinsics", "intrinsics", "anchor_mask"):
        moved[key] = packet[key].to(device, non_blocking=True)
    return moved


@torch.no_grad()
def evaluate_packets(
    model,
    loader,
    decode_camera,
    device,
    amp=torch.bfloat16,
    max_packets=0,
):
    model.eval()
    rows = []
    camera_rows = defaultdict(list)
    for packet_index, cpu_packet in enumerate(loader):
        if max_packets and packet_index >= max_packets:
            break
        packet = _move(cpu_packet, device)
        prediction = forward_camera(model, packet["images"], False, decode_camera, amp)
        pred_E = prediction["extrinsics"].squeeze(0)
        pred_K = prediction["intrinsics"].squeeze(0)
        height_width = tuple(int(value) for value in packet["images"].shape[-2:])
        metrics = packet_metrics(
            pred_E,
            pred_K,
            packet["extrinsics"],
            packet["intrinsics"],
            packet["anchor_mask"],
            height_width,
        )
        metrics.update(
            {
                "scene_id": cpu_packet["scene_id"],
                "frame_id": int(cpu_packet["frame_id"]),
                "target_camera": ",".join(cpu_packet["target_names"]),
            }
        )
        rows.append(metrics)
        for camera in cpu_packet["target_names"]:
            camera_rows[camera].append(metrics)
    return {
        "summary": aggregate_metrics(rows),
        "by_target_camera": {
            camera: aggregate_metrics(values) for camera, values in sorted(camera_rows.items())
        },
        "packets": rows,
    }
