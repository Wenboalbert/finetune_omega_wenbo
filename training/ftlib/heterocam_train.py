"""Conservative single-GPU VGGT-Omega camera-geometry finetuning."""
from __future__ import annotations

import json
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .heterocam_config import save_json
from .heterocam_data import HeteroCameraPacketDataset, scene_ids
from .heterocam_loss import camera_geometry_loss
from .heterocam_val import evaluate_packets
from .model_wrap import build_camera_geometry, forward_camera, save_bare


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def amp_dtype(name):
    mapping = {"bfloat16": torch.bfloat16, "float16": torch.float16, "none": None}
    if name not in mapping:
        raise ValueError(f"amp must be one of {sorted(mapping)}, got {name!r}")
    return mapping[name]


def build_dataset(config, manifest):
    data = config["data"]
    return HeteroCameraPacketDataset(
        manifest=manifest,
        anchors=data["anchors"],
        targets_per_packet=int(data.get("targets_per_packet", 1)),
        packets_per_record=int(data.get("packets_per_record", 0)),
        image_resolution=int(data.get("image_resolution", 512)),
        patch_size=int(data.get("patch_size", 16)),
        preprocess_mode=data.get("preprocess_mode", "balanced"),
        seed=int(config.get("seed", 42)),
    )


def build_loader(dataset, config, shuffle):
    workers = int(config["data"].get("num_workers", 2))
    return DataLoader(
        dataset,
        batch_size=None,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
    )


def _move(packet, device):
    moved = dict(packet)
    for key in ("images", "extrinsics", "intrinsics", "anchor_mask"):
        moved[key] = packet[key].to(device, non_blocking=True)
    return moved


def _set_training_modes(model, train_backbone):
    """Train only selected modules; keep a frozen aggregator deterministic."""
    model.train()
    if not train_backbone:
        model.aggregator.eval()
    if getattr(model, "dense_head", None) is not None:
        model.dense_head.eval()
    model.camera_head.train()


def _metric_score(metrics, baseline):
    keys = (
        "cross_relative_rotation_deg",
        "cross_baseline_direction_deg",
        "cross_baseline_mape",
        "target_hfov_mae_deg",
    )
    ratios = [metrics[key] / max(baseline[key], 1e-8) for key in keys]
    return sum(ratios) / len(ratios)


def _passes_gate(metrics, baseline, max_regression_ratio):
    return bool(
        metrics["cross_baseline_mape"] < baseline["cross_baseline_mape"]
        and metrics["cross_baseline_direction_deg"] < baseline["cross_baseline_direction_deg"]
        and metrics["cross_relative_rotation_deg"]
        <= baseline["cross_relative_rotation_deg"] * max_regression_ratio
        and metrics["target_hfov_mae_deg"]
        <= baseline["target_hfov_mae_deg"] * max_regression_ratio
    )


def _save_training_state(path, model, optimizer, scheduler, step, config, baseline, validation):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    full_state = model.state_dict()
    trainable_names = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    trainable_state = {
        name: full_state[name].detach().cpu() for name in sorted(trainable_names)
    }
    torch.save(
        {
            "step": step,
            "format": "omega_trainable_delta_v1",
            "base_checkpoint": config["model"]["init_checkpoint"],
            "trainable_model": trainable_state,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "config": config,
            "baseline": baseline,
            "validation": validation,
        },
        temporary,
    )
    temporary.replace(path)


def _hardlink_checkpoint(source, destination):
    """Atomically point another checkpoint name at the same immutable inode."""
    source, destination = Path(source), Path(destination)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()
    os.link(source, temporary)
    temporary.replace(destination)


def train(config, resume_path=None):
    seed_everything(int(config.get("seed", 42)))
    device = torch.device(config.get("device", "cuda"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; run through a GPU PBS job")

    train_manifest = config["data"]["train_manifest"]
    val_manifest = config["data"]["val_manifest"]
    overlap = scene_ids(train_manifest) & scene_ids(val_manifest)
    if overlap:
        raise RuntimeError(
            "train/validation scene leakage: " + ", ".join(sorted(overlap))
        )
    train_dataset = build_dataset(config, train_manifest)
    val_dataset = build_dataset(config, val_manifest)
    train_loader = build_loader(train_dataset, config, True)
    val_loader = build_loader(val_dataset, config, False)

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
    model = built["model"]
    decode_camera = built["decode_camera"]
    amp = amp_dtype(config.get("amp", "bfloat16"))

    validation_config = config.get("validation", {})
    max_val_packets = int(validation_config.get("max_packets", 0))
    baseline_result = evaluate_packets(
        model, val_loader, decode_camera, device, amp, max_val_packets
    )
    baseline = baseline_result["summary"]
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    save_json(output / "baseline_validation.json", baseline_result)
    print(f"[baseline] {baseline}", flush=True)

    optimizer_config = config.get("optimizer", {})
    optimizer = torch.optim.AdamW(
        built["param_groups"],
        weight_decay=float(optimizer_config.get("weight_decay", 0.01)),
        betas=tuple(optimizer_config.get("betas", [0.9, 0.95])),
    )
    total_steps = int(optimizer_config.get("steps", 2000))
    warmup = int(optimizer_config.get("warmup_steps", 100))

    def lr_factor(step):
        if warmup > 0 and step < warmup:
            return max((step + 1) / warmup, 1e-3)
        progress = (step - warmup) / max(total_steps - warmup, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    accumulation = int(optimizer_config.get("gradient_accumulation", 1))
    grad_clip = float(optimizer_config.get("grad_clip", 1.0))
    val_every = int(validation_config.get("every_steps", 100))
    max_regression = float(validation_config.get("max_regression_ratio", 1.05))
    best_score = float("inf")
    best_passed_score = float("inf")
    decision_path = output / "decision.json"
    decision = {"baseline": baseline, "evaluations": []}
    start_step = 0
    if resume_path:
        state = torch.load(resume_path, map_location="cpu", weights_only=False)
        if "trainable_model" in state:
            model.load_state_dict(state["trainable_model"], strict=False)
        else:
            model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_step = int(state["step"])
        if decision_path.is_file():
            decision = json.loads(decision_path.read_text(encoding="utf-8"))
            if "best_candidate" in decision:
                best_score = float(decision["best_candidate"]["score"])
            if "best_passed" in decision:
                best_passed_score = float(decision["best_passed"]["score"])
        print(f"[resume] {resume_path} at step={start_step}", flush=True)

    _set_training_modes(model, built["train_backbone"])
    optimizer.zero_grad(set_to_none=True)
    iterator = iter(train_loader)
    for step in range(start_step + 1, total_steps + 1):
        loss_sum = 0.0
        last_components = None
        for _ in range(accumulation):
            try:
                cpu_packet = next(iterator)
            except StopIteration:
                iterator = iter(train_loader)
                cpu_packet = next(iterator)
            packet = _move(cpu_packet, device)
            prediction = forward_camera(
                model,
                packet["images"],
                built["train_backbone"],
                decode_camera,
                amp,
            )
            pred_E = prediction["extrinsics"].squeeze(0)
            pred_K = prediction["intrinsics"].squeeze(0)
            loss, components = camera_geometry_loss(
                pred_E,
                pred_K,
                packet["extrinsics"],
                packet["intrinsics"],
                packet["anchor_mask"],
                tuple(int(value) for value in packet["images"].shape[-2:]),
                config.get("loss_weights"),
            )
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at step {step}: {components}")
            (loss / accumulation).backward()
            loss_sum += float(loss.detach().cpu()) / accumulation
            last_components = {key: float(value.detach().cpu()) for key, value in components.items()}
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip, error_if_nonfinite=True)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        if step == 1 or step % int(config.get("log_every_steps", 10)) == 0:
            print(
                f"[train] step={step}/{total_steps} loss={loss_sum:.6f} "
                f"lr={optimizer.param_groups[0]['lr']:.3e} components={last_components}",
                flush=True,
            )

        if step % val_every == 0 or step == total_steps:
            result = evaluate_packets(model, val_loader, decode_camera, device, amp, max_val_packets)
            metrics = result["summary"]
            score = _metric_score(metrics, baseline)
            passed = _passes_gate(metrics, baseline, max_regression)
            record = {"step": step, "score": score, "passed": passed, "summary": metrics}
            decision["evaluations"].append(record)
            save_json(output / f"validation_step_{step:06d}.json", result)
            candidate_saved = False
            if score < best_score:
                best_score = score
                save_bare(model, str(output / "best_candidate.pt"))
                candidate_saved = True
                decision["best_candidate"] = record
            if passed and score < best_passed_score:
                best_passed_score = score
                if candidate_saved:
                    _hardlink_checkpoint(
                        output / "best_candidate.pt", output / "best_passed.pt"
                    )
                else:
                    save_bare(model, str(output / "best_passed.pt"))
                decision["best_passed"] = record
            _save_training_state(
                output / "latest_training_state.pt",
                model,
                optimizer,
                scheduler,
                step,
                config,
                baseline,
                result,
            )
            save_json(output / "decision.json", decision)
            print(f"[validation] step={step} score={score:.4f} passed={passed} {metrics}", flush=True)
            _set_training_modes(model, built["train_backbone"])
    return decision
