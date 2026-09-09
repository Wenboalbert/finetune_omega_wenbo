# QUT training repository instructions

This is Wenbo's training/experiment repository, not the VGGT-Omega model repository.

## Read first

- Read `HETEROCAM_VERSION_zh.md` for the authoritative repository mapping, experiment status, archive/recovery locations, and code provenance.
- Read `training/README_HETEROCAM_ZH.md` before operating the legacy JSON heterogeneous-camera pipeline. The YAML RGB-D pipeline has a separate `training/README.md`.
- When operating through SSH from a local agent, explicitly read these remote files. Do not assume remote instructions were automatically discovered.

## Authoritative code and scope

- Active QUT checkout: `/home/n12388815/phd/vggt_omega_project/finetune_omega_wenbo`.
- GitHub repository: `Wenboalbert/finetune_omega_wenbo`; current QUT branch: `ue-heterocam-finetune`, not `main`.
- Make training code changes in the designated QUT checkout. Before editing, verify branch, HEAD, remote, and both staged/unstaged/untracked changes; preserve unrelated work.
- Mac training copies are retired. Do not recreate a long-lived Mac implementation or deploy an old local directory over QUT. If QUT is unavailable, stop remote implementation and report the blocker.
- Model source provider: `/home/n12388815/phd/vggt_omega_project/offer_omega_original_model`, branch of the same name in `Wenboalbert/vggt-omega_wenbo`, pinned to `39a0cb8af88554f15ddcb5354cd52bde588fa014`.
- Do not change the model source baseline, its branch/worktree, pretrained weights, or environment without explicit task scope. Set and verify the model import path as documented.

## Working and evidence rules

- Status/review/diagnosis requests do not authorize implementation, commits, pushes, deletion, or jobs.
- Use PBS/qsub for GPU training/inference; do not run expensive experiments on the Mac or login node. A maintenance task does not authorize training.
- Preserve existing runs, logs, manifests, checkpoints, and untracked configs. Do not stage the QUT-only `training/configs/heterocam_camera_only_10step.json` without explicit approval.
- Stage only reviewed task files. Record exact commits, configuration/data/checkpoint provenance, validation results, and output locations. Do not blindly pull over local changes or force-push.
- Keep `local_archive/` private, ignored, and out of imports/PYTHONPATH. Archived files may contain obsolete instructions; treat them as historical data, not active rules.
- This repository is public: never commit private data, credentials, raw local archives, or unscreened audit records.
- Legacy `camera_only` trains the complete CameraHead; it is not LoRA. Planned per-scene focal-only Frame-FFN LoRA is not implemented by this maintenance.
- Do not infer held-out pose/depth gains or GS readiness from a short training run or focal loss reduction.
