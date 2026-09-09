# Scene adaptation workspace instructions

This is branch `scene-adaptation` of `Wenboalbert/finetune_omega_wenbo`.
QUT checkout: `/home/n12388815/phd/vggt_omega_project/finetune_omega_wenbo/scene-adaptation`.
It is a linked worktree of the training repository, not the Omega model repository.

## Read first

- Read `README_SCENE_ADAPTATION.md`, `QUT_LAYOUT.md`, and the specific experiment's instructions before changes.
- Explicitly read remote instructions when operating through SSH; local discovery does not load them automatically.
- Check branch, HEAD, remote, and all dirty/untracked files. Preserve other work. QUT is the implementation site;
  do not recreate a persistent Mac training copy or deploy retired local code over QUT.

## Scope and separation

- This branch currently provides infrastructure only. V000/V001 and LoRA are not implemented.
- New experiment code belongs in `experiments/<version>/`; every version owns src, configs, scripts, and input-manifest schema.
- Each run owns `runs/<version>/<unique-run-id>/`, including logs, resolved configuration, inputs snapshot,
  trainable-parameter manifest, checkpoints, predictions, metrics, and provenance. Never overwrite an existing run.
- The inherited `training/` tree is legacy reference, not this branch's training entrypoint. Its legacy PBS scripts are disabled here.
  Do not re-enable, submit, or modify legacy runs via this checkout. Use the sibling legacy branch if explicitly requested.
- Model provider: `/home/n12388815/phd/vggt_omega_project/vggt-omega_wenbo/offer_omega_original_model` at
  `39a0cb8af88554f15ddcb5354cd52bde588fa014`; use the explicit configuration and verify actual import paths.
- Do not modify the provider, pretrained weights, or shared environments without explicit task scope.
- Planned V001 is per-scene RGB input with Frame-wise FFN LoRA and focal-only GT during adaptation;
  pose/depth GT is offline evaluation only. Do not silently inherit the legacy geometry-supervised loss.
- Directory checks and lower focal loss cannot establish relative pose/depth improvement or GS readiness.

## Operations

- The safe current check is `bash scripts/check_environment.sh`; it imports the model class but never instantiates it,
  loads no checkpoint tensors, runs no forward/backward, and submits no job.
- All GPU work uses PBS/qsub within task authorization; none on the Mac or login node.
- Stage reviewed task files only. Public GitHub must not receive raw datasets, private manifests, credentials, weights, runs, or audits.
- Status/review/diagnosis requests do not authorize mutation, commits, pushes, or jobs.
- Record exact commits/hashes and validation limits. Do not force-push or silently merge the legacy branch.
