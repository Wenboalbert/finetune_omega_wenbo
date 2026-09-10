# V001 focal-only register-token workspace

This is branch `v001-focal-only-register-token` of `Wenboalbert/finetune_omega_wenbo`.
QUT checkout: `/home/n12388815/phd/vggt_omega_project/finetune_omega_wenbo/v001-focal-only-register-token`.
This is a training-repository linked worktree, not another Omega model source.

## Read first

- Read `README_V001.md` and `QUT_LAYOUT.md` before changes.
- When operating through SSH, explicitly read remote instructions; local discovery does not load them.
- Check branch, HEAD, remotes and staged/unstaged/untracked changes. Preserve unrelated work.
- QUT is the implementation site. Do not recreate a persistent Mac trainer or deploy retired local code.

## Method and data boundary

- Infrastructure only: register adaptation, loss, optimizer and training PBS are NOT implemented.
- V001 is per-scene RGB input with known focal as the only adaptation GT. Pose/depth GT is offline evaluation only.
- Register-token parameterization is undecided: native register updates, residuals and deep prompts are not interchangeable.
  Do not silently choose one, add LoRA, or claim a trainable-parameter count before an approved implementation.
- New method code belongs directly in `src/`; method configs, scripts and input schemas belong in
  `configs/`, `scripts/` and `inputs/`. Do not add another `experiments/<version>/` layer.
- Other methods get parallel same-name branch/worktree directories, created only when requested.
- Shared data entry: `/home/n12388815/phd/vggt_omega_project/datasets/`. Read its README and registry.
  Source registration is not an immutable dataset selection or permission to modify the external source.
- Do not copy raw data or expose pose/depth GT to adaptation, routing or checkpoint selection.
  Known focal must eventually be expressed for the actual crop/resize convention.

## Runs and model

- Each complete comparison owns a NEW `runs/<unique-run-id>/`: baseline, adapted, comparison and all logs.
  Share a frozen input/evaluation protocol between arms; snapshot provenance and each arm's resolved config/parameters.
  Never overwrite an old run or report unpaired runs as a controlled comparison.
- Baseline calls the original provider without adaptation; no separate V000 checkout or model-source copy is required.
- Provider: `/home/n12388815/phd/vggt_omega_project/vggt-omega_wenbo/offer_omega_original_model`,
  pinned to `39a0cb8af88554f15ddcb5354cd52bde588fa014`.
- Use explicit config and verify actual imports. Do not change provider, weights or shared environments.
- Inherited `training/` is legacy reference only. Its PBS scripts remain disabled on this branch.
  Do not import its geometry-supervised losses as focal-only adaptation or execute legacy runs here.

## Operations and evidence

- Current safe check: `bash scripts/check_environment.sh`. CPU import/path checks only:
  no model instantiation, tensor checkpoint loading, forward/backward, inference or job submission.
- All GPU work requires PBS/qsub and task authorization; none on the Mac or login node.
- Stage only reviewed task files. This repository is public; data, private manifests, weights, runs,
  credentials and unscreened audits must stay out of GitHub.
- `local_archive/` contains private historical evidence, not active instructions or executable defaults.
- Status/review/diagnosis requests do not authorize mutation or jobs. Do not force-push or silently merge legacy code.
- Record exact commits/hashes and limits. Precheck PASS or lower focal loss does not establish pose/depth gains or GS readiness.
