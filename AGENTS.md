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

- Approved 2026-09-14: external activation residuals, frozen Omega, focal-only damped GN.
- Group 1 remains the historical one-step frame-40 smoke (scripts/v001_smoke.pbs).
- User authorized group 2 on 2026-09-14: same post_frame(14) D02 residual, independent
  0.3/1/2/4 percent cumulative budgets, explicit two-constraint GN, <=40 accepted steps/J.
- Read docs/BUDGET_LADDER_V001.md for frozen group-2 radius, acceptance and stop rules.
- Group 2 uses scripts/v001_budget_ladder.pbs; preflight must PASS before optimization.
  All four endpoints must freeze before any group-2 geometry GT evaluation.
- User approved group 3 on 2026-09-15: remove the 15% target early-stop ONLY.
  Read docs/ZERO_TARGET_V001.md. Threshold is 0.0; all other numerical settings,
  core modules, four budgets and independent zero starts are frozen.
  All four endpoints and group-2 trajectory-prefix audits must pass before geometry.
  Group-3 offline depth excludes user-confirmed source marker 65504 cm; compare
  to corrected group-2 depth. This does not affect optimization.
- V001 is per-scene RGB input with known focal as the only adaptation GT. Pose/depth GT is offline evaluation only.
- Native camera/register parameters remain frozen; no LoRA, no new tokens, no layer scan.
- See docs/IMPLEMENTATION_V001.md for current gates, numerical-policy limits and run entrypoints.
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
- User decision: experiment results, aggregate metrics and run identifiers stay on QUT/Mac.
  Public GitHub is for code and pre-registered protocols only. Result publication requires
  a new explicit authorization; do not infer it from permission to implement or push code.
- `local_archive/` contains private historical evidence, not active instructions or executable defaults.
- Status/review/diagnosis requests do not authorize mutation or jobs. Do not force-push or silently merge legacy code.
- Record exact commits/hashes and limits. Precheck PASS or lower focal loss does not establish pose/depth gains or GS readiness.

## Group 4 authorization (2026-09-15)

User approved docs/EXTENDED_V001.md: independent 4% at 80 steps/Jacobians,
320 candidate forwards; conditional independent 5% only if frozen D02
Emax>0.1%. Target remains 0; all other numerics/backends frozen. Prefix is
DIAGNOSTIC for this group only. All executed endpoints and focal-only branch
must close before offline geometry. Keep group 3 immutable and unevaluated.
Use scripts/prepare_extended.py and scripts/v001_extended.pbs; do not change
historical drivers. Results and run identifiers remain private QUT/Mac.
