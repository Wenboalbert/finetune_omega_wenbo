# V001 numerical smoke status: 2026-09-14

This is a screened operational handoff, not a benchmark or GS-readiness claim.
Training code lives in this QUT worktree and the same-name GitHub branch.
Private inputs, geometry GT, complete metrics, predictions and residuals remain on QUT.

## Completed sequence

| Run under runs/ | Tested commit | PBS job | Outcome |
| --- | --- | --- | --- |
| 20260914T010144Z_frame40_96f12f4e | bcdf98e3d5a6 | 25351724.aqua | Numerical STOP; no GN/evaluation |
| 20260914T012458Z_frame40_e3e2f8ce | e140ba9a3768 | 25351784.aqua | Preflight PASS, one GN accepted; evaluator wrote JSON then failed at final stdout print |
| 20260914T013110Z_frame40_25b9cf20 | f2089e735dd5 | 25351879.aqua | Full preflight -> one GN -> separate evaluation; exit 0 |

The third job ran on one NVIDIA A100-SXM4-80GB and finished in 00:01:01.
Eleven CPU tests passed, including an evaluator CLI regression and no-overwrite check.
Peak torch-allocated GPU memory was 9,734,543,360 bytes; this is not total process memory.
All three run directories are retained; nothing was overwritten.

## What changed between attempts

The first STOP was a random-direction finite-difference signal near FP32 resolution.
The row-space direction and algebraic damping/budget checks already passed.
The second configuration added larger L2 finite-difference radii, retaining the original
radii, seed, error/cosine gates, optimizer caps, damping and one-step policy.
The random direction passed at radii 0.1 and 0.3. No geometry GT evaluation had run
when this numerical setting was revised.

The third attempt only adds the missing evaluator json import and a synthetic
full-CLI regression test. Its resolved optimizer configuration is byte-identical
to the second attempt (SHA256
776364964aaadc642474b660aced9af9ff81dc1c6f72593944acb14217376df9).
No geometry metric was used to retune or select the repeated run.

## Verified mechanics and proof boundary

- Frozen provider remains 39a0cb8; all original parameters and weights unchanged.
- Frame 40, five RGB views, D02-only supervised focal and register residual.
- post_frame(14), 16,384 FP32 variables; no LoRA, layer scan or extra accepted step.
- Both baseline and adapted arms use FP32 with TF32 disabled, not default AMP.
- Provider preprocessing pixel equality, transformed K, exact no-op and real routing pass.
- Hook removal restores baseline pose/depth exactly; no handles remain.
- Layer 4/11 cached patches remain unchanged; layer 17/23 patches and depth change.
- The first candidate met both cumulative budgets and reduced supervised focal loss.
- Geometry evaluation is separate and occurs only after the residual is frozen.
- Each arm fits CCTV-support Sim(3); D02 is excluded from fitting, not unseen in RGB/focal.
- Geometry changes are small and mixed, with no relative-rotation improvement here.
  Do not call this validated geometry improvement, generalization, or GS-ready reconstruction.

## Read on QUT

For the third run, read run_completion_summary.json, numerical_preflight.json,
logs/preflight_completion.json, adapted/post14_D02_registers/optimization.json,
logs/smoke_completion.json, and comparison/geometry_metrics.json.
Input/configuration/provenance files and logs/cpu_tests.txt are in the same run.
run_manifest.json is the immutable preparation snapshot, not final job status.

No further scene/frame/layer scan or multi-step adaptation was launched.
A next accuracy experiment requires its own declared budget, control and evaluation
protocol. Preserve the current paired FP32 baseline, no-GT optimization boundary,
same-scale depth evaluation and failed-run history.
