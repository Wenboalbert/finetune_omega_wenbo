# V001 group 3: remove the 15-percent target early-stop

User-approved on 2026-09-15 (Australia/Brisbane). This is a single-variable
extension within V001, not a new model method and not a promise of exact calibration.

## Frozen optimization contract

The only numerical configuration change from each matching group-2 budget is
optimizer.target_mean_focal_relative_error: 0.15 -> 0.0.
The experiment_group label is metadata only. Original group-2 configs/runs remain
unchanged. New configs are zero_target_frame40_b{003,010,020,040}.json.

Keep all of these byte-identical or configuration-identical to the group-2 implementation:
D02-only log-focal loss and Jacobian; post_frame(14) D02 register activations only;
FP32 model/GT handling/residual storage, FP64 subproblem and norms; explicit
two-ball GN; cumulative budgets .003/.01/.02/.04; radius initial .0005,
maximum .0025, minimum .00001; 40 accepted steps and 40 Jacobians including
preflight J1; 8 candidates per J and 160 actual candidate forwards; acceptance
rho>=.1; accept then shrink for .1<=rho<.25; existing growth and casting rules;
tau=max(1e-7,1e-5*loss,10*epsilon_repeat); original frame40 RGB/preprocessing/seed,
model provider and checkpoint. No precision upgrade, tau adjustment, new loss,
camera-token intervention, CCTV supervision, LoRA or layer scan.

zero_target.verify_core compares eight numerical modules against group-2 code.
The old standalone absolute cap is not reintroduced. Neither budget nor compute
allowances may auto-expand in this study.

0.0 removes the practically triggerable positive target early-stop. It does NOT
assert that real-valued GT is exactly representable/reachable. The existing
unrounded mean absolute fx/fy relative-error diagnostic remains authoritative:
only a computed zero may produce TARGET_REACHED at threshold zero. Do not infer
success from a printed 0.000%, small loss, or FP32 log-loss zero. Preserve current
loss/diagnostic precision differences, record them, and retain all six original
termination labels and detailed reasons. Numerical/compute/budget limits are
not focal-alignment success.

## Execution and regression

All four runs start independently from zero residual, never from previous final
states. The new study links the corresponding group-2 runs privately by path and
immutable manifests/hashes. Fresh input arrays/focals must match the reference
study before PBS submission. Only the threshold and group metadata may differ
between corresponding configs.

Before geometry is released, compare each new trajectory with its old prefix:
baseline predictions, preflight J/r/h0, accepted a/r/pose_enc, logged J/r/a,
candidate ordering/acceptance/radius action, and numeric focal/solver fields.
Freeze rtol=1e-6, atol=1e-8 for finite equal-shaped numeric arrays; decisions must
match exactly. Also report whether every compared value was bitwise equal.
Ignore only termination metadata at the old target stop. If old stopping was
not TARGET_REACHED, require the same termination and counters.
The regression audit is read-only and never reads geometry GT.
If it fails, preserve all outputs and stop geometry release; do not loosen the
tolerance, change optimization or choose a different iteration automatically.

All four endpoints must be frozen first. After baseline parity and all four
prefix audits pass, release separate offline geometry evaluation. Geometry GT
must not select budgets, candidates, stopping points or reported endpoints.

## Evaluation-only validity correction

The user confirmed decoded EXR 65504 cm (displayed 655.04 m) is invalid GT.
For group-3 geometry evaluation, exclude EXACT source values before cm-to-m
conversion and apply the same crop/nearest-neighbour resize/padding mask.
Do not use a general distance cutoff or refit any scale. Retain per-arm
CCTV-only proper Sim(3), one arm scale for all depths, and all original pose
metrics. Compare depth to the corrected group-2 evaluation, not the historical
unfiltered table. This post-hoc mask correction cannot affect optimization.
Group-2 default evaluator behavior stays available for historical reproduction.

## Entry points

1. Run CPU tests and verify frozen core/config differences.
2. Commit reviewed code; keep result identifiers/metrics off public GitHub.
3. In existing env_finetune, prepare:
   scripts/prepare_budget_ladder.py --variant group3
     --source-root <same private source collection>
     --reference-study-run <first member of the original group-2 study>.
4. Reuse scripts/v001_budget_ladder.pbs with the NEW first run and tested commit.
   Set a group-3 PBS job name through qsub -N if desired.
5. Verify actual scheduler exit, numerical preflight, all endpoint hashes,
   logs/group2_prefix_audit.json, geometry release and final offline metrics.

All new artifacts belong to new runs in the same V001 worktree. Existing runs,
provider, environment and weights are untouched. Public GitHub stores code and
protocol only; run identifiers and results remain private QUT/Mac.
Any study of larger budgets, more compute or lower numerical thresholds is a
separate subsequent experiment requiring authorization.
