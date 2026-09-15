# V001 group 4: extended compute, conditional 5-percent budget

User approved on 2026-09-15 (Australia/Brisbane). This is a NEW protocol
within the same V001 branch. Group 3 remains immutable and unevaluated;
do not resume it, rewrite its FAIL or retrospectively release its geometry.

## Frozen numerical experiment

Use group-3 4-percent config at f502ba4823d1ec774be5a2a1c02060a6890d22a3.
Change only max_accepted_steps 40->80, max_jacobians 40->80 and
max_candidate_forwards 160->320, plus method-group/scope metadata.
Preflight J1 counts toward 80. Per-J generated candidates remain <=8.
Target remains 0.0; it does NOT promise exact calibration.

Keep D02-only log-focal supervision, frame40, ordered five RGB views, seed14,
post_frame(14) D02 registers, 16384 external FP32 activation variables,
frozen native parameters, preprocessing, model/checkpoint/environment,
FP32 model and actual-state storage, FP64 subproblem/norms and all backends.
Initial/max/min radius remain .0005/.0025/.00001.
Keep two-constraint GN, actual effective-step feasibility checks, casting
retreats, rho acceptance >=.1, shrink below .25 (accepted for .1<=rho<.25),
growth/streak rules and tau=max(1e-7,1e-5*loss,10*epsilon_repeat).
Eight numerical core modules and runner's model-execution body are verified
byte-identical. Geometry evaluator formulas remain byte-identical too.
No LoRA, camera-token intervention, extra loss, other supervised views or
additional numerical tuning. Do not repeat .3/1/2-percent arms.

## Pre-registered conditional execution

Prepare two independent zero-start arms, budget .04 and .05, with identical
pixels/focal/h0. Preparing the optional arm does not execute its model.
Never initialize one from the other, and never resume group 3.

1. Execute the 4-percent arm up to the frozen limits and freeze its endpoint.
2. Require preprocessing/numerical preflight, integrity, effective-FP32
   budgets, acceptance and execution audits to pass.
3. Compute unrounded D02 Emax=max(abs(fx/fx_star-1),abs(fy/fy_star-1))
   from the frozen endpoint, against the same focal manifest.
4. If Emax<=.001 (0.1 percent), record SKIPPED_NEAR_ZERO for the optional arm.
   Otherwise execute the independently zero-initialized 5-percent arm with
   the same 80/80/320 limits and controller, and freeze/audit it.
5. Only after this decision is immutable and every executed arm is frozen
   and valid, release separate offline geometry evaluation for all executed
   endpoints. Do not evaluate a skipped arm.

Near-zero is a post-run branch criterion ONLY, not an optimizer early-stop
or a new training target. No gray zone, geometry-based decision, budget >5
or automatic further compute expansion. Numerical/solver/integrity failure
stops the study, not an instruction to expand the budget.
All original termination labels/details remain; boundary contact alone does
not certify stationarity or lack of controllability.

## Reproducibility diagnostic, not geometry gate

Compare NEW 4 percent only against original group-3 4-percent prefix.
Keep rtol=1e-6, atol=1e-8; exact candidate decisions checked separately.
Save all numeric mismatches, missing prefix and decision mismatches rather
than aborting after the first difference. A shortened prefix is INCOMPLETE.
Old 40-step COMPUTE_LIMIT vs continued new run is expected: exclude final
termination/counter equality, not numerical/candidate comparisons.
After decision divergence, equal indices no longer imply identical states.
Do not require 5-percent and 4-percent optimization trajectories to agree;
check common input/baseline/r/h0 and record initial-J differences only.

Prefix PASS/FAIL/INCOMPLETE does not gate group-4 geometry. Missing audit
artifacts/integrity failures still do. Do not reinterpret nondeterminism as
established until diagnosed. No deterministic-mode/backend change this group.
Report optimization validity, prefix reproducibility and geometry completion
as separate statuses, with no generic accuracy PASS.

## Offline geometry and provenance

RGB and D02 focal are the only optimizer supervision. Stage pose/depth
references separately; only evaluator may consume them after release.
Use original per-arm proper CCTV-only Sim(3), D02 excluded from alignment,
one scale per arm for all Z-depth, cm-to-m .01, all original pose/pair metrics.
Exclude exact raw EXR source marker 65504 cm before unit conversion and
nearest-neighbour mask transformation. No confidence filtering or per-depth
scale fitting. Compare corrected depth semantics, never old unfiltered rows.
No geometry-based budget/iteration selection or third-group geometry access.

New runs own baseline/adapted/comparison/logs, immutable study/config/source
identity, validity audits, branch decision, prefix diagnostic and release.
scripts/prepare_extended.py requires explicit source root and private group-3
4-percent reference run. scripts/v001_extended.pbs is the new PBS entry.
Run CPU tests before commit/preparation/submission. Existing group2/group3
drivers, configs, results and semantics remain unchanged.
Only code and this protocol may be pushed to GitHub. Results, aggregate
metrics, run identifiers, private manifests/weights stay on QUT/Mac.
Frame40 remains development data; lower focal is not GS readiness.
