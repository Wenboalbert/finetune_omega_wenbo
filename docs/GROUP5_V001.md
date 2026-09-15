# V001 group 5: cumulative budget expansion only

Authorized 2026-09-15. This short protocol supersedes the unimplemented
6--10-percent/minimum-budget-search draft for group 5 only.

## Frozen setup

Use group-4 source at 5a3f75967dd608046ef75d2afe5552d46feadccf and its
extended_frame40_b050 config. Change only cumulative_budget_relative to
.07, .08, .09, and experiment_group/scope metadata. No optimizer change.
Keep frame40, ordered CCTV1--4/D02 RGB, D02-only focal, post_frame(14) D02
register residual (16384 FP32 variables), model/provider/checkpoint/env,
seed, preprocessing, loss, Jacobian, FP32/FP64, backends, preflight and
finite-difference probes unchanged. Native parameters stay frozen.

Each run starts with a=0. Its cumulative distance is RMS(a)/RMS(h0), where
h0 is the same selected baseline activation and its RMS is fixed.
Budget is an upper bound, neither a single step nor a required displacement.
No warm start, LoRA, new loss, layer scan, SAFE guard or geometry selection.

Keep 80 accepted steps, 80 Jacobians (including reused preflight J1),
320 candidate forwards, 8 generated candidates/J. Other preflight/repeat/
dense forwards retain separate counting. Initial/max/min step radius:
.0005/.0025/.00001. Keep target_mean_focal_relative_error=0.
Keep rho acceptance >=.1 and radius-shrink threshold .25 distinct.
Keep tau=max(1e-7,1e-5*loss,10*epsilon_repeat) and all casting retreats.
Zero target does not guarantee exact zero focal or zero loss.

## Fixed three-member sequence

Prepare all three immutable member manifests before GPU execution;
preparing optional members is not model execution. Then:

1. Execute 7 percent and freeze/audit its normal endpoint.
2. If valid, compute D02 Emax=max(abs(fx/fx_star-1),abs(fy/fy_star-1))
   from the unrounded frozen focal output. Emax<=.001 means near-zero.
3. Near-zero skips every remaining member. Otherwise independently execute
   8 percent, then 9 percent under the identical rule and per-run limits.
4. A valid non-near-zero COMPUTE_LIMIT, NUMERICAL_FLOOR,
   CONSTRAINED_STATIONARY or TR_RADIUS_MIN does NOT block the next budget.
   This does not prove budget is the sole bottleneck or predict improvement.
5. Solver failure, failed integrity/validity, nonfinite forward/endpoint,
   failed initialization or infrastructure failure stops the study.
   No automatic retries, new budgets, compute extensions or relaxed thresholds.
6. Only after all executed endpoints and all skipped members are accounted
   for in a checked immutable closure may offline geometry evaluate executed
   endpoints. Hard failures/incomplete studies have no release.

Near-zero is an outer decision only, never an optimizer early-stop.
No 6 percent, 10 percent, quarter-points or minimum-success-budget search.
At most three new model runs. Do not select an intermediate iteration.

## Reuse, checks and evidence

The old group-4 5-percent run is a read-only input/baseline/initialization
reference, not a lower bracket for a new budget search. Preserve old runs.
All old scripts/configs and numerical core remain byte-identical. Shared
runner/evaluator change only group dispatch before their frozen bodies.
Use scripts/prepare_group5.py and scripts/v001_group5.pbs.

Baseline arrays and initial r/h0 equality remain hard checks. Initial J
difference is diagnostic; no trajectory-prefix test across different budgets.
The unchanged group-4 saved-state audit repeats strict FP32 step/cumulative
checks, acceptance, radius, counters, checksums and residual-state equality.
When comparing its recorded/recomputed summary, ONLY max_step_relative uses
the existing audit close tolerance (rtol=1e-9, atol=1e-12) for CPU/BLAS low bits.
Every other summary field/hash is exact; this is not optimizer budget slack.
The group-4 audit implementation and old records are unchanged.

Each run saves resolved config, inputs, code/provider/weight hashes,
baseline/adapted predictions, residual, accepted states, Jacobians,
candidate logs, original termination and counters. Each executed member
gets a focal-only decision. Later members get immutable skip records if
appropriate; root closure is independently reconstructed before release.
Source preparation stages pose/depth references separately; optimizer and
outer decisions do not read them. Geometry runs in a separate process.

Reuse corrected group-4 evaluator: raw 65504 cm invalid-depth exclusion,
one proper CCTV-only Sim(3) per arm, its common scale for all Z-depth,
fixed masks, same focal/pose/pair/depth metrics. No own-scale fitting.
CCTV is in-sample alignment support; D02 is adapted, not unseen validation.
Lower focal does not prove geometry improvement or GS readiness.

Code/protocol only on public GitHub; all results, metrics and run IDs
remain QUT/Mac private. Mac keeps derived reports, never a trainer mirror.
