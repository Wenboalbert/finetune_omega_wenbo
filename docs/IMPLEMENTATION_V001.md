# V001 approved protocol and operational gates

This document preserves the GROUP-1 one-step smoke protocol below.
The later user-authorized GROUP-2 protocol is [BUDGET_LADDER_V001.md](BUDGET_LADDER_V001.md).
Group 2 keeps the same activation site/data/GT isolation, but supersedes the
one-step/coarse-damping/absolute-cap rules only for its own new runs.
Never apply new settings retroactively to old runs.

## Frozen task

Frame 40, canonical order CCTV_01, CCTV_02, CCTV_03, CCTV_04, Drone_02.
B=1; the general selectors support S and the smoke asserts S=5.
post_frame(14), editable=supervised=Drone_02, register-only.
Original weights including native register/camera parameters are frozen.
No new tokens, LoRA, layer scan, or additional geometry loss.

The source docs were reviewed on 2026-09-13/14; later user decisions override
their initial six-layer scan and multi-step proposals. Model source facts were
verified on provider 39a0cb8, not newer official main.

## Implementation and entrypoints

- specs.py: layer/view/scope resolution, duplicate/overlap rejection.
- intervention.py: FP32 functional state, out-of-place routing, scoped hooks.
- solver.py: log focal, dynamic Jacobian rows, dual damped solve; joint and per-view directions.
- preprocess.py: exact provider-compatible image transforms and K/ray audit.
- scripts/prepare_frame40.py: CPU-only preparation; requires an explicit source root.
- runner.py: PBS-only FP32 numerical preflight, then at most one accepted update.
- evaluate.py: separate process, reads geometry GT only after smoke completion.
- scripts/v001_smoke.pbs: new guarded GPU entrypoint. Never use inherited legacy PBS.

Run CPU unit tests with the existing environment and PYTHONPATH=src:<provider>.
Prepare inputs using the chosen source collection and configs/smoke_frame40.json.
Preparation creates a unique runs directory and prints its path.
Submit ONLY after CPU tests, source/branch review and preprocessing PASS:
`qsub -v V001_RUN=<prepared-run>,V001_COMMIT=<reviewed-commit> -o <prepared-run>/logs/scheduler.log scripts/v001_smoke.pbs`.
This PBS executes preflight, smoke and evaluator in separate processes, stopping
on any nonzero exit. All GPU forwards/backwards occur inside the PBS allocation.

## Numerical policy

Both paired arms use explicit FP32 aggregator + CameraHead calls, bypassing
the provider forward's internal AMP. TF32 is disabled; no provider file is changed.
No-op, real-hook routing, autograd Jacobian and central finite differences are
checked before smoke. A deterministic row-space direction and a random direction
are checked across the declared epsilon grid. Each must have a passing epsilon.
Raw errors are saved; tolerances are development numerical tolerances, not accuracy claims.

Epsilon is an L2 radius along a unit-length residual direction, not a per-element
amplitude. For 16,384 active elements epsilon=0.3 means perturbation RMS 0.00234375.
These symmetric diagnostic probes are never accepted optimizer updates and do not
change the separate cumulative-update caps. After the first preflight STOP on
2026-09-14, the grid added 0.03, 0.1 and 0.3 to the original 0.001--0.01 values:
the row direction passed but the random Jv norm was about 1.18e-4, leaving output
differences near FP32 resolution at the old radii. No optimizer/evaluator had run.
The relative-error (0.03) and cosine (0.995) gates, direction seed, damping policy
and update budgets are unchanged. Retain the failed run and create a new run.

The initial cumulative safety caps are absolute RMS 0.01 AND relative RMS 0.001,
per site and editable view, referenced to frozen zero-residual activations.
These conservative DEVELOPMENT caps are not optimal hyperparameters.
Preflight selects the first damping on a declared spectrum-relative grid whose
algebraic direction meets both caps and predicts focal decrease, without a
geometry evaluator or geometry GT. It freezes numerical_preflight.json.
If no candidate or finite-difference setting passes, STOP; do not loosen thresholds
after viewing geometry GT. Any revised policy requires a new run and provenance.

Multi-view per_view uses each J_ii to compute directions synchronously, then
evaluates the full candidate using total supervised focal loss. Cross-view
network propagation is never detached. This is not a block-diagonal physical model.
Joint retains all Jacobian blocks. Single-view D02 has identical partition semantics.
One accepted step is the limit; four attempts with increased damping are permitted.
No acceptable update is STOP_NO_ACCEPT, preserving the zero-residual output.

## Data/evaluation isolation

The optimizer reads only focal_manifest.json and rgb_packet.npz. Raw pose CSV and
depth file references live in private_evaluation/manifest.json, not optimizer inputs.
The evaluator is a separate command, only after smoke_completion PASS and frozen
residual checksum verification. Camera/depth GT cannot choose a GN candidate.

HFOV is horizontal degrees; depth is Z-depth cm, converted to metres by 0.01.
These conventions were confirmed by the user; exporter code has not been traced.
UE position conversion retains the audited legacy centimetre/OpenCV convention
from ue_camera.py at legacy commit 9b4bac03a7fefabc776522f4ec38f33f51e9d86f.
EXR channels must be single, identical RGB, or uniquely nonzero; ambiguous channels STOP.
RGB/depth original shapes, source hashes and exact crop/resize/pad must agree.
Valid depth uses fixed finite positive GT and padding exclusion; no confidence-based
selection and no fitted per-image scale. Positive sky/sentinel semantics beyond
this declared mask are not inferred or silently filtered.

Each arm separately fits one proper Sim(3) on the SAME four CCTV support centers;
D02 is excluded. That arm uses the SAME fitted scale for all depth images.
Degenerate supports STOP rather than changing alignment. Report support fit error,
held-out-from-alignment center/rotation, relative pose/baseline and depth coverage.
D02 RGB and focal are adapted: it is not an unseen-view generalization test.

## Artifacts and failures

runs/<id>/ holds immutable input hashes, transforms, config, provider/weight
identity, per-stage source hashes, GPU/job/precision, hooks, J/Gram/finite differences,
damping/budgets/acceptance, residual and paired predictions.
Preflight and smoke completion are separate from optimization acceptance status.
The PBS log records exceptions; no prior file or failed run is overwritten.
Original data, provider, weights and environment remain unchanged.

## Group 3: zero target threshold (2026-09-15)

Authorized single-variable experiment: remove the 15% early-stop by setting
its threshold to 0.0. Keep all other optimization settings and numerical core
unchanged. See [frozen protocol](ZERO_TARGET_V001.md) and actual private run
artifacts for execution status; source implementation is not a GPU result.

## Group 4: extended compute and conditional budget (2026-09-15)

Read [EXTENDED_V001.md](EXTENDED_V001.md) before execution.
Same V001 branch/worktree/provider/environment; new runs only. Independent
4% with 80 accepted steps / 80 Jacobians / 320 candidate forwards, followed
only if necessary by independent 5%. Near-zero D02 Emax<=0.1% is a post-freeze
branch test, not an optimizer stop. Prefix is diagnostic, not a geometry gate;
validity/integrity and all-executed-endpoints barriers remain hard gates.
Group 3 is neither resumed nor evaluated. New preparation/PBS entrypoints:
scripts/prepare_extended.py and scripts/v001_extended.pbs. No public results.

## Group 5: simplified budget expansion (2026-09-15)

User authorized [docs/GROUP5_V001.md](GROUP5_V001.md): independent
7 -> 8 -> 9 percent, stop subsequent arms at frozen D02 Emax<=0.1%.
Valid non-near-zero COMPUTE_LIMIT/NUMERICAL_FLOOR and other normal stops
continue to the next budget; hard failures stop with no geometry release.
All group-4 numerical settings and 80/80/320/8 limits remain unchanged.
No quarter-point refinement, minimum-budget search or warm start.
Use scripts/prepare_group5.py and scripts/v001_group5.pbs. Three fixed
member manifests; all executed endpoints/skip decisions close before GT.
Preserve every prior run and group entrypoint. Public code/protocol only.
