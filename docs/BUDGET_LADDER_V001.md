# V001 group 2: pre-registered focal-only cumulative budget ladder

Approved by the user on 2026-09-14 after the one-step smoke. This is the SECOND
experiment group inside the same V001 method branch/worktree, not V002.
Original smoke code/configuration and old runs remain historical and unchanged.

## Fixed experiment

Frame 40 is DEVELOPMENT, not held-out evidence. Five RGB views remain in the
canonical CCTV_01..04, Drone_02 order. Only D02 log-fx/log-fy supervise optimization.
External FP32 additive activation residual: post_frame(14), D02's 16 registers,
16,384 variables. All original model parameters are frozen. No LoRA, native
register tuning, extra token, layer scan, new loss, or other supervised camera.

Four independently prepared complete paired runs start from identical zero
residuals and pixels: cumulative relative RMS 0.003, 0.01, 0.02, 0.04.
The ONLY varying configuration field is cumulative_budget_relative.
Never initialize a larger-budget run from a smaller-budget endpoint.
Each run owns its baseline/adapted/comparison/logs/private manifests.
A shared, identical study_manifest.json links all four without a new worktree.

h0 is the original selected activation, captured before residual injection.
For n=16384, scale = sqrt(n)*RMS(h0), fixed for the entire run.
The state a=z-z0 is the CURRENT external activation displacement, not a sum
of past step lengths. norm_relative(a)=||a||_2/scale.
The old standalone absolute 0.01 RMS cap does NOT apply to group 2.
The actual absolute RMS cap is budget_relative*RMS(h0).

The historical frame-40 J extrapolates about 11.27% relative RMS for complete
linear focal correction; 15% mean focal error needs roughly 3.4-3.6%, depending
on the precise target constraint. These are LOCAL LINEAR estimates, NOT a
guarantee that 4% reaches the target or improves nonlinear geometry.

## Convex subproblem and proof boundary

At each accepted state recompute J and solve in FP64:

min_d ||r+Jd||^2
subject to ||d|| <= step_radius*scale
           ||a+d|| <= cumulative_budget*scale.

The objective's factor 1/2, if used to derive KKT, changes only multiplier
conventions; logged predicted and actual loss are BOTH sum(r^2), no 1/2.
KKT stationarity (multipliers in the half-objective convention) is:

(J.T J + (lambda+mu) I)d = -J.T r - mu*a.

lambda and mu are nonnegative multipliers PRODUCED by the step and cumulative
constraints. They are not two externally swept damping hyperparameters.

Projecting d onto span(J.T[:,0], J.T[:,1], a) leaves Jd unchanged and cannot
increase either constraint norm; hence a global optimum exists in dimension
at most three. Any orthogonal component also increases step norm. For a
non-unique least-squares optimum, explicitly choose the smallest feasible step
norm (including necessary null-space movement toward -a).
Otherwise solve the dual active constraints by monotone scalar bisections.
No new numerical dependency is installed; only existing NumPy FP64 is used.

Record primal feasibility, stationarity, complementarity, lambda/mu, active
constraints and subspace rank. KKT suffices for this convex SUBPROBLEM, not for
global optimality of the nonlinear network or correctness of R/t/depth.
Full-space checks reject an inaccurate reduced-space certificate.

Numerical certificate tolerances: primal relative violation <=1e-8;
stationarity relative <=1e-6 or absolute <=1e-12;
complementarity divided by current log-focal loss <=1e-7.
The current-state nonlinear stationarity test uses gradient norm tolerance
1e-10 + 1e-7*||g|| and a boundary proximity of 2e-7 relative. It is logged,
not inferred from failed/outward LM candidates.

## Radius, casting and acceptance (frozen before GPU execution)

- Initial relative radius 0.0005; max 0.0025; min 0.00001.
- <=40 accepted steps; <=40 Jacobians INCLUDING initial preflight J reused as J1.
- <=8 generated candidates per J; <=160 actual candidate forwards per run.
- Finite-difference/repeat/baseline/endpoint forwards are separately counted
  diagnostics, not candidate forwards and not accepted changes.
- Epsilon-repeat is the maximum pairwise focal-loss difference across three
  zero-state repeats, also covering initial grad/no-grad loss disagreement.
- tau=max(1e-7,1e-5*current_loss,10*epsilon_repeat).
- FP64 optimal state is cast to FP32. Recheck BOTH budgets strictly using
  FP64 norms of the actually stored FP32 state and its effective step.
- A rounding-only retreat along the current-to-candidate segment is tried
  deterministically at fractions 1, 1-2^-23,...,1-2^-17. This consumes NO extra
  model forwards, uses no losses and is at most 7.63 ppm of the proposed step.
  Record fraction and actual casting difference. If none is strictly feasible,
  do not forward it; shrink radius once. No hidden ball projection/clipping.
- pred_decrease = ||r||^2 - ||r+J*d_effective||^2.
- actual_decrease = current_loss - actual_candidate_loss.
- rho = actual_decrease / pred_decrease.
- Accept only finite feasible candidates with BOTH decreases >tau and rho>=0.1.
- **0.1<=rho<0.25 is ACCEPTED; halve the NEXT radius.**
- A rejection halves radius ONCE; rho<0.25 is NOT a rejection threshold.
- Grow x2 (capped) after two consecutive accepted steps with rho in [0.75,1.25]
  and >=80% radius utilization; reset streak after growth.
- Rejections, rho outside that interval or utilization<80% reset growth streak.
- Other cases keep radius. Do not double-halve one event.
- Immediately stop once D02 mean absolute fx/fy relative error<=0.15.
  Individual axis errors are logged; there is no hidden per-axis target.

Numerical preflight retains the proven two-direction FD grid and thresholds,
zero no-op and nonzero routing gates. If it fails, no optimization or evaluator
is released. The original baseline/provider/checkpoint/environment are unchanged.

## Termination and evidence

Exact labels:
TARGET_REACHED, NUMERICAL_FLOOR, TR_RADIUS_MIN, SOLVER_FAILURE,
COMPUTE_LIMIT, CONSTRAINED_STATIONARY.

NUMERICAL_FLOOR: effective predicted improvement<=tau without a valid current
stationarity certificate. CONSTRAINED_STATIONARY requires that certificate.
A solver/cast failure or cumulative boundary alone NEVER proves stationarity
or lack of register controllability. COMPUTE_LIMIT names the specific exhausted
counter. TR_RADIUS_MIN means rejection at the minimum radius.
A computationally frozen endpoint with SOLVER_FAILURE is NOT solver success.
Preflight/infrastructure exceptions produce no endpoint freeze marker.

Save each J/r/current state, singular values and left-singular residual-energy
fractions; every candidate's effective step, cumulative displacement, radius,
multipliers, feasibility/KKT, losses, rho, decision and counters; each accepted
state and all-camera pose/focal outputs. Dense prediction and propagation
diagnostics are saved for baseline and frozen endpoint, not each candidate.

All FOUR endpoints and residual/prediction/optimization hashes must be frozen
before ANY group-2 pose/depth GT is read. The PBS driver releases the separate
evaluator only after this barrier; the evaluator independently rechecks it.
No geometry GT is used to pick radius, budget, candidates, iteration or stopping.
Report every pre-registered budget endpoint, including failures. Do not select
the geometrically best iteration or budget as an unbiased result.

Offline geometry protocol is unchanged: proper per-arm CCTV-only Sim(3);
D02 excluded from alignment; one shared arm scale for all Z-depth; UE cm*0.01;
finite-positive GT plus existing preprocessing/padding masks, no confidence
filter. CCTV 655.04 m sentinel semantics remain unverified and are not silently
changed. Focal success does not establish geometry improvement or GS readiness.

## Execution / source of truth

QUT and same-name GitHub branch: v001-focal-only-register-token.
Public source: src/residual_tto/{constrained_gn,trust_policy,study,
ladder_optimizer,ladder_runner}.py; four configs/ladder_frame40_b*.json;
scripts/{prepare_budget_ladder,run_budget_ladder}.py; scripts/v001_budget_ladder.pbs.
Tests include the original smoke regressions, constrained boundary/null-space/
random KKT cases, casting and thresholds, synthetic closed-loop counters/stop,
independent zero starts and the four-endpoint barrier.

1. CPU tests in existing env_finetune with PYTHONPATH=src:<pinned provider>.
2. Commit reviewed code; prepare four runs with scripts/prepare_budget_ladder.py
   --source-root <explicit registered source collection>.
3. Verify manifests, config-only budget difference, identical input arrays,
   clean pinned branch/provider and output ownership.
4. qsub -v V001_STUDY_RUN=<first run>,V001_COMMIT=<reviewed commit>
   -o <first run>/logs/scheduler.log scripts/v001_budget_ladder.pbs
5. Inspect actual PBS exit, per-run numerical_preflight.json,
   logs/ladder_completion.json, optimization.json and comparison/geometry_metrics.json.
   Do not label CPU PASS or job submission as GPU/geometry success.

The driver refuses to resume/overwrite an attempted study. Preserve partial
runs after failures. A revised implementation requires new immutable runs.
Run-specific private data and outputs stay on QUT, out of public GitHub.
Mac report archives follow output/finetune_omega_wenbo/
v001-focal-only-register-token/runs/<EXACT-QUT-RUN-ID>/.
