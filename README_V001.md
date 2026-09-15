# V001: focal-only register activation residual

2026-09-14 status: the first complete one-step GPU smoke passed. See
[execution record and limits](docs/SMOKE_STATUS_20260914.md); this is not an accuracy claim.

Approved scope (2026-09-14): freeze all Omega pretrained parameters, optimize only
external additive activation residuals using known focal. This is neither native
register-parameter tuning, CameraHead tuning, nor LoRA.

- QUT worktree/branch: `v001-focal-only-register-token`.
- Repository: `Wenboalbert/finetune_omega_wenbo`.
- Shared provider: `offer_omega_original_model @ 39a0cb8af88554f15ddcb5354cd52bde588fa014`.
- Existing environment: `env_finetune`; no package/environment changes.
- First packet: frame 40, ordered CCTV_01..04 + Drone_02.
- First intervention: post_frame(14), Drone_02 registers, 16,384 FP32 activation variables.
- Group 1: at most one accepted GN update; its runs/configuration are preserved.
- Group 2 is now user-authorized: independent 0.3/1/2/4 percent cumulative budgets,
  explicit two-constraint GN and a four-endpoint GT barrier. Read
  [frozen group-2 protocol](docs/BUDGET_LADDER_V001.md); implementation is not a GPU result.
- Group 3: user-authorized removal of the 15% target early-stop only.
  See [zero-target frozen protocol](docs/ZERO_TARGET_V001.md). It reuses the
  numerical core and adds configuration/provenance/prefix regression checks;
  zero target does not guarantee zero error. No result publication is authorized.
- HFOV degrees and UE Z-depth cm are user-confirmed conventions, not exporter-source verification.

Read [AGENTS.md](AGENTS.md), [QUT_LAYOUT.md](QUT_LAYOUT.md), and
[implementation/gates](docs/IMPLEMENTATION_V001.md) before execution.
Historical layout and rename provenance remain in QUT_LAYOUT.md and private archives.

## Ownership

`src/residual_tto/` holds this version's implementation. `configs/`, `scripts/`,
`inputs/`, `tests/`, `docs/` and `runs/` belong directly to this worktree.
Inherited `training/` is reference only; never launch its disabled PBS here.
Other methods remain independent parallel branch/worktrees. Do not create another
Omega copy or speculative V000/V002 directory.

Each comparison owns a new private `runs/<run-id>/`:
baseline, adapted/post14_D02_registers, comparison, logs, immutable configuration,
focal-only inputs and separate private_evaluation inputs. Never overwrite a run.
Source data, private manifests, predictions, residuals, logs and checkpoints stay
on QUT and must not be staged or pushed to public GitHub.

## Evidence boundary

Code existing or CPU tests passing is not a GPU result. Consult the actual run's
numerical_preflight.json, logs/*_completion.json, optimization.json and
comparison/geometry_metrics.json. A failed gate stops execution. A one-packet smoke
is a numerical/propagation diagnostic, not held-out generalization or GS readiness.

## Group 4: extended compute and conditional budget (2026-09-15)

Read [docs/EXTENDED_V001.md](docs/EXTENDED_V001.md) before execution.
Same V001 branch/worktree/provider/environment; new runs only. Independent
4% with 80 accepted steps / 80 Jacobians / 320 candidate forwards, followed
only if necessary by independent 5%. Near-zero D02 Emax<=0.1% is a post-freeze
branch test, not an optimizer stop. Prefix is diagnostic, not a geometry gate;
validity/integrity and all-executed-endpoints barriers remain hard gates.
Group 3 is neither resumed nor evaluated. New preparation/PBS entrypoints:
scripts/prepare_extended.py and scripts/v001_extended.pbs. No public results.

## Group 5: simplified budget expansion (2026-09-15)

User authorized [docs/GROUP5_V001.md](docs/GROUP5_V001.md): independent
7 -> 8 -> 9 percent, stop subsequent arms at frozen D02 Emax<=0.1%.
Valid non-near-zero COMPUTE_LIMIT/NUMERICAL_FLOOR and other normal stops
continue to the next budget; hard failures stop with no geometry release.
All group-4 numerical settings and 80/80/320/8 limits remain unchanged.
No quarter-point refinement, minimum-budget search or warm start.
Use scripts/prepare_group5.py and scripts/v001_group5.pbs. Three fixed
member manifests; all executed endpoints/skip decisions close before GT.
Preserve every prior run and group entrypoint. Public code/protocol only.
