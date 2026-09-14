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
- Maximum accepted GN updates: one; finite retries and STOP are valid outcomes.
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
