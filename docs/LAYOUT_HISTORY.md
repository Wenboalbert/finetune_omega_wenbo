# Historical layout index (superseded 2026-09-10)

This is historical planning, not current instructions. The former `scene-adaptation` branch is now
`v001-focal-only-register-token`. Method versions are parallel same-name worktrees, not children of `experiments/`.
The first approved method is register-token, not the earlier planned Frame-FFN LoRA.
Neither of the experiments listed below was implemented by the old scaffold.
Use `../README_V001.md` and `../QUT_LAYOUT.md` for current instructions.

## Original planned index from be46a9c

# Planned experiment index

No model-adaptation experiment is implemented yet.

| Planned ID | Scope | Status |
|---|---|---|
| `v000-original` | Frozen original Omega comparison | Not implemented |
| `v001-focal-only-frame-ffn-lora` | Per-scene Frame-wise FFN LoRA; focal GT only | Not implemented |

Future versions own `src/`, `configs/`, `scripts/`, and `inputs/` schema under their own folder.
Each execution owns `runs/<version>/<unique-run-id>/`, with its logs inside that run.
Do not treat inherited `training/` or old CameraHead checkpoints as either of these experiments.
