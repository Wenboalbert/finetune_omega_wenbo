# Planned experiment index

No model-adaptation experiment is implemented yet.

| Planned ID | Scope | Status |
|---|---|---|
| `v000-original` | Frozen original Omega comparison | Not implemented |
| `v001-focal-only-frame-ffn-lora` | Per-scene Frame-wise FFN LoRA; focal GT only | Not implemented |

Future versions own `src/`, `configs/`, `scripts/`, and `inputs/` schema under their own folder.
Each execution owns `runs/<version>/<unique-run-id>/`, with its logs inside that run.
Do not treat inherited `training/` or old CameraHead checkpoints as either of these experiments.
