# V001 input contract (documentation only)

Shared private entry: `/home/n12388815/phd/vggt_omega_project/datasets/`.
Read its README and registry; current source registration has no selected/frozen dataset revision.

Before running, define an immutable scene/frame manifest with stable image IDs, source references/content hashes,
image dimensions and crop/resize transforms, known focal values with units and processed-image conventions,
context membership/order, seed and a disjoint offline-evaluation data reference.
The adaptation loader may read RGB and known focal only; do not expose pose/depth GT to adaptation,
routing, hyperparameter choice or checkpoint selection. Any other information requires an explicit protocol revision.

This folder stores public-safe schemas/docs/examples only. Do not commit actual private manifests or images.
Copy the approved actual manifest and its hash to the unique run and give baseline/adapted the same paired
input/preprocessing/evaluation protocol. Mutable source paths alone cannot establish reproducibility.
No loader, schema validator, split or data selection is implemented by this scaffold.
