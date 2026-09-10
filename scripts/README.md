# V001 entrypoints

Current safe entry: `bash scripts/check_environment.sh`.
It prints a CPU-only infrastructure report to stdout; it does not train, infer, allocate GPU resources,
instantiate the model or load checkpoint tensors. Dataset entry checks are not a data-readiness certificate.

Training, evaluation, paired-comparison and PBS entrypoints are not implemented.
When implemented, each execution must allocate a new `runs/<run-id>/` before submission and route
PBS stdout/stderr and orchestration logs into that run's `logs/`, with arm-specific logs below
`baseline/logs/` and `adapted/logs/`. Do not reuse legacy `training/qut/` scripts.
