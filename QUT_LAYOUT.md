# QUT repository / branch layout

Current handoff, 2026-09-09. Read this with the checkout's `AGENTS.md` before SSH operations.
Remote instruction files must be read explicitly; they are not automatically loaded by a local agent.

## Directory mapping

All paths below are under `/home/n12388815/phd/vggt_omega_project`.
The repository-named directories are grouping folders, not Git checkouts; run Git in a branch folder.

| QUT path | GitHub repository | Branch / role |
|---|---|---|
| `vggt-omega_wenbo/main` | `Wenboalbert/vggt-omega_wenbo` | `main`: primary checkout, shared Git administration, original weights |
| `vggt-omega_wenbo/offer_omega_original_model` | same model repository | same-name branch: original model provider, linked worktree |
| `finetune_omega_wenbo/ue-heterocam-finetune` | `Wenboalbert/finetune_omega_wenbo` | same-name branch: legacy training primary checkout |
| `finetune_omega_wenbo/scene-adaptation` | same training repository | same-name branch: independent per-scene experiment worktree |

The new scene branch starts from the migrated legacy branch, not training repository `main`.
The two training worktrees share Git objects/history but have separate working files; editing one does not edit the other.
They do not automatically share untracked manifests, runs, logs, or private archives.
Existing branch names and GitHub default branches were not renamed. Model source remains
`39a0cb8af88554f15ddcb5354cd52bde588fa014` in both model worktrees.

## Explicit model import contract

For either training branch, change into that branch folder, then:

```bash
export VGGT_OMEGA_PATH=/home/n12388815/phd/vggt_omega_project/vggt-omega_wenbo/offer_omega_original_model
export PYTHONPATH="$VGGT_OMEGA_PATH:$PWD/training"
export PYTHONDONTWRITEBYTECODE=1
/home/n12388815/phd/vggt_omega_project/env_finetune/bin/python -c 'import vggt_omega; print(vggt_omega.__file__)'
```

The printed package path must resolve inside the provider worktree, not the model `main` checkout.
Original checkpoint: `/home/n12388815/phd/vggt_omega_project/vggt-omega_wenbo/main/checkpoints/vggt_omega_1b_512.pt`.
Historical verified SHA256: `c02da418b18bb01d0392598d3f6147366bcde1bb70fd08a5e3bf7925b0667934`.
Current verification evidence is in the private migration audit; a listed expected hash is not by itself a new verification.

Existing `env`, `env_finetune`, and `env_gs_camera_refinement` were not recreated/upgraded.
Their editable finder, direct URL, and RECORD metadata were repaired to the relocated model `main`.
Explicit provider import is still required for training; activating an environment alone selects its default main checkout.
Never change global package installation merely to select an experiment model.

## Legacy and new experiment boundaries

- Legacy entrypoints remain `training/qut/{smoke,train,eval}_heterocam.pbs`; these are only for the legacy checkout.
- Legacy `camera_only` trains the whole CameraHead, not LoRA. Its geometry/FOV supervision is not focal-only.
- Legacy `runs/`, `logs/`, `training/manifests/`, and `local_archive/` moved with their original checkout.
  Original result/metadata bytes and historical paths were preserved. Never reuse an existing run directory for a new run.
- Default legacy `heterocam_camera_only.json` train/val manifests remain placeholders if absent. This migration does not choose training data.
- `scene-adaptation` has its own instructions and environment precheck. Inherited legacy training files are reference only;
  do not launch their PBS scripts from the new checkout because they target the legacy checkout.
- Planned new experiment structure: `experiments/<version>/{src,configs,scripts,inputs}/` for versioned code and
  input-manifest schemas; `runs/<version>/<unique-run-id>/` for resolved config, input snapshot, parameter manifest,
  checkpoints, predictions, metrics, and logs. Raw RGB/depth datasets and large weights remain external references.
- V000 frozen baseline and V001 focal-only Frame-wise FFN LoRA are planned, not implemented or trained here.
  Only focal GT is allowed in the planned V001 adaptation; pose/depth GT is offline evaluation only.
- A later promising experiment can be promoted to a dedicated branch/worktree from an exact scene-adaptation commit.
  Promotion is explicit, not an automatic merge or filesystem move.

## Migration / historical-path lookup

| Previous root | Current root |
|---|---|
| `finetune_omega_wenbo/` checkout | `finetune_omega_wenbo/ue-heterocam-finetune/` |
| `offer_omega_original_model/` | `vggt-omega_wenbo/offer_omega_original_model/` |
| `vggt-omega_wenbo/` checkout | `vggt-omega_wenbo/main/` |

These are directory moves, not additional training copies. No compatibility symlinks were created.
Do not bulk-rewrite old logs or audit records using this table.
Mac-retirement recovery archive is now
`/home/n12388815/phd/vggt_omega_project/finetune_omega_wenbo/ue-heterocam-finetune/local_archive/mac_retirement_20260909T014805Z/`.
Retain its hashes, recovery script, original audit, and permissions; never import or execute archived code as current defaults.
Migration audit and pre-change file contents are private under legacy
`local_archive/hierarchy_migration_20260909T025134Z/`; they are not published to this public repository.

## GS dependent maintenance

The GS repository is not moved. Only `qut/construction_cctv_inference.pbs` changes its model path to `vggt-omega_wenbo/main`.
Its independent maintenance commit is published to
`Wenboalbert/vggt_omega_gs_camera_refinement:maintenance/qut-hierarchy-20260909`.
GS remote `main` is not overwritten; local historical divergence is preserved.
When next submitting that GS job, set `EXPECTED_GS_COMMIT` to the reviewed new GS commit, not the pre-maintenance HEAD.
Do not infer GPU/renderer compatibility from a path/import check.

## GitHub and operational gates

Check `git status --short`, branch, HEAD, remotes, and `git worktree list --porcelain` before editing.
Preserve untracked QUT-only `training/configs/heterocam_camera_only_10step.json`; path maintenance did not publish it.
Private data, runs, weights, credentials, and raw audits stay out of public GitHub.
QUT SSH GitHub access is available. An HTTPS origin can be retained with a per-command SSH push override:

```bash
git -c remote.origin.pushurl=git@github.com:Wenboalbert/finetune_omega_wenbo.git push origin <explicit-branch>
```

Commit/push only within task authorization; no force pushes. Verify actual remote refs after pushing.
Git identity was preserved, not globally reconfigured. All GPU work needs PBS/qsub and explicit task scope.
Directory/import/preflight PASS is infrastructure evidence, not a successful forward/backward run or GS-ready geometry.
