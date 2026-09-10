# QUT repository / branch layout

Current handoff: 2026-09-10. Read this with this checkout's `AGENTS.md` before SSH operations.
Remote instructions must be read explicitly; local discovery does not load SSH-host files.

## Directory and GitHub mapping

All paths below are under `/home/n12388815/phd/vggt_omega_project`.
Repository-named parent folders are grouping directories, not Git checkouts. Run Git in a branch folder.

| QUT path | GitHub repository | Branch / role |
|---|---|---|
| `vggt-omega_wenbo/main` | `Wenboalbert/vggt-omega_wenbo` | `main`: primary checkout, shared Git administration, original weights |
| `vggt-omega_wenbo/offer_omega_original_model` | same model repository | same-name branch: original-model provider worktree |
| `finetune_omega_wenbo/ue-heterocam-finetune` | `Wenboalbert/finetune_omega_wenbo` | same-name branch: legacy CameraHead primary checkout |
| `finetune_omega_wenbo/v001-focal-only-register-token` | same training repository | same-name branch: V001 per-scene register-token scaffold |
| `datasets/` | outside Git checkouts | private shared data registry, manifests and prepared-data entry |

V001 is the renamed `scene-adaptation` worktree/branch, preserving ancestry at
`be46a9cb30b8925f279842430514fd268ee41c18`. It is not a fresh clone from repository `main`.
Both training worktrees share Git objects/history but have independent working files.
Untracked data, runs, logs and archives are NOT shared automatically.
GitHub default branches, the legacy branch name, and both model branches are unchanged.
Both model worktrees remain at `39a0cb8af88554f15ddcb5354cd52bde588fa014`.

## Original model and environment

Provider: `/home/n12388815/phd/vggt_omega_project/vggt-omega_wenbo/offer_omega_original_model`.
Checkpoint: `/home/n12388815/phd/vggt_omega_project/vggt-omega_wenbo/main/checkpoints/vggt_omega_1b_512.pt`.
Expected bytes: `4576706117`.
Historical verified SHA256: `c02da418b18bb01d0392598d3f6147366bcde1bb70fd08a5e3bf7925b0667934`.
The Sep-09 maintenance rehashed the full file; routine prechecks only verify file presence/size.
A listed expected hash is not a new full verification.

V001's safe entry is `bash scripts/check_environment.sh` from its worktree.
It uses `configs/qut_environment.json` and the existing `env_finetune`, clears inherited import overrides,
and verifies the actual package/class belongs to the pinned provider.
No model instantiation, checkpoint tensor loading, forward/backward or job submission occurs.
No register training implementation or GPU compatibility claim is implied by PASS.

Legacy training retains its own `training/qut/` entrypoints and explicit provider import:
```bash
export VGGT_OMEGA_PATH=/home/n12388815/phd/vggt_omega_project/vggt-omega_wenbo/offer_omega_original_model
export PYTHONPATH="$VGGT_OMEGA_PATH:$PWD/training"
```
Run that legacy setup only from its own checkout. V001 does not use `training/` as its implementation.
Existing `env`, `env_finetune` and `env_gs_camera_refinement` are not recreated/upgraded by this rename.
Their editable installation paths were repaired during the earlier Sep-09 hierarchy move.
Environment activation alone may select model `main`; always verify the explicit provider import.

## Methods, runs and data

- Each method gets a parallel branch/worktree with an identical name, created when requested.
  Current first method is `v001-focal-only-register-token`; no V000, FFN-LoRA or placeholder V002 directory is created.
- V001 owns `src/`, `configs/`, `scripts/`, `inputs/` directly, not under `experiments/<version>/`.
  Input folders hold schemas/docs, not private actual data. Cross-method implementation dependencies must be explicit and pinned.
- Each complete comparison owns a unique `runs/<run-id>/` inside its method worktree.
  Keep `baseline/`, `adapted/`, `comparison/` and all logs under that run, together with
  immutable input/preprocessing/provenance/config/parameter manifests. Never overwrite prior runs.
- The frozen baseline calls the same original provider/weights with no adaptation. No second source copy or V000 checkout is needed.
- Shared `datasets/` has `README.md`, `registry.yaml`, `external/`, `manifests/` and `prepared/`.
  Initial registration only: an existing external input collection is referenced, not copied, modified or selected.
  A source path is not a frozen dataset revision. Freeze scene/frame selection, preprocessing and hashes before any run.
- Private registry/data/manifests/prepared data remain on QUT; they are outside these code branches and are not GitHub-backed.
  Public data protocol documentation lives in V001 `inputs/README.md`; actual runs snapshot their private manifests.
- V001 plans RGB + known focal as the only adaptation GT. Pose/depth GT is offline evaluation only.
  Native register updates vs residuals/deep prompts is not yet decided; no training, loss, optimizer or PBS is implemented.
- Legacy `camera_only` trains the complete CameraHead, not LoRA, and its geometry/FOV supervision is not focal-only.
  Its existing runs/logs/manifests/archive and QUT-only 10-step config are preserved.
- In V001, inherited legacy PBS scripts remain disabled. Do not submit them or treat inherited training modules as V001.

## Historical paths and recovery

| Historical root | Current root |
|---|---|
| old `finetune_omega_wenbo/` checkout | `finetune_omega_wenbo/ue-heterocam-finetune/` |
| old `offer_omega_original_model/` | `vggt-omega_wenbo/offer_omega_original_model/` |
| old `vggt-omega_wenbo/` checkout | `vggt-omega_wenbo/main/` |
| `finetune_omega_wenbo/scene-adaptation/` | `finetune_omega_wenbo/v001-focal-only-register-token/` |

These are moves/renames, not duplicate active code copies. No compatibility symlinks are created.
Do not rewrite old run logs/audits: use this table to interpret their historical paths.
Prior private archives remain in the legacy checkout:
`local_archive/mac_retirement_20260909T014805Z/` and
`local_archive/hierarchy_migration_20260909T025134Z/`.
This rename's before/validation records are private in V001 `local_archive/rename_20260910T002513Z/`.
Keep archives out of imports, GitHub and active instructions. The previous layout index is retained in
V001 `docs/LAYOUT_HISTORY.md`; old `README_SCENE_ADAPTATION.md` content is recoverable from Git history/private backup.

Historical GS maintenance (Sep-09): its `qut/construction_cctv_inference.pbs` model path was updated
in `Wenboalbert/vggt_omega_gs_camera_refinement:maintenance/qut-hierarchy-20260909`.
GS remote `main` was not overwritten. This V001 rename changes no GS files/branches.
Use the reviewed GS commit for its own future `EXPECTED_GS_COMMIT`; do not infer renderer compatibility from import checks.

## Operations and future versions

Check status, HEAD, branch, remotes and `git worktree list --porcelain` before edits.
Preserve untracked `training/configs/heterocam_camera_only_10step.json`; do not publish it implicitly.
Commit and push reviewed code/docs from the designated QUT worktree within task authorization.
QUT SSH GitHub access is available; HTTPS origin can remain with a per-command push override:
```bash
git -c remote.origin.pushurl=git@github.com:Wenboalbert/finetune_omega_wenbo.git push origin <explicit-branch>
```
No force pushes or global identity/auth changes. Verify remote refs after pushing.
New methods must start from an explicitly recorded commit and declare inherited code; do not silently reuse a sibling's current files.
Do not create speculative branches or merge method changes into legacy/default branches without a request.
All GPU work requires PBS/qsub and task scope. Infrastructure PASS is not evidence of focal/pose/depth improvement or GS readiness.
