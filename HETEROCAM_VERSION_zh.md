# 异构相机微调版本记录与交接

## 当前状态：仓库 / 分支分层 — 2026-09-09

当前路径、导入方式、分支角色、运行隔离与恢复入口以 [QUT_LAYOUT.md](QUT_LAYOUT.md) 为准。
训练分支 `ue-heterocam-finetune` 是旧 CameraHead 实验；新分支 `scene-adaptation` 从迁移后的旧分支建立，
仅提供独立的新实验工作入口，尚未实现 focal-only Frame-FFN LoRA，也没有新训练结果。
模型两个 worktree 的源码均保持 `39a0cb8af88554f15ddcb5354cd52bde588fa014`。

本次只进行目录、引用路径和交接维护：旧结果/日志/manifest/私有归档随原 checkout 移动，
配置的训练超参数未变，QUT-only 10-step 配置只改路径、不提交。
旧审计和运行元数据保持原始字节；其中绝对路径需按 `QUT_LAYOUT.md` 的迁移表解析。
本次前的训练 HEAD 为 `a6e7fb6ae78e9227366f47b171de8ceb503c2ef7`；最新提交以分支 Git 记录为准。

## 历史快照（以下不是当前操作指令）

以下保留迁移前原文，仅供追溯；旧路径、旧的“本次不创建分支 / 不修改 editable 安装”等说法
描述当时维护，不覆盖上面的当前状态。不要直接运行历史命令。

## 当前交接入口：QUT + GitHub 双端代码管理 — 2026-09-09

本文件是本训练仓库的主要交接说明；`AGENTS.md` 保存简短操作规则。
Mac 仅作为远端操作、研究资料和结果查看入口，不再维护本项目的独立训练代码副本。
正式训练代码在指定 QUT checkout 修改、验证，然后按任务授权提交到 GitHub。
GitHub 保存已提交历史；QUT 未提交文件和运行产物不等于已被 GitHub 备份。

### 仓库、分支和源码基线

QUT 共同父目录：`/home/n12388815/phd/vggt_omega_project`。

| QUT 目录 | GitHub 仓库 | 分支 | 作用 |
|---|---|---|---|
| `finetune_omega_wenbo` | `Wenboalbert/finetune_omega_wenbo` | `ue-heterocam-finetune` | 当前训练代码、配置、PBS 入口 |
| `offer_omega_original_model` | `Wenboalbert/vggt-omega_wenbo` | `offer_omega_original_model` | 为训练提供原始 Ω 模型源码的 linked worktree |
| `vggt-omega_wenbo` | `Wenboalbert/vggt-omega_wenbo` | `main` | 模型主 checkout、共享 Git 管理目录、原始权重位置 |

- 整理前训练提交：`95265c6be0f191d4ad56365f4ae3ca7e791b1f5c`；本次只更新交接、规则和归档忽略项。
  之后的文档提交以本分支 `git log` 为准，不把整理前提交冒充最新 HEAD。
- 模型源码基线：`39a0cb8af88554f15ddcb5354cd52bde588fa014`；本次不向模型仓库提交任何修改。
- 训练仓库 `main` 在整理时为 `0f1e288308873fc8df77215dce60cdd3c4749f54`，不是当前 QUT 异构相机分支。
  本次不合并、不更名、不修改默认分支。打开 GitHub 时须选择 `ue-heterocam-finetune`。
- 模型原始权重：`/home/n12388815/phd/vggt_omega_project/vggt-omega_wenbo/checkpoints/vggt_omega_1b_512.pt`。
  它不是训练仓库提供的模型源码，也不是旧实验输出的微调 checkpoint。

### 远端工作入口及导入核验

通过 SSH 工作时，先读取此仓库的 `AGENTS.md`、本文件和适用的训练 README。
本地 agent 不应假定 SSH 远端的规则会被自动加载。

修改前至少核查：

```bash
cd /home/n12388815/phd/vggt_omega_project/finetune_omega_wenbo
git status --short
git branch --show-current
git rev-parse HEAD
git remote -v
```

若分支、路径或已有改动与任务不符，先核实；不要自动覆盖、重置、换分支或从 Mac 旧文件反向同步。
在 QUT 做源码修改和静态测试；正式 GPU 工作必须通过 PBS/qsub，并保留每次运行的配置快照、提交号和输出位置。
完成任务授权的提交后，确认远端分支指向预期提交；不要把提交成功等同于推送成功。

PBS 已使用以下组合，日常手动诊断也需显式设置并核验：

```bash
export VGGT_OMEGA_PATH=/home/n12388815/phd/vggt_omega_project/offer_omega_original_model
export PYTHONPATH="$VGGT_OMEGA_PATH:/home/n12388815/phd/vggt_omega_project/finetune_omega_wenbo/training"
```

环境位置：`/home/n12388815/phd/vggt_omega_project/env_finetune`。
环境原有 editable 安装可能仍默认指向主模型 checkout；仅激活环境并不能保证使用 provider worktree。
在已配置的环境中，用 `importlib.util.find_spec("vggt_omega").origin` 检查顶层包路径，
预期位于 `offer_omega_original_model/vggt_omega/__init__.py`；再按训练前门禁检查实际导入。
不要为整理目录重装环境或修改 editable 安装。

### 已有实验与未实现计划的边界

- YAML RGB-D 流程：`training/README.md`；与 JSON 异构相机流程分开理解。
- JSON 异构相机流程：`training/README_HETEROCAM_ZH.md`、`training/ftlib/model_wrap.py`。
  `camera_only` 解冻整个 CameraHead，`camera_bias` 只解冻 bias，`camera_plus_lastN` 额外解冻尾部块；
  这套旧实验不是 CameraHead LoRA。JSON 第一阶段冻结 dense head，使用相机几何和 FOV 监督，不是 focal-only。
- 已保留 2026-07-28 的 10-step 运行目录和日志；短运行仅能提供有限流程/诊断证据，
  不能作为已达到 GS 精度或泛化收益的证明。根 README 继承的通用 benchmark 表格不代表本次 QUT 实验结果。
- 计划中的逐场景 `RGB → Ω + Frame-wise FFN LoRA → focal/R/t/depth`，训练唯一 GT 为已知 focal，
  以及 V000/V001 独立实验布局，均未由此次维护建立。后续须另行设计、实现、验证，不能把旧实验改名当作新实验。
- 后续不同实验的代码、配置、输入 manifest、输出、日志、可训练参数和 checkpoint 来源应各自可追溯；
  不复用或覆盖旧 run 目录。本次不创建新实验分支或 worktree。

### 数据和非 Git 产物

以下路径相对于 QUT 训练 checkout，除另有注明外不保证已在 GitHub：

| 位置 | 内容及处理 |
|---|---|
| `training/manifests/` | 数据输入清单；JSONL 被忽略，实际源数据位置从 manifest 核查 |
| `runs/` | checkpoint、评估和决策文件；不得覆盖历史结果 |
| `logs/` | PBS/训练日志 |
| `training/configs/heterocam_camera_only_10step.json` | 整理时既存的 QUT-only 未跟踪配置；本次不修改或提交 |
| `local_archive/mac_retirement_20260909T014805Z/` | Mac 退役历史归档、逐文件清单和维护证据；不参与训练或导入 |

原始数据不属于本次迁移；训练配置中的示例路径不等于真实数据可用。
GitHub 忽略规则不提供备份功能：数据、权重和运行结果的额外备份需按存储政策单独安排。

### Mac 副本退役及恢复

退役对象仅为 Mac 原项目工作区下的 `work/vggt_omega_heterocam_finetune` 和
`staging/vggt_omega_heterocam_finetune`。它们不是 Git checkout。
前者源码已与当前训练源码核对，仍含旧路径/说明及缓存；后者是结构不同且不完整的早期实现，
不能把它的 README 所列能力当作已实现功能。两份目录均完整归档，而非直接丢弃差异。

归档根目录：
`/home/n12388815/phd/vggt_omega_project/finetune_omega_wenbo/local_archive/mac_retirement_20260909T014805Z`。

- `work_snapshot.tar.gz`、`staging_snapshot.tar.gz`：两份历史目录。
- `local_manifest.json`：原始相对路径、类型、大小、模式及逐文件 SHA256。
- `verify_restore.py`：检查恢复目录的文件集合、内容、大小和权限模式；不执行历史代码。
- `mac_code_retirement_20260909T014805Z.json`：实际完成状态、比较结果、校验、提交号、删除与恢复证据。
- 同目录保留此前三次训练/模型目录维护的审计 JSON，以及本地规则修改前的文本。
- 删除是否完成以审计 JSON 的状态为准。只有传输、实际解包校验、GitHub 交接确认和依赖检查均通过才可删除。

恢复时先核对审计记录中的压缩包 SHA256，创建全新的空目录，将两个压缩包解包进去，
然后运行 `python3 verify_restore.py local_manifest.json <恢复目录>`。
压缩包内保留 `work/...`、`staging/...` 相对结构。恢复仅用于查阅历史；
不得解包覆盖当前训练 checkout，也不得直接运行或采用其中的过期路径、规则和配置。

归档保持私有且被 `.gitignore` 排除，不上传公开 GitHub；忽略规则不是访问控制，应保持目录仅账户可访问。
Mac 的 `sources/`、研究报告、其他项目和原有审计记录不在删除范围。
`/Volumes/n12388815` 是 QUT 挂载，不是另一份 Mac 副本，不能从那里执行本地清理。

以下历史记录保持原文，旧目录名只用于追溯，不是当前运行入口。

## 当前目录关系 — 2026-09-09

- 训练仓库目录：`finetune_omega_wenbo`，对应 GitHub `Wenboalbert/finetune_omega_wenbo`，
  使用分支 `ue-heterocam-finetune`，负责训练代码、配置和结果。
- 原版模型主目录：`vggt-omega_wenbo`，对应 GitHub `Wenboalbert/vggt-omega_wenbo` 的
  `main` 分支；原始预训练权重仍保留在该目录的 `checkpoints/` 下。
- 模型源码提供目录：`offer_omega_original_model`，是模型仓库 `vggt-omega_wenbo` 的
  worktree，使用同名分支 `offer_omega_original_model`；它不是训练仓库的 worktree。

QUT 模型根路径：`/home/n12388815/phd/vggt_omega_project/offer_omega_original_model`。
本次变更前后，模型源码均为提交 `39a0cb8af88554f15ddcb5354cd52bde588fa014`。
训练仓库通过 `VGGT_OMEGA_PATH` / `PYTHONPATH` 和配置中的 `vggt_omega_root`
导入这里的 `VGGTOmega`，再独立加载原始预训练权重。

本次将模型目录 `vggt-omega_wenbo-wt-camerahead` 和分支 `experiment/finetune-v1`
统一更名为 `offer_omega_original_model`，并同步当前配置、PBS 入口和说明。
训练仓库名称、训练分支、模型源码、训练超参数和权重不变。下面的历史记录不改写。

## 目录改名记录 — 2026-09-08

训练仓库目录已由 `vggt_omega_heterocam_finetune` 更名为 `finetune_omega_wenbo`，
以与 GitHub 仓库名称对齐。下面的旧版本记录、历史运行日志和结果元数据保留原始命名；
当前训练配置和 PBS 入口已同步新路径。本次更名不改变模型结构、训练参数或实验方法。

模型 worktree 同日由 `vggt-omega-finetune` 更名为 `vggt-omega_wenbo-wt-camerahead`，
通过 `git worktree move` 更新关联；分支 `experiment/finetune-v1` 和模型提交 `39a0cb8` 不变。
名称中的 `camerahead` 表示旧实验用途，不表示模型源码已经修改。历史记录中的旧路径不改写。

## v0.1 — 2026-07-21

本版本只在新目录 `vggt_omega_heterocam_finetune` 中增加代码，不改动同级的
`vggt-omega-finetune`、`vggt-omega_wenbo`、`env_finetune` 或其他项目。

### 模型和基础设施来源

- 模型与实验主体：`Wenboalbert/finetune_omega_wenbo`；
- 实际训练模型代码：`/home/n12388815/phd/vggt_omega_project/vggt-omega-finetune`；
- 从 `vggt_wenbo` 训练实现中选择性采用的工程原则：AMP、warmup+cosine、梯度裁剪、
  完整恢复状态与裸模型权重分开保存；
- 初版为单 GPU PBS 训练，没有移植与当前实验无关的大型 DDP 框架。

### 新增能力

- UE `camera_parameters.csv` 到 OpenCV camera-from-world 的显式转换；
- 与 VGGT-Omega 相同的裁剪、缩放、混合尺寸 padding，并同步更新内参；
- 同一时刻 `4 CCTV anchors + 1 target camera` 的数据包；
- 不依赖全局最小二乘 Sim(3) 的几何监督：相对旋转、基线方向、CCTV 尺度归一化后的
  anchor/target 基线和水平/垂直 FOV；
- `camera_bias`、`camera_only` 和 `camera_plus_lastN` 三档冻结策略；
- 训练前 frozen baseline、scene-disjoint validation 和回归门禁；
- QUT Aquarius 的 smoke/train/eval PBS 脚本，不加载 CUDA module；
- 单元测试和一次 forward/backward 的 GPU smoke test。

### 有意暂不包含

- 未审计的 UE EXR depth 不进入第一阶段训练；
- 不使用错误的旧 `images.txt`；GT 与预测直接在张量中计算；
- 不允许按帧随机切 train/validation，必须按独立 scene 切分；
- 不因为 train loss 下降就自动采用 checkpoint。
