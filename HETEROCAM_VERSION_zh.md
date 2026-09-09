# 异构相机微调版本记录

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
