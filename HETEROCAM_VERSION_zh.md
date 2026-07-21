# 异构相机微调版本记录

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
