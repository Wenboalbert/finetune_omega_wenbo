# Scene adaptation: 独立实验工作区

状态：只有目录 / 分支 / 环境预检基础设施；尚无 V000/V001 训练实现或新实验结果。

- GitHub：`Wenboalbert/finetune_omega_wenbo`，分支 `scene-adaptation`。
- QUT：`/home/n12388815/phd/vggt_omega_project/finetune_omega_wenbo/scene-adaptation`。
- 来源：从完成分层迁移的 `ue-heterocam-finetune` 提交建立；精确父提交见 `git log --first-parent`。
- Ω 源码：`/home/n12388815/phd/vggt_omega_project/vggt-omega_wenbo/offer_omega_original_model`，固定 `39a0cb8`。
- 环境：既有 `env_finetune`；权重仍引用模型 `main/checkpoints/` 中的原始文件。

## 现在可运行的入口

```bash
cd /home/n12388815/phd/vggt_omega_project/finetune_omega_wenbo/scene-adaptation
bash scripts/check_environment.sh
```

它检查分支、路径、模型提交、实际包/类导入、checkpoint 文件大小和运行环境，输出 JSON 到 stdout。
不会加载 1B 权重、实例化模型、执行推理/训练或申请 GPU。预检 PASS 不等于训练已经跑通。
默认配置为 `configs/qut_environment.json`；脚本路径相对于自身解析，不依赖启动时的工作目录。
原始权重完整 SHA256 的复核记录在此次私有迁移审计中；日常预检不重复读取 4.5GB。

## 后续版本 / run 约定

```text
scene-adaptation/
├── AGENTS.md
├── README_SCENE_ADAPTATION.md
├── QUT_LAYOUT.md
├── configs/qut_environment.json      # 公共模型 / 环境定位，不含训练超参数
├── scripts/check_environment.{sh,py} # 当前唯一的新工作区可运行入口
├── experiments/README.md             # 实验索引；版本目录尚未创建
└── runs/<version>/<unique-run-id>/   # 运行时创建，不入 Git
    ├── resolved_config.json
    ├── inputs_manifest.json
    ├── trainable_parameters.json
    ├── provenance.json
    ├── logs/
    ├── checkpoints/
    ├── predictions/
    └── metrics/
```

每版未来各自保留 `experiments/<version>/{src,configs,scripts,inputs}/`。
inputs 保存格式定义、公开安全示例或数据引用；私有实际输入清单放到该 run，不向公开 GitHub 上传。
原始图像和大权重不复制到代码仓库。每次执行创建新 run，配置 / 可训练参数 / 输入 / commit 均快照化。
跨版本复用实现需显式声明依赖和锁定来源，不能偷偷读取兄弟实验的当前文件。

规划的 V000 为 frozen Ω，V001 为 focal-only Frame-wise FFN LoRA；唯一训练 GT 是已知 focal。
本次未实现 LoRA、损失、优化器、训练 PBS、数据划分、GT 离线评估或 GS-ready 判定。
继承的 `training/` 是旧 CameraHead/RGB-D 代码，不可当作 V001；三个旧 PBS 在本分支直接 STOP。
新分支没有复制旧 manifests、runs、logs 或归档；它们只在 sibling legacy checkout。

下一步先确定 V000/V001 的输入协议和实验设计，再建立对应版本目录与训练代码。
所有目录 / GitHub 映射和历史恢复信息见 [QUT_LAYOUT.md](QUT_LAYOUT.md)。
