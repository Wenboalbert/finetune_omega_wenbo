# V001：focal-only register-token

状态：**目录、分支、共享数据入口与 CPU 环境预检已准备；微调算法尚未实现。**
本版本不是 Frame-FFN LoRA，也不是旧 CameraHead 全量微调。

- GitHub：`Wenboalbert/finetune_omega_wenbo:v001-focal-only-register-token`。
- QUT：`/home/n12388815/phd/vggt_omega_project/finetune_omega_wenbo/v001-focal-only-register-token`。
- 原分支 `scene-adaptation` 从 `be46a9cb30b8925f279842430514fd268ee41c18` 改名，保留 Git 历史。
- 模型：共用 `vggt-omega_wenbo/offer_omega_original_model @ 39a0cb8`，不复制 Ω 源码。
- 环境：现有 `/home/n12388815/phd/vggt_omega_project/env_finetune`；不新建环境。

## 当前入口与目录

```bash
cd /home/n12388815/phd/vggt_omega_project/finetune_omega_wenbo/v001-focal-only-register-token
bash scripts/check_environment.sh
```

该入口读取 `configs/qut_environment.json`，检查分支、模型提交和实际包/类导入路径、
环境、checkpoint 大小，以及共享数据入口的存在。JSON 报告输出到 stdout。
它不实例化模型、不读取 checkpoint 张量、不训练/推理、不提交 PBS，也不验证数据划分。
预检 PASS 不能解释为训练跑通；4.5GB 权重的期望 SHA256 不是本次重新完整哈希的结果。

```text
v001-focal-only-register-token/
├── AGENTS.md
├── README_V001.md
├── QUT_LAYOUT.md
├── src/                    # 本版核心代码；目前只有职责说明
├── configs/                # 本版配置；目前只有环境/模型/数据入口定位
├── scripts/                # 本版入口；目前只有安全预检
├── inputs/                 # 输入格式和公开安全示例；不放私有实际清单
├── docs/                   # 当前运行规范与历史布局说明
├── runs/                   # 每次完整对照，QUT 私有，不入 Git
└── training/               # 继承的旧代码，仅供参考；旧 PBS 被禁用
```

代码按方法版本分开；结果按一次完整对照聚合。后续 V002 等是本目录的兄弟 worktree，
目录名等于分支名，不建立在本目录之内。当前不创建 V000、FFN LoRA 或 V002。

## 每次完整对照的出口

```text
runs/<UTC-time>_<label>_<unique-id>/
├── run_manifest.json        # run_id、两组配对关系、协议和所有来源
├── inputs_manifest.json     # 相同场景/帧、预处理、focal 单位/坐标约定及内容哈希
├── resolved_config.json
├── provenance.json          # 代码/模型 commit、权重哈希、环境、seed、PBS job id
├── logs/                    # 调度器 stdout/stderr、总控日志
├── baseline/
│   ├── metadata.json        # 原始 Ω、同一权重、训练参数为空
│   ├── resolved_config.json
│   ├── trainable_parameters.json
│   ├── logs/
│   ├── predictions/
│   └── metrics/
├── adapted/
│   ├── metadata.json
│   ├── resolved_config.json
│   ├── trainable_parameters.json  # 精确参数名、shape、数量、更新方式
│   ├── logs/
│   ├── checkpoints/         # 未来每场景优化的参数，而非覆盖原始权重
│   ├── predictions/
│   └── metrics/
└── comparison/             # 相同评估协议的配对表、图、检查和结论
```

以上是规范，不是已经产生的结果或已实现的 writer。
run 内记录每组的状态、输入协议哈希和实际配置；失败/中断 run 也保留。
新执行必须使用新目录，不能覆写先前结果；独立运行的 baseline 不能靠文件名猜测配对。
如果以后复用 baseline，必须显式锁定来源 run、输入/模型/协议哈希并验证相同。
重复执行相同数据也创建新 run。数据原件仍在共享入口引用的源目录，不复制到 runs。

## 数据与下一步边界

共享入口是 `/home/n12388815/phd/vggt_omega_project/datasets/`，不是本 Git checkout 的子目录。
其 `registry.yaml` 只登记现有外部 source collection；当前未选择场景/帧、未冻结 dataset revision，
`external/`、`manifests/`、`prepared/` 暂为空。未复制、移动或加工原始数据。
私有 registry/真实 manifests 由 QUT 保存，不因代码 push 而得到 GitHub 备份。
代码中的数据协议和公开安全示例可入 Git；运行时将实际选择及哈希快照到该 run。

本版计划仅用 RGB 与已知 focal 进行每场景适配；pose/depth GT 仅离线评估，
不得参与适配、路由、参数/检查点选择。下一步须先确定 register 参数化和输入/focal
预处理协议，再实现训练与对照入口。不能仅凭目录名断言已采用 native register、
residual 或 deep prompts；当前未选择其中一种。未实现损失、优化器、训练 PBS 或 GS-ready 评估。

当前模型/环境/分支映射、未来版本规则和历史恢复入口见 [QUT_LAYOUT.md](QUT_LAYOUT.md)。
