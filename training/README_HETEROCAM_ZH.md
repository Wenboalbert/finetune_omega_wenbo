# VGGT-Omega 异构相机几何微调（QUT HPC）

这是独立于原有代码的保守训练入口。第一阶段只微调 camera head，不训练 depth head，目的不是
立刻提高所有指标，而是验证“远距、大基线、高空/不利视角”的误差能否在独立场景上稳定降低。

## 1. 为什么不用全局最小二乘 Sim(3) 当训练损失

一个错误很大的目标相机会拉动最小二乘解，使本来较正确的 CCTV 也出现残差。本实现改用四种
不依赖世界原点、整体朝向和全局尺度的量：

1. 两相机之间的相对旋转；
2. 在源相机坐标系中观察到的基线方向；
3. 用 CCTV-CCTV 中位基线估计尺度后，CCTV-CCTV 与 CCTV-target 的距离比例；
4. 由处理后 `K` 和图像尺寸计算的 HFOV/VFOV。

因此 UE world 和 VGGT-Omega world 不一致并不会制造损失。CCTV 仍然提供尺度基准，但 bad target
不能通过参与一次全局 least-squares fit 把 anchors 拉偏。

## 2. 数据分组

每个训练样本必须来自同一个 scene、同一个 frame：

```text
CCTV_01 + CCTV_02 + CCTV_03 + CCTV_04 + 一个 Drone/Phone target
```

推荐 `targets_per_packet=1`。每个目标相机分别产生一条可解释的验证记录。训练集、验证集和最终
测试集必须由不同 UE scene 构成；同一 scene 的不同 frame 不能分别放入 train 和 validation。

## 3. 建立 manifests

例子（把实际独立 scene 路径替换进去）：

```bash
cd /home/n12388815/phd/vggt_omega_project/vggt_omega_heterocam_finetune
export PYTHONPATH="$PWD/training"

python training/build_ue_manifest.py \
  --scene-root /home/n12388815/vgg_omega_4dgs/datasets/inputs/TRAIN_SCENE_A \
  --scene-root /home/n12388815/vgg_omega_4dgs/datasets/inputs/TRAIN_SCENE_B \
  --output training/manifests/train.jsonl \
  --anchors CCTV_01,CCTV_02,CCTV_03,CCTV_04

python training/build_ue_manifest.py \
  --scene-root /home/n12388815/vgg_omega_4dgs/datasets/inputs/INDEPENDENT_VAL_SCENE \
  --output training/manifests/val.jsonl \
  --anchors CCTV_01,CCTV_02,CCTV_03,CCTV_04
```

UE CSV 的 `fov` 被明确当作水平 FOV；位置由厘米转为米；姿态转成 OpenCV camera-from-world。

## 4. 运行门禁

登录节点先做不加载大模型的检查：

```bash
source /mnt/weka/pkg/rhel94/AuthenticAMD-25/software/Anaconda3/2024.02-1/etc/profile.d/conda.sh
conda activate /home/n12388815/phd/vggt_omega_project/env_finetune
export VGGT_OMEGA_PATH=/home/n12388815/phd/vggt_omega_project/vggt-omega-finetune
export PYTHONPATH="$VGGT_OMEGA_PATH:/home/n12388815/phd/vggt_omega_project/vggt_omega_heterocam_finetune/training"
python training/preflight_heterocam.py --config training/configs/heterocam_camera_only.json
```

它会拒绝：路径缺失、anchor 缺失、图像/相机矩阵非有限值、导入了旧
`vggt-omega_wenbo`、或 train/validation scene 重叠。

## 5. GPU smoke、训练和评估

```bash
qsub training/qut/smoke_heterocam.pbs
qsub training/qut/train_heterocam.pbs
qsub -v CHECKPOINT=/绝对路径/best_passed.pt training/qut/eval_heterocam.pbs
```

从完整状态恢复：

```bash
qsub -v RESUME=/绝对路径/latest_training_state.pt training/qut/train_heterocam.pbs
```

PBS 脚本使用 `env_finetune`，并把 `vggt-omega-finetune` 放到 `PYTHONPATH` 最前面。它不会修改
环境，也不加载 CUDA module。smoke 只做一个 packet 的 forward/backward，不执行 optimizer step。

## 6. checkpoint 采用规则

- `best_candidate.pt`：综合验证分数最低，但可能没有通过安全门禁；
- `best_passed.pt`：只看 held-out target 的 cross-camera baseline 与方向必须改善，
  cross-camera rotation 和 target HFOV 最多回归 5%；
- `latest_training_state.pt`：仅含可训练参数 delta、optimizer/scheduler，供恢复训练，避免再复制
  一份 4.5GB 的冻结 backbone；
- `decision.json`：每次验证是否通过门禁的完整证据。

只有 `best_passed.pt` 才建议进入独立 test scene 和后续 4DGS A/B 测试。若没有该文件，结论应是
这次微调没有可靠收益，而不是改用 train loss 最低的权重。

## 7. 分阶段策略

1. 先用 `camera_bias` 做诊断；
2. 默认 `camera_only`；
3. 只有多个独立验证场景都显示一致收益，才使用 `camera_plus_lastN`；
4. depth head 在完成 EXR 深度定义、单位、裁剪和投影审计前始终冻结。
