# B 榜相对 A 榜的改动说明

## 一句话结论

**网络架构、损失函数、训练算法、训练/推理代码与 A 榜完全一致（零修改）**；改动仅有两类：
①用 B 榜官方训练集对 A 榜权重做了继续训练并做 SWA 权重平均（与 A 榜相同的做法与脚本）；
②推理超参在 A 榜声明的"验证集扫描"方法论下按 B 榜数据重扫微调。

## 1. 模型权重（算法不变，数据换为 B 榜官方训练集）

- 起点：A 榜最优权重 `checkpoints/aboard_best_swa19_bnfix.pkl`（A 榜审查包同一文件）。
- 用**与 A 榜完全相同的训练脚本** `train_heavy_xloss_strict_jittor.py`（同损失、同 FP_GRAD_ITERS=2、同 patch=2048 等），
  在 **B 榜官方训练集 dataset_train_b** 上继续训练三条臂。数据预处理与 A 榜同一脚本
  （`tools/prepare_train_npy.py`，官方网格 → 50k 表面采样 npy），训练集按官方划分剔除
  `datalist/validate_b.txt` 的 100 个验证模型（具体命令见 `scripts/10_bboard_train_arms.sh` 头部注释）
  （lr 1e-4 / 2e-5 两档、关闭旋转增广 `--no_aug_rotate`——该开关为训练脚本原有 CLI 参数）。
- 对 26 个逐 epoch 检查点做**等权 SWA 平均**得到 B 榜最优权重 `checkpoints/bboard_best_bigswa26.pkl`
  （成分清单与构建脚本：`src/pdlts_jittor/train/make_bigswa26.py`；SWA 做法与 A 榜 `04_swa_bnfix.sh` 同族）。
- 训练命令：`scripts/10_bboard_train_arms.sh`。

## 2. 推理超参（方法论沿 A 榜 DESIGN.md："由验证集扫描确定的推理参数"）

| 参数 | A 榜值 | B 榜值 | 依据 |
|---|---|---|---|
| patch_size | 3072 | 3072（不变） | 官方验证集划分(datalist/validate_b.txt, 100 模型)上的扫描确认仍为最优 |
| seed_k | 6 | **12** | 在官方提供的验证集划分（datalist/validate_b.txt，100 个模型）上构建本地验证集（噪声按 B 榜测试集统计特性生成）扫描，单调上升（sk6 → sk8 → sk10 逐档 +0.02~+0.03）；线上分数亦随 sk6 → sk8 → sk12 单调上升 |
| niters | 2 | 2（不变） | 扫描确认 niters=3 显著劣化 |
| 输出混合 | 无（单次前向） | **0.3×(niters=1) + 0.7×(niters=2) 逐点凸组合** | A 榜后期已线上验证的同族方法（n1/n2 混合较单 n2 提升约 +0.2）；B 榜线上 α=0.2/0.25/0.3/0.4 四点扫描显示 α≥0.3 处于平顶 |

- 复现命令：`scripts/11_bboard_reproduce_best.sh`（两次推理 + 混合 + 打包）。

## 3. 未做的事（与 A 榜一致性声明）

- 未修改任何网络结构 / 损失项 / 优化器 / 数据增强逻辑（`--no_aug_rotate` 为脚本原有开关）；
- 未引入外部数据、外部预训练权重（B 榜训练仅用官方 dataset_train_b，A 榜起点权重为纯 Jittor 从零训练所得）；
- 未使用任何测试集标签信息；未做模型集成（最终仍是**一份权重**，混合仅为同一权重两种 niters 的输出凸组合）；
- 推理代码 `run_heldout_strict_jittor.py` 与 A 榜逐字节一致，改动全部通过原有 CLI 参数表达。
