# jittor-AiTuiChiDe1220-PointCloudDenoising

第六届计图人工智能挑战赛 · 赛道二「基于深度学习的三维点云降噪」参赛代码，**纯 Jittor 实现，从零训练**。

仓库名按官方规范 `jittor-[战队名]-[项目名]`（GitHub 不支持中文仓库名，战队名以拼音表示）。

**方法概要**：EdgeConv 图卷积特征 + 谱归一化可逆残差流（参考 CVPR 2024 PD-LTS 架构，Jittor 独立实现）；
对齐评分指标的复合损失（EMD + 可微 CD + repulsion + Sinkhorn）；定点求逆的 K=2 截断 Neumann 反传；
两阶段 patch 尺度对齐训练；宽窗口 SWA 权重平均 + 谱 scale 重算 + BN 重估。设计依据与消融见 [`docs/DESIGN.md`](docs/DESIGN.md)。

**A / B 榜权重**：

| 阶段 | 权重 | 说明 |
|---|---|---|
| A 榜 | `checkpoints/aboard_best_swa19_bnfix.pkl` | 从零训练两阶段 + 19 点 SWA |
| B 榜 | `checkpoints/bboard_best_bigswa26.pkl` | 以 A 榜权重为起点、用官方 B 榜训练集续训 + 26 点 SWA |

B 榜相对 A 榜：网络 / 损失 / 训练与推理代码**零修改**，仅续训 + 推理超参重扫，见 [`docs/BBOARD_CHANGES.md`](docs/BBOARD_CHANGES.md)。

## 目录结构

```
README.md  LICENSE(MIT)  NOTICE(第三方引用声明)  requirements.txt  .gitignore
configs/                     最终配置记录 (aboard_final.yaml / bboard_final.yaml, 与脚本取值一一对应)
src/pdlts_jittor/            核心代码 (纯 Jittor, 无 torch 依赖)
├── pdlts_model.py               网络结构: EdgeConv + 谱归一化可逆流
├── strict_jittor_denoise.py     patch 化去噪核心 (FPS / KNN / 融合 / 迭代)
├── strict_jittor_io.py          jt.save / jt.load 权重读写 (含谱 scale 烘焙)
├── run_heldout_strict_jittor.py 推理入口 (断点续跑, 多卡分片)
├── run_meta.py                  运行配置/命令落盘 + 缺路径报错
└── train/
    ├── train_heavy_xloss_strict_jittor.py  训练入口
    ├── data.py  losses_jt.py  emd_jt.py    数据管线 / 损失
    ├── model_train.py  spectral.py         训练版模型 / 谱归一化
    ├── make_swa_ckpt.py  make_bigswa26.py  SWA 权重平均 (A 榜 19 点 / B 榜 26 点)
    └── bn_recalibrate.py                   谱 scale 重算 + BN 重估
scripts/                     一键脚本 (只需填 00_env.sh 里的数据路径)
├── 00_env.sh                    路径变量
├── 01~06_*.sh                   A 榜: 数据 → 训练两阶段 → SWA → 推理 → 打包
└── 10~11_bboard_*.sh            B 榜: 三臂续训 / 最终结果复现
tools/                       prepare_train_npy.py (官方网格 → 训练 npy)  package_result.py (打包+硬校验)  heldout_151.txt
checkpoints/                 两份最终权重 (各 6MB, SHA256 见 SHA256SUMS.txt)
data/README.md               数据获取与目录配置说明 (不放数据)
docs/                        DESIGN.md (设计思路与消融)  BBOARD_CHANGES.md (B 榜改动说明)
outputs/                     运行产物 (数据采样 / 权重 / 结果 / 日志), 自动生成, 不入库
```

## 1. 环境安装

- OS：Ubuntu 20.04 / 22.04；CUDA 12.x（组委会推荐 12.4）；单卡显存 ≥ 24 GB（训练 bs4×patch2048 约 20.4 GB）
- Python 3.9 或 3.10；Jittor 1.3.10.0

```bash
conda create -n jittor python=3.9 -y && conda activate jittor
pip install -r requirements.txt        # jittor==1.3.10.0 numpy scipy trimesh
python -c "import jittor as jt; jt.flags.use_cuda=1; print(jt.__version__)"   # 首次运行会 JIT 编译 CUDA 算子
```

## 2. 数据准备

从官网「评测数据」页下载官方数据（结构见 [`data/README.md`](data/README.md)），然后填 `scripts/00_env.sh` 顶部三个路径
（或运行前 `export` 同名环境变量，环境变量优先）：

```bash
export DATASET_TRAIN=/path/to/dataset_train          # A 榜训练集 (OBJ 网格)
export DATASET_TEST=/path/to/dataset_test_noisy      # A 榜测试集 (noisy.npy)
export DATASET_TEST_B=/path/to/dataset_test_noisy_b  # B 榜测试集 (noisy.npy)
```

官方网格 → 训练用 npy（50k 点/模型，面积加权均匀采样，仅 CPU，约 1–2 h，可断点续跑）：

```bash
bash scripts/01_prepare_data.sh        # 输出 outputs/data/, 自动剔除 tools/heldout_151.txt 的 151 个本地验证模型
```

## 3. 训练

A 榜权重（从零训练，单卡 RTX 4090 约 270 GPU·h）：

```bash
GPU=0 bash scripts/02_train_stage1.sh          # 阶段一 ep1-78,  patch=1024, lr=2e-4 (~2.7 h/epoch)
GPU=0 bash scripts/03_train_stage2_bigpatch.sh # 阶段二 ep79-97, patch=2048 (~3.2 h/epoch)
GPU=0 bash scripts/04_swa_bnfix.sh             # ep79-97 共 19 点 SWA → 谱 scale 重算 → BN 重估 (~0.5 h)
#  → outputs/runs/swa19_bnfix.pkl  (= checkpoints/aboard_best_swa19_bnfix.pkl)
```

B 榜权重（以 A 榜权重为起点，用官方 B 榜训练集续训三臂，再做 26 点 SWA）：

```bash
GPU=0 bash scripts/10_bboard_train_arms.sh /path/to/dataset_train_b /path/to/datalist
python src/pdlts_jittor/train/make_bigswa26.py --ckpt_dir outputs/runs/bboard --out outputs/runs/bboard/bboard_best_bigswa26.pkl
#  → 等价于 checkpoints/bboard_best_bigswa26.pkl
```

单条训练命令示例（脚本内部即此，全部关键参数由命令行给出，`FP_GRAD_ITERS=2` 为环境变量读取的必需超参）：

```bash
cd src/pdlts_jittor/train
CUDA_VISIBLE_DEVICES=0 FP_GRAD_ITERS=2 python -u train_heavy_xloss_strict_jittor.py \
  --data_root ../../../outputs/data --out_dir ../../../outputs/runs/stage1 --tag strict-jt \
  --max_epochs 78 --save_every 1 --batch_size 4 --num_patches 1 --train_patch_size 1024 --lr 2e-4 \
  --emd_w 0.10 --w_cd 40 --w_rep 40 --r0 0.05 --w_sink 40 --sink_iters 30 --sink_eps 0.005 \
  --emd_workers 4 --emd_max_points 512 --emd_subset_mode random --emd_subset_scale inverse_fraction \
  --feature_hidden 64 --log_every 20 --seed 2023
```

每次运行会在 `--out_dir` 下落盘 `<tag>-config.json`（实际参数）与 `command.txt`（命令与环境变量）；训练日志在标准输出，建议 `| tee outputs/runs/train.log`。

## 4. 评测 / 推理

**复现 B 榜最终提交结果**（同一权重两次推理 niters=2 与 niters=1，逐点凸组合 0.3·n1 + 0.7·n2，再打包）：

```bash
GPU=0 bash scripts/11_bboard_reproduce_best.sh                 # 单卡全量约 8 h
# 多卡分片 (每卡一分片, 断点自动续; 全部分片跑完后任意再执行一次即完成混合与打包):
GPU=0 SHARD=0 NSHARD=4 bash scripts/11_bboard_reproduce_best.sh &
GPU=1 SHARD=1 NSHARD=4 bash scripts/11_bboard_reproduce_best.sh &  # ...
#  → outputs/result_b_best.zip  (提交格式: shapenet/<syn>/<mid>/denoised.npy)
```

**复现 A 榜最终提交结果**：

```bash
cp checkpoints/aboard_best_swa19_bnfix.pkl outputs/runs/swa19_bnfix.pkl   # 跳过训练时
GPU=0 bash scripts/05_infer_test.sh     # patch=3072, seed_k=6, niters=2; 单卡约 5.5 h, 支持 SHARD/NSHARD 分片
bash scripts/06_package.sh              # → outputs/result_a.zip (含点数=输入 / float32 / 无 NaN 硬校验)
```

单条推理命令（含 ckpt 路径）：

```bash
cd src/pdlts_jittor
CUDA_VISIBLE_DEVICES=0 JT_PATCH_STEP=8 python -u run_heldout_strict_jittor.py \
  --ckpt ../../checkpoints/bboard_best_bigswa26.pkl --data /path/to/dataset_test_noisy_b \
  --base ../../outputs/out_b --tag b_n2 --patch_size 3072 --seed_k 12 --niters 2 --seed 0
```

推理参数含义：`--patch_size` KNN patch 点数；`--seed_k` FPS 种子覆盖倍率（patch 数 = seed_k·N/patch_size）；`--niters` 去噪迭代轮数；`JT_PATCH_STEP` 每批 patch 数（24 GB 显存用 8，更小显存改 4）。

## 5. 结果说明

**指标**（官方）：`score = 0.5·CD_score + 0.5·P2S_score`，其中每项 `= clamp(100·(1 − metric_pred / metric_noisy), 0, 100)`；
CD 为去噪点云与真值点云的双向最近邻距离，P2S 为去噪点到原网格表面的均方距离，均在真值单位球归一化下计算。

| 提交 | 权重 | 推理配置 |
|---|---|---|
| A 榜 | `aboard_best_swa19_bnfix.pkl` | patch 3072 / seed_k 6 / niters 2 |
| B 榜 | `bboard_best_bigswa26.pkl` | patch 3072 / seed_k 12 / 0.3·niters1 + 0.7·niters2 |

**与线上提交结果的差异说明**：

- 用仓库内权重直接推理：输出与线上提交文件**不逐位相同**（GPU 并行归约的浮点顺序不确定，绝大多数点差异 < 1e-5，极少数处于两个 patch 等距边界的点归属翻转），但在有真值的样本上实测 CD 得分完全一致（小数点后四位相同）。
- 从零重新训练：训练含数据增广随机性，逐位复现不保证；配方即最终配方（无隐藏步骤），SWA 覆盖 19 / 26 个 epoch 对训练随机性有平滑作用，预期总分波动在 ±0.3 以内。
- 本地验证分与线上分存在固定偏移（本地验证集噪声分布 ≠ 线上），文档中的消融数字均为同一验证集内的相对比较。

## 合规与可复现性

- 纯 Jittor：`grep -rn "import torch" src/` 为空；权重为 `jt.save` 的 `.pkl`（Jittor 惯例，等同 `.ckpt`）。
- 无外部数据 / 无预训练权重：随机初始化从零训练；训练 npy 可由 `tools/prepare_train_npy.py` 从官方网格完整重建。
- 单模型、一份权重：B 榜的 n1/n2 混合是同一权重两种迭代轮数的输出凸组合，非模型集成。
- 随机种子：训练 `--seed`（默认 2023，线上所用），推理 `--seed`（推理无随机采样，FPS 起点固定）。
- 权重校验：`cd checkpoints && sha256sum -c SHA256SUMS.txt`。
- 本仓库与提交组委会审查的代码包相比，**算法零改动**，仅按官方《开源代码规范》做了规范性调整：目录重排（`src/`、`outputs/`）、去除写死的本机路径（改为必填参数）、运行配置与命令落盘、缺路径时的明确报错、新增 `configs/` 记录与 `LICENSE`/`NOTICE`。

## 引用

```
PD-LTS: Mao, A. et al. Denoising Point Clouds in Latent Space via Graph Convolution and Invertible Neural Network. CVPR 2024.
Jittor: Hu, S.-M. et al. Jittor: a novel deep learning framework with meta-operators and unified graph execution. Sci China Inf Sci, 2020.
```
