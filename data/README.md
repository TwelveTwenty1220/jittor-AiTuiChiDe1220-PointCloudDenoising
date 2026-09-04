# 数据说明（本目录不放任何数据文件）

所有数据均为组委会提供的官方数据，从 https://www.educoder.net/competitions/Jittor-7 的「评测数据」页下载。

## 官方目录结构

```
dataset_train/                 # A 榜训练集: ShapeNet OBJ 网格 (~15.8k 模型)
└── shapenet/<synset_id>/<model_id>/models/model_normalized.obj
dataset_train_b/               # B 榜训练集 (同结构)
datalist/                      # B 榜官方划分: train_b.txt / validate_b.txt / test_b.txt
dataset_test_noisy/            # A 榜测试集: 带噪点云
└── shapenet/<synset_id>/<model_id>/noisy.npy      # (N,3) float32
dataset_test_noisy_b/          # B 榜测试集 (200 模型, 同结构)
```

## 如何配置数据根目录

编辑 `scripts/00_env.sh` 顶部三行（或运行前 `export` 同名环境变量，环境变量优先）：

| 变量 | 含义 |
|---|---|
| `DATASET_TRAIN` | A 榜官方训练集根（含 `shapenet/`） |
| `DATASET_TEST` | A 榜官方测试集根（含 `shapenet/`） |
| `DATASET_TEST_B` | B 榜官方测试集根（含 `shapenet/`） |

B 榜训练集与 datalist 目录以命令行参数传给 `scripts/10_bboard_train_arms.sh`。

## 预处理产物

`scripts/01_prepare_data.sh` 把官方网格做面积加权均匀表面采样（50k 点/模型），写到
`outputs/data/`（B 榜为 `outputs/data_b/`），单位球归一化后保存为 npy。该目录约 4 GB，
默认被 `.gitignore` 排除，可随时从官方网格重建。

- A 榜训练集中 151 个模型（`tools/heldout_151.txt`）划为本地验证并整体从训练集剔除；
- B 榜训练集按官方 `datalist/validate_b.txt`（100 模型）剔除验证划分。
