#!/usr/bin/env python
"""构建 B 榜最优权重 bboard_best_bigswa26.pkl —— 26 个训练检查点的等权 SWA 平均。

成分清单(共 26 点, 三条继续训练臂, 全部由本仓库 train_heavy_xloss_strict_jittor.py 产生):
  1) norot 臂  ep118..ep133 (16 点): 由 A 榜权重 warm-start, B 榜官方训练集继续训练,
     lr=1e-4, 关闭旋转增广(--no_aug_rotate), 其余训练超参与 A 榜一致;
  2) a2nd-e129 臂 ep130..ep134 (5 点): 从 anneal 臂 ep129 起 lr=2e-5 退火续训;
  3) a2nd-p12 臂  ep130..ep134 (5 点): 从 norot 臂 ep118-129 的 12 点 SWA 起 lr=2e-5 退火续训。

用法:
  python make_bigswa26.py --ckpt_dir <存放上述27个pkl的目录> --out bboard_best_bigswa26.pkl
验证: 生成文件应与 checkpoints/bboard_best_bigswa26.pkl 的 SHA256 一致
     (逐参数最大误差 < 2e-6, 为 float32 舍入)。
"""
import argparse
import numpy as np
import jittor as jt

ap = argparse.ArgumentParser()
ap.add_argument('--ckpt_dir', required=True)
ap.add_argument('--out', default='bboard_best_bigswa26.pkl')
a = ap.parse_args()

paths = (
    [f'{a.ckpt_dir}/bboard-norot-ep{e}.pkl' for e in range(118, 134)] +
    [f'{a.ckpt_dir}/bboard-a2nd-ep{e}.pkl' for e in range(130, 135)] +
    [f'{a.ckpt_dir}/bboard-a2ndp-ep{e}.pkl' for e in range(130, 135)]
)
assert len(paths) == 26
acc, last = {}, None
for p in paths:
    d = jt.load(p)
    last = d
    for k, v in d.items():
        arr = v.numpy() if hasattr(v, 'numpy') else np.asarray(v)
        if np.issubdtype(arr.dtype, np.floating):
            acc[k] = acc.get(k, 0) + arr.astype(np.float64)
out = {}
for k, v in last.items():
    arr = v.numpy() if hasattr(v, 'numpy') else np.asarray(v)
    out[k] = (acc[k] / len(paths)).astype(arr.dtype) if k in acc else v
jt.save(out, a.out)
print(f'saved {a.out}: 26-point SWA, {len(out)} keys')
