#!/usr/bin/env bash
# B榜第1步: B榜数据预处理 + 三条继续训练臂 (训练代码/损失/架构与A榜完全一致, 仅换官方B榜数据并调 lr)
# 起点: A榜最优权重 checkpoints/aboard_best_swa19_bnfix.pkl
# 用法: bash 10_bboard_train_arms.sh <dataset_train_b解压目录> <官方datalist目录>
# 三臂有依赖顺序(臂2/3 从臂1的中间产物出发), 脚本按序执行; 单臂单卡约 3-4h/epoch。
# 全部结束后: python "$WORK/src/pdlts_jittor/train/make_bigswa26.py" --ckpt_dir "$RUNS/bboard" --out bboard_best_bigswa26.pkl
set -e; source "$(dirname "$0")/00_env.sh"
TRAIN_B="${1:?用法: bash 10_bboard_train_arms.sh <dataset_train_b目录> <datalist目录>}"
DATALIST="${2:?需提供官方 datalist 目录(含 train_b.txt/validate_b.txt)}"
CK_A="$WORK/checkpoints/aboard_best_swa19_bnfix.pkl"
OUT="$RUNS/bboard"; mkdir -p "$OUT"
cd "$WORK/src/pdlts_jittor/train"

# ---- 0. B榜数据预处理(与A榜同一脚本): 官方网格 -> 50k采样npy, 剔除官方验证集划分 ----
if [ ! -d "$DATA_B" ]; then
  sed 's|^shapenet/||' "$DATALIST/validate_b.txt" > "$OUT/val_b_heldout.txt"
  python "$WORK/tools/prepare_train_npy.py" --dataset_train "$TRAIN_B" \
    --out "$DATA_B" --points 50000 --split train --heldout_list "$OUT/val_b_heldout.txt" --workers 8
fi

COMMON="--data_root $DATA_B --save_every 1 --batch_size 4 --num_patches 1 --train_patch_size 2048 \
  --num_workers 4 --emd_w 0.05 --w_cd 40 --w_rep 40 --r0 0.05 --w_sink 40 --sink_iters 30 --sink_eps 0.005 \
  --no_aug_rotate --emd_workers 4 --emd_max_points 512 --emd_subset_mode random --emd_subset_scale inverse_fraction \
  --feature_hidden 64 --log_every 20 --seed $SEED"
TS=train_heavy_xloss_strict_jittor.py

# ---- 臂1 norot: A榜权重起点, lr1e-4, 关旋转增广, ep101..133 (SWA取 ep118..133 共16点) ----
CUDA_VISIBLE_DEVICES=${GPU:-0} FP_GRAD_ITERS=2 python -u $TS \
  --out_dir "$OUT" --tag bboard-norot \
  --resume_jittor "$CK_A" --start_epoch 100 --max_epochs 33 --lr 1e-4 $COMMON

# ---- 臂1b anneal(臂2的母臂): 从臂1 ep121 起 lr5e-5 退火, ep122..129 ----
CUDA_VISIBLE_DEVICES=${GPU:-0} FP_GRAD_ITERS=2 python -u $TS \
  --out_dir "$OUT" --tag bboard-anneal \
  --resume_jittor "$OUT/bboard-norot-ep121.pkl" --start_epoch 121 --max_epochs 8 --lr 5e-5 $COMMON

# ---- 臂2 a2nd-e129: 从 anneal ep129 起 lr2e-5, ep130..134 (SWA取5点) ----
CUDA_VISIBLE_DEVICES=${GPU:-0} FP_GRAD_ITERS=2 python -u $TS \
  --out_dir "$OUT" --tag bboard-a2nd \
  --resume_jittor "$OUT/bboard-anneal-ep129.pkl" --start_epoch 129 --max_epochs 5 --lr 2e-5 $COMMON

# ---- 臂3 a2nd-p12: 从臂1 ep118-129 的12点SWA起 lr2e-5, ep130..134 (SWA取5点) ----
python - <<'PYEOF'
import os, numpy as np, jittor as jt
out = os.environ['RUNS'] + '/bboard'
paths = [f'{out}/bboard-norot-ep{e}.pkl' for e in range(118, 130)]
acc, last = {}, None
for p in paths:
    d = jt.load(p); last = d
    for k, v in d.items():
        a = v.numpy() if hasattr(v, 'numpy') else np.asarray(v)
        if np.issubdtype(a.dtype, np.floating):
            acc[k] = acc.get(k, 0) + a.astype(np.float64)
res = {k: ((acc[k]/len(paths)).astype((v.numpy() if hasattr(v,'numpy') else np.asarray(v)).dtype) if k in acc else v) for k, v in last.items()}
jt.save(res, f'{out}/norot_pswa12.pkl')
print('norot_pswa12.pkl saved (12-point SWA of ep118-129)')
PYEOF
CUDA_VISIBLE_DEVICES=${GPU:-0} FP_GRAD_ITERS=2 python -u $TS \
  --out_dir "$OUT" --tag bboard-a2ndp \
  --resume_jittor "$OUT/norot_pswa12.pkl" --start_epoch 129 --max_epochs 5 --lr 2e-5 $COMMON

echo "三臂完成。构建B榜最优权重:"
echo "  python $WORK/src/pdlts_jittor/train/make_bigswa26.py --ckpt_dir $OUT --out $OUT/bboard_best_bigswa26.pkl"
