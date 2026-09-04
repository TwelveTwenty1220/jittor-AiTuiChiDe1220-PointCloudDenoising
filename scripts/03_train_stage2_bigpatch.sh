#!/usr/bin/env bash
# 第3步: 阶段二 大patch续训 (patch=2048 对齐推理, ep79..97; 约 3.2h/epoch)
# 注意 emd_w 0.10 -> 0.05: emd_subset_scale=num_points/512 会随 patch 翻倍被动放大 EMD 权重, 减半抵消
set -e; source "$(dirname "$0")/00_env.sh"
cd "$WORK/src/pdlts_jittor/train"
CUDA_VISIBLE_DEVICES=${GPU:-0} FP_GRAD_ITERS=2 python -u train_heavy_xloss_strict_jittor.py \
  --data_root "$DATA_ROOT" --out_dir "$RUNS/stage2" --tag strict-jt \
  --resume_jittor "$RUNS/stage1/strict-jt-ep78.pkl" \
  --max_epochs 19 --save_every 1 --start_epoch 78 \
  --batch_size 4 --num_patches 1 --train_patch_size 2048 --num_workers 4 --lr 2e-4 \
  --emd_w 0.05 --w_cd 40 --w_rep 40 --r0 0.05 --w_sink 40 --sink_iters 30 --sink_eps 0.005 \
  --emd_workers 4 --emd_max_points 512 --emd_subset_mode random --emd_subset_scale inverse_fraction \
  --feature_hidden 64 --log_every 20 --seed "$SEED"
