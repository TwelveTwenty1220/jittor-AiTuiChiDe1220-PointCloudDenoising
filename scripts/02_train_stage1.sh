#!/usr/bin/env bash
# 第2步: 从零训练 阶段一 (patch=1024, ep1..78; 单卡 24G 约 2.7h/epoch)
# FP_GRAD_ITERS=2 是定点求逆反传的 Neumann 截断阶数, 本路线关键超参(见 DESIGN.md §3)
set -e; source "$(dirname "$0")/00_env.sh"
cd "$WORK/src/pdlts_jittor/train"
CUDA_VISIBLE_DEVICES=${GPU:-0} FP_GRAD_ITERS=2 python -u train_heavy_xloss_strict_jittor.py \
  --data_root "$DATA_ROOT" --out_dir "$RUNS/stage1" --tag strict-jt \
  --max_epochs 78 --save_every 1 --start_epoch 0 \
  --batch_size 4 --num_patches 1 --train_patch_size 1024 --num_workers 4 --lr 2e-4 \
  --emd_w 0.10 --w_cd 40 --w_rep 40 --r0 0.05 --w_sink 40 --sink_iters 30 --sink_eps 0.005 \
  --emd_workers 4 --emd_max_points 512 --emd_subset_mode random --emd_subset_scale inverse_fraction \
  --feature_hidden 64 --log_every 20 --seed "$SEED"
