#!/usr/bin/env bash
# 第1步: 官方网格 -> 50k 均匀表面采样 npy (约 1-2 h, 仅 CPU)
set -e; source "$(dirname "$0")/00_env.sh"
python "$WORK/tools/prepare_train_npy.py" \
  --dataset_train "$DATASET_TRAIN" --out "$DATA_ROOT" \
  --points 50000 --split train --heldout_list "$WORK/tools/heldout_151.txt" --workers 8
