#!/usr/bin/env bash
# 第5步: 测试集推理 (patch=3072, seed_k=6, niters=2; 50k点/模型约 100s, 200模型单卡约 5.5h)
# 多卡加速: GPU=0 SHARD=0 NSHARD=3 bash 05_infer_test.sh (每卡一个分片, 断点自动续)
set -e; source "$(dirname "$0")/00_env.sh"
cd "$WORK/src/pdlts_jittor"
CUDA_VISIBLE_DEVICES=${GPU:-0} JT_PATCH_STEP=8 python -u run_heldout_strict_jittor.py \
  --ckpt "$RUNS/swa19_bnfix.pkl" --data "$DATASET_TEST" \
  --base "$OUT_A" --tag final \
  --patch_size 3072 --seed_k 6 --niters 2 \
  --start "${SHARD:-0}" --stride "${NSHARD:-1}"
