#!/usr/bin/env bash
# 第4步: SWA(ep79..97 共19个ckpt 权重平均) -> 谱scale重算 + BN重估 (约 0.5h)
set -e; source "$(dirname "$0")/00_env.sh"
cd "$WORK/src/pdlts_jittor/train"
CKS=""; for e in $(seq 79 97); do CKS="$CKS $RUNS/stage2/strict-jt-ep${e}.pkl"; done
python make_swa_ckpt.py "$RUNS/swa19.pkl" $CKS
CUDA_VISIBLE_DEVICES=${GPU:-0} python -u bn_recalibrate.py \
  --ckpt "$RUNS/swa19.pkl" --out "$RUNS/swa19_bnfix.pkl" \
  --data_root "$DATA_ROOT" --patch_size 2048 --batches 120
