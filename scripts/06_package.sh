#!/usr/bin/env bash
# 第6步: 打包 result.zip(含全部硬校验: 点数=输入/float32/无NaN/结构)
set -e; source "$(dirname "$0")/00_env.sh"
python "$WORK/tools/package_result.py" \
  --pred_dir "$OUT_A/final" --noisy_dir "$DATASET_TEST" --out "$OUTPUTS/result_a.zip"
