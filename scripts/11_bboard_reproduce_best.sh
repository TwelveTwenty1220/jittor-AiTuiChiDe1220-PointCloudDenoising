#!/usr/bin/env bash
# B榜第2步: 复现 B 榜最终提交结果
# 权重: checkpoints/bboard_best_bigswa26.pkl (或由 10_bboard_train_arms.sh + make_bigswa26.py 复现)
# 配方: 同一权重两次推理 (niters=2 与 niters=1), 输出逐点凸组合 0.3*n1 + 0.7*n2
# 先在 00_env.sh 里填好 DATASET_TEST_B。单卡全量约 8h; 多卡分片:
#   GPU=0 SHARD=0 NSHARD=4 bash 11_bboard_reproduce_best.sh   (每卡一分片, 断点自动续)
# 全部分片跑完后, 任意再执行一次本脚本(或单独执行第3步)完成混合与打包。
set -e; source "$(dirname "$0")/00_env.sh"
cd "$WORK/src/pdlts_jittor"
CKPT="$WORK/checkpoints/bboard_best_bigswa26.pkl"
[ -f "$CKPT" ] || { echo "缺权重 $CKPT"; exit 1; }
[ -d "$DATASET_TEST_B" ] || { echo "请先在 00_env.sh 填 DATASET_TEST_B (当前: $DATASET_TEST_B)"; exit 1; }

# 1) niters=2 主推理 (约 100s/模型/卡)
CUDA_VISIBLE_DEVICES=${GPU:-0} JT_PATCH_STEP=8 python -u run_heldout_strict_jittor.py \
  --ckpt "$CKPT" --data "$DATASET_TEST_B" --base "$OUT_B" --tag b_n2 \
  --patch_size 3072 --seed_k 12 --niters 2 --start "${SHARD:-0}" --stride "${NSHARD:-1}"

# 2) niters=1 辅推理 (约 50s/模型/卡)
CUDA_VISIBLE_DEVICES=${GPU:-0} JT_PATCH_STEP=8 python -u run_heldout_strict_jittor.py \
  --ckpt "$CKPT" --data "$DATASET_TEST_B" --base "$OUT_B" --tag b_n1 \
  --patch_size 3072 --seed_k 12 --niters 1 --start "${SHARD:-0}" --stride "${NSHARD:-1}"

# 3) 齐 200 模型后: 逐点混合 0.3*n1+0.7*n2 -> float32 -> 打包 (未齐则提示后退出)
N2CNT=$(ls "$OUT_B"/b_n2/shapenet/*/*/denoised.npy 2>/dev/null | wc -l)
N1CNT=$(ls "$OUT_B"/b_n1/shapenet/*/*/denoised.npy 2>/dev/null | wc -l)
if [ "$N2CNT" -lt 200 ] || [ "$N1CNT" -lt 200 ]; then
  echo "推理未齐 (n2=$N2CNT/200, n1=$N1CNT/200): 其余分片完成后再执行一次本脚本即可打包"; exit 0
fi
python - <<'PYEOF'
import numpy as np, glob, os
outb = os.environ['OUT_B']
n2r, n1r, out = f'{outb}/b_n2', f'{outb}/b_n1', f'{outb}/b_blend30'
n2s = sorted(glob.glob(f'{n2r}/shapenet/*/*/denoised.npy'))
assert len(n2s) == 200
for f2 in n2s:
    rel = os.path.relpath(f2, n2r)
    a2, a1 = np.load(f2), np.load(f'{n1r}/{rel}')
    bl = (0.3 * a1 + 0.7 * a2).astype(np.float32)
    assert np.isfinite(bl).all()
    o = f'{out}/{rel}'
    os.makedirs(os.path.dirname(o), exist_ok=True)
    np.save(o, bl)
print('blend done: 200 shapes')
PYEOF
python "$WORK/tools/package_result.py" --pred_dir "$OUT_B/b_blend30" \
  --noisy_dir "$DATASET_TEST_B" --out "$OUTPUTS/result_b_best.zip"
echo "B榜最优复现完成: $OUTPUTS/result_b_best.zip"
