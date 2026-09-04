"""SWA checkpoint 的 BatchNorm 重估 + 谱归一化 scale 重算.

背景(2026-07-22 审查):
  make_swa_ckpt.py 无差别平均所有 float 数组, 包含 BN 的 running_mean/running_var
  和谱归一化的 u/v/scale. 这两者都不是"可平均"的量:
    - BN 统计量: SWA 原论文明确要求在平均后的权重上【重新估计】, 而不是平均各 ckpt 的统计量.
    - 谱 scale: 正确值是 sigma(W_avg), 但 mean(sigma(W_i)) >= sigma(W_avg) (谱范数是凸的),
      实测 stored/true 均值 1.000228 / 最大 1.0107 / 180 层里 9 层偏差 >0.1%.
  另: 实测同 ckpt 同输入下 model.train() 与 model.eval() 输出差 = 去噪位移量的 28-34%,
  说明 BN 统计量对该模型影响很大, 而推理只跑过 eval().

用法:
  python bn_recalibrate.py --ckpt <in.pkl> --out <out.pkl> [--patch_size 2048] [--batches 120]
"""
import argparse
import os
import pickle
import sys

import jittor as jt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

from data import PairedPatchDataset            # noqa: E402
from model_train import HeavyDenoiseFlowTrain  # noqa: E402
from spectral import InducedNormLinearJT       # noqa: E402
from strict_jittor_io import load_jittor_state, save_jittor_state  # noqa: E402


def update_lipschitz(model):
    """与 train_heavy_xloss_strict_jittor.py:37 同一实现(那里是内联定义的)."""
    count = 0
    for module in model.modules():
        if isinstance(module, InducedNormLinearJT):
            module.compute_weight(update=True)
            count += 1
    return count


def reset_bn(model):
    """把所有 BN 的 running 统计量清零, 并记录模块列表."""
    mods = []
    for m in model.modules():
        if hasattr(m, "running_mean") and hasattr(m, "running_var"):
            m.running_mean.assign(jt.zeros_like(m.running_mean))
            m.running_var.assign(jt.ones_like(m.running_var))
            mods.append(m)
    return mods


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--data_root", required=True, help="训练 npy 根目录(与训练脚本 --data_root 相同)")
    ap.add_argument("--patch_size", type=int, default=2048,
                    help="重估用的 patch 大小; 默认对齐推理的 2048 而非训练的 1024")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--batches", type=int, default=120)
    ap.add_argument("--noise_min", type=float, default=0.004)
    ap.add_argument("--noise_max", type=float, default=0.017)
    ap.add_argument("--seed", type=int, default=20260722)
    ap.add_argument("--skip_spectral", action="store_true")
    args = ap.parse_args()

    jt.flags.use_cuda = 1
    jt.set_global_seed(args.seed)
    np.random.seed(args.seed)

    model = HeavyDenoiseFlowTrain(feature_hidden=64)
    load_jittor_state(model, args.ckpt, verbose=False)

    if not args.skip_spectral:
        n = update_lipschitz(model)
        print(f"[spectral] 用平均后的权重重算 {n} 层的谱 scale", flush=True)

    mods = reset_bn(model)
    print(f"[bn] 重置 {len(mods)} 个 BN 层, patch={args.patch_size} "
          f"bs={args.batch_size} batches={args.batches}", flush=True)

    loader = PairedPatchDataset(
        root=args.data_root, patch_size=args.patch_size, num_patches=1,
        noise_min=args.noise_min, noise_max=args.noise_max, aug_rotate=True,
        batch_size=args.batch_size, num_workers=4, shuffle=True)

    model.train()
    seen = 0
    with jt.no_grad():
        for batch in loader:
            # 累积移动平均: momentum = 1/(k+1) 使 running 统计量等于所有 batch 的无偏均值
            mom = 1.0 / (seen + 1)
            for m in mods:
                m.momentum = mom
            noisy = batch["pcl_noisy"]
            seed = batch["seed_pnts"].broadcast(noisy.shape)
            model(noisy - seed)
            seen += 1
            if seen % 20 == 0:
                print(f"  [bn] {seen}/{args.batches}", flush=True)
            if seen >= args.batches:
                break

    # 还原一个常规 momentum, 免得存进 ckpt 的值误导后续续训
    for m in mods:
        m.momentum = 0.1

    save_jittor_state(model, args.out)

    old = pickle.load(open(args.ckpt, "rb"))
    new = pickle.load(open(args.out, "rb"))
    dm = [float(np.abs(np.asarray(new[k]) - np.asarray(old[k])).mean())
          for k in old if "running_mean" in k]
    dv = [float(np.abs(np.asarray(new[k]) - np.asarray(old[k])).mean())
          for k in old if "running_var" in k]
    sm = [float(np.abs(np.asarray(old[k])).mean()) for k in old if "running_mean" in k]
    sv = [float(np.abs(np.asarray(old[k])).mean()) for k in old if "running_var" in k]
    print(f"[bn] running_mean 平均改动 {np.mean(dm):.5f} (原量级 {np.mean(sm):.5f}, "
          f"相对 {100*np.mean(dm)/max(np.mean(sm),1e-12):.1f}%)")
    print(f"[bn] running_var  平均改动 {np.mean(dv):.5f} (原量级 {np.mean(sv):.5f}, "
          f"相对 {100*np.mean(dv)/max(np.mean(sv),1e-12):.1f}%)")
    print(f"[bn] 写出 -> {args.out}")


if __name__ == "__main__":
    main()
