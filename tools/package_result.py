"""把推理输出打包成提交 zip，并做全部硬校验（第 6 步）。

校验项(任一不过即拒绝打包):
  - 模型数量与 --noisy_dir 完全一致(不缺不多)
  - 每个 denoised.npy 的点数与对应 noisy.npy 完全相同(赛题硬约束)
  - dtype 为 float32, 无 NaN/Inf
  - zip 内路径为 shapenet/<synset_id>/<model_id>/denoised.npy

用法:
  python tools/package_result.py \
      --pred_dir <推理输出>/<tag> --noisy_dir <path>/dataset_test_noisy --out result.zip
"""
import argparse
import glob
import os
import zipfile

import numpy as np


def main():
    """打包入口: 校验推理输出与测试集一一对应后写入提交 zip。

    关键参数: --pred_dir 含 shapenet/<syn>/<mid>/denoised.npy 的推理输出目录;
    --noisy_dir 官方测试集根目录; --out 输出 zip 路径(已存在则先删除)。
    校验: 模型集合一致、每个 denoised (N, 3) 与对应 noisy 同形状、float32、无 NaN/Inf。
    zip 内路径为 shapenet/<syn>/<mid>/denoised.npy(ZIP_STORED 不压缩), 打印平均位移。
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_dir", required=True, help="含 shapenet/<syn>/<mid>/denoised.npy")
    ap.add_argument("--noisy_dir", required=True, help="官方测试集根(含 shapenet/.../noisy.npy)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    noisy = sorted(glob.glob(os.path.join(args.noisy_dir, "shapenet", "*", "*", "noisy.npy")))
    preds = sorted(glob.glob(os.path.join(args.pred_dir, "shapenet", "*", "*", "denoised.npy")))
    rel = lambda p, root: os.path.relpath(os.path.dirname(p), root)  # noqa: E731
    need = {rel(p, args.noisy_dir) for p in noisy}
    have = {rel(p, args.pred_dir) for p in preds}
    assert need, "测试集为空?"
    assert need == have, f"模型集不一致: 缺 {sorted(need - have)[:3]} 多 {sorted(have - need)[:3]}"

    shifts = []
    if os.path.exists(args.out):
        os.remove(args.out)
    with zipfile.ZipFile(args.out, "w", zipfile.ZIP_STORED) as z:
        for p in preds:
            r = rel(p, args.pred_dir)
            a = np.load(p)
            x = np.load(os.path.join(args.noisy_dir, r, "noisy.npy"))
            assert a.shape == x.shape, f"{r}: 点数 {a.shape} != 输入 {x.shape}"
            assert a.dtype == np.float32, f"{r}: dtype {a.dtype}"
            assert np.isfinite(a).all(), f"{r}: 含 NaN/Inf"
            shifts.append(float(np.linalg.norm(
                a.astype(np.float64) - x.astype(np.float64), axis=1).mean()))
            z.write(p, os.path.join(r, "denoised.npy"))
    print(f"[package] OK: {len(preds)} 个模型 -> {args.out} "
          f"({os.path.getsize(args.out)} bytes), 平均位移 {np.mean(shifts):.5f}")


if __name__ == "__main__":
    main()
