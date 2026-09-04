"""从官方 dataset_train 网格重建训练点云 npy（数据准备，第 1 步）。

官方训练集: dataset_train/shapenet/<synset_id>/<model_id>/models/model_normalized.obj
输出布局  : <out>/OurShapeNet/pointclouds/<split>/50000_poisson/<synset_id>__<model_id>.npy
            （目录命名沿用 PD-LTS 开源代码的数据布局，便于 train/data.py 直接读取；
              实际采样方式为**面积加权均匀表面采样**，并非泊松盘 —— 目录名只是历史命名。）

说明:
  - 本比赛除官方提供的 ShapeNet 数据外不使用任何外部数据。本脚本的唯一输入就是
    官方 dataset_train 的网格；训练所用的全部 npy 均可由本脚本重建。
  - --heldout_list 给出的形状(每行 "synset_id/model_id")不会写入 train split，
    用于本地验证，保证验证集与训练集零交集。
  - 少数退化网格(无面片/加载失败)会被跳过并在结束时列出。
  - 可断点续跑：已存在且形状正确的 npy 会跳过。

用法:
  python tools/prepare_train_npy.py \
      --dataset_train <path>/dataset_train --out <path>/data \
      [--points 50000] [--split train] [--heldout_list heldout_151.txt] [--workers 8]
"""
import argparse
import glob
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import trimesh


def sample_one(job):
    obj_path, out_path, n_points = job
    try:
        mesh = trimesh.load(obj_path, force="mesh", process=False)
        if mesh.faces is None or len(mesh.faces) == 0:
            return (obj_path, "no_faces")
        pts, _ = trimesh.sample.sample_surface(mesh, n_points)  # 面积加权均匀采样
        pts = np.asarray(pts, dtype=np.float32)
        if pts.shape != (n_points, 3) or not np.isfinite(pts).all():
            return (obj_path, "bad_sample")
        tmp = out_path + ".tmp.npy"
        np.save(tmp, pts)
        os.replace(tmp, out_path)
        return None
    except Exception as e:  # noqa: BLE001
        return (obj_path, repr(e)[:80])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_train", required=True,
                    help="官方训练集根目录(含 shapenet/<syn>/<mid>/models/model_normalized.obj)")
    ap.add_argument("--out", required=True, help="数据输出根目录(即训练脚本的 --data_root)")
    ap.add_argument("--points", type=int, default=50000)
    ap.add_argument("--split", default="train")
    ap.add_argument("--heldout_list", default=None,
                    help="可选: 每行 synset_id/model_id, 这些形状不写入该 split(用作本地验证)")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    held = set()
    if args.heldout_list and os.path.isfile(args.heldout_list):
        for line in open(args.heldout_list, encoding="utf-8"):
            line = line.strip()
            if line and not line.startswith("#"):
                held.add(tuple(line.split("/")[:2]))
        print(f"[prepare] held-out 排除 {len(held)} 个形状")

    objs = sorted(glob.glob(os.path.join(
        args.dataset_train, "shapenet", "*", "*", "models", "model_normalized.obj")))
    assert objs, f"未在 {args.dataset_train} 下找到官方网格"
    out_dir = os.path.join(args.out, "OurShapeNet", "pointclouds", args.split, "50000_poisson")
    os.makedirs(out_dir, exist_ok=True)

    jobs = []
    skipped_held = skipped_done = 0
    for obj in objs:
        parts = obj.split(os.sep)
        syn, mid = parts[-4], parts[-3]
        if (syn, mid) in held:
            skipped_held += 1
            continue
        out_path = os.path.join(out_dir, f"{syn}__{mid}.npy")
        if os.path.isfile(out_path):
            try:
                if np.load(out_path, mmap_mode="r").shape == (args.points, 3):
                    skipped_done += 1
                    continue
            except Exception:  # noqa: BLE001
                pass
        jobs.append((obj, out_path, args.points))
    print(f"[prepare] 官方网格 {len(objs)} | held-out 排除 {skipped_held} | "
          f"已完成跳过 {skipped_done} | 待采样 {len(jobs)}")

    failures = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, r in enumerate(ex.map(sample_one, jobs, chunksize=8), 1):
            if r is not None:
                failures.append(r)
            if i % 500 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)} (失败 {len(failures)})", flush=True)

    total = len(glob.glob(os.path.join(out_dir, "*.npy")))
    print(f"[prepare] 完成: {out_dir} 共 {total} 个 npy")
    if failures:
        print(f"[prepare] {len(failures)} 个网格采样失败(退化网格, 训练时自然缺席):")
        for p, why in failures[:20]:
            print(f"    {why}  {p}")


if __name__ == "__main__":
    main()
