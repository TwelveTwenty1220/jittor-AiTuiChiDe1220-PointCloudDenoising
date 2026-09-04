import argparse
import glob
import os
import sys
import time

import jittor as jt
import numpy as np

jt.flags.use_cuda = 1

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from pdlts_model import HeavyDenoiseFlow
from strict_jittor_denoise import denoise_loop
from run_meta import dump_run_meta, require_path
from strict_jittor_io import load_jittor_state


def valid_existing_output(path, noisy):
    if not os.path.exists(path):
        return False
    try:
        arr = np.load(path, mmap_mode="r")
        return arr.shape == noisy.shape and arr.dtype == np.float32
    except Exception:
        return False


def save_output_atomic(path, arr):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = f"{path}.tmp.{os.getpid()}"
    with open(tmp_path, "wb") as f:
        np.save(f, arr.astype(np.float32))
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--tag", default="strict_jt")
    parser.add_argument("--base", required=True, help="输出根目录; 结果写到 <base>/<tag>/shapenet/<syn>/<mid>/denoised.npy")
    parser.add_argument("--data", required=True, help="官方测试集根目录(含 shapenet/<syn>/<mid>/noisy.npy)")
    parser.add_argument("--patch_size", type=int, default=2048)
    parser.add_argument("--seed_k", type=int, default=3)
    parser.add_argument("--niters", type=int, default=2)
    parser.add_argument("--feature_hidden", type=int, default=64)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0,
                        help="随机种子(推理本身无随机采样, FPS 起点固定; 仅为可复现规范统一设置)")
    args = parser.parse_args()

    jt.set_global_seed(args.seed)
    np.random.seed(args.seed)
    require_path(args.ckpt, "权重文件", "使用 checkpoints/ 下的 .pkl, 或先完成训练与 SWA 步骤")
    require_path(os.path.join(args.data, "shapenet"), "测试集", "在 scripts/00_env.sh 填写 DATASET_TEST(_B), 其下应有 shapenet/<syn>/<mid>/noisy.npy")
    dump_run_meta(os.path.join(args.base, args.tag), args, name="infer-config")

    model = HeavyDenoiseFlow(feature_hidden=args.feature_hidden)
    load_jittor_state(model, args.ckpt, verbose=True, bake_spectral=True)
    model.eval()

    files = sorted(glob.glob(os.path.join(args.data, "shapenet", "*", "*", "noisy.npy")))
    if args.start or args.stride != 1:
        files = files[args.start::args.stride]
    if args.limit:
        files = files[:args.limit]
    if not files:
        sys.stderr.write(f"[无输入] 在 {args.data}/shapenet/*/*/ 下找不到 noisy.npy, 请检查测试集目录结构\n")
        sys.exit(2)
    print(f"[strict-run] {len(files)} files tag={args.tag} ps={args.patch_size} sk={args.seed_k} niters={args.niters}", flush=True)

    start_time = time.time()
    for index, path in enumerate(files, 1):
        parts = path.split(os.sep)
        synset_id, model_id = parts[-3], parts[-2]
        out_dir = os.path.join(args.base, args.tag, "shapenet", synset_id, model_id)
        out_path = os.path.join(out_dir, "denoised.npy")
        noisy = np.load(path).astype(np.float32)
        if valid_existing_output(out_path, noisy):
            continue
        item_start = time.time()
        denoised = denoise_loop(model, noisy, args.patch_size, args.seed_k, args.niters)
        assert denoised.shape == noisy.shape, f"shape {denoised.shape} != {noisy.shape}"
        save_output_atomic(out_path, denoised)
        print(f"  [{index}/{len(files)}] {synset_id}/{model_id} {denoised.shape} ({time.time() - item_start:.1f}s)", flush=True)
    print(f"[strict-run] done in {time.time() - start_time:.1f}s", flush=True)


if __name__ == "__main__":
    main()
