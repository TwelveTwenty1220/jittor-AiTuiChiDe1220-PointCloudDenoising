import argparse
import os
import sys
import time

import jittor as jt
import numpy as np

jt.flags.use_cuda = 1

HERE = os.path.dirname(os.path.abspath(__file__))
PDLTS_JT = os.path.dirname(HERE)
sys.path.insert(0, PDLTS_JT)
sys.path.insert(0, HERE)

from data import PairedPatchDataset
from emd_jt import EMDLoss
from losses_jt import chamfer_distance_unit_sphere, coverage_loss, infocd_unit_sphere, paired_l2_unit_sphere, repulsion_loss, sinkhorn_unit_sphere, sliced_wasserstein_unit_sphere
from model_train import HeavyDenoiseFlowTrain
from spectral import InducedNormLinearJT
from run_meta import dump_run_meta, require_path
from strict_jittor_io import load_jittor_state, save_jittor_state


def init_noise_edgeconv_linear5(model):
    """Match the reference noiseEdgeConv.linear5 initialization."""
    count = 0
    for module in model.modules():
        if all(hasattr(module, name) for name in ("linear1", "linear2", "linear3", "linear4", "linear5")):
            module.linear5.weight.assign(jt.randn(module.linear5.weight.shape) * 0.05)
            if getattr(module.linear5, "bias", None) is not None:
                module.linear5.bias.assign(jt.zeros_like(module.linear5.bias))
            count += 1
    return count


def update_lipschitz(model):
    count = 0
    for module in model.modules():
        if isinstance(module, InducedNormLinearJT):
            module.compute_weight(update=True)
            count += 1
    return count


def emd_subset_indices(num_points, max_points, mode="first"):
    if max_points <= 0 or num_points <= max_points:
        return None
    if mode == "first":
        return np.arange(max_points, dtype=np.int64)
    if mode == "even":
        return np.linspace(0, num_points - 1, max_points).round().astype(np.int64)
    if mode == "random":
        return np.sort(np.random.choice(num_points, size=max_points, replace=False)).astype(np.int64)
    raise ValueError(f"unknown emd subset mode: {mode}")


def subset_for_emd(x, indices):
    if indices is None:
        return x
    return x[:, indices, :]


def emd_subset_scale(num_points, indices, mode="none"):
    if mode == "none" or indices is None:
        return 1.0
    if mode == "inverse_fraction":
        return float(num_points) / float(len(indices))
    raise ValueError(f"unknown emd subset scale mode: {mode}")


def build_emd_loss(workers=8, reorder_noisy=False):
    return EMDLoss(workers=workers, reorder_noisy=reorder_noisy)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True, help="tools/prepare_train_npy.py 生成的训练 npy 根目录")
    parser.add_argument("--train_list", default="", help="Optional newline-delimited .npy list for train subset.")
    parser.add_argument("--max_epochs", type=int, default=24)
    parser.add_argument("--save_every", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_patches", type=int, default=1)
    parser.add_argument("--train_patch_size", type=int, default=1024)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--noise_min", type=float, default=0.004)
    parser.add_argument("--noise_max", type=float, default=0.017)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--emd_w", type=float, default=0.1)
    parser.add_argument("--w_cd", type=float, default=40.0)
    parser.add_argument("--w_rep", type=float, default=40.0)
    parser.add_argument("--w_cover", type=float, default=0.0)
    parser.add_argument("--w_infocd", type=float, default=0.0)
    parser.add_argument("--w_sink", type=float, default=0.0)
    parser.add_argument("--w_swd", type=float, default=0.0)
    # 逐点配对 L2: 数据里 pcl_clean 与 pcl_noisy 同索引配对(data.py:98), 真实曲面位置精确已知,
    # 但主输出的所有损失都是集合损失, 丢掉了对应关系 -> 最近邻/OT 匹配带 ~半点间距(0.0025) 歧义,
    # 恰等于实测残余离面 0.0022. 单位与 w_cd 对齐, 故 w_pair 起点取 w_cd 同尺度.
    parser.add_argument("--w_pair", type=float, default=0.0)
    parser.add_argument("--swd_max", type=int, default=0)  # 1=max-sliced(最差投影), 0=mean
    parser.add_argument("--w_long", type=float, default=0.0)  # HybridPF long-range velocity supervision
    parser.add_argument("--long_k", type=int, default=0)  # KNN for LongEncoder; 0 -> num_neighbors
    parser.add_argument("--sink_eps", type=float, default=0.01)
    parser.add_argument("--sink_iters", type=int, default=30)
    parser.add_argument("--r0", type=float, default=0.05)
    parser.add_argument("--feature_hidden", type=int, default=64)
    parser.add_argument("--emd_workers", type=int, default=2)
    parser.add_argument("--emd_reorder_noisy", action="store_true")
    parser.add_argument("--emd_max_points", type=int, default=0)
    parser.add_argument("--emd_subset_mode", choices=("first", "even", "random"), default="first")
    parser.add_argument("--emd_subset_scale", choices=("none", "inverse_fraction"), default="none")
    parser.add_argument("--resume_jittor", default="", help="Optional Jittor .pkl checkpoint to continue from.")
    parser.add_argument("--start_epoch", type=int, default=0, help="Epoch offset for resumed checkpoint names/logs.")
    parser.add_argument("--out_dir", default=os.path.join(HERE, "runs_strict_jt_heavy_xloss_scratch"))
    parser.add_argument("--tag", default="strict-jt-heavy-xloss-scratch")
    parser.add_argument("--seed", type=int, default=2023)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--max_steps", type=int, default=0)
    args = parser.parse_args()

    jt.set_global_seed(args.seed)
    np.random.seed(args.seed)
    require_path(args.data_root, "训练数据", "先运行 scripts/01_prepare_data.sh 生成 npy, 或用 --data_root 指定其目录")
    if args.resume_jittor:
        require_path(args.resume_jittor, "起点权重", "检查 --resume_jittor 路径(见 scripts/03、10 脚本的依赖顺序)")
    dump_run_meta(args.out_dir, args, name=f"{args.tag}-config")

    use_long = args.w_long > 0
    model = HeavyDenoiseFlowTrain(feature_hidden=args.feature_hidden,
                                  use_long=use_long, long_k=(args.long_k or None))
    n_linear5 = init_noise_edgeconv_linear5(model)
    n_spectral = update_lipschitz(model)
    if args.resume_jittor:
        loaded, missing, extra = load_jittor_state(model, args.resume_jittor, verbose=True)
        if missing:
            # HybridPF: warm from 非-long ckpt 时, 长程分支(long_encoder/long_proj)的键允许 missing,
            # 走新初始化(long_proj 零初始化=恒等起步, 不改初始去噪行为)。非长程键 missing 仍报错。
            long_missing = [k for k in missing if "long" in k.lower()]
            hard_missing = [k for k in missing if "long" not in k.lower()]
            if hard_missing:
                raise RuntimeError(f"resume_jittor missing {len(hard_missing)} non-long keys")
            print(f"[HybridPF] warm from non-long ckpt: {len(long_missing)} long-branch keys new-init (恒等起步)", flush=True)
        print(
            f"[strict] resumed from Jittor checkpoint: {args.resume_jittor} "
            f"loaded={len(loaded)} extra={len(extra)}",
            flush=True,
        )
    else:
        print("[strict] scratch init: no warm_start, no imported checkpoint", flush=True)
    print(f"[strict] init parity: linear5={n_linear5} spectral_preupdate={n_spectral}", flush=True)
    model.train()

    optimizer = jt.optim.Adam(model.parameters(), lr=args.lr)
    emd = build_emd_loss(workers=args.emd_workers, reorder_noisy=args.emd_reorder_noisy)
    loader = PairedPatchDataset(
        root=args.data_root,
        patch_size=args.train_patch_size,
        num_patches=args.num_patches,
        noise_min=args.noise_min,
        noise_max=args.noise_max,
        aug_rotate=True,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        file_list=args.train_list or None,
    )

    print(
        f"[cfg] strict_jt epochs={args.max_epochs} bs={args.batch_size} "
        f"patch={args.train_patch_size} emd_points={args.emd_max_points} "
        f"emd_subset={args.emd_subset_mode} emd_subset_scale={args.emd_subset_scale} "
        f"emd_reorder_noisy={args.emd_reorder_noisy} "
        f"num_patches={args.num_patches} lr={args.lr} emd_w={args.emd_w} "
        f"w_cd={args.w_cd} w_rep={args.w_rep} w_cover={args.w_cover} r0={args.r0} w_pair={args.w_pair} "
        f"w_long={args.w_long} long_k={args.long_k or 'num_neighbors'} use_long={use_long} "
        f"feature_hidden={args.feature_hidden} train_list={args.train_list or 'FULL'} "
        f"resume_jittor={args.resume_jittor or 'NONE'} start_epoch={args.start_epoch} "
        f"FP_GRAD_ITERS={os.environ.get('FP_GRAD_ITERS','12')}",
        flush=True,
    )

    global_step = 0
    for epoch in range(args.max_epochs):
        epoch_id = args.start_epoch + epoch + 1
        epoch_losses = []
        epoch_start = time.time()
        for batch in loader:
            step_start = time.time()
            noisy = batch["pcl_noisy"]
            clean = batch["pcl_clean"]
            seed = batch["seed_pnts"]
            std = batch["pcl_std"]
            scale = batch["scale"]

            seed_rep = seed.broadcast(noisy.shape)
            noisy = noisy - seed_rep
            clean = clean - seed_rep

            sig0 = (std * scale / 5.3).reshape((-1, 1, 1))
            sig1 = (std * scale / (5.3 * 5.3)).reshape((-1, 1, 1))
            target_mid = clean + jt.randn_like(clean) * sig0
            target_small = clean + jt.randn_like(clean) * sig1

            if use_long:
                denoised, vel = model(noisy)
                # flow[0] velocity supervised to the full push toward clean surface.
                long_loss = ((vel - (clean - noisy)) ** 2).mean()
            else:
                denoised = model(noisy)
                long_loss = jt.array(0.0)
            emd_idx = emd_subset_indices(noisy.shape[1], args.emd_max_points, args.emd_subset_mode)
            emd_scale = emd_subset_scale(noisy.shape[1], emd_idx, args.emd_subset_scale)
            emd_loss = (
                emd(subset_for_emd(denoised[0], emd_idx),
                    subset_for_emd(target_mid, emd_idx),
                    subset_for_emd(noisy, emd_idx))
                + emd(subset_for_emd(denoised[1], emd_idx),
                      subset_for_emd(target_small, emd_idx),
                      subset_for_emd(noisy, emd_idx))
                + emd(subset_for_emd(denoised[2], emd_idx),
                      subset_for_emd(clean, emd_idx),
                      subset_for_emd(noisy, emd_idx))
            ) * emd_scale
            cd = chamfer_distance_unit_sphere(denoised[2], clean)
            rep = repulsion_loss(denoised[2], r0=args.r0) if args.w_rep > 0 else jt.array(0.0)
            cov = coverage_loss(clean, denoised[2]) if args.w_cover > 0 else jt.array(0.0)
            icd = infocd_unit_sphere(denoised[2], clean) if args.w_infocd > 0 else jt.array(0.0)
            snk = sinkhorn_unit_sphere(denoised[2], clean, eps=args.sink_eps, iters=args.sink_iters) if args.w_sink > 0 else jt.array(0.0)
            swd = sliced_wasserstein_unit_sphere(denoised[2], clean, use_max=bool(args.swd_max)) if args.w_swd > 0 else jt.array(0.0)
            pair = paired_l2_unit_sphere(denoised[2], clean) if args.w_pair > 0 else jt.array(0.0)
            loss = (args.emd_w * emd_loss + args.w_cd * cd + args.w_rep * rep
                    + args.w_cover * cov + args.w_infocd * icd + args.w_sink * snk
                    + args.w_swd * swd + args.w_long * long_loss
                    + args.w_pair * pair)

            optimizer.zero_grad()
            optimizer.backward(loss)
            optimizer.clip_grad_norm(1e-3)
            optimizer.step()
            update_lipschitz(model)

            loss_value = float(loss.item())
            epoch_losses.append(loss_value)
            global_step += 1
            if global_step <= 5 or global_step % args.log_every == 0:
                print(
                    f"  ep{epoch_id} step{global_step} loss={loss_value:.5f} "
                    f"emd*w={float((args.emd_w * emd_loss).item()):.5f} "
                    f"cd*w={float((args.w_cd * cd).item()):.5f} "
                    f"rep*w={float((args.w_rep * rep).item()):.5f} "
                    f"cover*w={float((args.w_cover * cov).item()):.5f} "
                    f"icd*w={float((args.w_infocd * icd).item()):.5f} "
                    f"snk*w={float((args.w_sink * snk).item()):.5f} "
                    f"swd*w={float((args.w_swd * swd).item()):.5f} "
                    f"pair*w={float((args.w_pair * pair).item()):.5f} "
                    f"long*w={float((args.w_long * long_loss).item()):.5f} "
                    f"({time.time() - step_start:.2f}s/step)",
                    flush=True,
                )
            if args.max_steps and global_step >= args.max_steps:
                save_jittor_state(model, os.path.join(args.out_dir, f"{args.tag}-debug.pkl"))
                return

        print(
            f"[epoch {epoch_id}] mean_loss={np.mean(epoch_losses):.5f} "
            f"({time.time() - epoch_start:.1f}s)",
            flush=True,
        )
        if (epoch + 1) % args.save_every == 0:
            save_jittor_state(model, os.path.join(args.out_dir, f"{args.tag}-ep{epoch_id}.pkl"))

    save_jittor_state(model, os.path.join(args.out_dir, f"{args.tag}-final.pkl"))


if __name__ == "__main__":
    main()
