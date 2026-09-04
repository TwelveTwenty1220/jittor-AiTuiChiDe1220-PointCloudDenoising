"""Jittor data pipeline mirroring dataset/scoredenoise/dataset.py for OurShapeNet.

Per item (PairedPatchDataset on_the_fly, num_patches controls epoch length):
  pick a clean cloud (50000,3) -> NormalizeUnitSphere -> AddLaplacianNoise(b in
  [noise_min,noise_max]) -> RandomScale([0.8,1.2]) -> RandomRotate(x,y,z) ->
  one random patch: seed = random point of NOISY cloud; patch = K nearest NOISY
  points; the CLEAN patch uses the SAME indices (point-paired). Returns
  pcl_noisy/pcl_clean/seed_pnts in WORLD coords (the trainer subtracts the seed).

All numpy/scipy in the worker (no Jittor) to stay multiprocessing-safe; the
Dataset.collate_batch stacks into Jittor Vars.
"""
import os
import glob
import random
import numpy as np
from scipy.spatial import cKDTree
import jittor as jt
from jittor.dataset import Dataset


def _normalize_unit_sphere(pcl):
    p_max = pcl.max(axis=0, keepdims=True)
    p_min = pcl.min(axis=0, keepdims=True)
    center = (p_max + p_min) / 2
    pcl = pcl - center
    scale = np.sqrt((pcl ** 2).sum(axis=1)).max()
    pcl = pcl / scale
    return pcl


def _rot_matrix(axis, deg):
    a = np.pi * deg / 180.0
    s, c = np.sin(a), np.cos(a)
    if axis == 0:
        return np.array([[1, 0, 0], [0, c, s], [0, -s, c]], dtype=np.float32)
    if axis == 1:
        return np.array([[c, 0, -s], [0, 1, 0], [s, 0, c]], dtype=np.float32)
    return np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]], dtype=np.float32)


class PairedPatchDataset(Dataset):
    def __init__(self, root, dataset="OurShapeNet", resolution="50000_poisson",
                 split="train", patch_size=1024, num_patches=50,
                 noise_min=0.004, noise_max=0.017, aug_rotate=True,
                 batch_size=16, num_workers=8, shuffle=True, file_list=None):
        super().__init__()
        self.pcl_dir = os.path.join(root, dataset, "pointclouds", split, resolution)
        if file_list:
            self.files = []
            with open(file_list, "r", encoding="utf-8") as f:
                for line in f:
                    item = line.strip()
                    if not item or item.startswith("#"):
                        continue
                    if not os.path.isabs(item):
                        item = os.path.join(self.pcl_dir, item)
                    self.files.append(item)
        else:
            self.files = sorted(glob.glob(os.path.join(self.pcl_dir, "*.npy")))
        assert len(self.files) > 0, f"no .npy in {self.pcl_dir}"
        missing = [p for p in self.files if not os.path.exists(p)]
        assert not missing, f"missing files in file_list: {missing[:5]}"
        self.clouds = [np.load(f).astype(np.float32) for f in self.files]
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.noise_min = noise_min
        self.noise_max = noise_max
        self.aug_rotate = aug_rotate
        n_total = len(self.clouds) * num_patches
        src = f" file_list={file_list}" if file_list else ""
        print(f"[data] {len(self.clouds)} clouds x {num_patches} patches = {n_total} items{src}",
              flush=True)
        self.set_attrs(total_len=n_total, batch_size=batch_size,
                       num_workers=num_workers, shuffle=shuffle)

    def __getitem__(self, idx):
        clean = self.clouds[idx % len(self.clouds)]
        # transforms (NormalizeUnitSphere -> Laplace -> scale -> rotate)
        clean = _normalize_unit_sphere(clean)
        b = random.uniform(self.noise_min, self.noise_max)
        noisy = clean + np.random.laplace(0.0, b, size=clean.shape).astype(np.float32)
        sc = random.uniform(0.8, 1.2)
        clean = clean * sc
        noisy = noisy * sc
        if self.aug_rotate:
            for ax in (0, 1, 2):
                R = _rot_matrix(ax, random.uniform(-180.0, 180.0))
                clean = clean @ R
                noisy = noisy @ R
        # one patch: seed = random NOISY point; KNN over NOISY; clean uses same idx
        N = noisy.shape[0]
        seed_i = random.randrange(N)
        tree = cKDTree(noisy)
        _, knn = tree.query(noisy[seed_i], k=self.patch_size)
        knn = np.asarray(knn, dtype=np.int64)
        pat_noisy = noisy[knn]                          # (M,3)
        pat_clean = clean[knn]                          # (M,3) paired
        seed = noisy[seed_i:seed_i + 1]                 # (1,3)
        return (jt.array(pat_noisy), jt.array(pat_clean), jt.array(seed),
                np.float32(b), np.float32(sc))

    def collate_batch(self, batch):
        noisy = jt.stack([b[0] for b in batch], dim=0)  # (B,M,3)
        clean = jt.stack([b[1] for b in batch], dim=0)
        seed = jt.stack([b[2] for b in batch], dim=0)   # (B,1,3)
        noise_std = jt.array(np.array([b[3] for b in batch], dtype=np.float32))
        scale = jt.array(np.array([b[4] for b in batch], dtype=np.float32))
        return {"pcl_noisy": noisy, "pcl_clean": clean, "seed_pnts": seed,
                "pcl_std": noise_std, "scale": scale}
