#!/usr/bin/env python
"""Average N jittor .pkl checkpoints (SWA). Unpickling may import jittor (needs g++
in PATH -> run with conda env bin prepended). Values converted to numpy before averaging.

Usage: python make_swa_ckpt.py out.pkl in1.pkl in2.pkl [...]
Float arrays are averaged; non-float / non-array values taken from the first ckpt.
"""
import sys, pickle
import numpy as np


def to_np(v):
    """把 ckpt 中的值转成 numpy: ndarray 原样返回, 带 .numpy() 的(jt.Var)转换, 其余原样返回。"""
    if isinstance(v, np.ndarray):
        return v
    if hasattr(v, "numpy"):
        return v.numpy()
    return v


out, ins = sys.argv[1], sys.argv[2:]
assert len(ins) >= 2, "need >=2 ckpts"
dicts = []
for p in ins:
    with open(p, "rb") as f:
        d = pickle.load(f)
    dicts.append({k: to_np(v) for k, v in d.items()})
keys = list(dicts[0].keys())
for d in dicts[1:]:
    assert list(d.keys()) == keys, "key mismatch between ckpts"
avg, n_avg, n_copy = {}, 0, 0
for k in keys:
    v0 = dicts[0][k]
    if isinstance(v0, np.ndarray) and np.issubdtype(v0.dtype, np.floating):
        avg[k] = np.mean([d[k].astype(np.float64) for d in dicts], axis=0).astype(v0.dtype)
        n_avg += 1
    else:
        avg[k] = v0
        n_copy += 1
with open(out, "wb") as f:
    pickle.dump(avg, f)
print(f"SWA over {len(ins)} ckpts -> {out}  (averaged={n_avg} copied={n_copy})")
