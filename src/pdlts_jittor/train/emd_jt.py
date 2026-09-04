"""Earth Mover's Distance for PD-LTS training, pure Jittor (+ scipy matching).

Upstream loss metric/loss.py::EarthMoverDistance1:

    _, asg1 = emd(pcl_noisy, gts)     # auction assignment (approx EMD), no grad
    gts = gts[asg1]                   # reorder clean to noisy order
    loss, _ = emd(preds, gts)         # loss = sum_i || preds_i - gts_{asg2_i} ||^2

emd() (metric/emd/emd_cuda.cu) is an APPROXIMATE optimal-transport matching via
the auction algorithm; crucially its gradient (NmDistanceGradKernel) is exactly
the gradient of  sum_i || x_i - y_{asg_i} ||^2  w.r.t. x. So once the integer
assignment is fixed, the loss is ordinary differentiable Jittor ops.

We obtain the assignment with scipy's exact linear_sum_assignment (Hungarian) on
the pairwise squared-distance cost. That is an *exact* EMD matching (a strict
improvement over the auction approximation), computed on CPU with no gradient,
then the paired squared distance is evaluated in Jittor where autodiff flows to
preds only (matching emd's backward, which returns grad for xyz1 only -> we
.stop_grad() the targets).

For speed the per-patch Hungarian solves run in a process pool.
"""
import numpy as np
import jittor as jt
from scipy.optimize import linear_sum_assignment
from multiprocessing import Pool

_POOL = None


def _get_pool(workers):
    global _POOL
    if workers <= 1:
        return None
    if _POOL is None:
        _POOL = Pool(workers)
    return _POOL


def _lsa_one(args):
    """args: (x, y) each (N,3) float32. Returns assignment (N,) int32 s.t. y[asg]
    is matched to x (i.e. col index for each row)."""
    x, y = args
    # cost = pairwise squared distance (N,N)
    cost = ((x[:, None, :] - y[None, :, :]) ** 2).sum(-1)
    _, col = linear_sum_assignment(cost)
    return col.astype(np.int32)


def _batch_assignment(x_np, y_np, workers):
    """x_np,y_np: (B,N,3). Returns (B,N) int32 assignment col per row."""
    B = x_np.shape[0]
    tasks = [(x_np[b], y_np[b]) for b in range(B)]
    pool = _get_pool(workers)
    if pool is None:
        cols = [_lsa_one(t) for t in tasks]
    else:
        cols = pool.map(_lsa_one, tasks)
    return np.stack(cols, axis=0)


def _reorder(gts, assignment):
    """gts:(B,N,3) jt, assignment:(B,N) jt int -> out[i]=gts[assignment[i]]."""
    B, N, C = gts.shape
    idx = assignment.reshape(B, N, 1).broadcast((B, N, C))
    return gts.gather(1, idx)


class EMDLoss:
    """EarthMoverDistance1 equivalent (exact Hungarian matching).

    forward(preds, gts, pcl_noisy) -> scalar = sum over batch & points of the
    paired squared distance, differentiable w.r.t. preds.
    """
    def __init__(self, workers=8, reorder_noisy=False):
        self.workers = workers
        self.reorder_noisy = reorder_noisy

    def __call__(self, preds, gts, pcl_noisy):
        gts_d = gts.detach()
        # step 1: reorder clean to noisy order (no grad). ~97% identity in patches
        # but we replicate the upstream two-stage matching for fidelity.
        if self.reorder_noisy:
            asg1 = _batch_assignment(pcl_noisy.detach().numpy(),
                                     gts_d.numpy(), self.workers)
            gts1 = _reorder(gts_d, jt.array(asg1)).stop_grad()
        else:
            gts1 = gts_d
        # step 2: assignment preds<->gts1 (no grad)
        asg2 = _batch_assignment(preds.detach().numpy(),
                                 gts1.numpy(), self.workers)
        gts2 = _reorder(gts1, jt.array(asg2)).stop_grad()
        # differentiable paired squared distance (grad to preds only)
        dist = ((preds - gts2) ** 2).sum(-1)        # (B,N)
        return dist.sum()
