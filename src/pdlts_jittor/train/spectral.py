"""Jittor implementation of the live spectral-norm linear layer used in
PD-LTS training.

At inference time the upstream repo bakes  W_eff = W / max(1, sigma/coeff)  into a
plain Linear (see pdlts_jittor/load_weights.py). During TRAINING the normalization
must be "live":
  - `weight` is a learnable parameter (raw W),
  - `u`, `v`, `scale` are non-grad buffers,
  - compute_weight(update=True) runs power iteration to refresh u/v (and scale),
  - forward() uses update=False: recompute sigma from the current u/v and return
    W / max(1, sigma/coeff), so the *normalized* weight is what multiplies the
    input and carries gradient to the raw W.

This follows the original compute_weight logic; update_lipschitz() calls
compute_weight(update=True) on every such layer once per optimizer step.
"""
import math
import jittor as jt
from jittor import nn


def _l2_normalize(x, eps=1e-12):
    return x / jt.maximum(jt.sqrt((x * x).sum()), jt.array(eps))


class InducedNormLinearJT(nn.Module):
    """训练期"实时"谱归一化线性层(L2 诱导范数), 对应原仓库 InducedNormLinear。

    参数 in_features/out_features: 输入/输出维; bias: 是否带偏置; coeff: Lipschitz 上限
    系数(前向用 W/max(1, sigma/coeff)); n_iterations: 幂迭代次数(None 则按 atol/rtol
    收敛判据, 上限 200 次); atol/rtol: 幂迭代收敛容差。
    可学习 weight (out, in); 非梯度缓冲 u (out,), v (in,), scale (1,) 存最近一次 sigma。
    """
    def __init__(self, in_features, out_features, bias=True, coeff=0.98,
                 domain=2, codomain=2, n_iterations=None, atol=None, rtol=None,
                 **unused):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.coeff = coeff
        # light config: n_iterations=None, atol=rtol=1e-3 (sn_atol/sn_rtol)
        self.n_iterations = n_iterations
        self.atol = atol
        self.rtol = rtol
        assert domain == 2 and codomain == 2, "light uses L2/L2 induced norm"

        # raw weight (learnable), shape (out, in).
        # Kaiming-uniform placeholder (warm-start overwrites this anyway).
        bound = math.sqrt(6.0 / ((1 + 5) * in_features))  # a=sqrt(5) kaiming_uniform
        self.weight = (jt.rand((out_features, in_features)) * 2 - 1) * bound
        if bias:
            b_bound = 1.0 / math.sqrt(in_features)
            b = (jt.rand((out_features,)) * 2 - 1) * b_bound
            self.bias = b
        else:
            self.bias = None

        # buffers (non-grad). Initialized random-normal then normalized.
        u = jt.randn((out_features,)).stop_grad()
        v = jt.randn((in_features,)).stop_grad()
        self.u = _l2_normalize(u).stop_grad()
        self.v = _l2_normalize(v).stop_grad()
        self.scale = jt.zeros((1,)).stop_grad()

    def compute_weight(self, update=True, n_iterations=None):
        """返回谱归一化后的权重 W / max(1, sigma/coeff), 形状 (out, in)。

        输入 update: True 时先做幂迭代刷新 u/v 缓冲(无梯度), False 只用当前 u/v;
        n_iterations: 覆盖本层默认迭代次数。sigma = u^T W v 对 W 保留梯度,
        同时把 sigma 写入 scale 缓冲(推理端烘焙权重时读取)。
        """
        u = self.u
        v = self.v
        weight = self.weight

        if update:
            n_iterations = self.n_iterations if n_iterations is None else n_iterations
            atol, rtol = self.atol, self.rtol
            max_itrs = 200 if n_iterations is None else n_iterations
            use_tol = (n_iterations is None and atol is not None and rtol is not None)

            with jt.no_grad():
                w = weight.detach()
                u_ = u.detach()
                v_ = v.detach()
                for _ in range(max_itrs):
                    if use_tol:
                        old_u = u_.clone()
                        old_v = v_.clone()
                    # u = normalize(W v); v = normalize(W^T u)
                    u_ = _l2_normalize(jt.matmul(w, v_))
                    v_ = _l2_normalize(jt.matmul(w.transpose(0, 1), u_))
                    if use_tol:
                        err_u = jt.sqrt(((u_ - old_u) ** 2).sum()) / (u_.numel() ** 0.5)
                        err_v = jt.sqrt(((v_ - old_v) ** 2).sum()) / (v_.numel() ** 0.5)
                        tol_u = atol + rtol * u_.max()
                        tol_v = atol + rtol * v_.max()
                        if float(err_u.item()) < float(tol_u.item()) and \
                           float(err_v.item()) < float(tol_v.item()):
                            break
                self.u.assign(u_)
                self.v.assign(v_)
                u = u_
                v = v_

        # sigma = u^T W v  -- with gradient to W (u,v are constants here)
        sigma = (u.detach() * jt.matmul(weight, v.detach())).sum()
        self.scale.assign(sigma.detach().reshape(1))
        factor = jt.maximum(jt.array(1.0), sigma / self.coeff)
        return weight / factor

    def execute(self, x):
        """线性变换 y = x @ W_norm^T + b。输入 x: (..., in_features); 返回 (..., out_features)。"""
        weight = self.compute_weight(update=False)
        # x: (..., in_features) -> (..., out_features)
        y = jt.matmul(x, weight.transpose(0, 1))
        if self.bias is not None:
            y = y + self.bias
        return y
