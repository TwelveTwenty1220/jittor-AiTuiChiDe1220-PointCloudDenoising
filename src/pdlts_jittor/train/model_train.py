"""Trainable Jittor port of PD-LTS light DenoiseFlow.

Reuses the (already numerically-verified) inference feature-extraction blocks
from pdlts_jittor/pdlts_model.py and replaces the three training-sensitive parts:

  1. FCNet  : plain nn.Linear -> live InducedNormLinearJT (spectral norm).
  2. IMonotoneBlock : detached banach fixed-point -> UNROLLED fixed-point so
     Jittor autodiff flows through to both the input and the FCNet parameters.
     (Mathematically equivalent to the implicit gradient at the fixed point.)
  3. ActNorm : add an `initialized` buffer (warm-start sets it to 1 so the
     pretrained weight/bias are kept, never re-initialized from a batch).

Module NAMING is kept byte-identical to the inference model so that:
  - warm-start can load the official light .ckpt, and
  - saved state_dicts (raw weight + scale/u/v buffers) are loadable by the
    inference load_weights.py (which bakes W_eff = W / max(1, sigma/coeff)).
"""
import os
import sys
import math
import jittor as jt
from jittor import nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))  # pdlts_jittor/
# reuse verified inference blocks
from pdlts_model import (  # noqa: E402
    _FEAT_CFG, knn_idx, Swish, PreConv, EdgeConv, FeatMergeUnit, NoiseEdgeConv,
)
from spectral import InducedNormLinearJT  # noqa: E402

# HybridPF long-injection tanh clamp: bound each per-injector perturbation to
# +-_LONG_TANH_SCALE. tanh(0)=0 preserves the zero-init identity start; the
# bounded output stops the forward blow-up that made the un-clamped run diverge.
_LONG_TANH_SCALE = float(os.environ.get("LONG_TANH_SCALE", "0.1"))


# ---------------------------------------------------------------------------
class ActNorm(nn.Module):
    """y = (x + bias) * exp(weight); inverse x = y*exp(-weight) - bias.

    Adds an `initialized` buffer. When warm-started from a trained ckpt the
    buffer is 1 and the data-dependent init is skipped.
    """
    def __init__(self, num_features):
        super().__init__()
        self.num_features = num_features
        self.weight = jt.zeros((num_features,))
        self.bias = jt.zeros((num_features,))
        self.initialized = jt.array(0).stop_grad()

    def execute(self, x):
        if int(self.initialized.item()) == 0:
            with jt.no_grad():
                c = x.shape[2]
                x_t = x.detach().transpose(0, 2).reshape((c, -1))
                batch_mean = x_t.mean(dim=1)
                batch_var = x_t.sqr().mean(dim=1) - batch_mean.sqr()
                batch_var = jt.maximum(batch_var, jt.array(0.2))
                self.bias.assign(-batch_mean)
                self.weight.assign(-0.5 * jt.log(batch_var))
                self.initialized.assign(jt.array(1))
        bias = self.bias.reshape(1, 1, -1)
        weight = self.weight.reshape(1, 1, -1)
        return (x + bias) * jt.exp(weight)

    def inverse(self, y):
        bias = self.bias.reshape(1, 1, -1)
        weight = self.weight.reshape(1, 1, -1)
        return y * jt.exp(-weight) - bias


# ---------------------------------------------------------------------------
class FCNet(nn.Module):
    """Spectral-norm MLP with LIVE InducedNormLinearJT layers.

    Layout matches deflow.FCNet (densenet=False, swish, nhidden=2):
      preact=False: [Lin, Swish, Lin, Swish, Lin]            (indices 0..4)
      preact=True : [Swish, Lin, Swish, Lin, Swish, Lin]     (indices 0..5)
    Naming `nnet.{i}` follows the original sequential layer layout.
    """
    def __init__(self, channel, idim, nhidden, preact, coeff=0.98,
                 sn_atol=1e-3, sn_rtol=1e-3):
        super().__init__()
        layers = []
        if preact:
            layers.append(Swish())
        last = channel
        for _ in range(nhidden):
            layers.append(InducedNormLinearJT(last, idim, bias=True, coeff=coeff,
                                              domain=2, codomain=2,
                                              n_iterations=None,
                                              atol=sn_atol, rtol=sn_rtol))
            layers.append(Swish())
            last = idim
        layers.append(InducedNormLinearJT(last, channel, bias=True, coeff=coeff,
                                          domain=2, codomain=2,
                                          n_iterations=None,
                                          atol=sn_atol, rtol=sn_rtol))
        self.nnet = nn.Sequential(*layers)

    def execute(self, x):
        return self.nnet(x)


# ---------------------------------------------------------------------------
# Banach fixed point of  x = y - G(x)  (mirrors solvers.find_fixed_point, which
# starts x = y - G(y) and caps at i>10 ~ 12 evals).
#
# To save memory/time this finds the fixed point
# WITHOUT gradient (detached), then take ONE more G evaluation WITH gradient:
#     w* = fixed_point(G, y)            # detached, _FP_ITERS evals, no graph
#     w  = y - G(w*.stop_grad())        # one graphed eval -> grad to G params & y
# At the fixed point w==w*, and d w/d(params,y) from this single step equals the
# implicit-function gradient to first order; the param grad path is identical:
# w_proxy = sqrt2*x - nnet(w_value)). This keeps the autograd graph to a single
# FCNet forward per block instead of _FP_ITERS of them.
_FP_ITERS = 12
# Number of TRAILING iterations kept in the autograd graph.
#   _GRAD_ITERS >= _FP_ITERS : full unroll (most accurate, matches "let jt
#       autodiff the fixed-point"; gives true grad to both params and input).
#   _GRAD_ITERS == 1         : detached fixed point + one graphed
#       G eval); same param grad, approximate input grad; far cheaper.
_GRAD_ITERS = int(os.environ.get("FP_GRAD_ITERS", str(_FP_ITERS)))


def _unrolled_find_root(Gnet, y):
    n_detached = max(0, _FP_ITERS - _GRAD_ITERS)
    if n_detached > 0:
        with jt.no_grad():
            x = y - Gnet(y)
            for _ in range(n_detached - 1):
                x = y - Gnet(x)
        x = x.detach()
    else:
        x = y - Gnet(y)
    for _ in range(_GRAD_ITERS):
        x = y - Gnet(x)
    return x


class IMonotoneBlock(nn.Module):
    """Monotone residual flow (differentiable training version).

    forward:  solve w = s2*x - g(w);  y = s2*w - x
    inverse:  solve w = s2*y + g(w);  x = s2*w - y
    """
    def __init__(self, nnet):
        super().__init__()
        self.nnet = nnet

    def execute(self, x):
        s2 = math.sqrt(2)
        w = _unrolled_find_root(lambda z: self.nnet(z), s2 * x)
        return s2 * w - x

    def inverse(self, y):
        s2 = math.sqrt(2)
        w = _unrolled_find_root(lambda z: -self.nnet(z), s2 * y)
        return s2 * w - y


# ---------------------------------------------------------------------------
class FlowAssembly(nn.Module):
    """chain = [IMonotoneBlock, ActNorm, IMonotoneBlock(preact=True), ActNorm]."""
    def __init__(self, channel, idim, nhidden, coeff=0.98, sn_atol=1e-3, sn_rtol=1e-3):
        super().__init__()
        self.chain = nn.ModuleList([
            IMonotoneBlock(FCNet(channel, idim, nhidden, preact=False, coeff=coeff,
                                 sn_atol=sn_atol, sn_rtol=sn_rtol)),
            ActNorm(channel),
            IMonotoneBlock(FCNet(channel, idim, nhidden, preact=True, coeff=coeff,
                                 sn_atol=sn_atol, sn_rtol=sn_rtol)),
            ActNorm(channel),
        ])

    def execute(self, x):
        for layer in self.chain:
            x = layer(x)
        return x

    def inverse(self, y):
        for i in range(len(self.chain) - 1, -1, -1):
            y = self.chain[i].inverse(y)
        return y


# ---------------------------------------------------------------------------
class LongEncoder(nn.Module):
    """HybridPF-style long-range velocity encoder (arXiv:2508.08542).

    Input noisy xyz (B,N,3). A 3-layer Dynamic-EdgeConv stack (3 -> 64 -> 128 ->
    feature_hidden) aggregates a growing neighborhood and produces a long-range
    per-point feature E_L (B,N,feature_hidden). A small MLP velocity head maps
    E_L -> v (B,N,3); at train time v is supervised to approximate (clean - noisy),
    i.e. the near-constant "push toward the clean surface".

    E_L is what conditions every flow step (via DenoiseFlow.long_proj); v is a
    training-only auxiliary supervision that shapes E_L into a surface-pointing
    signal. Reuses the numerically-verified EdgeConv block; concat=False keeps the
    output width exactly equal to the requested out_channel.
    """
    def __init__(self, feature_hidden=64, k=32):
        super().__init__()
        self.k = k
        self.feature_hidden = feature_hidden
        self.conv1 = EdgeConv(3, 64, 64, concat=False)
        self.conv2 = EdgeConv(64, 128, 128, concat=False)
        self.conv3 = EdgeConv(128, feature_hidden, feature_hidden, concat=False)
        self.vel_head = nn.Sequential(
            nn.Linear(feature_hidden, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
        )

    def execute(self, xyz):
        idx = knn_idx(xyz, xyz, self.k)          # spatial graph, reused per layer
        f = self.conv1(xyz, idx)                 # (B,N,64)
        f = self.conv2(f, idx)                   # (B,N,128)
        e_l = self.conv3(f, idx)                 # (B,N,feature_hidden)
        v = self.vel_head(e_l)                   # (B,N,3)
        return e_l, v


# ---------------------------------------------------------------------------
class DenoiseFlowTrain(nn.Module):
    """Trainable PD-LTS DenoiseFlow.

    Defaults to the light config. Passing the heavy config mirrors
    models/model_heavy/denoise.py for one flow in the three-flow stack.

    When ``use_long=True`` a HybridPF long-range branch is attached: a shared
    LongEncoder produces E_L (and a velocity v), and per-injector ``long_proj``
    layers (zero-initialized => identity start) fold E_L additively into the SAME
    inj_f[i] that ``f`` adds and ``g`` subtracts, so reversibility is preserved by
    construction. ``use_long=False`` (default) is byte-identical to the original
    model: no extra params, no changed return type.
    """
    def __init__(self, aug_channel=48, n_injector=12, cut_channel=24,
                 nflow_module=12, num_neighbors=32, idim=64, nhidden=2,
                 feat_cfg="light", coeff=0.98, sn_atol=1e-3, sn_rtol=1e-3,
                 feature_hidden=64, use_long=False, long_k=None):
        super().__init__()
        self.pc_channel = 3
        self.aug_channel = aug_channel
        self.n_injector = n_injector
        self.num_neighbors = num_neighbors
        self.cut_channel = cut_channel
        self.nflow_module = nflow_module
        self.idim = idim
        self.nhidden = nhidden
        self.feature_hidden = feature_hidden
        self.use_long = use_long
        channel = self.pc_channel + self.aug_channel  # 51

        self.noise_params = NoiseEdgeConv(self.pc_channel, 32, self.aug_channel)
        self.PreConv = PreConv(self.pc_channel, 16)
        cfg = _FEAT_CFG[feat_cfg]
        in_channelE = cfg["in_channelE"]
        in_channelA = cfg["in_channelA"]
        out_channel = cfg["out_channel"]
        concat_off = cfg["concat_off"]
        hidden_channel = feature_hidden
        self.feat_Conv = nn.ModuleList()
        self.AdaptConv = nn.ModuleList()
        for i in range(self.n_injector):
            concat = i not in concat_off
            self.feat_Conv.append(EdgeConv(in_channelE[i], hidden_channel, out_channel[i], concat=concat))
            self.AdaptConv.append(FeatMergeUnit(in_channelA[i], hidden_channel, channel))

        self.flow_assemblies = nn.ModuleList([
            FlowAssembly(channel, self.idim, self.nhidden, coeff=coeff,
                         sn_atol=sn_atol, sn_rtol=sn_rtol)
            for _ in range(self.nflow_module)
        ])

        if self.use_long:
            lk = self.num_neighbors if long_k is None else long_k
            self.long_encoder = LongEncoder(feature_hidden=feature_hidden, k=lk)
            self.long_proj = nn.ModuleList()
            for _ in range(self.n_injector):
                proj = nn.Linear(feature_hidden, channel)
                # zero-init => long injection starts at 0 (identity warm gate);
                # denoising path is unchanged until denoise-loss grads open it up.
                proj.weight.assign(jt.zeros_like(proj.weight))
                if proj.bias is not None:
                    proj.bias.assign(jt.zeros_like(proj.bias))
                self.long_proj.append(proj)

    def feat_extract(self, xyz):
        """提取逐层注入特征。输入 xyz: (B, N, 3) patch 局部坐标。

        返回 cs: 长度 n_injector 的列表, 每项 (B, N, pc_channel+aug_channel);
        use_long=True 时返回 (cs, e_l, v), e_l (B, N, feature_hidden), v (B, N, 3)。
        KNN 图(k=num_neighbors)只建一次并复用于全部 EdgeConv 层。
        """
        idx = knn_idx(xyz, xyz, self.num_neighbors)
        f = self.PreConv(xyz, idx)
        cs = []
        for i in range(self.n_injector):
            f = self.feat_Conv[i](f, idx)
            cs.append(self.AdaptConv[i](f))
        if self.use_long:
            e_l, v = self.long_encoder(xyz)
            return cs, e_l, v
        return cs

    def _inject_long(self, cs, e_l):
        """inj_f[i] = cs[i] + long_proj[i](E_L). Built ONCE and passed to both f
        and g, so the f-adds / g-subtracts symmetry (reversibility) is intact."""
        return [cs[i] + _LONG_TANH_SCALE * jt.tanh(self.long_proj[i](e_l)) for i in range(self.n_injector)]

    def unit_coupling(self, xyz):
        idx = knn_idx(xyz, xyz, self.num_neighbors)
        return self.noise_params(xyz, idx)

    def f(self, x, inj_f):
        for i in range(self.nflow_module):
            if i < self.n_injector:
                x = x + inj_f[i]
            x = self.flow_assemblies[i].execute(x)
        return x

    def g(self, z, inj_f):
        for i in range(self.nflow_module - 1, -1, -1):
            z = self.flow_assemblies[i].inverse(z)
            if i < self.n_injector:
                z = z - inj_f[i]
        return z

    def execute(self, x):
        if self.use_long:
            cs, e_l, v = self.feat_extract(x)
            inj_f = self._inject_long(cs, e_l)
        else:
            inj_f = self.feat_extract(x)
        aug = self.unit_coupling(x)
        xc = jt.concat([x, aug], dim=-1)
        z = self.f(xc, inj_f)
        # FBM: zero out the cut channels (channel_mask). Build a new tensor so the
        # in-place style assignment does not break autodiff.
        keep = self.pc_channel + self.aug_channel - self.cut_channel
        z = jt.concat([z[:, :, :keep], jt.zeros_like(z[:, :, keep:])], dim=-1)
        full = self.g(z, inj_f)
        den = full[..., :self.pc_channel]
        if self.use_long:
            return den, v
        return den

    def denoise(self, noisy_pc):
        """单流去噪接口。输入 noisy_pc: (B, N, 3); 返回 (B, N, 3) 去噪坐标(use_long 时丢弃 v)。"""
        out = self.execute(noisy_pc)
        return out[0] if self.use_long else out


def build_heavy_flow_train(feature_hidden=64, use_long=False, long_k=None):
    """构造 heavy 配置的单个可训练流(aug_channel=32, n_injector=10, cut_channel=16)。

    输入 feature_hidden: EdgeConv 隐层宽度; use_long: 是否挂 HybridPF 长程分支;
    long_k: 长程分支 KNN 的 k(None 则用 num_neighbors=32)。返回 DenoiseFlowTrain 实例。
    """
    return DenoiseFlowTrain(aug_channel=32, n_injector=10, cut_channel=16,
                            nflow_module=10, num_neighbors=32, idim=64,
                            nhidden=2, feat_cfg="heavy", feature_hidden=feature_hidden,
                            use_long=use_long, long_k=long_k)


class HeavyDenoiseFlowTrain(nn.Module):
    """Trainable PD-LTS heavy stack.

    execute() returns all three intermediate denoised outputs, matching the
    heavy training contract used by the xloss recipe:
    den[0] -> medium-noise target, den[1] -> small-noise target, den[2] -> clean.

    With ``use_long=True`` every flow carries a HybridPF long-range branch and
    execute() ALSO returns the velocity v of flow[0] (whose input is the original
    noisy patch, so v is supervised against clean - noisy). The other flows still
    learn their long branches through the denoising loss (their long_proj opens up
    from the zero-init identity start). ``use_long=False`` (default) keeps the
    original list-only return so every existing caller is unaffected.
    """
    def __init__(self, feature_hidden=64, use_long=False, long_k=None):
        super().__init__()
        self.feature_hidden = feature_hidden
        self.use_long = use_long
        self.flows = nn.ModuleList([
            build_heavy_flow_train(feature_hidden=feature_hidden, use_long=use_long, long_k=long_k)
            for _ in range(3)
        ])

    def execute(self, x):
        """依次通过三个流。输入 x: (B, N, 3); 返回 outs: 3 个 (B, N, 3) 中间输出的列表。

        use_long=True 时返回 (outs, v0), v0 (B, N, 3) 为 flows[0] 的长程速度分支输出。
        """
        p = x
        outs = []
        v0 = None
        for i, flow in enumerate(self.flows):
            if self.use_long:
                p, v = flow.execute(p)
                if i == 0:
                    v0 = v
            else:
                p = flow.execute(p)
            outs.append(p)
        if self.use_long:
            return outs, v0
        return outs

    def denoise(self, noisy_pc):
        """三级串联去噪, 只返回末级输出。输入 noisy_pc: (B, N, 3); 返回 (B, N, 3)。"""
        out = self.execute(noisy_pc)
        outs = out[0] if self.use_long else out
        return outs[-1]
