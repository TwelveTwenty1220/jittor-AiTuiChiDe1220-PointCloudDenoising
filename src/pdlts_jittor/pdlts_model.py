"""Jittor implementation of the PD-LTS *light* (DenoiseFlow) inference path.

Only the forward (x->z) and inverse (z->x) of each flow layer are needed for
denoising; all log-det / NLL / Neumann machinery is training-only and omitted.

Spectral-norm linear layers are baked: at inference the effective weight is
    W_eff = weight / max(1, sigma / coeff)
where sigma is the stored `scale` buffer. We therefore load plain Linear layers
with the baked weight computed at load time.

Layer correspondence to the original repo:
  - PreConv / EdgeConv / FeatMergeUnit / noiseEdgeConv : models/model_light/layer.py
  - iMonotoneBlock (monotone residual flow) : models/layers/iMonotoneBlock.py
  - ActNorm : models/layers/normalize.py
  - InvertibleLinear : models/layers/glow.py  (unused in light cfg's chain)
  - Swish activation : models/layers/base/activations.py
  - RootFind banach fixed point : models/layers/solvers.py
"""
import math
import os
import numpy as np
import jittor as jt
from jittor import nn

# HybridPF long-injection tanh clamp (must match train.model_train._LONG_TANH_SCALE).
# Bounds each per-injector perturbation to +-_LONG_TANH_SCALE; tanh(0)=0 keeps the
# zero-init identity start so a non-long ckpt still loads byte-identically.
_LONG_TANH_SCALE = float(os.environ.get("LONG_TANH_SCALE", "0.1"))


# ----------------------------------------------------------------------------
def knn_idx(query, source, k):
    """Return indices (B, M, k) of the k nearest source points for each query.

    Mirrors KNN(query, source, K).idx using jt.misc.knn,
    which has signature knn(unknown, known, k) -> (dist, idx).
    """
    if query.shape[-1] == 3:
        _, idx = jt.misc.knn(query, source, k)
    else:
        dist = ((query.unsqueeze(2) - source.unsqueeze(1)) ** 2).sum(-1)
        _, idx = jt.topk(dist, k=k, dim=-1, largest=False)
    return idx  # (B, M, k) int


def knn_gather(feat, idx):
    """feat: (B, N, C), idx: (B, M, k) -> (B, M, k, C). Mirrors knn_gather."""
    B, N, C = feat.shape
    _, M, k = idx.shape
    idx_b = jt.arange(B).reshape(B, 1)
    flat = idx.reshape(B, M * k)
    y = feat[idx_b, flat]            # (B, M*k, C)
    return y.reshape(B, M, k, C)


# ----------------------------------------------------------------------------
# Activations
class Swish(nn.Module):
    """forward: (x * sigmoid(x * softplus(beta))) / 1.1  (beta learnable)."""
    def __init__(self):
        super().__init__()
        self.beta = jt.array([0.5])

    def execute(self, x):
        return (x * jt.sigmoid(x * nn.softplus(self.beta))) / 1.1


# ----------------------------------------------------------------------------
# Feature extraction blocks (from layer.py)
class PreConv(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        in_channel = in_channel * 2
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=(1, 1)),
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(0.05),
        )

    def execute(self, f, idx):
        # f: (B, N, C), idx: (B, N, k)
        knn_feat = knn_gather(f, idx)               # (B, N, k, 3)
        f_tiled = f.unsqueeze(2).broadcast(knn_feat.shape)
        x = jt.concat([f_tiled, knn_feat - f_tiled], dim=-1)  # (B,N,k,6)
        x = x.permute(0, 3, 1, 2)                    # (B,6,N,k)
        x = self.conv(x)
        x = x.max(dim=-1)                            # (B,out,N)
        x = x.transpose(1, 2)                        # (B,N,out)
        return x


class EdgeConv(nn.Module):
    def __init__(self, in_channel, hidden_channel, out_channel, concat=True):
        super().__init__()
        self.concat = concat
        if not concat:
            hidden_channel = hidden_channel + 32
        self.convs = nn.ModuleList()
        self.convs.append(nn.Sequential(
            nn.Conv2d(in_channel * 2, hidden_channel, kernel_size=(1, 1)),
            nn.BatchNorm2d(hidden_channel),
            nn.LeakyReLU(0.05),
        ))
        self.convs.append(nn.Sequential(
            nn.Conv2d(hidden_channel, out_channel, kernel_size=(1, 1), bias=True),
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(0.05),
        ))

    def execute(self, f, idx):
        knn_feat = knn_gather(f, idx)               # (B,N,k,C)
        f_tiled = f.unsqueeze(2).broadcast(knn_feat.shape)
        x = jt.concat([f_tiled, knn_feat - f_tiled], dim=-1)  # (B,N,k,2C)
        x = x.permute(0, 3, 1, 2)                    # (B,2C,N,k)
        for conv in self.convs:
            x = conv(x)
        x = x.max(dim=-1)                            # (B,out,N)
        x = x.transpose(1, 2)                        # (B,N,out)
        if self.concat:
            x = jt.concat([x, f], dim=-1)
        return x


class FeatMergeUnit(nn.Module):
    def __init__(self, in_channel, hidden_channel, out_channel):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(nn.Sequential(
            nn.Conv1d(in_channel, hidden_channel, kernel_size=1),
            nn.BatchNorm1d(hidden_channel),
            nn.ReLU(),
        ))
        self.convs.append(nn.Sequential(
            nn.Conv1d(hidden_channel, out_channel, kernel_size=1),
            nn.BatchNorm1d(out_channel),
            nn.ReLU(),
        ))

    def execute(self, x):
        x = x.transpose(1, 2)                        # (B,C,N)
        for conv in self.convs:
            x = conv(x)
        x = x.transpose(1, 2)                        # (B,N,C)
        return x


class noiseEdgeConv(nn.Module):
    def __init__(self, in_channel, hidden_channel, out_channel):
        super().__init__()
        self.linear1 = nn.Linear(in_channel * 2, hidden_channel)
        self.linear2 = nn.Linear(hidden_channel, hidden_channel)
        self.linear3 = nn.Linear(in_channel, hidden_channel)
        self.linear4 = nn.Linear(hidden_channel, hidden_channel)
        self.linear5 = nn.Linear(hidden_channel, out_channel)

    def execute(self, f, idx):
        knn_feat = knn_gather(f, idx)               # (B,N,k,C)
        f_tiled = f.unsqueeze(2).broadcast(knn_feat.shape)
        x = jt.concat([knn_feat, knn_feat - f_tiled], dim=-1)
        x = nn.relu(self.linear1(x))
        x = nn.relu(self.linear2(x))
        x = x.max(dim=2)                            # (B,N,h)
        ff = nn.relu(self.linear3(f))
        ff = nn.relu(self.linear4(ff))
        x = x + ff
        x = self.linear5(x)
        return x


# ----------------------------------------------------------------------------
# Flow layers
class ActNorm(nn.Module):
    """y = (x + bias) * exp(weight); inverse x = y*exp(-weight) - bias."""
    def __init__(self, num_features):
        super().__init__()
        self.weight = jt.zeros((num_features,))
        self.bias = jt.zeros((num_features,))

    def execute(self, x):
        bias = self.bias.reshape(1, 1, -1)
        weight = self.weight.reshape(1, 1, -1)
        return (x + bias) * jt.exp(weight)

    def inverse(self, y):
        bias = self.bias.reshape(1, 1, -1)
        weight = self.weight.reshape(1, 1, -1)
        return y * jt.exp(-weight) - bias


class FCNet(nn.Module):
    """Spectral-norm MLP, baked to plain Linear layers. Mirrors FCNet in deflow.py
    with densenet=False, activation='swish'.

    preact=False: [Linear, Swish, Linear, Swish, Linear]
    preact=True : [Swish, Linear, Swish, Linear, Swish, Linear]
    """
    def __init__(self, channel, idim, nhidden, preact):
        super().__init__()
        layers = []
        if preact:
            layers.append(Swish())
        last = channel
        for _ in range(nhidden):
            layers.append(nn.Linear(last, idim))
            layers.append(Swish())
            last = idim
        layers.append(nn.Linear(last, channel))
        self.nnet = nn.Sequential(*layers)

    def execute(self, x):
        return self.nnet(x)


# Fixed number of Banach iterations. The original find_fixed_point breaks as soon as
# (x-x_prev)^2/tol < 1 (typically a few iters) and hard-caps at i>10 (i.e. up to
# 12 evaluations). Running a FIXED count avoids a GPU->CPU sync per iteration
# (huge slowdown in Jittor) and only refines the same fixed point. We use 12 to
# match the original cap; convergence is verified numerically against the golden.
_FP_ITERS = int(os.environ.get("JT_FP_ITERS", "12"))


def _banach_find_root(Gnet, x):
    """Solve w = y - Gnet(w) with y=x, mirroring solvers.find_fixed_point:
        x_cur, x_prev = y - Gnet(y), y
        loop: x_prev = x_cur; x_cur = y - Gnet(x_cur)
    Runs a fixed _FP_ITERS iterations (no data-dependent break)."""
    y = x
    x_cur = y - Gnet(y)
    for _ in range(_FP_ITERS):
        x_cur = y - Gnet(x_cur)
    return x_cur


class iMonotoneBlock(nn.Module):
    """Monotone residual flow (deterministic inference only).

    forward:  w = (Id+g)^{-1}(sqrt2*x);  y = sqrt2*w - x
              i.e. solve w + g(w) = sqrt2*x  ->  w = sqrt2*x - g(w)
    inverse:  w = (Id-g)^{-1}(sqrt2*y);  x = sqrt2*w - y
              i.e. solve w - g(w) = sqrt2*y  ->  w = sqrt2*y + g(w)
    """
    def __init__(self, nnet):
        super().__init__()
        self.nnet = nnet

    def execute(self, x):
        s2 = math.sqrt(2)
        x0 = (s2 * x)
        # fixed point of  w = s2*x - g(w)  via find_fixed_point with y=s2*x, G=g
        w = _banach_find_root(lambda z: self.nnet(z), x0)
        # Original proxy: w_proxy = s2*x0 - nnet(w_value); here x0 already = s2*x,
        # and find_fixed_point returns w s.t. w = s2*x - g(w). y = s2*w - x.
        y = s2 * w - x
        return y

    def inverse(self, y):
        s2 = math.sqrt(2)
        y0 = (s2 * y)
        # fixed point of  w = s2*y + g(w) = s2*y - (-g)(w)  -> G = -g
        w = _banach_find_root(lambda z: -self.nnet(z), y0)
        x = s2 * w - y
        return x


class FlowAssembly(nn.Module):
    """chain = [iMonotoneBlock, ActNorm, iMonotoneBlock(preact=True), ActNorm]."""
    def __init__(self, channel, idim, nhidden):
        super().__init__()
        self.chain = nn.ModuleList([
            iMonotoneBlock(FCNet(channel, idim, nhidden, preact=False)),
            ActNorm(channel),
            iMonotoneBlock(FCNet(channel, idim, nhidden, preact=True)),
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


# Per-config feature-injector channel schedules. These come straight from the
# original get_denoise_net definitions (models/model_{light,heavy}/deflow.py) and
# are validated layer-by-layer against the ckpt weight shapes.
#   light: n_injector=12, concat=False at i in {7, 11}
#   heavy: n_injector=10, concat=False at i in {5, 9}
_FEAT_CFG = {
    "light": dict(
        in_channelE=[16, 48, 80, 112, 144, 176, 208, 240, 96, 120, 144, 168],
        in_channelA=[48, 80, 112, 144, 176, 208, 240, 96, 120, 144, 168, 96],
        out_channel=[32, 32, 32, 32, 32, 32, 32, 96, 24, 24, 24, 96],
        concat_off=(7, 11),
    ),
    "heavy": dict(
        in_channelE=[16, 48, 80, 112, 144, 176, 96, 120, 144, 168],
        in_channelA=[48, 80, 112, 144, 176, 96, 120, 144, 168, 96],
        out_channel=[32, 32, 32, 32, 32, 96, 24, 24, 24, 96],
        concat_off=(5, 9),
    ),
}


class LongEncoder(nn.Module):
    """HybridPF-style long-range velocity encoder (arXiv:2508.08542).

    Inference twin of train.model_train.LongEncoder with identical submodule
    names so trained weights load 1:1. Input noisy xyz (B,N,3); a 3-layer
    EdgeConv stack (3 -> 64 -> 128 -> feature_hidden) yields the long-range
    feature E_L (B,N,feature_hidden); a small MLP head yields v (B,N,3). At
    inference only E_L is used (v is discarded).
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
        idx = knn_idx(xyz, xyz, self.k)
        f = self.conv1(xyz, idx)
        f = self.conv2(f, idx)
        e_l = self.conv3(f, idx)
        v = self.vel_head(e_l)
        return e_l, v


class DenoiseFlow(nn.Module):
    """Single DenoiseFlow. Defaults to the *light* config from get_denoise_net
    (models/model_light/denoise.py). Pass a different config dict for heavy.

    The flow layers (FlowAssembly/FCNet/iMonotoneBlock/ActNorm) are identical
    across light/heavy; only the channel widths and injector schedule differ.

    ``use_long=True`` attaches the HybridPF long branch (LongEncoder + zero-init
    long_proj) exactly as in the trainable model. It is gated so the default keeps
    the model byte-identical to the original inference path (existing ckpts load
    with zero missing/extra keys).
    """
    def __init__(self, aug_channel=48, n_injector=12, cut_channel=24,
                 nflow_module=12, num_neighbors=32, idim=64, nhidden=2,
                 feat_cfg="light", feature_hidden=64, use_long=False, long_k=None):
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
        channel = self.pc_channel + self.aug_channel

        cfg = _FEAT_CFG[feat_cfg]
        in_channelE = cfg["in_channelE"]
        in_channelA = cfg["in_channelA"]
        out_channel = cfg["out_channel"]
        concat_off = cfg["concat_off"]
        assert len(in_channelE) >= n_injector, "feat_cfg too short for n_injector"

        self.noise_params = noiseEdgeConv(self.pc_channel, 32, self.aug_channel)
        self.PreConv = PreConv(self.pc_channel, 16)
        hidden_channel = feature_hidden
        self.feat_Conv = nn.ModuleList()
        self.AdaptConv = nn.ModuleList()
        for i in range(self.n_injector):
            concat = i not in concat_off
            self.feat_Conv.append(EdgeConv(in_channelE[i], hidden_channel, out_channel[i], concat=concat))
            self.AdaptConv.append(FeatMergeUnit(in_channelA[i], hidden_channel, channel))

        self.flow_assemblies = nn.ModuleList([
            FlowAssembly(channel, self.idim, self.nhidden) for _ in range(self.nflow_module)
        ])

        if self.use_long:
            lk = self.num_neighbors if long_k is None else long_k
            self.long_encoder = LongEncoder(feature_hidden=feature_hidden, k=lk)
            self.long_proj = nn.ModuleList()
            for _ in range(self.n_injector):
                proj = nn.Linear(feature_hidden, channel)
                proj.weight.assign(jt.zeros_like(proj.weight))
                if proj.bias is not None:
                    proj.bias.assign(jt.zeros_like(proj.bias))
                self.long_proj.append(proj)

    def feat_extract(self, xyz):
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
        """inj_f[i] = cs[i] + long_proj[i](E_L). Built once, passed to both f and
        g -> f adds and g subtracts the SAME tensor -> reversibility preserved."""
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
            cs, e_l, _v = self.feat_extract(x)     # v is training-only; discard
            inj_f = self._inject_long(cs, e_l)
        else:
            inj_f = self.feat_extract(x)
        aug = self.unit_coupling(x)
        xc = jt.concat([x, aug], dim=-1)
        z = self.f(xc, inj_f)
        z[:, :, -self.cut_channel:] = 0
        full = self.g(z, inj_f)
        return full[..., :self.pc_channel]

    def denoise(self, noisy_pc):
        return self.execute(noisy_pc)


def build_heavy_flow(feature_hidden=64, use_long=False, long_k=None):
    """One DenoiseFlow with the *heavy* config (models/model_heavy/denoise.py:
    aug_channel=32, n_injector=10, cut_channel=16, nflow_module=10)."""
    return DenoiseFlow(aug_channel=32, n_injector=10, cut_channel=16,
                       nflow_module=10, num_neighbors=32, idim=64, nhidden=2,
                       feat_cfg="heavy", feature_hidden=feature_hidden,
                       use_long=use_long, long_k=long_k)


class HeavyDenoiseFlow(nn.Module):
    """PD-LTS *heavy*: three DenoiseFlow stacked in a ModuleList.

    Mirrors models/model_heavy/denoise.py get_denoise_net (3x DenoiseFlow) and
    its patch_denoise, where each patch is pushed sequentially through the three
    flows:  p = noisy; for i in 0,1,2: p = flow[i](p); output = p.

    state_dict keys are prefixed "0." / "1." / "2." (the ModuleList index).
    """
    def __init__(self, feature_hidden=64, use_long=False, long_k=None):
        super().__init__()
        self.feature_hidden = feature_hidden
        self.use_long = use_long
        self.flows = nn.ModuleList([
            build_heavy_flow(feature_hidden=feature_hidden, use_long=use_long, long_k=long_k)
            for _ in range(3)
        ])

    def execute(self, x):
        p = x
        for flow in self.flows:
            p = flow.execute(p)
        return p

    def denoise(self, noisy_pc):
        return self.execute(noisy_pc)
