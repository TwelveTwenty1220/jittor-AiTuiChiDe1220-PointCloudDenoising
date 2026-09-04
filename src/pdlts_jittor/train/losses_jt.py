"""Pure-Jittor losses for the PD-LTS heavy xloss route."""
import numpy as np
import jittor as jt


def _normalize_like_ref(gen, ref):
    p_max = ref.max(dim=1, keepdims=True)
    p_min = ref.min(dim=1, keepdims=True)
    center = (p_max + p_min) / 2
    ref_n = ref - center
    scale = (ref_n * ref_n).sum(dim=-1, keepdims=True).sqrt().max(dim=1, keepdims=True)
    ref_n = ref_n / scale
    gen_n = (gen - center) / scale
    return gen_n, ref_n


def _pairwise_d2(a, b):
    diff = a.unsqueeze(2) - b.unsqueeze(1)
    return (diff * diff).sum(dim=-1)


def chamfer_distance_unit_sphere(gen, ref):
    """Symmetric Chamfer distance after normalizing by the reference sphere.

    The reference Chamfer objective uses the sum of both directional means,
    not their half-average.
    """
    gen_n, ref_n = _normalize_like_ref(gen, ref)
    d2 = _pairwise_d2(gen_n, ref_n)
    d_gen = d2.min(dim=2)
    d_ref = d2.min(dim=1)
    return d_gen.mean() + d_ref.mean()


def paired_l2_unit_sphere(gen, ref):
    """逐点配对 L2(平方距离均值), 与 chamfer_distance_unit_sphere 同一归一化/同单位.

    依据(2026-07-25, probe_tangential3): 训练数据里 pcl_clean 与 pcl_noisy 是【同索引配对】
    的(data.py:98 `pat_clean = clean[knn]`, 噪声逐点加), 即每个噪声点的真实曲面位置精确已知;
    但主输出上的所有损失(emd/cd/rep/cover/icd/sink/swd)都是集合损失, 把这个对应关系扔掉了,
    改用最近邻/最优传输匹配 —— 那种匹配天生带 ~半个点间距(0.0025) 的歧义,
    而实测残余离面误差正是 0.0022 = 半间距. 配对 L2 没有这个下限.
    上限参考(独立50k口径): 完全收敛到配对目标 = rand50k 构型 = CD score 75.25 (现状 68.71);
    与 repulsion 的均匀化叠加后理论更高(vox50k 78.16).

    注意与 train 脚本 line199 的 long_loss 区分: 那个配对 L2 作用在已判死的 HybridPF
    long-range `vel` 分支上(w_long=0), 不是主输出.
    """
    gen_n, ref_n = _normalize_like_ref(gen, ref)
    d = gen_n - ref_n
    return (d * d).sum(dim=-1).mean()


def infocd_unit_sphere(gen, ref, tau_half=0.5, lam=1e-7):
    """InfoCD (NeurIPS 2023, official calc_cd_like_InfoV2) in mean reduction.

    Official per-direction form: -log( exp(-0.5*d_i) / (sum_j exp(-0.5*d_j))^lam )
      = 0.5*d_i + lam*log(sum_j exp(-0.5*d_j)),  d = EUCLIDEAN nn distance (L1-CD
    main term + contrastive log-sum-exp regularizer that spreads matched points).
    Mean-equivalent: mean_i(0.5*d_i) + lam*mean_over_batch(logsumexp), both
    directions averaged. Same normalization as chamfer_distance_unit_sphere.
    """
    gen_n, ref_n = _normalize_like_ref(gen, ref)
    d2 = _pairwise_d2(gen_n, ref_n)
    d_gen = jt.sqrt(jt.maximum(d2.min(dim=2), jt.array(1e-9)))  # (B, Ngen)
    d_ref = jt.sqrt(jt.maximum(d2.min(dim=1), jt.array(1e-9)))  # (B, Nref)

    def one_dir(d):
        """单方向 InfoCD。输入 d: (B, N) 最近邻欧氏距离。

        返回标量 mean(tau_half*d) + lam*mean_B(logsumexp_N(-tau_half*d))。
        """
        main = (tau_half * d).mean()
        # stable logsumexp over the point dim
        m = (-tau_half * d).max(dim=1, keepdims=True)
        lse = m.squeeze(1) + jt.log(jt.exp(-tau_half * d - m).sum(dim=1) + 1e-7)
        return main + lam * lse.mean()

    return 0.5 * (one_dir(d_gen) + one_dir(d_ref))


def _sinkhorn_potentials(d2, eps, iters):
    """Log-domain symmetric Sinkhorn on cost d2 (B,N,M), uniform weights.

    Returns potentials (f, g) with NO grad tracking (call inside jt.no_grad()).
    """
    B, N, M = d2.shape
    log_mu = -float(np.log(N))
    log_nu = -float(np.log(M))
    f = jt.zeros((B, N))
    g = jt.zeros((B, M))
    for _ in range(iters):
        # f_i = -eps * LSE_j( (g_j - C_ij)/eps + log_nu )
        z = (g.unsqueeze(1) - d2) / eps + log_nu          # (B,N,M)
        m = z.max(dim=2, keepdims=True)
        f = -eps * (m.squeeze(2) + jt.log(jt.exp(z - m).sum(dim=2) + 1e-30))
        z = (f.unsqueeze(2) - d2) / eps + log_mu          # (B,N,M)
        m = z.max(dim=1, keepdims=True)
        g = -eps * (m.squeeze(1) + jt.log(jt.exp(z - m).sum(dim=1) + 1e-30))
    return f, g


def _ot_eps(a, b, eps, iters):
    """Entropic OT_eps(a,b) with Danskin gradient: potentials solved without
    grad, plunged back through the differentiable cost (soft-assignment form).
    Gradient flows to a and b via the transport-plan-weighted cost."""
    d2 = _pairwise_d2(a, b)
    with jt.no_grad():
        f, g = _sinkhorn_potentials(d2.detach(), eps, iters)
        N, M = a.shape[1], b.shape[1]
        # transport plan P_ij = exp((f_i+g_j-C_ij)/eps) * mu_i * nu_j
        logP = (f.unsqueeze(2) + g.unsqueeze(1) - d2.detach()) / eps \
               - float(np.log(N)) - float(np.log(M))
        P = jt.exp(logP)
    return (P * d2).sum(dim=2).sum(dim=1).mean()


def sinkhorn_unit_sphere(gen, ref, eps=0.01, iters=30):
    """Debiased Sinkhorn divergence S_eps = OT(x,y) - (OT(x,x)+OT(y,y))/2.

    Non-local coupling: every gen point is softly matched to ref mass, so
    coverage holes receive gradient (unlike nearest-neighbor CD). eps is in
    unit-sphere squared-distance units (0.01 ~ (0.1 diameter)^2 blur).
    """
    gen_n, ref_n = _normalize_like_ref(gen, ref)
    s = _ot_eps(gen_n, ref_n, eps, iters)
    s_xx = _ot_eps(gen_n, gen_n, eps, iters)
    s_yy = _ot_eps(ref_n, ref_n, eps, iters)
    return s - 0.5 * (s_xx + s_yy)


def sliced_wasserstein_unit_sphere(gen, ref, n_proj=128, use_max=True):
    """Max-Sliced-Wasserstein-2 覆盖损失(非局部 OT)。

    随机把点云投影到 1D 方向 -> 各自排序 -> 逐位 L2 匹配。每条投影是真·全局
    1D 最优传输, 耦合半径=整个 patch, 用来治 CD 的"切向覆盖塌缩"(点聚集留洞)。
    这是理论(arXiv:2603.09925 Corollary 1)许可的合法非局部耦合中最轻的一个。
    要求 gen/ref 同点数(patch 内成立)。use_max=最差投影(梯度更强)。
    """
    gen_n, ref_n = _normalize_like_ref(gen, ref)                 # (B,N,3)
    theta = jt.randn(n_proj, 3)
    theta = theta / (theta.sqr().sum(dim=1, keepdims=True).sqrt() + 1e-8)  # (P,3) 单位向量
    pg = jt.matmul(gen_n, theta.transpose(0, 1))                 # (B,N,P) 投影坐标
    pr = jt.matmul(ref_n, theta.transpose(0, 1))
    _, pg_s = jt.argsort(pg, dim=1)                              # 沿点维排序, 取排序后的值
    _, pr_s = jt.argsort(pr, dim=1)
    cost = (pg_s - pr_s).sqr().mean(dim=1)                       # (B,P) 每投影 1D-W2^2
    if use_max:
        return cost.max(dim=1).mean()                           # max-sliced: 取最差投影
    return cost.mean()


def repulsion_loss(pcl, k=5, r0=0.05):
    """Penalize close non-self neighbors, matching train_ours_heavy_xloss.py."""
    d2 = _pairwise_d2(pcl, pcl)
    B, N, _ = d2.shape
    eye = jt.array(np.eye(N, dtype=np.float32)).reshape(1, N, N).broadcast((B, N, N))
    d2 = d2 + eye * 1e6
    vals, _ = jt.topk(d2, k=k, dim=-1, largest=False)
    d = jt.sqrt(jt.maximum(vals, jt.array(1e-10)))
    return nn_relu(r0 - d).sqr().mean()


def coverage_loss(clean, denoised):
    """Single-direction clean->denoised squared distance."""
    d2 = _pairwise_d2(clean, denoised)
    return d2.min(dim=2).mean()


def nn_relu(x):
    """逐元素 ReLU: max(x, 0)。输入 x: 任意形状 jt.Var; 返回同形状 Var。"""
    return jt.maximum(x, jt.zeros_like(x))
