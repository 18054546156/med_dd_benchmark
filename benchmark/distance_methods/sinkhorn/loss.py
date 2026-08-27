"""Log-domain, differentiable debiased Sinkhorn divergence.

Based on Feydy et al., *Interpolating between Optimal Transport and MMD using
Sinkhorn Divergences* (AISTATS 2019).  It is intentionally compact so that
the NCFM audit can expose epsilon, iteration count and marginal residuals.
For production-scale batches, compare with the authors' maintained GeomLoss.
"""

from __future__ import annotations

import torch


def _validate(real: torch.Tensor, synthetic: torch.Tensor, epsilon: float) -> None:
    # 两组输入都是 [样本数, 特征维度] 的点云，且特征维度必须一致。
    if real.ndim != 2 or synthetic.ndim != 2 or not len(real) or not len(synthetic):
        raise ValueError("real and synthetic must be non-empty [N, D] tensors")
    if real.shape[1] != synthetic.shape[1]:
        raise ValueError("real and synthetic feature dimensions must match")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")


def _cost(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    # 成本矩阵 C[i,j] = 1/2 * ||left[i] - right[j]||_2^2。
    # 输出形状为 [left样本数, right样本数]。
    return 0.5 * torch.cdist(left, right, p=2).square()


def sinkhorn_plan(
    real: torch.Tensor,
    synthetic: torch.Tensor,
    *,
    epsilon: float = 0.1,
    iterations: int = 200,
) -> torch.Tensor:
    """Return the uniform-marginal entropic transport plan in log domain."""
    _validate(real, synthetic, epsilon)
    if iterations < 1:
        raise ValueError("iterations must be positive")
    # 计算真实点到合成点的运输成本矩阵。
    cost = _cost(real, synthetic)
    # Gibbs kernel K[i,j] = exp(-C[i,j] / epsilon)。
    # epsilon 越大，运输计划越平滑；越小，运输关系越尖锐。
    log_kernel = -cost / float(epsilon)
    log_a = real.new_full((len(real),), -torch.log(torch.tensor(float(len(real)), dtype=real.dtype, device=real.device)))
    log_b = synthetic.new_full((len(synthetic),), -torch.log(torch.tensor(float(len(synthetic)), dtype=synthetic.dtype, device=synthetic.device)))
    log_u = torch.zeros_like(log_a)
    log_v = torch.zeros_like(log_b)
    for _ in range(iterations):
        # 交替归一化行和列，让运输计划逐渐满足真实/合成的目标边际。
        log_u = log_a - torch.logsumexp(log_kernel + log_v.unsqueeze(0), dim=1)
        log_v = log_b - torch.logsumexp(log_kernel + log_u.unsqueeze(1), dim=0)
    # plan[i,j] 表示真实点 i 向合成点 j 运输的质量。
    return torch.exp(log_kernel + log_u.unsqueeze(1) + log_v.unsqueeze(0))


def entropic_ot(
    real: torch.Tensor,
    synthetic: torch.Tensor,
    *,
    epsilon: float = 0.1,
    iterations: int = 200,
) -> torch.Tensor:
    """Return OT_epsilon with KL(plan || uniform-product) regularisation."""
    # 先求满足边际约束的运输计划，再计算运输成本和熵正则项。
    plan = sinkhorn_plan(real, synthetic, epsilon=epsilon, iterations=iterations)
    cost = _cost(real, synthetic)
    reference_mass = 1.0 / (len(real) * len(synthetic))
    kl = (plan * (plan.clamp_min(torch.finfo(plan.dtype).tiny).log() - torch.log(torch.as_tensor(reference_mass, dtype=plan.dtype, device=plan.device)))).sum()
    # OT_epsilon = <plan, cost> + epsilon * KL(plan || 均匀参考分布)。
    return (plan * cost).sum() + float(epsilon) * kl


def debiased_sinkhorn_divergence(
    real: torch.Tensor,
    synthetic: torch.Tensor,
    *,
    epsilon: float = 0.1,
    iterations: int = 200,
) -> torch.Tensor:
    """Return OT_e(real, synthetic) - .5 OT_e(real, real) - .5 OT_e(syn, syn)."""
    # 去掉 real-real 与 synthetic-synthetic 的自比较偏差。
    return entropic_ot(real, synthetic, epsilon=epsilon, iterations=iterations) - 0.5 * entropic_ot(real, real, epsilon=epsilon, iterations=iterations) - 0.5 * entropic_ot(synthetic, synthetic, epsilon=epsilon, iterations=iterations)
