"""Exact empirical one-dimensional and sliced 1-Wasserstein distances.

The projection construction follows Bonneel, Rabin, Peyré and Pfister (2015).
Unlike the historical NCFM K=64 code, unequal sample counts are evaluated with
the exact uniform empirical quantile coupling rather than interpolation.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def projection_bank(feature_dim: int, count: int, seed: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    # 生成 K 个 D 维单位投影方向，返回形状 [K, D]。
    # K=4096 就表示从 4096 个方向观察特征分布。
    if feature_dim < 1 or count < 1:
        raise ValueError("feature_dim and count must be positive")
    generator = torch.Generator(device="cpu").manual_seed(int(seed) + feature_dim * 1009 + count * 9176)
    return F.normalize(torch.randn((count, feature_dim), generator=generator).to(device=device, dtype=dtype), dim=1)


def _quantile_indices(real_count: int, synthetic_count: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    common = math.lcm(real_count, synthetic_count)
    real_edges = torch.arange(0, common + 1, common // real_count)
    syn_edges = torch.arange(0, common + 1, common // synthetic_count)
    edges = torch.unique(torch.cat((real_edges, syn_edges)), sorted=True)
    left, right = edges[:-1], edges[1:]
    midpoint_twice = left + right
    real_index = torch.div(midpoint_twice * real_count, 2 * common, rounding_mode="floor").clamp_max(real_count - 1).long()
    syn_index = torch.div(midpoint_twice * synthetic_count, 2 * common, rounding_mode="floor").clamp_max(synthetic_count - 1).long()
    weights = (right - left).to(torch.float64) / common
    return real_index.to(device), syn_index.to(device), weights.to(device)


def exact_uniform_w1(real: torch.Tensor, synthetic: torch.Tensor) -> torch.Tensor:
    """Exact W1 between two one-dimensional uniform empirical measures."""
    if real.ndim != 1 or synthetic.ndim != 1 or not len(real) or not len(synthetic):
        raise ValueError("real and synthetic must be non-empty vectors")
    # 一维 Wasserstein-1 通过排序后的经验分位数直接计算。
    real_sorted, syn_sorted = real.sort().values, synthetic.sort().values
    i, j, weights = _quantile_indices(len(real), len(synthetic), real.device)
    # W1 使用绝对差：|真实分位数 - 合成分位数|，不是平方差。
    return (weights.to(dtype=real.dtype) * (real_sorted[i] - syn_sorted[j]).abs()).sum()


def exact_sliced_wasserstein(
    real: torch.Tensor,
    synthetic: torch.Tensor,
    *,
    projections: torch.Tensor | None = None,
    projection_count: int = 256,
    seed: int = 1701,
) -> torch.Tensor:
    """Average exact one-dimensional W1 over a fixed bank of unit projections."""
    if real.ndim != 2 or synthetic.ndim != 2 or not len(real) or not len(synthetic):
        raise ValueError("real and synthetic must be non-empty [N, D] tensors")
    if real.shape[1] != synthetic.shape[1]:
        raise ValueError("real and synthetic feature dimensions must match")
    # projections 的形状为 [K, D]，每一行是一个单位观察方向。
    if projections is None:
        projections = projection_bank(real.shape[1], projection_count, seed, device=real.device, dtype=real.dtype)
    if projections.ndim != 2 or projections.shape[1] != real.shape[1]:
        raise ValueError("projections must have shape [K, D]")
    # 每个方向把高维特征投影为一维样本，再计算该方向的一维 W1。
    # 最后平均 K 个方向的距离，得到 Sliced Wasserstein distance。
    return torch.stack([exact_uniform_w1(real @ direction, synthetic @ direction) for direction in projections]).mean()
