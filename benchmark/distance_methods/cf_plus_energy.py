"""保留原始 CF 距离并添加 Energy Distance 约束。"""

from __future__ import annotations

import torch

from .energy.loss import energy_distance


def cf_plus_energy(
    cf_loss: torch.Tensor,
    real_features: torch.Tensor,
    synthetic_features: torch.Tensor,
    energy_weight: float = 0.01,
) -> torch.Tensor:
    """返回 CF + lambda * Energy；CF 梯度方向和原目标仍被保留。"""
    if cf_loss.ndim != 0:
        raise ValueError("cf_loss 必须是标量")
    return cf_loss + energy_weight * energy_distance(real_features, synthetic_features)
