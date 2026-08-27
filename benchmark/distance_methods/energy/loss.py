"""Differentiable energy distance.

Formula follows Székely and Rizzo, *Energy statistics* (2013), and is checked
against the authors' `energy` R package at the statistic level.  This is a
PyTorch implementation, not a vendored copy of that R/C++ package.
"""

from __future__ import annotations

import torch


def _features(name: str, value: torch.Tensor) -> torch.Tensor:
    # 特征必须是二维张量：[样本数 N, 特征维度 D]。
    # 同时要求至少包含一个样本。
    if value.ndim != 2 or value.shape[0] == 0:
        raise ValueError(f"{name} must have shape [N, D] with N > 0")
    return value


def _off_diagonal_mean(distance: torch.Tensor) -> torch.Tensor:
    # 计算距离矩阵中非对角元素的平均值。
    # 对角线表示样本与自身的距离，通常全部为 0，不能用于 U-statistic。
    n = distance.shape[0]
    # 无偏估计至少需要两个样本，否则 n * (n - 1) 为 0。
    if n < 2:
        raise ValueError("the unbiased estimator requires at least two samples per distribution")
    # distance.sum() 包含所有元素；减去对角线后只保留不同样本之间的距离。
    return (distance.sum() - distance.diagonal().sum()) / (n * (n - 1))


def energy_distance(
    real: torch.Tensor,
    synthetic: torch.Tensor,
    *,
    unbiased: bool = False,
) -> torch.Tensor:
    """计算真实特征分布与合成特征分布之间的经验 Energy distance。

    Energy distance 的基本形式为：

        2 * E[真实-合成距离]
        - E[真实-真实距离]
        - E[合成-合成距离]

    unbiased=False 使用 V-statistic，通常非负；
    unbiased=True 使用 U-statistic，有限样本下可能为负。
    """
    # 检查两组特征的形状。
    real, synthetic = _features("real", real), _features("synthetic", synthetic)
    # 真实特征和合成特征必须位于相同维度的特征空间。
    if real.shape[1] != synthetic.shape[1]:
        raise ValueError("real and synthetic feature dimensions must match")

    # 真实特征与合成特征之间的平均欧氏距离：E[||X - Y||]。
    # cross[i, j] = ||X_i - Y_j||，即真实特征到合成特征的两两距离。
    # cross.mean() 对应数学公式中的 E[||X - Y||]。
    cross = torch.cdist(real, synthetic, p=2).mean()

    # 真实特征内部两两距离矩阵：||X_i - X_j||。
    # rr[i, j] = ||X_i - X_j||，描述真实特征分布内部的距离结构。
    rr = torch.cdist(real, real, p=2)

    # 合成特征内部两两距离矩阵：||Y_i - Y_j||。
    # ss[i, j] = ||Y_i - Y_j||，描述合成特征分布内部的距离结构。
    ss = torch.cdist(synthetic, synthetic, p=2)

    if unbiased:
        # U-statistic：去掉样本与自身的距离，降低有限样本偏差。
        within_real = _off_diagonal_mean(rr)
        within_synthetic = _off_diagonal_mean(ss)
    else:
        # V-statistic：包含对角线，计算更简单且通常保持非负。
        within_real = rr.mean()
        within_synthetic = ss.mean()

    # Energy distance：
    # 真实-合成距离越小越好；
    # 同时保留真实分布和合成分布各自的内部结构。
    # Energy = 2 * 跨分布距离 - 真实分布内部距离 - 合成分布内部距离。
    # 它不建立点到点运输计划，而是比较两组点云的整体距离结构。
    return 2.0 * cross - within_real - within_synthetic
