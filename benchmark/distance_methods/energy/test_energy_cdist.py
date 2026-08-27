"""验证 Energy distance 中 torch.cdist 的计算过程。"""

from __future__ import annotations

import torch

try:
    from .loss import energy_distance
except ImportError:
    # 兼容直接执行脚本的情况：python test_energy_cdist.py
    from loss import energy_distance


def test_cdist_matrices() -> None:
    """验证真实-真实、合成-合成和真实-合成距离矩阵。"""
    real = torch.tensor(
        [
            [0.0, 0.0],
            [3.0, 4.0],
            [1.0, 0.0],
        ]
    )
    synthetic = torch.tensor(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [3.0, 4.0],
        ]
    )

    # rr[i, j] 是第 i 个真实特征与第 j 个真实特征的欧氏距离。
    rr = torch.cdist(real, real, p=2)
    print("rr = torch.cdist(real, real, p=2) =\n", rr)

    # ss[i, j] 是第 i 个合成特征与第 j 个合成特征的欧氏距离。
    ss = torch.cdist(synthetic, synthetic, p=2)

    # cross[i, j] 是第 i 个真实特征与第 j 个合成特征的欧氏距离。
    cross = torch.cdist(real, synthetic, p=2)

    expected_rr = torch.tensor(
        [
            [0.0, 5.0, 1.0],
            [5.0, 0.0, (20.0) ** 0.5],
            [1.0, (20.0) ** 0.5, 0.0],
        ]
    )
    expected_ss = torch.tensor(
        [
            [0.0, 1.0, 5.0],
            [1.0, 0.0, (20.0) ** 0.5],
            [5.0, (20.0) ** 0.5, 0.0],
        ]
    )
    expected_cross = torch.tensor(
        [
            [0.0, 1.0, 5.0],
            [5.0, (20.0) ** 0.5, 0.0],
            [1.0, 0.0, (20.0) ** 0.5],
        ]
    )

    assert torch.allclose(rr, expected_rr, atol=1e-6)
    assert torch.allclose(ss, expected_ss, atol=1e-6)
    assert torch.allclose(cross, expected_cross, atol=1e-6)


def test_energy_v_statistic_matches_manual_calculation() -> None:
    """验证 V-statistic：2*cross - real_internal - synthetic_internal。"""
    real = torch.tensor([[0.0, 0.0], [3.0, 4.0], [1.0, 0.0]])
    synthetic = torch.tensor([[0.0, 0.0], [1.0, 0.0], [3.0, 4.0]])

    cross = torch.cdist(real, synthetic, p=2).mean()
    within_real = torch.cdist(real, real, p=2).mean()
    within_synthetic = torch.cdist(synthetic, synthetic, p=2).mean()
    expected = 2.0 * cross - within_real - within_synthetic

    actual = energy_distance(real, synthetic, unbiased=False)

    # 两组点完全相同，只是顺序不同，因此 Energy V-statistic 应为 0。
    assert torch.allclose(actual, expected, atol=1e-6)
    assert torch.allclose(actual, torch.zeros(()), atol=1e-6)


def test_energy_has_synthetic_gradient() -> None:
    """验证 Energy loss 可以对合成特征反向传播。"""
    real = torch.tensor([[0.0, 0.0], [2.0, 0.0]])
    synthetic = torch.tensor(
        [[0.5, 0.0], [2.5, 0.0]],
        requires_grad=True,
    )

    loss = energy_distance(real, synthetic, unbiased=False)
    loss.backward()

    assert synthetic.grad is not None
    assert torch.isfinite(synthetic.grad).all()


if __name__ == "__main__":
    # 允许直接执行：python test_energy_cdist.py
    test_cdist_matrices()
    test_energy_v_statistic_matches_manual_calculation()
    test_energy_has_synthetic_gradient()
    print("Energy cdist tests passed.")
