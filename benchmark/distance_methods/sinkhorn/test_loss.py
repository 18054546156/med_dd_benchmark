import torch

from benchmark.distance_methods.sinkhorn.loss import debiased_sinkhorn_divergence, sinkhorn_plan


def test_sinkhorn_plan_has_uniform_marginals() -> None:
    torch.manual_seed(7)
    real, synthetic = torch.randn(7, 3, dtype=torch.float64), torch.randn(5, 3, dtype=torch.float64)
    plan = sinkhorn_plan(real, synthetic, epsilon=0.3, iterations=300)
    assert torch.allclose(plan.sum(dim=1), torch.full((7,), 1 / 7, dtype=plan.dtype), atol=1e-7)
    assert torch.allclose(plan.sum(dim=0), torch.full((5,), 1 / 5, dtype=plan.dtype), atol=1e-7)


def test_debiased_sinkhorn_identity_and_gradient() -> None:
    torch.manual_seed(11)
    real = torch.randn(8, 2, dtype=torch.float64)
    assert abs(debiased_sinkhorn_divergence(real, real, epsilon=0.2, iterations=300).item()) < 1e-8
    synthetic = (real[:6] + 1.0).detach().requires_grad_(True)
    loss = debiased_sinkhorn_divergence(real, synthetic, epsilon=0.2, iterations=300)
    assert loss.item() > 0
    loss.backward()
    assert torch.isfinite(synthetic.grad).all()
