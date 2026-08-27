import torch

from benchmark.distance_methods.sliced_wasserstein.loss import exact_sliced_wasserstein, exact_uniform_w1, projection_bank


def test_exact_one_dimensional_w1_for_unequal_samples() -> None:
    real = torch.tensor([0.0, 2.0], dtype=torch.float64)
    synthetic = torch.tensor([1.0, 3.0], dtype=torch.float64)
    assert torch.allclose(exact_uniform_w1(real, synthetic), torch.tensor(1.0, dtype=torch.float64))


def test_sliced_identity_permutation_and_gradient() -> None:
    torch.manual_seed(13)
    real = torch.randn(8, 4)
    bank = projection_bank(4, 32, 19, device=real.device, dtype=real.dtype)
    assert torch.allclose(exact_sliced_wasserstein(real, real, projections=bank), torch.zeros(()), atol=1e-6)
    synthetic = (real[:5] + 1.5).detach().requires_grad_(True)
    loss = exact_sliced_wasserstein(real, synthetic, projections=bank)
    assert loss.item() > 0.1
    loss.backward()
    assert torch.isfinite(synthetic.grad).all()
