import torch

from benchmark.distance_methods.energy.loss import energy_distance


def test_energy_identity_and_permutation() -> None:
    torch.manual_seed(3)
    x = torch.randn(11, 5, dtype=torch.float64)
    assert torch.allclose(energy_distance(x, x), torch.zeros((), dtype=x.dtype), atol=1e-10)
    assert torch.allclose(energy_distance(x, x[torch.randperm(len(x))]), torch.zeros((), dtype=x.dtype), atol=1e-10)


def test_energy_responds_to_shift_and_backpropagates() -> None:
    torch.manual_seed(5)
    real = torch.randn(12, 4)
    synthetic = (real[:7] + 2.0).detach().requires_grad_(True)
    loss = energy_distance(real, synthetic)
    assert loss.item() > 0.1
    loss.backward()
    assert torch.isfinite(synthetic.grad).all()
