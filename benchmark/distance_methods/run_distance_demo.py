"""在 CPU 上验证四种距离实验的最小可执行闭环。"""

from __future__ import annotations

import torch

from benchmark.distance_methods.cf_plus_energy import cf_plus_energy
from benchmark.distance_methods.energy.loss import energy_distance
from benchmark.distance_methods.sinkhorn.loss import debiased_sinkhorn_divergence
from benchmark.distance_methods.sliced_wasserstein.loss import exact_sliced_wasserstein, projection_bank


def main() -> None:
    torch.manual_seed(2026)
    real = torch.randn(16, 8)
    synthetic = torch.randn(10, 8, requires_grad=True)
    projections = projection_bank(8, 128, 1701, device=real.device, dtype=real.dtype)

    losses = {
        "energy_replacement": energy_distance(real, synthetic),
        "cf_plus_energy": cf_plus_energy(torch.tensor(1.0), real, synthetic, energy_weight=0.01),
        "sinkhorn_replacement": debiased_sinkhorn_divergence(real, synthetic, epsilon=0.1, iterations=100),
        "sliced_wasserstein": exact_sliced_wasserstein(real, synthetic, projections=projections),
    }
    for name, loss in losses.items():
        if not torch.isfinite(loss):
            raise RuntimeError(f"{name} produced a non-finite loss")
        print(f"{name}: {loss.detach().item():.8f}")

    sum(losses.values()).backward()
    if synthetic.grad is None or not torch.isfinite(synthetic.grad).all():
        raise RuntimeError("distance gradients are missing or non-finite")
    print("backward: passed")


if __name__ == "__main__":
    main()
