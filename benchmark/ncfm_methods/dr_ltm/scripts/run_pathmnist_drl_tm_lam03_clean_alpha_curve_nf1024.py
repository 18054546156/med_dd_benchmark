"""Clean PathMNIST DR-LTM alpha-curve rerun.

This wrapper keeps only the alpha values that must be rerun after the shared
condense output collision found on 2026-06-29. It reuses
run_pathmnist_drl_tm_seed0.py, whose save_dir is group-specific.
"""

import run_pathmnist_drl_tm_seed0 as base


base.JOBS = [
    dict(
        group="DRLTM_lam03_a050_L1_nf1024_T1024_sampling",
        lam=0.3,
        alpha=0.50,
        layers="[1]",
        nf=1024,
        mode="topk",
    ),
    dict(
        group="DRLTM_lam03_a035_L1_nf1024_T1024_sampling",
        lam=0.3,
        alpha=0.35,
        layers="[1]",
        nf=1024,
        mode="topk",
    ),
    dict(
        group="DRLTM_lam03_a025_L1_nf1024_T1024_sampling",
        lam=0.3,
        alpha=0.25,
        layers="[1]",
        nf=1024,
        mode="topk",
    ),
]


if __name__ == "__main__":
    base.main()
