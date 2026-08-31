"""PathMNIST DR-LTM position-wise local descriptor optimization.

This launcher keeps the method family fixed:

    global NCFM + position-wise local descriptor CF risk

It only searches the most direct knobs around the current clean best line:
lambda strength and feature layer. It intentionally does not introduce OT,
coverage, or learned attention so the result isolates the position-wise
descriptor matching hypothesis.
"""

import run_pathmnist_drl_tm_seed0 as base


base.JOBS = [
    # Strength around existing alpha=1.0 mean local risk.
    dict(
        group="DRLTM_lam02_a100_L1_nf1024_T1024_sampling",
        lam=0.2,
        alpha=1.0,
        layers="[1]",
        nf=1024,
        mode="topk",
    ),
    dict(
        group="DRLTM_lam04_a100_L1_nf1024_T1024_sampling",
        lam=0.4,
        alpha=1.0,
        layers="[1]",
        nf=1024,
        mode="topk",
    ),
    # Strength around existing alpha=0.5 mild-tail/local risk.
    dict(
        group="DRLTM_lam02_a050_L1_nf1024_T1024_sampling",
        lam=0.2,
        alpha=0.5,
        layers="[1]",
        nf=1024,
        mode="topk",
    ),
    dict(
        group="DRLTM_lam04_a050_L1_nf1024_T1024_sampling",
        lam=0.4,
        alpha=0.5,
        layers="[1]",
        nf=1024,
        mode="topk",
    ),
    # Layer check at the known good strength. L1 is the current reference.
    dict(
        group="DRLTM_lam03_a100_L0_nf1024_T1024_sampling",
        lam=0.3,
        alpha=1.0,
        layers="[0]",
        nf=1024,
        mode="topk",
    ),
    dict(
        group="DRLTM_lam03_a100_L2_nf1024_T1024_sampling",
        lam=0.3,
        alpha=1.0,
        layers="[2]",
        nf=1024,
        mode="topk",
    ),
]


if __name__ == "__main__":
    base.main()
