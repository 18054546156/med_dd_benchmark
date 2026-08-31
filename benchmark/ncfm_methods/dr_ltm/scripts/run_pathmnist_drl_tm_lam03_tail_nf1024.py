from run_pathmnist_drl_tm_seed0 import JOBS, main


JOBS[:] = [
    dict(
        group="DRLTM_lam03_a050_L1_nf1024_T1024_sampling",
        lam=0.3,
        alpha=0.50,
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
    dict(
        group="DRLTM_lam03_a010_L1_nf1024_T1024_sampling",
        lam=0.3,
        alpha=0.10,
        layers="[1]",
        nf=1024,
        mode="topk",
    ),
]


if __name__ == "__main__":
    main()
