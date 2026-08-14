#!/usr/bin/env python3
"""Run auditable Phase 2 frequency-method analyses on real NCFM artifacts.

This module measures candidate estimators on an explicit real teacher/synthetic
feature pair. It does not silently rewrite or claim to have rerun the NCFM
condenser; a downstream accuracy claim requires a separately recorded rerun.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from real_phase1 import (
    DATASETS,
    RunConfig,
    classwise_d_omega,
    classwise_per_frequency_error,
    frequency_bank,
    load_features,
    load_synthetic_features,
    percentile_ci,
    project_root,
    sha256,
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def dump(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def qmc(
    config: RunConfig,
    real: torch.Tensor,
    real_labels: torch.Tensor,
    synthetic: torch.Tensor,
    synthetic_labels: torch.Tensor,
) -> dict:
    sweep = {}
    for count in config.t_values:
        mc = []
        sobol = []
        mc_per_class = []
        qmc_per_class = []
        for replica in range(config.replicas):
            seed = config.seed + replica
            mc_value, mc_classes = classwise_d_omega(
                real, real_labels, synthetic, synthetic_labels,
                frequency_bank(count, real.shape[1], seed, "mc"),
            )
            qmc_value, qmc_classes = classwise_d_omega(
                real, real_labels, synthetic, synthetic_labels,
                frequency_bank(count, real.shape[1], seed, "qmc"),
            )
            mc.append(mc_value)
            sobol.append(qmc_value)
            mc_per_class.append(mc_classes)
            qmc_per_class.append(qmc_classes)
        mc_stats = percentile_ci(mc, config.seed)
        qmc_stats = percentile_ci(sobol, config.seed)
        sweep[str(count)] = {
            "mc": mc_stats,
            "qmc": qmc_stats,
            "mc_per_class": {
                class_id: percentile_ci(
                    [record[class_id] for record in mc_per_class], config.seed
                )
                for class_id in sorted(mc_per_class[0], key=int)
            },
            "qmc_per_class": {
                class_id: percentile_ci(
                    [record[class_id] for record in qmc_per_class], config.seed
                )
                for class_id in sorted(qmc_per_class[0], key=int)
            },
            "std_reduction_fraction": float(
                (mc_stats["std"] - qmc_stats["std"]) / max(mc_stats["std"], 1e-12)
            ),
        }
    return {"method": "qmc", "sweep": sweep,
            "claim_scope": "estimator-only until a real condenser rerun exists"}


def importance_values(
    real: torch.Tensor,
    real_labels: torch.Tensor,
    synthetic: torch.Tensor,
    synthetic_labels: torch.Tensor,
    count: int,
    seed: int,
    mean_shift: float,
) -> dict:
    generator = torch.Generator().manual_seed(seed)
    proposal = torch.randn(count, real.shape[1], generator=generator)
    proposal[:, 0] += mean_shift
    # For p=N(0,I), q=N(mu,I): log(p/q) = -mu dot w + ||mu||^2/2.
    mu = torch.zeros(real.shape[1])
    mu[0] = mean_shift
    log_weights = -(proposal @ mu) + 0.5 * mu.square().sum()
    weights = log_weights.exp()
    values, class_values = classwise_per_frequency_error(
        real, real_labels, synthetic, synthetic_labels, proposal
    )
    estimate = float((values * weights).mean())
    ess = float(weights.sum().square() / weights.square().sum())
    return {
        "estimate": estimate,
        "ess": ess,
        "ess_fraction": ess / count,
        "max_weight": float(weights.max()),
        "weight_mean": float(weights.mean()),
        "per_class_estimate": {
            class_id: float((errors * weights).mean())
            for class_id, errors in class_values.items()
        },
    }


def importance(
    config: RunConfig,
    real: torch.Tensor,
    real_labels: torch.Tensor,
    synthetic: torch.Tensor,
    synthetic_labels: torch.Tensor,
    mean_shift: float,
) -> dict:
    reference_count = max(config.num_freqs * 4, 16384)
    reference, reference_per_class = classwise_d_omega(
        real, real_labels, synthetic, synthetic_labels,
        frequency_bank(reference_count, real.shape[1], config.seed + 90000, "mc"),
    )
    records = [importance_values(real, real_labels, synthetic, synthetic_labels,
                                 config.num_freqs, config.seed + i, mean_shift)
               for i in range(config.replicas)]
    errors = [abs(item["estimate"] - reference) for item in records]
    return {
        "method": "importance",
        "proposal": {"distribution": "N(mu,I)", "mu_first_coordinate": mean_shift},
        "target": {"distribution": "N(0,I)", "reference_count": reference_count,
                    "reference_estimate": reference,
                    "reference_per_class": reference_per_class},
        "records": records,
        "absolute_error": percentile_ci(errors, config.seed),
        "claim_scope": "exact-weight estimator; no clipping; estimator-only until condenser rerun",
    }


def empirical_bernstein(values: torch.Tensor, delta: float) -> dict:
    n = int(values.numel())
    if n < 2:
        raise ValueError("certificate requires at least two independent holdout banks")
    bound = 4.0
    mean = float(values.mean())
    variance = float(values.var(unbiased=True))
    log_term = math.log(3.0 / delta)
    radius = math.sqrt(2.0 * variance * log_term / n) + 3.0 * bound * log_term / n
    return {
        "n_banks": n,
        "mean": mean,
        "variance": variance,
        "range_bound": [0.0, bound],
        "delta": delta,
        "upper_bound": mean + radius,
        "lower_bound": max(0.0, mean - radius),
        "radius": radius,
        "interpretation": "conditional bound for fixed features and independent holdout frequencies",
    }


def certificate(
    config: RunConfig,
    real: torch.Tensor,
    real_labels: torch.Tensor,
    synthetic: torch.Tensor,
    synthetic_labels: torch.Tensor,
    delta: float,
) -> dict:
    values = []
    per_class_records = []
    for replica in range(config.replicas):
        bank = frequency_bank(config.num_freqs, real.shape[1], config.seed + 50000 + replica, "mc")
        aggregate, per_class = classwise_d_omega(
            real, real_labels, synthetic, synthetic_labels, bank
        )
        values.append(aggregate)
        per_class_records.append(per_class)
    return {
        "method": "certificate",
        "holdout": {"frequency_kind": "mc", "num_freqs": config.num_freqs,
                     "replicas": config.replicas},
        "bank_means": values,
        "per_class_confidence": {
            class_id: empirical_bernstein(
                torch.tensor([record[class_id] for record in per_class_records]), delta
            )
            for class_id in sorted(per_class_records[0], key=int)
        },
        "confidence": empirical_bernstein(torch.tensor(values), delta),
        "claim_scope": "conditional feature-pair certificate; not a certificate for optimized training or unseen backbones",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    parser.add_argument("--method", choices=["qmc", "importance", "certificate"], required=True)
    parser.add_argument("--synthetic-data")
    parser.add_argument("--teacher-dir")
    parser.add_argument("--artifact-manifest",
                        help="Filled manifest with explicit synthetic and teacher paths")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-freqs", type=int, default=4096)
    parser.add_argument("--replicas", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mean-shift", type=float, default=0.5)
    parser.add_argument("--delta", type=float, default=0.05)
    parser.add_argument("--t-values", default="256,512,1024,2048,4096,8192")
    args = parser.parse_args()
    if args.replicas < 5:
        raise ValueError("formal Phase 2 requires at least five independent replicas")
    if not (0.0 < args.delta < 1.0):
        raise ValueError("delta must be in (0,1)")

    root = project_root()
    manifest_meta = None
    if args.artifact_manifest:
        manifest_path = Path(args.artifact_manifest)
        if not manifest_path.is_absolute():
            manifest_path = (root / manifest_path).resolve()
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        entry = payload.get("datasets", payload).get(args.dataset, {})
        if not isinstance(entry, dict):
            raise ValueError(f"manifest entry is not an object: {args.dataset}")
        if not args.synthetic_data:
            args.synthetic_data = entry.get("synthetic")
        if not args.teacher_dir:
            args.teacher_dir = entry.get("teacher_dir")
        if not args.synthetic_data or "<" in str(args.synthetic_data) or ">" in str(args.synthetic_data):
            raise ValueError("manifest synthetic path is missing or still contains a placeholder")
        if not args.teacher_dir or "<" in str(args.teacher_dir) or ">" in str(args.teacher_dir):
            raise ValueError("manifest teacher directory is missing or still contains a placeholder")
        manifest_meta = {"path": str(manifest_path), "sha256": sha256(manifest_path)}
    if not args.synthetic_data or not args.teacher_dir:
        raise ValueError("provide --synthetic-data and --teacher-dir, or --artifact-manifest")
    random.seed(args.seed)
    np.random.seed(args.seed)
    config = RunConfig(
        dataset=args.dataset,
        experiment=f"phase2.{args.method}",
        run_id=args.run_id,
        num_freqs=args.num_freqs,
        replicas=args.replicas,
        seed=args.seed,
        synthetic_data=args.synthetic_data,
        teacher_dir=args.teacher_dir,
        t_values=tuple(int(value) for value in args.t_values.split(",") if value),
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    real, real_labels, real_meta = load_features(config, root, "train")
    synthetic, synthetic_labels, synthetic_meta = load_synthetic_features(config, root)
    if args.method == "qmc":
        result = qmc(config, real, real_labels, synthetic, synthetic_labels)
    elif args.method == "importance":
        result = importance(
            config, real, real_labels, synthetic, synthetic_labels, args.mean_shift
        )
    else:
        result = certificate(
            config, real, real_labels, synthetic, synthetic_labels, args.delta
        )
    result.update({
        "status": "complete",
        "dataset": args.dataset,
        "run_id": args.run_id,
        "created_at": now(),
        "source": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__).resolve())},
        "runtime": {"python": platform.python_version(), "torch": torch.__version__,
                    "cuda": torch.version.cuda, "device": config.device},
        "real_features": real_meta,
        "synthetic_features": synthetic_meta,
        "artifact_manifest": manifest_meta,
        "seed": args.seed,
        "replicas": args.replicas,
        "input_policy": {"real_data_required": True, "toy_fallback": False,
                          "synthetic_path_explicit": True},
    })
    dump(args.output, result)
    print(json.dumps({"status": result["status"], "method": args.method,
                      "dataset": args.dataset, "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
