#!/usr/bin/env python3
"""Build explicit E3.1/E4.2 replication entries from named RUN_IDs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path


DATASETS = ("PathMNIST", "COVID", "Kvasir")
SLUGS = {"PathMNIST": "pathmnist", "COVID": "covid", "Kvasir": "kvasir"}
SEEDS = tuple(range(5))
VARIANTS = ("baseline_seed", "pixel_mean")


def mathematical_root(root: Path) -> Path:
    configured = os.environ.get("NCFM_MATH_ROOT")
    return Path(configured).expanduser().resolve() if configured else root / "research" / "ncfm_mathematical_analysis"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_run(root: Path, dataset: str, run_id: str, variant: str) -> tuple[dict, Path]:
    path = mathematical_root(root) / "runs" / "ncfm" / SLUGS[dataset] / run_id / "run_manifest.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    contract = payload.get("method_contract", {})
    if payload.get("status") != "complete" or payload.get("method") != "NCFM" or payload.get("dataset") != dataset:
        raise ValueError(f"invalid run manifest identity: {path}")
    if contract.get("experiment_variant") != variant:
        raise ValueError(f"{path}: expected experiment_variant={variant}, got {contract.get('experiment_variant')}")
    expected_objective = "pixel_mean" if variant == "pixel_mean" else "cf"
    if contract.get("objective") != expected_objective:
        raise ValueError(f"{path}: expected objective={expected_objective}, got {contract.get('objective')}")
    return payload, path


def eval_path(root: Path, dataset: str, variant: str, run_id: str) -> Path:
    return root / "results" / "controlled_eval" / "replications" / variant / dataset / "ConvNet" / f"{run_id}.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--tag", required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    entries = []
    grouped = {dataset: {variant: [] for variant in VARIANTS} for dataset in DATASETS}
    source_runs = {}
    for dataset in DATASETS:
        for variant in VARIANTS:
            for seed in SEEDS:
                run_id = f"{args.tag}-{variant}-{SLUGS[dataset]}-seed{seed}"
                payload, manifest_path = read_run(root, dataset, run_id, variant)
                synthetic = Path(payload["synthetic"]["path"])
                if not synthetic.is_absolute():
                    synthetic = (root / synthetic).resolve()
                if not synthetic.is_file():
                    raise FileNotFoundError(synthetic)
                output = eval_path(root, dataset, variant, run_id)
                entry = {
                    "method": "NCFM",
                    "variant": variant,
                    "dataset": dataset,
                    "architecture": "ConvNet",
                    "seed": seed,
                    "run_id": run_id,
                    "synthetic": str(synthetic),
                    "run_manifest": str(manifest_path),
                    "path": str(output),
                }
                entries.append(entry)
                grouped[dataset][variant].append(entry)
                source_runs[run_id] = {"path": str(manifest_path), "sha256": sha256(manifest_path)}

    output = mathematical_root(root) / "phase1_replication_manifest.json"
    payload = {
        "status": "complete",
        "protocol": "ncfm-phase1-replications-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "tag": args.tag,
        "seeds": list(SEEDS),
        "variants": list(VARIANTS),
        "source_runs": source_runs,
        "evaluations": entries,
        "datasets": {
            dataset: {
                "condense_seeds": [
                    {"seed": item["seed"], "run_id": item["run_id"], "synthetic": item["synthetic"], "evaluation": item["path"], "run_manifest": item["run_manifest"]}
                    for item in grouped[dataset]["baseline_seed"]
                ],
                "pairs": [
                    {"seed": item["seed"], "run_id": item["run_id"], "synthetic": item["synthetic"], "evaluation": item["path"], "run_manifest": item["run_manifest"]}
                    for item in grouped[dataset]["baseline_seed"]
                ],
                "pixel_space_pairs": [
                    {
                        "seed": seed,
                        "feature_synthetic": grouped[dataset]["baseline_seed"][seed]["synthetic"],
                        "pixel_synthetic": grouped[dataset]["pixel_mean"][seed]["synthetic"],
                        "feature_run_manifest": grouped[dataset]["baseline_seed"][seed]["run_manifest"],
                        "pixel_run_manifest": grouped[dataset]["pixel_mean"][seed]["run_manifest"],
                        "feature_evaluation": grouped[dataset]["baseline_seed"][seed]["path"],
                        "pixel_evaluation": grouped[dataset]["pixel_mean"][seed]["path"],
                    }
                    for seed in SEEDS
                ],
            }
            for dataset in DATASETS
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete", "output": str(output), "entries": len(entries)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
