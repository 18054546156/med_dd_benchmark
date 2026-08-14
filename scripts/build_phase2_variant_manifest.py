#!/usr/bin/env python3
"""Build an explicit manifest for the three real NCFM Phase 2 variants."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path


DATASETS = ("PathMNIST", "COVID", "Kvasir")
SLUGS = {"PathMNIST": "pathmnist", "COVID": "covid", "Kvasir": "kvasir"}
VARIANTS = ("qmc", "importance", "learned_frequency")


def mathematical_root(root: Path) -> Path:
    configured = os.environ.get("NCFM_MATH_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    return (root / "research" / "ncfm_mathematical_analysis").resolve()


EXPECTED_CONTRACT = {
    "qmc": {"frequency_sampler": "qmc", "sampling_net": False, "frequency_variant": "qmc", "objective": "cf"},
    "importance": {"frequency_sampler": "importance", "sampling_net": False, "frequency_variant": "importance", "objective": "cf"},
    "learned_frequency": {"frequency_sampler": "mc", "sampling_net": True, "frequency_variant": "learned_frequency", "objective": "cf"},
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--phase2-tag", required=True)
    parser.add_argument("--seeds", default="0,1,2,3,4",
                        help="Comma-separated independent condenser seeds")
    args = parser.parse_args()
    root = args.root.resolve()
    math_root = mathematical_root(root)
    try:
        seeds = tuple(dict.fromkeys(int(value) for value in args.seeds.split(",") if value.strip()))
    except ValueError as exc:
        raise ValueError(f"invalid --seeds={args.seeds!r}; use comma-separated integers") from exc
    if len(seeds) != 5:
        raise ValueError("formal Phase 2 currently requires exactly five independent condenser seeds")
    entries = []
    source_runs = {}
    for dataset in DATASETS:
        for variant in VARIANTS:
            for seed in seeds:
                run_id = f"{args.phase2_tag}-{variant}-{SLUGS[dataset]}-seed{seed}"
                manifest = math_root / "runs" / "ncfm" / SLUGS[dataset] / run_id / "run_manifest.json"
                if not manifest.is_file():
                    raise FileNotFoundError(manifest)
                payload = json.loads(manifest.read_text(encoding="utf-8"))
                if (payload.get("status") != "complete"
                        or payload.get("method") != "NCFM"
                        or payload.get("dataset") != dataset
                        or payload.get("run_id") != run_id):
                    raise ValueError(f"invalid variant run manifest: {manifest}")
                contract = payload.get("method_contract", {})
                expected_contract = EXPECTED_CONTRACT[variant]
                for key, expected in expected_contract.items():
                    actual = contract.get(key)
                    if actual != expected:
                        raise ValueError(
                            f"{manifest}: {variant} contract mismatch for {key}: "
                            f"expected {expected!r}, got {actual!r}"
                        )
                synthetic = Path(payload["synthetic"]["path"])
                if not synthetic.is_absolute():
                    synthetic = (root / synthetic).resolve()
                if not synthetic.is_file():
                    raise FileNotFoundError(synthetic)
                source_runs[f"{variant}/{dataset}/seed{seed}"] = {
                    "seed": seed, "path": str(manifest), "sha256": sha256(manifest)
                }
                for architecture in ("ConvNet", "ResNet18"):
                    entries.append({
                        "method": "NCFM",
                        "variant": variant,
                        "dataset": dataset,
                        "architecture": architecture,
                        "seed": seed,
                        "run_id": run_id,
                        "synthetic": str(synthetic),
                        "run_manifest": str(manifest),
                        "path": str(root / "results" / "controlled_eval" / "ncfm_variants" / variant / dataset / architecture / f"{variant}-{SLUGS[dataset]}-seed{seed}-{architecture}.json"),
                    })
    output = mathematical_root(root) / "phase2_variant_manifest.json"
    payload = {
        "protocol": "ncfm-phase2-real-v2-learned-frequency",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "variants": list(VARIANTS),
        "seeds": list(seeds),
        "source_runs": source_runs,
        "evaluations": entries,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete", "output": str(output), "entries": len(entries)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
