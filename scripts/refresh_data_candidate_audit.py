#!/usr/bin/env python3
"""Validate a deduplicated data candidate and write a compact ready record."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


DATASETS = ("COVID", "Kvasir")


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split-seed", type=int, default=20260810)
    args = parser.parse_args()

    root = args.candidate_root.resolve()
    datasets = {}
    for dataset in DATASETS:
        prepared = root / "prepared" / dataset
        manifest_path = prepared / "manifest.json"
        statistics_path = prepared / "statistics.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        statistics_payload = json.loads(statistics_path.read_text(encoding="utf-8"))
        statistics = statistics_payload.get("statistics", {})
        if statistics_payload.get("status") != "complete":
            raise ValueError(f"{dataset}: statistics status is not complete")
        if int(statistics.get("duplicate_file_count", -1)) != 0:
            raise ValueError(f"{dataset}: duplicate RGB pixel files remain")
        deduplication = manifest.get("deduplication")
        if not isinstance(deduplication, dict):
            raise ValueError(f"{dataset}: prepared manifest has no deduplication audit")
        datasets[dataset] = {
            "prepared_root": str(prepared),
            "manifest_sha256": digest(manifest_path),
            "statistics_sha256": digest(statistics_path),
            "split_counts": statistics["split_counts"],
            "mean": statistics["mean"],
            "std": statistics["std"],
            "duplicate_file_count": statistics["duplicate_file_count"],
            "deduplication": {
                key: deduplication[key]
                for key in (
                    "hash_algorithm",
                    "source_count",
                    "retained_count",
                    "same_class_duplicate_group_count",
                    "same_class_duplicate_file_count",
                    "ambiguous_group_count",
                    "ambiguous_file_count",
                )
            },
        }

    payload = {
        "status": "ready",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "split_seed": args.split_seed,
        "candidate_root": str(root),
        "transform_contract": "PIL RGB bicubic resize, then ToTensor in [0,1]",
        "datasets": datasets,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
