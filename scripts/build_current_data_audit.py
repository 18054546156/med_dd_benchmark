#!/usr/bin/env python3
"""Build or verify the exact prepared-data audit used by production runs."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


SPECS = {
    "PathMNIST": {"size": [32, 32], "classes": 9},
    "COVID": {"size": [112, 112], "classes": 4},
    "Kvasir": {"size": [128, 128], "classes": 8},
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inspect_dataset(root: Path, dataset: str) -> dict:
    prepared = (root / "data" / "prepared" / dataset).resolve()
    manifest_path = prepared / "manifest.json"
    statistics_path = prepared / "statistics.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"{dataset}: missing prepared manifest: {manifest_path}")
    if not statistics_path.is_file():
        raise FileNotFoundError(f"{dataset}: missing statistics: {statistics_path}")
    statistics_payload = json.loads(statistics_path.read_text(encoding="utf-8"))
    statistics = statistics_payload.get("statistics", {})
    contract = statistics_payload.get("contract", {})
    expected = SPECS[dataset]
    if statistics_payload.get("status") != "complete":
        raise ValueError(f"{dataset}: statistics status is not complete")
    if contract.get("size") != expected["size"] or contract.get("num_classes") != expected["classes"]:
        raise ValueError(f"{dataset}: statistics contract mismatch: {contract}")
    if statistics.get("statistics_split") != "train":
        raise ValueError(f"{dataset}: statistics must use train split only")
    if int(statistics.get("duplicate_file_count", 0)) != 0:
        raise ValueError(f"{dataset}: duplicate RGB pixels remain")
    counts = statistics.get("split_counts")
    if not isinstance(counts, dict) or set(counts) != {"train", "val", "test"}:
        raise ValueError(f"{dataset}: invalid split counts: {counts}")
    if any(not isinstance(value, int) or value <= 0 for value in counts.values()):
        raise ValueError(f"{dataset}: split counts must be positive integers")
    mean, std = statistics.get("mean"), statistics.get("std")
    if not isinstance(mean, list) or not isinstance(std, list) or len(mean) != 3 or len(std) != 3:
        raise ValueError(f"{dataset}: invalid RGB mean/std")
    if any(float(value) <= 0 for value in std):
        raise ValueError(f"{dataset}: standard deviations must be positive")
    return {
        "prepared_root": str(prepared),
        "manifest": {"path": str(manifest_path), "sha256": sha256(manifest_path)},
        "statistics": {"path": str(statistics_path), "sha256": sha256(statistics_path)},
        "split_counts": counts,
        "mean": [float(value) for value in mean],
        "std": [float(value) for value in std],
        "duplicate_file_count": int(statistics.get("duplicate_file_count", 0)),
        "contract": expected,
    }


def build(root: Path) -> dict:
    return {
        "status": "ready",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "benchmark_root": str(root),
        "transform_contract": "PIL RGB bicubic resize, then ToTensor in [0,1]",
        "statistics_policy": "train split only",
        "datasets": {dataset: inspect_dataset(root, dataset) for dataset in SPECS},
    }


def verify(root: Path, output: Path) -> dict:
    recorded = json.loads(output.read_text(encoding="utf-8"))
    if recorded.get("status") != "ready" or set(recorded.get("datasets", {})) != set(SPECS):
        raise ValueError("current data audit is incomplete")
    current = {dataset: inspect_dataset(root, dataset) for dataset in SPECS}
    for dataset in SPECS:
        expected = recorded["datasets"][dataset]
        actual = current[dataset]
        for key in ("prepared_root", "manifest", "statistics", "split_counts", "mean", "std", "duplicate_file_count", "contract"):
            if expected.get(key) != actual.get(key):
                raise ValueError(f"{dataset}: current data audit mismatch for {key}")
    return {"status": "verified", "audit": str(output), "sha256": sha256(output)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output if args.output.is_absolute() else (root / args.output).resolve()
    if args.verify_only:
        result = verify(root, output)
    else:
        result = build(root)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        result = {"status": "ready", "audit": str(output), "sha256": sha256(output)}
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
