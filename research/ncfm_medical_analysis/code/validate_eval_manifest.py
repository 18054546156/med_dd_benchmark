#!/usr/bin/env python3
"""Validate the explicit controlled-evaluation input manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


DATASETS = {"PathMNIST", "COVID", "Kvasir"}
METHODS = {"NCFM", "HoP"}
ARCHITECTURES = {"ConvNet", "ResNet18"}
SOURCE_METHODS = {"NCFM": "NCFM", "HoP": "HoP-TM"}
SOURCE_CONTRACTS = {
    "NCFM": {
        "pretrain_teachers": 20,
        "condense_iterations": 20000,
        "num_freqs": 4096,
        "sampling_net": False,
        "frequency_variant": "baseline",
        "objective": "cf",
        "experiment_variant": "baseline",
    },
    "HoP": {
        "buffer_experts": 100,
        "distill_iterations": 10000,
        "augmentation": "DSA",
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (root / path).resolve()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    manifest = resolve(root, str(args.manifest))
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    entries = payload.get("evaluations")
    errors = []
    seen = set()
    checked = []
    if not isinstance(entries, list):
        errors.append("evaluations must be a list")
        entries = []
    for index, entry in enumerate(entries):
        try:
            if not isinstance(entry, dict):
                raise ValueError("entry must be an object")
            method, dataset, architecture = (entry.get(key) for key in ("method", "dataset", "architecture"))
            if method not in METHODS or dataset not in DATASETS or architecture not in ARCHITECTURES:
                raise ValueError(f"invalid method/dataset/architecture: {method}/{dataset}/{architecture}")
            key = (method, dataset, architecture)
            if key in seen:
                raise ValueError(f"duplicate evaluation key: {key}")
            seen.add(key)
            synthetic_value = str(entry.get("synthetic", ""))
            if not synthetic_value or "<" in synthetic_value or ">" in synthetic_value:
                raise ValueError("synthetic path is not explicitly filled")
            synthetic = resolve(root, synthetic_value)
            if not synthetic.is_file():
                raise FileNotFoundError(f"synthetic artifact missing: {synthetic}")
            source_value = entry.get("source_run_manifest")
            source_meta = None
            if source_value:
                source = resolve(root, str(source_value))
                if not source.is_file():
                    raise FileNotFoundError(f"source run manifest missing: {source}")
                source_payload = json.loads(source.read_text(encoding="utf-8"))
                expected_source_method = entry.get("source_method", SOURCE_METHODS[method])
                if (expected_source_method != SOURCE_METHODS[method]
                        or source_payload.get("status") != "complete"
                        or source_payload.get("method") != expected_source_method
                        or source_payload.get("dataset") != dataset):
                    raise ValueError(
                        f"source run manifest identity mismatch: {source}"
                    )
                contract = source_payload.get("method_contract")
                if not isinstance(contract, dict):
                    raise ValueError(f"source run manifest has no method_contract: {source}")
                mismatches = {
                    key: (contract.get(key), expected)
                    for key, expected in SOURCE_CONTRACTS[method].items()
                    if contract.get(key) != expected
                }
                if mismatches:
                    raise ValueError(f"source run manifest contract mismatch: {source}: {mismatches}")
                declared = resolve(
                    root, str(source_payload.get("synthetic", {}).get("path", ""))
                )
                if declared != synthetic.resolve():
                    raise ValueError(
                        f"source run manifest synthetic does not match entry: {source}"
                    )
                source_meta = {"path": str(source), "sha256": sha256(source)}
            output_value = str(entry.get("path", ""))
            if not output_value or "<" in output_value or ">" in output_value:
                raise ValueError("output path is not explicitly filled")
            checked_entry = {**entry, "synthetic": {"path": str(synthetic), "sha256": sha256(synthetic)}, "index": index}
            if source_meta:
                checked_entry["source_run_manifest"] = source_meta
            checked.append(checked_entry)
        except Exception as exc:
            errors.append(f"entry {index}: {exc}")
    expected = {(method, dataset, architecture) for method in METHODS for dataset in DATASETS for architecture in ARCHITECTURES}
    if seen != expected:
        errors.append(f"manifest must contain all 12 method/dataset/architecture entries; missing={sorted(expected - seen)} extra={sorted(seen - expected)}")
    result = {"status": "failed" if errors else "complete", "manifest": str(manifest), "manifest_sha256": sha256(manifest), "entries": checked, "errors": errors}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"status": result["status"], "output": str(args.output), "errors": errors}))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
