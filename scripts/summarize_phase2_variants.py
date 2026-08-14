#!/usr/bin/env python3
"""Summarize explicit Phase 2 variant evaluations without mtime selection."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path


DATASETS = ("PathMNIST", "COVID", "Kvasir")
VARIANTS = ("qmc", "importance", "learned_frequency")
ARCHITECTURES = ("ConvNet", "ResNet18")
EXPECTED_CONTRACT = {
    "qmc": {"frequency_sampler": "qmc", "sampling_net": False, "frequency_variant": "qmc", "objective": "cf"},
    "importance": {"frequency_sampler": "importance", "sampling_net": False, "frequency_variant": "importance", "objective": "cf"},
    "learned_frequency": {"frequency_sampler": "mc", "sampling_net": True, "frequency_variant": "learned_frequency", "objective": "cf"},
}


def mathematical_root(root: Path) -> Path:
    configured = os.environ.get("NCFM_MATH_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    return (root / "research" / "ncfm_mathematical_analysis").resolve()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    manifest_path = args.manifest if args.manifest.is_absolute() else root / args.manifest
    if not manifest_path.is_file():
        report = {
            "status": "insufficient_evidence",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "protocol": "ncfm-phase2-real-v2-learned-frequency",
            "manifest": {"path": str(manifest_path.resolve()), "sha256": None},
            "expected_count": 90,
            "observed_count": 0,
            "errors": [f"Phase 2 variant manifest is missing: {manifest_path}"],
            "baseline_missing": [],
            "results": [],
            "interpretation": "At least one real Phase 2 production job did not produce a verifiable variant manifest.",
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.with_suffix(".json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        args.output.with_suffix(".md").write_text(
            "# NCFM Phase 2 Variant Report\n\nStatus: `insufficient_evidence`\n\n"
            + report["errors"][0] + "\n",
            encoding="utf-8",
        )
        print(json.dumps({"status": report["status"], "output": str(args.output.with_suffix('.json'))}))
        return 1
    manifest = read(manifest_path)
    entries = manifest.get("evaluations", [])
    seeds = tuple(int(seed) for seed in manifest.get("seeds", ()))
    if len(seeds) != 5 or len(set(seeds)) != len(seeds):
        raise ValueError("Phase 2 manifest must declare exactly five unique seeds")
    expected = {(variant, dataset, architecture, seed)
                for variant in VARIANTS for dataset in DATASETS
                for architecture in ARCHITECTURES for seed in seeds}
    rows = []
    errors = []
    seen = set()
    for index, entry in enumerate(entries):
        try:
            key = (entry["variant"], entry["dataset"], entry["architecture"], int(entry["seed"]))
            if key not in expected or key in seen:
                raise ValueError(f"unexpected or duplicate entry: {key}")
            seen.add(key)
            path = Path(entry["path"])
            if not path.is_absolute():
                path = root / path
            result = read(path)
            if result.get("status") != "complete" or result.get("variant") != entry["variant"]:
                raise ValueError(f"incomplete or mismatched evaluator result: {path}")
            source_manifest = Path(entry.get("run_manifest", ""))
            if not source_manifest.is_absolute():
                source_manifest = root / source_manifest
            source = read(source_manifest)
            if source.get("status") != "complete" or source.get("method") != "NCFM" or source.get("dataset") != entry["dataset"]:
                raise ValueError(f"invalid source run manifest: {source_manifest}")
            contract = source.get("method_contract", {})
            for contract_key, expected_value in EXPECTED_CONTRACT[entry["variant"]].items():
                if contract.get(contract_key) != expected_value:
                    raise ValueError(
                        f"source contract mismatch for {entry['variant']} at {source_manifest}: "
                        f"{contract_key}={contract.get(contract_key)!r}, expected {expected_value!r}"
                    )
            declared_synthetic = Path(source.get("synthetic", {}).get("path", ""))
            if not declared_synthetic.is_absolute():
                declared_synthetic = root / declared_synthetic
            entry_synthetic = Path(entry["synthetic"])
            if not entry_synthetic.is_absolute():
                entry_synthetic = root / entry_synthetic
            if declared_synthetic.resolve() != entry_synthetic.resolve():
                raise ValueError(f"synthetic path does not match source manifest: {source_manifest}")
            accuracy = result.get("test_accuracy", {}).get("mean")
            if not isinstance(accuracy, (int, float)):
                raise ValueError(f"missing test accuracy: {path}")
            rows.append({
                "variant": entry["variant"], "dataset": entry["dataset"],
                "architecture": entry["architecture"], "seed": int(entry["seed"]),
                "path": str(path.resolve()),
                "sha256": sha256(path), "test_accuracy": result["test_accuracy"],
                "synthetic": entry["synthetic"],
                "source_run_manifest": str(source_manifest.resolve()),
                "source_run_manifest_sha256": sha256(source_manifest),
            })
        except Exception as exc:
            errors.append(f"entry {index}: {exc}")

    baseline = {}
    baseline_manifest = root / "research" / "ncfm_medical_analysis" / "formal_eval_manifest.json"
    if baseline_manifest.is_file():
        for entry in read(baseline_manifest).get("evaluations", []):
            if entry.get("method") != "NCFM":
                continue
            path = Path(entry["path"])
            if not path.is_absolute():
                path = root / path
            if path.is_file():
                payload = read(path)
                value = payload.get("test_accuracy", {}).get("mean")
                if isinstance(value, (int, float)):
                    baseline[(entry["dataset"], entry["architecture"])] = {"mean": float(value), "path": str(path.resolve()), "sha256": sha256(path)}

    baseline_missing = []
    for row in rows:
        base = baseline.get((row["dataset"], row["architecture"]))
        row["baseline_ncfm"] = base
        row["delta_vs_baseline_pp"] = (float(row["test_accuracy"]["mean"]) - base["mean"]) if base else None
        if base is None:
            baseline_missing.append((row["dataset"], row["architecture"]))

    aggregates = []
    for variant in VARIANTS:
        for dataset in DATASETS:
            for architecture in ARCHITECTURES:
                group = [row for row in rows if row["variant"] == variant
                         and row["dataset"] == dataset
                         and row["architecture"] == architecture]
                values = [float(row["test_accuracy"]["mean"]) for row in group]
                base = baseline.get((dataset, architecture))
                if not values:
                    continue
                mean = sum(values) / len(values)
                variance = sum((value - mean) ** 2 for value in values) / max(len(values) - 1, 1)
                aggregates.append({
                    "variant": variant,
                    "dataset": dataset,
                    "architecture": architecture,
                    "seed_count": len(values),
                    "seed_means": values,
                    "mean": mean,
                    "std": variance ** 0.5,
                    "baseline_mean": base["mean"] if base else None,
                    "delta_vs_baseline_pp": mean - base["mean"] if base else None,
                })

    status = "complete" if not errors and seen == expected and not baseline_missing else "insufficient_evidence"
    math_root = mathematical_root(root)
    report = {
        "status": status,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": "ncfm-phase2-real-v2-learned-frequency",
        "mathematical_analysis_root": str(math_root),
        "manifest": {"path": str(manifest_path.resolve()), "sha256": sha256(manifest_path)},
        "expected_count": len(expected), "observed_count": len(seen),
        "seeds": list(seeds),
        "errors": errors,
        "baseline_missing": sorted(set(baseline_missing)),
        "baseline_manifest": str(baseline_manifest.resolve()) if baseline_manifest.is_file() else None,
        "results": rows,
        "aggregates": aggregates,
        "interpretation": "Deltas are paired controlled-evaluation observations; they are not causal claims without preregistered replication and artifact gates.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    lines = ["# NCFM Phase 2 Variant Report", "", f"Status: `{status}`", "",
             "| Variant | Dataset | Architecture | Seeds | Mean test acc | Std | Delta vs baseline (pp) |",
             "|---|---|---|---:|---:|---:|---:|"]
    for row in sorted(aggregates, key=lambda item: (item["variant"], item["dataset"], item["architecture"])):
        delta = row["delta_vs_baseline_pp"]
        lines.append(f"| {row['variant']} | {row['dataset']} | {row['architecture']} | {row['seed_count']} | {row['mean']:.4f} | {row['std']:.4f} | {delta:.4f} |" if delta is not None else f"| {row['variant']} | {row['dataset']} | {row['architecture']} | {row['seed_count']} | {row['mean']:.4f} | {row['std']:.4f} | insufficient_evidence |")
    args.output.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "output": str(args.output.with_suffix('.json')), "errors": errors}))
    return 0 if status == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
