#!/usr/bin/env python3
"""Validate the explicit real-data artifact manifest before Phase 1 submission."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import torch


DATASETS = {
    "PathMNIST": {"slug": "pathmnist", "shape": (3, 32, 32), "classes": 9, "ipc": 10},
    "COVID": {"slug": "covid", "shape": (3, 112, 112), "classes": 4, "ipc": 10},
    "Kvasir": {"slug": "kvasir", "shape": (3, 128, 128), "classes": 8, "ipc": 10},
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


def check_synthetic(path: Path, spec: dict) -> dict:
    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, (tuple, list)) and len(payload) == 2:
        data, labels = payload
    elif isinstance(payload, dict) and {"data", "label"}.issubset(payload):
        data, labels = payload["data"], payload["label"]
    elif isinstance(payload, dict) and {"images", "labels"}.issubset(payload):
        data, labels = payload["images"], payload["labels"]
    else:
        raise ValueError(f"{path}: expected (data, labels), data/label or images/labels")
    data = torch.as_tensor(data)
    labels = torch.as_tensor(labels).reshape(-1)
    if data.ndim != 4 or tuple(data.shape[1:]) != spec["shape"]:
        raise ValueError(f"{path}: expected NCHW [N,{spec['shape'][0]},{spec['shape'][1]},{spec['shape'][2]}], got {tuple(data.shape)}")
    expected_count = spec["classes"] * spec["ipc"]
    if len(data) != expected_count:
        raise ValueError(f"{path}: expected IPC={spec['ipc']} ({expected_count} samples), got {len(data)}")
    if len(data) != len(labels):
        raise ValueError(f"{path}: invalid data/label contract: {tuple(data.shape)}, {tuple(labels.shape)}")
    if float(data.min()) < -1e-5 or float(data.max()) > 1.00001:
        raise ValueError(f"{path}: data is not in raw [0,1]")
    unique = sorted(int(v) for v in labels.unique().tolist())
    if unique and (min(unique) < 0 or max(unique) >= spec["classes"]):
        raise ValueError(f"{path}: labels outside [0,{spec['classes'] - 1}]: {unique}")
    return {"path": str(path), "sha256": sha256(path), "shape": list(data.shape), "labels": unique}


def check_optional_entries(root: Path, entry: dict, field: str, minimum: int = 0, dataset: str | None = None) -> list[dict]:
    values = entry.get(field, [])
    if not isinstance(values, list):
        raise ValueError(f"{field} must be a list")
    if len(values) < minimum:
        raise ValueError(f"{field} requires at least {minimum} explicit entries, got {len(values)}")
    checked = []
    for index, item in enumerate(values):
        if not isinstance(item, dict):
            raise ValueError(f"{field}[{index}] must be an object")
        checked_item = {"index": index}
        for key in ("synthetic", "evaluation", "feature_synthetic", "pixel_synthetic", "feature_evaluation", "pixel_evaluation", "run_manifest", "feature_run_manifest", "pixel_run_manifest"):
            if key not in item:
                continue
            path = resolve(root, str(item[key]))
            if not path.is_file():
                raise FileNotFoundError(f"{field}[{index}] missing {key}: {path}")
            if key.endswith("run_manifest"):
                payload = json.loads(path.read_text(encoding="utf-8"))
                if payload.get("status") != "complete" or payload.get("method") != "NCFM" or (dataset and payload.get("dataset") != dataset):
                    raise ValueError(f"{field}[{index}] invalid {key}: {path}")
                synthetic_key = {
                    "run_manifest": "synthetic",
                    "feature_run_manifest": "feature_synthetic",
                    "pixel_run_manifest": "pixel_synthetic",
                }[key]
                declared = resolve(root, str(payload.get("synthetic", {}).get("path", "")))
                expected = resolve(root, str(item.get(synthetic_key, "")))
                if declared != expected:
                    raise ValueError(f"{field}[{index}] {key} does not bind its synthetic artifact")
            checked_item[key] = {"path": str(path), "sha256": sha256(path)}
        checked.append(checked_item)
    return checked


def check_provenance(root: Path, entry: dict) -> dict:
    """Require explicit provenance for every artifact used in a claim."""
    provenance = entry.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError("provenance must declare config, command, stdout and stderr")
    checked = {}
    for key in ("config", "command", "stdout", "stderr"):
        value = provenance.get(key)
        if not value:
            raise ValueError(f"provenance.{key} is missing")
        path = resolve(root, str(value))
        if not path.is_file():
            raise FileNotFoundError(f"missing provenance.{key}: {path}")
        checked[key] = {"path": str(path), "sha256": sha256(path)}
    return checked


def check_run_binding(root: Path, entry: dict, dataset: str, teacher_dir: Path, synthetic: Path) -> dict | None:
    value = entry.get("run_manifest")
    if not value:
        raise ValueError(f"{dataset}: formal artifact must declare run_manifest")
    path = resolve(root, str(value))
    if not path.is_file():
        raise FileNotFoundError(f"missing run_manifest: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "complete" or payload.get("method") != "NCFM" or payload.get("dataset") != dataset:
        raise ValueError(f"run_manifest identity/status mismatch: {path}")
    contract = payload.get("method_contract")
    if not isinstance(contract, dict):
        raise ValueError(f"run_manifest has no method_contract: {path}")
    expected_contract = {
        "pretrain_teachers": 20,
        "condense_iterations": 20000,
        "num_freqs": 4096,
        "sampling_net": False,
        "frequency_variant": "baseline",
        "objective": "cf",
        "experiment_variant": "baseline",
    }
    mismatches = {
        key: (contract.get(key), expected)
        for key, expected in expected_contract.items()
        if contract.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"baseline NCFM method_contract mismatch in {path}: {mismatches}")
    declared_teacher = resolve(root, str(payload.get("pretrained_dir", {}).get("path", "")))
    declared_synthetic = resolve(root, str(payload.get("synthetic", {}).get("path", "")))
    if declared_teacher != teacher_dir.resolve():
        raise ValueError(f"teacher directory does not match run_manifest: {path}")
    if declared_synthetic != synthetic.resolve():
        raise ValueError(f"synthetic artifact does not match run_manifest: {path}")
    return {
        "path": str(path),
        "sha256": sha256(path),
        "run_id": payload.get("run_id"),
        "method_contract": contract,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    manifest_path = resolve(root, str(args.manifest))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = manifest.get("datasets", manifest)
    report = {"status": "complete", "manifest": str(manifest_path), "manifest_sha256": sha256(manifest_path), "datasets": {}}
    errors = []
    for dataset, spec in DATASETS.items():
        entry = entries.get(dataset, {})
        stats_path = resolve(root, entry.get("statistics", ""))
        teacher_dir = resolve(root, entry.get("teacher_dir", ""))
        synthetic_value = entry.get("synthetic")
        try:
            if not stats_path.is_file():
                raise FileNotFoundError(f"missing statistics: {stats_path}")
            stats_payload = json.loads(stats_path.read_text(encoding="utf-8"))
            stats = stats_payload.get("statistics", stats_payload)
            if len(stats.get("mean", [])) != 3 or len(stats.get("std", [])) != 3:
                raise ValueError(f"invalid train-only statistics: {stats_path}")
            if stats.get("statistics_split") != "train":
                raise ValueError(f"statistics must be computed from train split only: {stats_path}")
            if any(float(value) <= 0 for value in stats.get("std", [])):
                raise ValueError(f"statistics std must be positive: {stats_path}")
            expected_ids = set(range(20))
            trained = sorted(teacher_dir.glob("premodel*_trained.pth.tar"))
            initialized = sorted(teacher_dir.glob("premodel*_init.pth.tar"))
            trained_ids = {
                int(match.group(1)) for path in trained
                if (match := re.fullmatch(r"premodel(\d+)_trained\.pth\.tar", path.name))
            }
            initialized_ids = {
                int(match.group(1)) for path in initialized
                if (match := re.fullmatch(r"premodel(\d+)_init\.pth\.tar", path.name))
            }
            if trained_ids != expected_ids or initialized_ids != expected_ids:
                raise ValueError(f"expected exactly teacher IDs 0..19, got init={sorted(initialized_ids)}, trained={sorted(trained_ids)} in {teacher_dir}")
            if not synthetic_value or "<formal-run>" in str(synthetic_value):
                raise ValueError(f"synthetic artifact is not explicitly filled: {synthetic_value}")
            synthetic = check_synthetic(resolve(root, synthetic_value), spec)
            synthetic_path = Path(synthetic["path"])
            report["datasets"][dataset] = {
                "statistics": {"path": str(stats_path), "sha256": sha256(stats_path), "mean": stats["mean"], "std": stats["std"]},
                "teacher": {"dir": str(teacher_dir), "init_count": len(initialized), "trained_count": len(trained),
                            "trained_sha256": {path.name: sha256(path) for path in trained}},
                "synthetic": synthetic,
                "run_binding": check_run_binding(root, entry, dataset, teacher_dir, synthetic_path),
                "provenance": check_provenance(root, entry),
                "optional_entries": {
                    "condense_seeds": check_optional_entries(root, entry, "condense_seeds", dataset=dataset),
                    "pairs": check_optional_entries(root, entry, "pairs", dataset=dataset),
                    "pixel_space_pairs": check_optional_entries(root, entry, "pixel_space_pairs", dataset=dataset),
                },
            }
        except Exception as exc:
            errors.append(f"{dataset}: {exc}")
            report["datasets"][dataset] = {"status": "failed", "error": str(exc)}
    if errors:
        report["status"] = "failed"
        report["errors"] = errors
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"status": report["status"], "output": str(args.output), "errors": errors}))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
