#!/usr/bin/env python3
"""Build formal benchmark manifests from six explicit completed runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path


DATASETS = ("PathMNIST", "COVID", "Kvasir")
SLUGS = {"PathMNIST": "pathmnist", "COVID": "covid", "Kvasir": "kvasir"}


def mathematical_root(root: Path) -> Path:
    configured = os.environ.get("NCFM_MATH_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    return (root / "research" / "ncfm_mathematical_analysis").resolve()


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read_run(root: Path, method: str, dataset: str, run_id: str) -> tuple[dict, Path]:
    path = mathematical_root(root) / "runs" / ("ncfm" if method == "NCFM" else "hop_tm") / SLUGS[dataset] / run_id / "run_manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing {method} {dataset} run manifest: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "complete" or payload.get("method") != method or payload.get("dataset") != dataset:
        raise ValueError(f"invalid run manifest identity/status: {path}")
    contract = payload.get("method_contract")
    if not isinstance(contract, dict):
        raise ValueError(f"missing method_contract in {path}")
    if method == "NCFM":
        expected = {
            "pretrain_teachers": 20,
            "condense_iterations": 20000,
            "num_freqs": 4096,
            "sampling_net": False,
            "frequency_variant": "baseline",
            "objective": "cf",
            "experiment_variant": "baseline",
        }
    else:
        expected = {
            "buffer_experts": 100,
            "distill_iterations": 10000,
            "augmentation": "DSA",
        }
    mismatches = {
        key: (contract.get(key), value)
        for key, value in expected.items()
        if contract.get(key) != value
    }
    if mismatches:
        raise ValueError(f"{method} baseline contract mismatch in {path}: {mismatches}")
    return payload, path


def explicit_path(root: Path, value: str, label: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = root / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")
    return path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    for dataset in DATASETS:
        parser.add_argument(f"--ncfm-{SLUGS[dataset]}-run-id", required=True)
        parser.add_argument(f"--hop-{SLUGS[dataset]}-run-id", required=True)
    args = parser.parse_args()
    root = args.root.resolve()

    ncfm = {}
    hop = {}
    for dataset in DATASETS:
        ncfm[dataset] = read_run(root, "NCFM", dataset, getattr(args, f"ncfm_{SLUGS[dataset]}_run_id"))
        hop[dataset] = read_run(root, "HoP-TM", dataset, getattr(args, f"hop_{SLUGS[dataset]}_run_id"))

    artifact_datasets = {}
    for dataset in DATASETS:
        run, run_path = ncfm[dataset]
        synthetic = explicit_path(root, run["synthetic"]["path"], f"NCFM {dataset} synthetic")
        teacher_dir = Path(run["pretrained_dir"]["path"])
        if not teacher_dir.is_absolute():
            teacher_dir = (root / teacher_dir).resolve()
        if not teacher_dir.is_dir():
            raise FileNotFoundError(f"NCFM {dataset} teacher directory: {teacher_dir}")
        provenance = run["provenance"]
        artifact_datasets[dataset] = {
            "statistics": f"data/prepared/{dataset}/statistics.json",
            "teacher_dir": str(teacher_dir),
            "synthetic": str(synthetic),
            "provenance": {
                "config": run["config"]["path"],
                "command": provenance["command"]["path"],
                "stdout": provenance["stdout"]["path"],
                "stderr": provenance["stderr"]["path"],
            },
            "condense_seeds": [],
            "pairs": [],
            "pixel_space_pairs": [],
            "run_manifest": str(run_path),
        }

    artifact_manifest = {
        "protocol": "ncfm-medical-phase1-real-v3",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "datasets": artifact_datasets,
        "source_run_manifests": {
            dataset: {"ncfm": str(ncfm[dataset][1]), "hop": str(hop[dataset][1])}
            for dataset in DATASETS
        },
    }
    artifact_path = root / "research" / "ncfm_medical_analysis" / "formal_artifact_manifest.json"
    artifact_path.write_text(json.dumps(artifact_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    evaluations = []
    for dataset in DATASETS:
        ncfm_path = explicit_path(root, ncfm[dataset][0]["synthetic"]["path"], f"NCFM {dataset} synthetic")
        hop_path = explicit_path(root, hop[dataset][0]["synthetic"]["path"], f"HoP {dataset} synthetic")
        for method, synthetic in (("NCFM", ncfm_path), ("HoP", hop_path)):
            source_manifest = ncfm[dataset][1] if method == "NCFM" else hop[dataset][1]
            for architecture in ("ConvNet", "ResNet18"):
                evaluations.append({
                    "method": method,
                    "source_method": ncfm[dataset][0]["method"] if method == "NCFM" else hop[dataset][0]["method"],
                    "dataset": dataset,
                    "architecture": architecture,
                    "synthetic": str(synthetic),
                    "source_run_manifest": str(source_manifest),
                    "path": str(root / "results" / "controlled_eval" / method.lower() / dataset / architecture / f"{method.lower()}-{SLUGS[dataset]}-{architecture}.json"),
                })
    eval_manifest = {
        "protocol": "controlled-eval-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "instruction": "All paths are explicit outputs of the six completed production runs.",
        "evaluations": evaluations,
    }
    eval_path = root / "research" / "ncfm_medical_analysis" / "formal_eval_manifest.json"
    eval_path.write_text(json.dumps(eval_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    production = {
        "status": "complete",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "artifact_manifest": {"path": str(artifact_path), "sha256": digest(artifact_path)},
        "evaluation_manifest": {"path": str(eval_path), "sha256": digest(eval_path)},
        "runs": {
            dataset: {
                "ncfm": {"path": str(ncfm[dataset][1]), "sha256": digest(ncfm[dataset][1])},
                "hop": {"path": str(hop[dataset][1]), "sha256": digest(hop[dataset][1])},
            }
            for dataset in DATASETS
        },
    }
    production_path = root / "research" / "ncfm_medical_analysis" / "production_manifest.json"
    production_path.write_text(json.dumps(production, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete", "artifact_manifest": str(artifact_path), "evaluation_manifest": str(eval_path), "production_manifest": str(production_path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
