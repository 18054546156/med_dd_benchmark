#!/usr/bin/env python3
"""Merge verified Phase 1 replication paths into the formal artifact manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


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
    parser.add_argument("--artifact-manifest", type=Path, required=True)
    parser.add_argument("--replication-manifest", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    artifact_path = resolve(root, str(args.artifact_manifest))
    replication_path = resolve(root, str(args.replication_manifest))
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    replication = json.loads(replication_path.read_text(encoding="utf-8"))
    if replication.get("status") != "complete":
        raise ValueError("replication manifest is not complete")
    datasets = artifact.get("datasets", artifact)
    replication_datasets = replication.get("datasets", {})
    for dataset, entry in datasets.items():
        if dataset not in replication_datasets:
            raise ValueError(f"replication manifest missing {dataset}")
        values = replication_datasets[dataset]
        for field in ("condense_seeds", "pairs", "pixel_space_pairs"):
            if not isinstance(values.get(field), list) or len(values[field]) < 5:
                raise ValueError(f"{dataset}.{field} requires five entries")
        for field in ("condense_seeds", "pairs"):
            for item in values[field]:
                for key in ("synthetic", "evaluation"):
                    if not resolve(root, str(item[key])).is_file():
                        raise FileNotFoundError(f"{dataset}.{field}.{key}: {item[key]}")
        for item in values["pixel_space_pairs"]:
            for key in ("feature_synthetic", "pixel_synthetic", "feature_run_manifest", "pixel_run_manifest", "feature_evaluation", "pixel_evaluation"):
                if not resolve(root, str(item[key])).is_file():
                    raise FileNotFoundError(f"{dataset}.pixel_space_pairs.{key}: {item[key]}")
        entry["condense_seeds"] = values["condense_seeds"]
        entry["pairs"] = values["pairs"]
        entry["pixel_space_pairs"] = values["pixel_space_pairs"]
        entry["replication_manifest"] = {"path": str(replication_path), "sha256": sha256(replication_path)}
    artifact["protocol"] = "ncfm-medical-phase1-real-v4-with-replications"
    artifact["replication_manifest"] = {"path": str(replication_path), "sha256": sha256(replication_path)}
    artifact_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    production_path = root / "research" / "ncfm_medical_analysis" / "production_manifest.json"
    if production_path.is_file():
        production = json.loads(production_path.read_text(encoding="utf-8"))
        production.setdefault("artifact_manifest", {})["path"] = str(artifact_path)
        production["artifact_manifest"]["sha256"] = sha256(artifact_path)
        production["replication_manifest"] = {"path": str(replication_path), "sha256": sha256(replication_path)}
        production_path.write_text(json.dumps(production, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete", "output": str(artifact_path), "sha256": sha256(artifact_path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
