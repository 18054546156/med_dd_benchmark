#!/usr/bin/env python3
"""Register one completed HoP-TM run and its explicit synthetic artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path


DATASETS = {"PathMNIST": "pathmnist", "COVID": "covid", "Kvasir": "kvasir"}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def required(path: Path, label: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"{label}: {path}")
    return path.resolve()


def source_provenance(root: Path) -> dict:
    relative_files = (
        "adapted/hop_tm/buffer/buffer_FTD.py",
        "adapted/hop_tm/distill/distill_high_order_spl.py",
        "adapted/hop_tm/distill/evaluation.py",
        "utils/medical_dataset_utils.py",
    )
    files = {}
    for relative in relative_files:
        path = required(root / relative, f"source file {relative}")
        files[relative] = digest(path)
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None
    return {"git_revision": revision, "files_sha256": files}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--buffer-dir", type=Path, required=True)
    parser.add_argument("--synthetic", type=Path, required=True)
    parser.add_argument("--stdout", type=Path, required=True)
    parser.add_argument("--stderr", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()

    def from_root(path: Path) -> Path:
        return path if path.is_absolute() else root / path

    config = required(from_root(args.config), "config")
    synthetic = required(from_root(args.synthetic), "synthetic")
    stdout = required(from_root(args.stdout), "stdout")
    stderr = required(from_root(args.stderr), "stderr")
    statistics = required(
        root / "data" / "prepared" / args.dataset / "statistics.json",
        "train-only statistics",
    )
    buffer_dir = from_root(args.buffer_dir).resolve()
    buffer_files = sorted(buffer_dir.glob("replay_buffer_*.pt"))
    if len(buffer_files) != 10:
        raise ValueError(f"expected 10 HoP replay buffers, found {len(buffer_files)} in {buffer_dir}")
    if f"/{args.run_id}/" not in str(synthetic).replace("\\", "/"):
        raise ValueError(f"synthetic path is not scoped to RUN_ID={args.run_id}: {synthetic}")

    slug = DATASETS[args.dataset]
    math_root = Path(os.environ.get(
        "NCFM_MATH_ROOT", root / "research" / "ncfm_mathematical_analysis"
    )).expanduser().resolve()
    output = math_root / "runs" / "hop_tm" / slug / args.run_id / "run_manifest.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    command = output.parent / "command.txt"
    command.write_text(" ".join(__import__("sys").argv) + "\n", encoding="utf-8")
    payload = {
        "status": "complete",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "method": "HoP-TM",
        "dataset": args.dataset,
        "run_id": args.run_id,
        "statistics": {"path": str(statistics), "sha256": digest(statistics)},
        "config": {"path": str(config), "sha256": digest(config)},
        "source_provenance": source_provenance(root),
        "buffer": {
            "path": str(buffer_dir),
            "trajectory_files": {path.name: digest(path) for path in buffer_files},
            "trajectory_count": 100,
            "trajectory_states": 101,
        },
        "synthetic": {"path": str(synthetic), "sha256": digest(synthetic)},
        "provenance": {
            "command": {"path": str(command), "sha256": digest(command)},
            "stdout": {"path": str(stdout), "sha256": digest(stdout)},
            "stderr": {"path": str(stderr), "sha256": digest(stderr)},
        },
        "method_contract": {
            "buffer_experts": 100,
            "distill_iterations": 10000,
            "augmentation": "DSA",
            "downstream_evaluation": "separate controlled evaluator",
        },
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"status": "complete", "manifest": str(output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
