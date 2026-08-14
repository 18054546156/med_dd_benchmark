#!/usr/bin/env python3
"""Register one completed NCFM run without implicit artifact selection."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path


DATASETS = {
    "PathMNIST": {"slug": "pathmnist", "classes": 9, "size": [3, 32, 32]},
    "COVID": {"slug": "covid", "classes": 4, "size": [3, 112, 112]},
    "Kvasir": {"slug": "kvasir", "classes": 8, "size": [3, 128, 128]},
}


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
        "adapted/ncfm/pretrain/pretrain_script.py",
        "adapted/ncfm/condense/condense_script.py",
        "adapted/ncfm/condenser/Condenser.py",
        "adapted/ncfm/condenser/compute_loss.py",
        "adapted/ncfm/NCFM/NCFM.py",
        "adapted/ncfm/utils/utils.py",
        "adapted/ncfm/utils/init_script.py",
        "utils/medical_dataset_utils.py",
        "scripts/ncfm_pipeline.sbatch",
        "scripts/ncfm_condense_variant.sbatch",
        "scripts/ncfm_variant_seed_sweep.sbatch",
        "scripts/write_ncfm_run_manifest.py",
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


def teachers(directory: Path) -> dict[str, list[Path]]:
    pattern = re.compile(r"premodel(\d+)_(init|trained)\.pth\.tar$")
    result: dict[str, list[Path]] = {}
    for kind in ("init", "trained"):
        paths = sorted(directory.glob(f"premodel*_{kind}.pth.tar"))
        ids = {int(match.group(1)) for path in paths if (match := pattern.fullmatch(path.name))}
        if ids != set(range(20)):
            raise ValueError(f"{kind} teacher IDs must be exactly 0..19, got {sorted(ids)}")
        result[kind] = paths
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--pretrain-dir", type=Path, required=True)
    parser.add_argument("--synthetic", type=Path, required=True)
    parser.add_argument("--stdout", type=Path, required=True)
    parser.add_argument("--stderr", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    math_root = Path(os.environ.get(
        "NCFM_MATH_ROOT", root / "research" / "ncfm_mathematical_analysis"
    )).expanduser().resolve()

    def from_root(path: Path) -> Path:
        return path if path.is_absolute() else root / path

    config = required(from_root(args.config), "config")
    pretrain = from_root(args.pretrain_dir).resolve()
    synthetic = required(from_root(args.synthetic), "synthetic")
    stdout = required(from_root(args.stdout), "stdout")
    stderr = required(from_root(args.stderr), "stderr")
    statistics = required(
        root / "data" / "prepared" / args.dataset / "statistics.json",
        "train-only statistics",
    )
    prepared_manifest = required(
        root / "data" / "prepared" / args.dataset / "manifest.json",
        "prepared dataset manifest",
    )
    data_audit = required(
        math_root / "data_audit" / "current_ready.json",
        "current prepared-data audit",
    )
    file_sets = teachers(pretrain)
    if f"_{args.run_id}" not in str(synthetic.parent):
        raise ValueError(f"synthetic path is not scoped to RUN_ID={args.run_id}: {synthetic}")

    slug = DATASETS[args.dataset]["slug"]
    output = math_root / "runs" / "ncfm" / slug / args.run_id / "run_manifest.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    command = output.parent / "command.txt"
    command.write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
    sampling_net_enabled = os.environ.get("NCFM_SAMPLING_NET", "false").lower() in {"1", "true", "yes"}
    frequency_sampler = os.environ.get("NCFM_FREQUENCY_SAMPLER", "mc")
    frequency_variant = "learned_frequency" if sampling_net_enabled else (
        frequency_sampler if frequency_sampler != "mc" else "baseline"
    )
    experiment_variant = os.environ.get("NCFM_EXPERIMENT_VARIANT", frequency_variant)
    objective = os.environ.get("NCFM_OBJECTIVE", "cf")
    payload = {
        "status": "complete",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "method": "NCFM",
        "dataset": args.dataset,
        "run_id": args.run_id,
        "dataset_contract": DATASETS[args.dataset],
        "prepared_manifest": {
            "path": str(prepared_manifest),
            "sha256": digest(prepared_manifest),
        },
        "data_audit": {"path": str(data_audit), "sha256": digest(data_audit)},
        "statistics": {"path": str(statistics), "sha256": digest(statistics)},
        "config": {"path": str(config), "sha256": digest(config)},
        "source_provenance": source_provenance(root),
        "pretrained_dir": {
            "path": str(pretrain),
            "teacher_count": 20,
            "init_sha256": {path.name: digest(path) for path in file_sets["init"]},
            "trained_sha256": {path.name: digest(path) for path in file_sets["trained"]},
        },
        "synthetic": {"path": str(synthetic), "sha256": digest(synthetic)},
        "provenance": {
            "command": {"path": str(command), "sha256": digest(command)},
            "stdout": {"path": str(stdout), "sha256": digest(stdout)},
            "stderr": {"path": str(stderr), "sha256": digest(stderr)},
        },
        "method_contract": {
            "pretrain_teachers": 20,
            "condense_iterations": 20000,
            "num_freqs": 4096,
            "sampling_net": sampling_net_enabled,
            "frequency_sampler": frequency_sampler,
            "frequency_variant": frequency_variant,
            "experiment_variant": experiment_variant,
            "objective": objective,
            "importance_mean_shift": os.environ.get("NCFM_IMPORTANCE_MEAN_SHIFT"),
            "downstream_evaluation": "separate controlled evaluator",
        },
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    # This writer is deliberately silent.  It runs as the final command in
    # the Slurm script, after the script has printed the output path, so the
    # stdout/stderr hashes above remain valid when the job closes its logs.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
