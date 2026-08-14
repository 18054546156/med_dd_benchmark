#!/usr/bin/env bash
# Build leak-free COVID/Kvasir candidates without modifying legacy prepared data.
#SBATCH --job-name=medical-dedup
#SBATCH --partition=cpu-amd9754
#SBATCH --qos=qos-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=12:00:00
#SBATCH --output=logs/medical-dedup-%j.out
#SBATCH --error=logs/medical-dedup-%j.err

set -Eeuo pipefail
export PYTHONUNBUFFERED=1

CODE_ROOT="${CODE_ROOT:-/project/prj-sis01/xiaoyu_xu/med_dd_project/dd_benchmark_worktrees/formal-20260814}"
PROJECT_ROOT="${BENCHMARK_ROOT:-/project/prj-sis01/xiaoyu_xu/med_dd_project/dd_benchmark}"
MATH_ROOT="${NCFM_MATH_ROOT:-$PROJECT_ROOT/research/ncfm_mathematical_analysis}"
PY="${PYTHON:-/home/xiaoyuxu2/.conda/envs/meddd/bin/python}"
DATA_TAG="${DATA_TAG:-deduplicated-20260814}"
CANDIDATE_ROOT="${CANDIDATE_ROOT:-$PROJECT_ROOT/data_candidates/$DATA_TAG}"
COVID_SOURCE="${COVID_SOURCE:-$PROJECT_ROOT/../dd_benchmark_archive_20260813/data/raw/COVID/source/COVID-19_Radiography_Dataset}"
KVASIR_SOURCE="${KVASIR_SOURCE:-/mnt/prj01/hgrp-1502-5TB/xiaoyu_xu/med_dd_project/datasets/kvasirv2}"
SEED="${SPLIT_SEED:-20260810}"

mkdir -p "$PROJECT_ROOT/logs" "$MATH_ROOT/data_audit/$DATA_TAG"
cd "$CODE_ROOT"
test -x "$PY"
test -d "$COVID_SOURCE"
test -d "$KVASIR_SOURCE"
test ! -e "$CANDIDATE_ROOT/prepared/COVID"
test ! -e "$CANDIDATE_ROOT/prepared/Kvasir"

for item in "COVID:$COVID_SOURCE" "Kvasir:$KVASIR_SOURCE"; do
  dataset="${item%%:*}"
  source_dir="${item#*:}"
  "$PY" scripts/prepare_medical_data.py --data-root "$CANDIDATE_ROOT" prepare \
    --dataset "$dataset" --source-dir "$source_dir" --seed "$SEED" \
    --test-ratio 0.2 --val-ratio 0.1
  "$PY" scripts/prepare_medical_data.py --data-root "$CANDIDATE_ROOT" validate \
    --dataset "$dataset"
  "$PY" research/ncfm_medical_analysis/code/medical_stats.py \
    --dataset "$dataset" --data-root "$CANDIDATE_ROOT" \
    --output "$CANDIDATE_ROOT/prepared/$dataset/statistics.json"
  cp "$CANDIDATE_ROOT/prepared/$dataset/manifest.json" \
    "$MATH_ROOT/data_audit/$DATA_TAG/${dataset}_manifest.json"
  cp "$CANDIDATE_ROOT/prepared/$dataset/statistics.json" \
    "$MATH_ROOT/data_audit/$DATA_TAG/${dataset}_statistics.json"
done

"$PY" - "$CANDIDATE_ROOT" "$MATH_ROOT/data_audit/$DATA_TAG/ready.json" "$SEED" <<'PY'
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

root, output, seed = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
datasets = {}
for dataset in ("COVID", "Kvasir"):
    prepared = root / "prepared" / dataset
    manifest = prepared / "manifest.json"
    statistics = prepared / "statistics.json"
    stats = json.loads(statistics.read_text(encoding="utf-8"))
    if stats.get("status") != "complete":
        raise SystemExit(f"{dataset}: statistics status is not complete")
    if stats["statistics"].get("duplicate_file_count") != 0:
        raise SystemExit(f"{dataset}: duplicate images remain")
    datasets[dataset] = {
        "prepared_root": str(prepared),
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "statistics_sha256": hashlib.sha256(statistics.read_bytes()).hexdigest(),
        "split_counts": stats["statistics"]["split_counts"],
        "mean": stats["statistics"]["mean"],
        "std": stats["statistics"]["std"],
    }
payload = {
    "status": "ready",
    "created_at": datetime.now(timezone.utc).isoformat(),
    "split_seed": seed,
    "candidate_root": str(root),
    "datasets": datasets,
}
output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
print(json.dumps(payload, indent=2))
PY

echo "status=ready candidate_root=$CANDIDATE_ROOT"
