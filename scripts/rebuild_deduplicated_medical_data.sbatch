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

"$PY" scripts/refresh_data_candidate_audit.py \
  --candidate-root "$CANDIDATE_ROOT" \
  --output "$MATH_ROOT/data_audit/$DATA_TAG/ready.json" \
  --split-seed "$SEED"

echo "status=ready candidate_root=$CANDIDATE_ROOT"
