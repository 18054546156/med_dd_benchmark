#!/usr/bin/env bash
# Generate train-only raw-pixel statistics before any formal Phase 1 run.
#SBATCH --job-name=ncfm-medical-stats
#SBATCH --partition=cpu-amd9754
#SBATCH --qos=qos-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=04:00:00
#SBATCH --output=logs/ncfm-medical-stats_%A_%a.out
#SBATCH --error=logs/ncfm-medical-stats_%A_%a.err

set -Eeuo pipefail
export PYTHONUNBUFFERED=1
ROOT="${BENCHMARK_ROOT:-/project/prj-sis01/xiaoyu_xu/med_dd_project/dd_benchmark}"
export NCFM_MATH_ROOT="${NCFM_MATH_ROOT:-$ROOT/research/ncfm_mathematical_analysis}"
PY="/home/xiaoyuxu2/.conda/envs/meddd/bin/python"
cd "$ROOT"
test -x "$PY"

for DATASET in PathMNIST COVID Kvasir; do
  "$PY" research/ncfm_medical_analysis/code/medical_stats.py \
    --dataset "$DATASET" \
    --data-root data \
    --output "data/prepared/$DATASET/statistics.json"
done

"$PY" scripts/build_current_data_audit.py \
  --root "$ROOT" \
  --output "$NCFM_MATH_ROOT/data_audit/current_ready.json"
