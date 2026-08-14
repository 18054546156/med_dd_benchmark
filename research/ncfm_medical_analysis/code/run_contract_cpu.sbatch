#!/usr/bin/env bash
# Loader contract checks are CPU-only and run sequentially to stay within the
# account's qos-normal memory limit while the HoP jobs are active.
#SBATCH --job-name=ncfm-contract
#SBATCH --partition=cpu-amd9754
#SBATCH --qos=qos-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --output=logs/ncfm-contract_%j.out
#SBATCH --error=logs/ncfm-contract_%j.err

set -Eeuo pipefail
export PYTHONUNBUFFERED=1
ROOT="${BENCHMARK_ROOT:-/project/prj-sis01/xiaoyu_xu/med_dd_project/dd_benchmark}"
export NCFM_MATH_ROOT="${NCFM_MATH_ROOT:-$ROOT/research/ncfm_mathematical_analysis}"
PY="/home/xiaoyuxu2/.conda/envs/meddd/bin/python"
cd "$ROOT"
"$PY" scripts/build_current_data_audit.py \
  --root "$ROOT" \
  --output "$NCFM_MATH_ROOT/data_audit/current_ready.json" \
  --verify-only
for DATASET in PathMNIST COVID Kvasir; do
  "$PY" research/ncfm_medical_analysis/code/contract_test.py --dataset "$DATASET"
done
