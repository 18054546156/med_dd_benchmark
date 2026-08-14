#!/usr/bin/env bash
# Generate figures after the formal report exists.
#SBATCH --job-name=ncfm-formal-figures
#SBATCH --partition=cpu-amd9754
#SBATCH --qos=qos-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=logs/ncfm-formal-figures_%j.out
#SBATCH --error=logs/ncfm-formal-figures_%j.err

set -Eeuo pipefail
export PYTHONUNBUFFERED=1
ROOT="${BENCHMARK_ROOT:-/project/prj-sis01/xiaoyu_xu/med_dd_project/dd_benchmark}"
export NCFM_MATH_ROOT="${NCFM_MATH_ROOT:-$ROOT/research/ncfm_mathematical_analysis}"
PY="/home/xiaoyuxu2/.conda/envs/meddd/bin/python"
cd "$ROOT"
test -x "$PY"
exec "$PY" research/ncfm_medical_analysis/code/generate_visualizations.py \
  --root "$ROOT" \
  --output "$NCFM_MATH_ROOT/figures"
