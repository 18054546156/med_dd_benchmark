#!/usr/bin/env bash
# Generate evidence-linked medical tuning hypotheses after formal reporting.
# Missing evidence is recorded and never replaced by defaults.
#SBATCH --job-name=ncfm-tuning-plan
#SBATCH --partition=cpu-amd9754
#SBATCH --qos=qos-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=logs/ncfm-tuning-plan_%j.out
#SBATCH --error=logs/ncfm-tuning-plan_%j.err

set -Eeuo pipefail
ROOT="${BENCHMARK_ROOT:-/project/prj-sis01/xiaoyu_xu/med_dd_project/dd_benchmark}"
export NCFM_MATH_ROOT="${NCFM_MATH_ROOT:-$ROOT/research/ncfm_mathematical_analysis}"
PY="${PYTHON:-/home/xiaoyuxu2/.conda/envs/meddd/bin/python}"
cd "$ROOT"
test -x "$PY"
exec "$PY" research/ncfm_medical_analysis/code/generate_tuning_plan.py \
  --root "$ROOT" \
  --formal-report "$NCFM_MATH_ROOT/formal_report.json" \
  --output "$NCFM_MATH_ROOT/medical_tuning_plan"
