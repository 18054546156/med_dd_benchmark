#!/usr/bin/env bash
# Produce the auditable Phase 2 variant report after all 18 eval entries.
#SBATCH --job-name=ncfm-p2-report
#SBATCH --partition=cpu-amd9754
#SBATCH --qos=qos-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=logs/ncfm-p2-report_%j.out
#SBATCH --error=logs/ncfm-p2-report_%j.err

set -Eeuo pipefail
ROOT="${BENCHMARK_ROOT:-/project/prj-sis01/xiaoyu_xu/med_dd_project/dd_benchmark}"
MATH_ROOT="${NCFM_MATH_ROOT:-$ROOT/research/ncfm_mathematical_analysis}"
PY="${PYTHON:-/home/xiaoyuxu2/.conda/envs/meddd/bin/python}"
cd "$ROOT"
test -x "$PY"
exec "$PY" scripts/summarize_phase2_variants.py \
  --root "$ROOT" \
  --manifest "$MATH_ROOT/phase2_variant_manifest.json" \
  --output "$MATH_ROOT/phase2_variant_report"
