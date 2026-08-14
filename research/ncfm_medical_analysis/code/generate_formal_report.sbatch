#!/usr/bin/env bash
# Generate the final evidence report from actual JSON artifacts.
# This job reports missing evidence; it never fabricates an accuracy or defect.
#SBATCH --job-name=ncfm-formal-report
#SBATCH --partition=cpu-amd9754
#SBATCH --qos=qos-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=logs/ncfm-formal-report_%j.out
#SBATCH --error=logs/ncfm-formal-report_%j.err

set -Eeuo pipefail
export PYTHONUNBUFFERED=1
ROOT="${BENCHMARK_ROOT:-/project/prj-sis01/xiaoyu_xu/med_dd_project/dd_benchmark}"
export NCFM_MATH_ROOT="${NCFM_MATH_ROOT:-$ROOT/research/ncfm_mathematical_analysis}"
PY="/home/xiaoyuxu2/.conda/envs/meddd/bin/python"
cd "$ROOT"
test -x "$PY"

ARGS=(
  --root "$ROOT"
  --artifact-validation "${ARTIFACT_VALIDATION:-$NCFM_MATH_ROOT/formal_artifact_validation.json}"
  --output "$NCFM_MATH_ROOT/formal_report"
)
if [[ -n "${CONTROLLED_EVAL_MANIFEST:-}" ]]; then
  ARGS+=(--evaluation-manifest "$CONTROLLED_EVAL_MANIFEST")
fi
if [[ -n "${PHASE1_ARRAY_ID:-}" ]]; then
  ARGS+=(--phase1-array-id "$PHASE1_ARRAY_ID")
fi
if [[ -n "${PHASE2_ARRAY_ID:-}" ]]; then
  ARGS+=(--phase2-array-id "$PHASE2_ARRAY_ID")
fi
if [[ -n "${PHASE2_VARIANT_REPORT:-}" ]]; then
  ARGS+=(--phase2-variant-report "$PHASE2_VARIANT_REPORT")
fi
exec "$PY" research/ncfm_medical_analysis/code/generate_formal_report.py "${ARGS[@]}"
