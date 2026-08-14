#!/usr/bin/env bash
# Stage 4: submit the final report and its dependent tuning/figure jobs.
set -Eeuo pipefail

ROOT="${BENCHMARK_ROOT:?}"
MATH_ROOT="${NCFM_MATH_ROOT:?}"
cd "$ROOT"

REPORT_JOB="$(sbatch --parsable \
  --export="ALL,BENCHMARK_ROOT=$ROOT,NCFM_MATH_ROOT=$MATH_ROOT,ARTIFACT_VALIDATION=$MATH_ROOT/formal_artifact_validation_${GATE_JOB_ID:?}.json,CONTROLLED_EVAL_MANIFEST=${CONTROLLED_EVAL_MANIFEST:?},PHASE1_ARRAY_ID=${PHASE1_JOB_ID:?},PHASE2_ARRAY_ID=${PHASE2_DIAGNOSTIC_JOB_ID:?},PHASE2_VARIANT_REPORT=$MATH_ROOT/phase2_variant_report.json" \
  research/ncfm_medical_analysis/code/generate_formal_report.sbatch)"
TUNING_JOB="$(sbatch --parsable --dependency="afterany:$REPORT_JOB" \
  --export="ALL,BENCHMARK_ROOT=$ROOT,NCFM_MATH_ROOT=$MATH_ROOT" \
  research/ncfm_medical_analysis/code/generate_tuning_plan.sbatch)"
FIGURE_JOB="$(sbatch --parsable --dependency="afterany:$REPORT_JOB" \
  --export="ALL,BENCHMARK_ROOT=$ROOT,NCFM_MATH_ROOT=$MATH_ROOT" \
  research/ncfm_medical_analysis/code/generate_visualizations.sbatch)"

printf 'stage=4\nreport=%s\ntuning_plan=%s\nfigures=%s\nphase2_variant_report_job=%s\n' \
  "$REPORT_JOB" "$TUNING_JOB" "$FIGURE_JOB" "${PHASE2_VARIANT_REPORT_JOB_ID:?}"
