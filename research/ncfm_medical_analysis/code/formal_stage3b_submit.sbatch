#!/usr/bin/env bash
# Stage 3b: after all real Phase 2 condenser variants exist, build their
# manifest, run grouped controlled evaluation, summarize it, and schedule the
# final report.  This continuation avoids submitting 90 array tasks at once.
#SBATCH --job-name=formal-stage3b-submit
#SBATCH --partition=cpu-amd9754
#SBATCH --qos=qos-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:20:00
#SBATCH --output=logs/formal-stage3b-%j.out
#SBATCH --error=logs/formal-stage3b-%j.err

set -Eeuo pipefail

ROOT="${BENCHMARK_ROOT:?}"
MATH_ROOT="${NCFM_MATH_ROOT:?}"
PHASE2_TAG="${PHASE2_TAG:?}"
PHASE2_SEEDS="${PHASE2_SEEDS:-0,1,2,3,4}"
PHASE2_SEEDS_CSV="${PHASE2_SEEDS//:/,}"
PY="${PYTHON:-/home/xiaoyuxu2/.conda/envs/meddd/bin/python}"
cd "$ROOT"

MANIFEST_JOB="$(sbatch --parsable --job-name=ncfm-p2-manifest \
  --export="ALL,NCFM_MATH_ROOT=${MATH_ROOT},PHASE2_TAG=${PHASE2_TAG}" \
  --wrap="cd '$ROOT' && '$PY' scripts/build_phase2_variant_manifest.py --root '$ROOT' --phase2-tag '$PHASE2_TAG' --seeds '$PHASE2_SEEDS_CSV'")"
EVAL_JOB="$(sbatch --parsable --job-name=ncfm-p2-eval --dependency="afterok:${MANIFEST_JOB}" \
  --export="ALL,NCFM_MATH_ROOT=${MATH_ROOT},PHASE2_VARIANT_MANIFEST=${MATH_ROOT}/phase2_variant_manifest.json" \
  scripts/run_phase2_variant_eval.sbatch)"
REPORT_JOB="$(sbatch --parsable --job-name=ncfm-p2-report --dependency="afterany:${EVAL_JOB}" \
  --export="ALL,NCFM_MATH_ROOT=${MATH_ROOT}" \
  scripts/summarize_phase2_variants.sbatch)"

STAGE4_JOB="$(sbatch --parsable \
  --dependency="afterany:${PHASE2_DIAGNOSTIC_JOB_ID:?}:${REPORT_JOB}" \
  --export="ALL,BENCHMARK_ROOT=$ROOT,NCFM_MATH_ROOT=$MATH_ROOT,ARTIFACT_MANIFEST=${ARTIFACT_MANIFEST:?},CONTROLLED_EVAL_MANIFEST=${CONTROLLED_EVAL_MANIFEST:?},PHASE1_JOB_ID=${PHASE1_JOB_ID:?},EVAL_JOB_ID=${EVAL_JOB_ID:?},GATE_JOB_ID=${GATE_JOB_ID:?},PHASE2_DIAGNOSTIC_JOB_ID=$PHASE2_DIAGNOSTIC_JOB_ID,PHASE2_VARIANT_REPORT_JOB_ID=$REPORT_JOB" \
  research/ncfm_medical_analysis/code/formal_stage4_report.sbatch)"

printf 'stage=3b\nmanifest=%s\nevaluation=%s\nvariant_report=%s\nstage4_report=%s\n' \
  "$MANIFEST_JOB" "$EVAL_JOB" "$REPORT_JOB" "$STAGE4_JOB"
