#!/usr/bin/env bash
# Stage 2: artifact gates, task analysis, Phase 1 diagnostics and controlled
# baseline evaluation. Stage 3 starts only when both GPU analyses complete.
set -Eeuo pipefail

ROOT="${BENCHMARK_ROOT:?}"
MATH_ROOT="${NCFM_MATH_ROOT:?}"
MANIFEST="${ARTIFACT_MANIFEST:?}"
EVAL_MANIFEST="${CONTROLLED_EVAL_MANIFEST:?}"
cd "$ROOT"

GATE_JOB="$(sbatch --parsable \
  --export="ALL,BENCHMARK_ROOT=$ROOT,NCFM_MATH_ROOT=$MATH_ROOT,ARTIFACT_MANIFEST=$MANIFEST" \
  research/ncfm_medical_analysis/code/run_validate_artifacts_cpu.sbatch)"
TASK_JOB="$(sbatch --parsable --dependency="afterok:$GATE_JOB" \
  --export="ALL,BENCHMARK_ROOT=$ROOT,NCFM_MATH_ROOT=$MATH_ROOT,ARTIFACT_MANIFEST=$MANIFEST" \
  research/ncfm_medical_analysis/code/run_task_analysis_gpu.sbatch)"
PHASE1_JOB="$(sbatch --parsable --dependency="afterok:$GATE_JOB:$TASK_JOB" \
  --export="ALL,BENCHMARK_ROOT=$ROOT,NCFM_MATH_ROOT=$MATH_ROOT,ARTIFACT_MANIFEST=$MANIFEST" \
  research/ncfm_medical_analysis/code/run_real_phase1.sbatch)"
EVAL_GATE_JOB="$(sbatch --parsable --dependency="afterok:$GATE_JOB" \
  --export="ALL,BENCHMARK_ROOT=$ROOT,NCFM_MATH_ROOT=$MATH_ROOT,CONTROLLED_EVAL_MANIFEST=$EVAL_MANIFEST" \
  research/ncfm_medical_analysis/code/run_validate_eval_manifest_cpu.sbatch)"
EVAL_JOB="$(sbatch --parsable --dependency="afterok:$EVAL_GATE_JOB" \
  --export="ALL,BENCHMARK_ROOT=$ROOT,NCFM_MATH_ROOT=$MATH_ROOT,CONTROLLED_EVAL_MANIFEST=$EVAL_MANIFEST" \
  research/ncfm_medical_analysis/code/run_eval_manifest_gpu.sbatch)"

STAGE3_JOB="$(sbatch --parsable \
  --job-name=formal-stage3-submit \
  --partition=cpu-amd9754 --qos=qos-normal --cpus-per-task=1 --mem=2G --time=00:20:00 \
  --output=logs/formal-stage3-%j.out --error=logs/formal-stage3-%j.err \
  --dependency="afterok:$PHASE1_JOB:$EVAL_JOB" \
  --export="ALL,BENCHMARK_ROOT=$ROOT,NCFM_MATH_ROOT=$MATH_ROOT,ARTIFACT_MANIFEST=$MANIFEST,CONTROLLED_EVAL_MANIFEST=$EVAL_MANIFEST,PHASE1_JOB_ID=$PHASE1_JOB,EVAL_JOB_ID=$EVAL_JOB,GATE_JOB_ID=$GATE_JOB" \
  research/ncfm_medical_analysis/code/formal_stage3_submit.sbatch)"

printf 'stage=2\ngate=%s\ntask_analysis=%s\nphase1=%s\neval_gate=%s\neval=%s\nstage3_submit=%s\n' \
  "$GATE_JOB" "$TASK_JOB" "$PHASE1_JOB" "$EVAL_GATE_JOB" "$EVAL_JOB" "$STAGE3_JOB"
