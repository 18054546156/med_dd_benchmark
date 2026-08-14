#!/usr/bin/env bash
# Stage 1b: register and evaluate the six completed Phase 1 replication
# sweeps, then enter Stage 2.  Delaying these submissions keeps the total
# submitted-job count below the cluster limit while the sweeps are queued.
#SBATCH --job-name=formal-stage1b-submit
#SBATCH --partition=cpu-amd9754
#SBATCH --qos=qos-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:20:00
#SBATCH --output=logs/formal-stage1b-%j.out
#SBATCH --error=logs/formal-stage1b-%j.err

set -Eeuo pipefail

ROOT="${BENCHMARK_ROOT:?}"
MATH_ROOT="${NCFM_MATH_ROOT:?}"
TAG="${PHASE1_REPLICATION_TAG:?}"
MANIFEST="${ARTIFACT_MANIFEST:?}"
EVAL_MANIFEST="${CONTROLLED_EVAL_MANIFEST:?}"
PY="${PYTHON:-/home/xiaoyuxu2/.conda/envs/meddd/bin/python}"
cd "$ROOT"

BUILD_JOB="$(sbatch --parsable --job-name=ncfm-p1-rep-manifest \
  --export="ALL,NCFM_MATH_ROOT=${MATH_ROOT},PHASE1_REPLICATION_TAG=${TAG}" \
  --wrap="cd '$ROOT' && '$PY' scripts/build_phase1_replication_manifest.py --root '$ROOT' --tag '$TAG'")"
EVAL_JOB="$(sbatch --parsable --job-name=ncfm-p1-rep-eval --dependency="afterok:${BUILD_JOB}" \
  --export="ALL,NCFM_MATH_ROOT=${MATH_ROOT},PHASE1_REPLICATION_MANIFEST=${MATH_ROOT}/phase1_replication_manifest.json" \
  scripts/run_phase1_replication_eval.sbatch)"
MERGE_JOB="$(sbatch --parsable --job-name=ncfm-p1-rep-merge --dependency="afterok:${EVAL_JOB}" \
  --wrap="cd '$ROOT' && '$PY' scripts/merge_phase1_replication_manifest.py --root '$ROOT' --artifact-manifest '$MANIFEST' --replication-manifest '$MATH_ROOT/phase1_replication_manifest.json'")"

STAGE2_JOB="$(sbatch --parsable \
  --dependency="afterok:${ANALYSIS_JOB_ID:?}:${RUNTIME_JOB_ID:?}:${MERGE_JOB}" \
  --export="ALL,BENCHMARK_ROOT=$ROOT,NCFM_MATH_ROOT=$MATH_ROOT,ARTIFACT_MANIFEST=$MANIFEST,CONTROLLED_EVAL_MANIFEST=$EVAL_MANIFEST" \
  research/ncfm_medical_analysis/code/formal_stage2_submit.sbatch)"

printf 'stage=1b\nbuild=%s\nevaluation=%s\nmerge=%s\nstage2_submit=%s\n' \
  "$BUILD_JOB" "$EVAL_JOB" "$MERGE_JOB" "$STAGE2_JOB"
