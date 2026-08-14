#!/usr/bin/env bash
# Submit five baseline-seed and five pixel-space-control runs per dataset.
set -Eeuo pipefail

ROOT="${BENCHMARK_ROOT:-/project/prj-sis01/xiaoyu_xu/med_dd_project/dd_benchmark}"
MATH_ROOT="${NCFM_MATH_ROOT:-$ROOT/research/ncfm_mathematical_analysis}"
TAG="${PHASE1_REPLICATION_TAG:-phase1-rep-$(date +%Y%m%d-%H%M%S)}"
NCFM_PATHMNIST_RUN_ID="${NCFM_PATHMNIST_RUN_ID:?baseline PathMNIST RUN_ID}"
NCFM_COVID_RUN_ID="${NCFM_COVID_RUN_ID:?baseline COVID RUN_ID}"
NCFM_KVASIR_RUN_ID="${NCFM_KVASIR_RUN_ID:?baseline Kvasir RUN_ID}"
PHASE1_DEPENDENCY="${PHASE1_DEPENDENCY:-}"
cd "$ROOT"
mkdir -p "$ROOT/logs"
mkdir -p "$MATH_ROOT"

declare -a jobs=()
submit_sweep() {
  local dataset="$1" slug="$2" variant="$3" teacher="$4"
  local nproc
  case "$dataset" in
    PathMNIST) nproc=3 ;;
    COVID) nproc=4 ;;
    Kvasir) nproc=8 ;;
  esac
  local dependency_args=()
  if [[ -n "$PHASE1_DEPENDENCY" ]]; then
    dependency_args+=(--dependency="$PHASE1_DEPENDENCY")
  fi
  jobs+=("$(sbatch --parsable "${dependency_args[@]}" --job-name="NCFM-${variant}-${slug}-sweep" \
    --gres="gpu:${nproc}" \
    --export="ALL,NCFM_MATH_ROOT=${MATH_ROOT},DATASET=${dataset},VARIANT=${variant},SWEEP_TAG=${TAG},TEACHER_RUN_ID=${teacher},NPROC_PER_NODE=${nproc},VARIANT_SEEDS=0:1:2:3:4" \
    scripts/ncfm_variant_seed_sweep.sbatch)")
}

for item in "PathMNIST pathmnist ${NCFM_PATHMNIST_RUN_ID}" "COVID covid ${NCFM_COVID_RUN_ID}" "Kvasir kvasir ${NCFM_KVASIR_RUN_ID}"; do
  read -r dataset slug teacher <<< "$item"
  submit_sweep "$dataset" "$slug" baseline_seed "$teacher"
  submit_sweep "$dataset" "$slug" pixel_mean "$teacher"
done

dep="afterok:$(IFS=:; echo "${jobs[*]}")"
BUILD_JOB="$(sbatch --parsable --job-name=ncfm-p1-rep-manifest --dependency="$dep" \
  --export="ALL,NCFM_MATH_ROOT=${MATH_ROOT},PHASE1_REPLICATION_TAG=${TAG}" \
  --wrap="cd '$ROOT' && '/home/xiaoyuxu2/.conda/envs/meddd/bin/python' scripts/build_phase1_replication_manifest.py --root '$ROOT' --tag '$TAG'")"
EVAL_JOB="$(sbatch --parsable --job-name=ncfm-p1-rep-eval --dependency="afterok:${BUILD_JOB}" \
    --export="ALL,NCFM_MATH_ROOT=${MATH_ROOT},PHASE1_REPLICATION_MANIFEST=${MATH_ROOT}/phase1_replication_manifest.json" \
  scripts/run_phase1_replication_eval.sbatch)"
MERGE_JOB="$(sbatch --parsable --job-name=ncfm-p1-rep-merge --dependency="afterok:${EVAL_JOB}" \
  --wrap="cd '$ROOT' && '/home/xiaoyuxu2/.conda/envs/meddd/bin/python' scripts/merge_phase1_replication_manifest.py --root '$ROOT' --artifact-manifest '$ROOT/research/ncfm_medical_analysis/formal_artifact_manifest.json' --replication-manifest '$MATH_ROOT/phase1_replication_manifest.json'")"

printf 'phase1_replication_tag=%s\ncondense_jobs=%s\nbuild=%s\nevaluation=%s\nmerge=%s\n' "$TAG" "${jobs[*]}" "$BUILD_JOB" "$EVAL_JOB" "$MERGE_JOB"
