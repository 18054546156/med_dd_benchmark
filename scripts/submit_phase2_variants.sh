#!/usr/bin/env bash
# Submit real QMC, exact importance, and learned-frequency NCFM variants.
# Requires the six production runs and their formal manifests to be complete.
set -Eeuo pipefail
ROOT="${BENCHMARK_ROOT:-/project/prj-sis01/xiaoyu_xu/med_dd_project/dd_benchmark}"
MATH_ROOT="${NCFM_MATH_ROOT:-$ROOT/research/ncfm_mathematical_analysis}"
PHASE2_TAG="${PHASE2_TAG:-phase2-$(date +%Y%m%d-%H%M%S)}"
PHASE2_SEEDS="${PHASE2_SEEDS:-0,1,2,3,4}"
NCFM_PATHMNIST_RUN_ID="${NCFM_PATHMNIST_RUN_ID:?baseline PathMNIST RUN_ID}"
NCFM_COVID_RUN_ID="${NCFM_COVID_RUN_ID:?baseline COVID RUN_ID}"
NCFM_KVASIR_RUN_ID="${NCFM_KVASIR_RUN_ID:?baseline Kvasir RUN_ID}"
BASELINE_EVAL_DEPENDENCY="${BASELINE_EVAL_DEPENDENCY:-}"
cd "$ROOT"
mkdir -p "$ROOT/logs"
mkdir -p "$MATH_ROOT"

IFS=',' read -r -a phase2_seed_list <<< "$PHASE2_SEEDS"
if (( ${#phase2_seed_list[@]} != 5 )); then
  echo "formal Phase 2 currently allocates exactly five seeds; got $PHASE2_SEEDS" >&2
  exit 2
fi
IFS=$' \t\n'

submit_variant() {
  local dataset="$1" slug="$2" variant="$3" seed="$4" teacher_var="$5"
  local nproc
  case "$dataset" in
    PathMNIST) nproc=3 ;;
    COVID) nproc=4 ;;
    Kvasir) nproc=8 ;;
  esac
  local dependency_args=()
  if [[ -n "$BASELINE_EVAL_DEPENDENCY" ]]; then
    dependency_args+=(--dependency="$BASELINE_EVAL_DEPENDENCY")
  fi
  sbatch --parsable "${dependency_args[@]}" --job-name="NCFM-${variant}-${slug}-s${seed}" \
    --gres="gpu:${nproc}" \
    --export="ALL,NCFM_MATH_ROOT=${MATH_ROOT},DATASET=${dataset},VARIANT=${variant},SEED=${seed},RUN_ID=${PHASE2_TAG}-${variant}-${slug}-seed${seed},TEACHER_RUN_ID=${!5},NPROC_PER_NODE=${nproc}" \
    scripts/ncfm_condense_variant.sbatch
}

declare -a jobs=()
for item in "PathMNIST pathmnist NCFM_PATHMNIST_RUN_ID" "COVID covid NCFM_COVID_RUN_ID" "Kvasir kvasir NCFM_KVASIR_RUN_ID"; do
  read -r dataset slug teacher_var <<< "$item"
  for variant in qmc importance learned_frequency; do
    for seed in ${PHASE2_SEEDS//,/ }; do
      jobs+=("$(submit_variant "$dataset" "$slug" "$variant" "$seed" "$teacher_var")")
    done
  done
done

dep="afterok:$(IFS=:; echo "${jobs[*]}")"
MANIFEST_JOB="$(sbatch --parsable --job-name=ncfm-p2-manifest --dependency="$dep" \
  --export="ALL,NCFM_MATH_ROOT=${MATH_ROOT},PHASE2_TAG=${PHASE2_TAG},PHASE2_SEEDS=${PHASE2_SEEDS}" \
  --wrap="cd '$ROOT' && '/home/xiaoyuxu2/.conda/envs/meddd/bin/python' scripts/build_phase2_variant_manifest.py --root '$ROOT' --phase2-tag '$PHASE2_TAG' --seeds '$PHASE2_SEEDS'")"
EVAL_JOB="$(sbatch --parsable --job-name=ncfm-p2-eval --dependency="afterok:${MANIFEST_JOB}" \
  --export="ALL,NCFM_MATH_ROOT=${MATH_ROOT},PHASE2_VARIANT_MANIFEST=${MATH_ROOT}/phase2_variant_manifest.json" \
  scripts/run_phase2_variant_eval.sbatch)"
REPORT_JOB="$(sbatch --parsable --job-name=ncfm-p2-report --dependency="afterany:${EVAL_JOB}" \
  --export="ALL,NCFM_MATH_ROOT=${MATH_ROOT}" \
  scripts/summarize_phase2_variants.sbatch)"
printf 'phase2_tag=%s\nphase2_seeds=%s\nbaseline_eval_dependency=%s\nvariant_job_count=%s\nvariant_jobs=%s\nmanifest=%s\nevaluation=%s\nreport=%s\n' \
  "$PHASE2_TAG" "$PHASE2_SEEDS" "${BASELINE_EVAL_DEPENDENCY:-none}" "${#jobs[@]}" "${jobs[*]}" "$MANIFEST_JOB" "$EVAL_JOB" "$REPORT_JOB"
