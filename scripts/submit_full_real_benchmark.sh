#!/usr/bin/env bash
# Submit the complete real-data benchmark with Slurm dependencies.
#
# Production graph:
#   NCFM(PathMNIST/COVID/Kvasir) + HoP(PathMNIST/COVID/Kvasir)
#     -> explicit manifest builder
#     -> statistics/contracts/task analysis
#     -> controlled evaluation + Phase 1 + Phase 2
#     -> formal report + figures
#
# This submitter never discovers an old result by mtime.  Each run receives a
# unique RUN_ID and every downstream job receives the resulting manifest.
set -Eeuo pipefail

ROOT="${BENCHMARK_ROOT:-/project/prj-sis01/xiaoyu_xu/med_dd_project/dd_benchmark}"
export NCFM_MATH_ROOT="${NCFM_MATH_ROOT:-$ROOT/research/ncfm_mathematical_analysis}"
RUN_TAG="${RUN_TAG:-formal-$(date +%Y%m%d-%H%M%S)}"

cd "$ROOT"
mkdir -p "$ROOT/logs"

STATS_JOB="$(sbatch --parsable research/ncfm_medical_analysis/code/run_statistics_cpu.sbatch)"
CONTRACT_JOB="$(sbatch --parsable --dependency="afterok:${STATS_JOB}" \
  research/ncfm_medical_analysis/code/run_contract_cpu.sbatch)"
RUNTIME_JOB="$(sbatch --parsable --dependency="afterok:${STATS_JOB}:${CONTRACT_JOB}" \
  research/ncfm_medical_analysis/code/run_runtime_contract_gpu.sbatch)"
PRECHECK_DEP="afterok:${STATS_JOB}:${CONTRACT_JOB}:${RUNTIME_JOB}"

submit_ncfm() {
  local dataset="$1" slug="$2" dependency="$3"
  local nproc
  case "$dataset" in
    PathMNIST) nproc=3 ;;
    COVID) nproc=4 ;;
    Kvasir) nproc=8 ;;
  esac
  sbatch --parsable \
    --job-name="NCFM-${slug}" \
    --gres="gpu:${nproc}" \
    --dependency="$dependency" \
    --export="ALL,NCFM_MATH_ROOT=${NCFM_MATH_ROOT},DATASET=${dataset},RUN_ID=${RUN_TAG}-ncfm-${slug},NPROC_PER_NODE=${nproc}" \
    scripts/ncfm_pipeline.sbatch
}

submit_hop() {
  local dataset="$1" slug="$2" dependency="$3"
  sbatch --parsable \
    --job-name="HoP-${slug}" \
    --dependency="$dependency" \
    --export="ALL,NCFM_MATH_ROOT=${NCFM_MATH_ROOT},DATASET=${dataset},RUN_ID=${RUN_TAG}-hop-${slug}" \
    scripts/hop_tm_pipeline_4gpu.sbatch
}

NCFM_PATHMNIST_JOB="$(submit_ncfm PathMNIST pathmnist "$PRECHECK_DEP")"
NCFM_COVID_JOB="$(submit_ncfm COVID covid "$PRECHECK_DEP")"
NCFM_KVASIR_JOB="$(submit_ncfm Kvasir kvasir "$PRECHECK_DEP")"
HOP_PATHMNIST_JOB="$(submit_hop PathMNIST pathmnist "$PRECHECK_DEP")"
HOP_COVID_JOB="$(submit_hop COVID covid "$PRECHECK_DEP")"
HOP_KVASIR_JOB="$(submit_hop Kvasir kvasir "$PRECHECK_DEP")"

# The follow-up must run after all production jobs settle, including failure.
# Its wrapper records insufficient_evidence instead of leaving a dependency
# chain stuck at DependencyNeverSatisfied.
MANIFEST_JOB="$(sbatch --parsable \
  --job-name=dd-manifests \
  --dependency="afterany:${NCFM_PATHMNIST_JOB}:${NCFM_COVID_JOB}:${NCFM_KVASIR_JOB}:${HOP_PATHMNIST_JOB}:${HOP_COVID_JOB}:${HOP_KVASIR_JOB}" \
  --export="ALL,NCFM_MATH_ROOT=${NCFM_MATH_ROOT},NCFM_PATHMNIST_RUN_ID=${RUN_TAG}-ncfm-pathmnist,NCFM_COVID_RUN_ID=${RUN_TAG}-ncfm-covid,NCFM_KVASIR_RUN_ID=${RUN_TAG}-ncfm-kvasir,HOP_PATHMNIST_RUN_ID=${RUN_TAG}-hop-pathmnist,HOP_COVID_RUN_ID=${RUN_TAG}-hop-covid,HOP_KVASIR_RUN_ID=${RUN_TAG}-hop-kvasir" \
  scripts/build_formal_manifests.sbatch)"

# This follow-up either submits the full dependency-aware analysis or writes
# an explicit insufficient-evidence report when manifest construction failed.
ANALYSIS_SUBMIT_JOB="$(sbatch --parsable \
  --job-name=dd-analysis-submit \
  --partition=cpu-amd9754 \
  --qos=qos-normal \
  --time=00:15:00 \
  --dependency="afterany:${MANIFEST_JOB}" \
  --export="ALL,NCFM_MATH_ROOT=${NCFM_MATH_ROOT},BENCHMARK_ROOT=${ROOT},BUILD_JOB_ID=${MANIFEST_JOB},RUN_TAG=${RUN_TAG}" \
  scripts/finalize_production.sbatch)"

# Persist the dependency graph because the login/VPN session may disconnect
# after submission. This record contains IDs only; artifacts remain under the
# run-specific manifests produced by the jobs.
SUBMISSION_DIR="$NCFM_MATH_ROOT/submissions"
mkdir -p "$SUBMISSION_DIR"
SUBMISSION_RECORD="$SUBMISSION_DIR/${RUN_TAG}.env"
{
  printf 'run_tag=%s\n' "$RUN_TAG"
  printf 'statistics=%s\n' "$STATS_JOB"
  printf 'contract=%s\n' "$CONTRACT_JOB"
  printf 'runtime=%s\n' "$RUNTIME_JOB"
  printf 'ncfm_pathmnist=%s\n' "$NCFM_PATHMNIST_JOB"
  printf 'ncfm_covid=%s\n' "$NCFM_COVID_JOB"
  printf 'ncfm_kvasir=%s\n' "$NCFM_KVASIR_JOB"
  printf 'hop_pathmnist=%s\n' "$HOP_PATHMNIST_JOB"
  printf 'hop_covid=%s\n' "$HOP_COVID_JOB"
  printf 'hop_kvasir=%s\n' "$HOP_KVASIR_JOB"
  printf 'manifest=%s\n' "$MANIFEST_JOB"
  printf 'analysis_submit=%s\n' "$ANALYSIS_SUBMIT_JOB"
} > "$SUBMISSION_RECORD"

cat <<EOF
run_tag=${RUN_TAG}
statistics=${STATS_JOB}
contract=${CONTRACT_JOB}
runtime=${RUNTIME_JOB}
ncfm_pathmnist=${NCFM_PATHMNIST_JOB}
ncfm_covid=${NCFM_COVID_JOB}
ncfm_kvasir=${NCFM_KVASIR_JOB}
hop_pathmnist=${HOP_PATHMNIST_JOB}
hop_covid=${HOP_COVID_JOB}
hop_kvasir=${HOP_KVASIR_JOB}
manifest=${MANIFEST_JOB}
analysis_submit=${ANALYSIS_SUBMIT_JOB}
submission_record=${SUBMISSION_RECORD}
phase2_submit=handled_by_${ANALYSIS_SUBMIT_JOB}
EOF
