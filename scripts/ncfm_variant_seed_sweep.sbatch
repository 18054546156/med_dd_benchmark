#!/usr/bin/env bash
# Run five independent condenser seeds sequentially inside one allocation.
# This preserves independent random initializations while respecting the
# cluster's MaxSubmitJobsPerUser=15 policy.
#SBATCH --job-name=NCFM-seed-sweep
#SBATCH --partition=gpu-rtx4090
#SBATCH --qos=qos-high-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=7-00:00:00
#SBATCH --output=logs/ncfm-seed-sweep-%j.out
#SBATCH --error=logs/ncfm-seed-sweep-%j.err

set -Eeuo pipefail
export PYTHONUNBUFFERED=1

ROOT="${BENCHMARK_ROOT:-/project/prj-sis01/xiaoyu_xu/med_dd_project/dd_benchmark}"
DATASET="${DATASET:?set DATASET=PathMNIST|COVID|Kvasir}"
VARIANT="${VARIANT:?set a condenser variant}"
SWEEP_TAG="${SWEEP_TAG:?set the Phase 1/2 sweep tag}"
TEACHER_RUN_ID="${TEACHER_RUN_ID:?set the explicit teacher run ID}"
NPROC="${NPROC_PER_NODE:?set the dataset-specific process count}"
SEEDS="${VARIANT_SEEDS:-0:1:2:3:4}"

case "$DATASET" in
  PathMNIST) SLUG=pathmnist ;;
  COVID) SLUG=covid ;;
  Kvasir) SLUG=kvasir ;;
  *) echo "unsupported DATASET=$DATASET" >&2; exit 2 ;;
esac

cd "$ROOT"
IFS=':' read -r -a seed_list <<< "${SEEDS//,/:}"
if (( ${#seed_list[@]} != 5 )); then
  echo "formal sweep requires exactly five seeds; got $SEEDS" >&2
  exit 2
fi
for seed in "${seed_list[@]}"; do
  if [[ ! "$seed" =~ ^[0-9]+$ ]]; then
    echo "invalid integer seed: $seed" >&2
    exit 2
  fi
done

for seed in "${seed_list[@]}"; do
  run_id="${SWEEP_TAG}-${VARIANT}-${SLUG}-seed${seed}"
  run_stdout="$ROOT/logs/ncfm-variant-${SLURM_JOB_ID}-${run_id}.out"
  run_stderr="$ROOT/logs/ncfm-variant-${SLURM_JOB_ID}-${run_id}.err"
  echo "===== variant=$VARIANT dataset=$DATASET seed=$seed run_id=$run_id start=$(date -Is) ====="
  echo "stdout=$run_stdout"
  echo "stderr=$run_stderr"
  BENCHMARK_ROOT="$ROOT" \
  DATASET="$DATASET" \
  VARIANT="$VARIANT" \
  SEED="$seed" \
  RUN_ID="$run_id" \
  TEACHER_RUN_ID="$TEACHER_RUN_ID" \
  NPROC_PER_NODE="$NPROC" \
  SLURM_STDOUT_PATH="$run_stdout" \
  SLURM_STDERR_PATH="$run_stderr" \
  bash scripts/ncfm_condense_variant.sbatch >"$run_stdout" 2>"$run_stderr"
  echo "===== variant=$VARIANT dataset=$DATASET seed=$seed run_id=$run_id done=$(date -Is) ====="
done

echo "status=complete dataset=$DATASET variant=$VARIANT seeds=$SEEDS sweep_tag=$SWEEP_TAG"
