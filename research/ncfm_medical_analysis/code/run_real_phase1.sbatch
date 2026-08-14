#!/usr/bin/env bash
# Real-data Phase 1 diagnostic array. This is intentionally separate from
# the legacy toy probes and refuses to run without the authoritative stats.
#SBATCH --job-name=ncfm-real-phase1
#SBATCH --partition=gpu-rtx4090
#SBATCH --qos=qos-high-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
# Three Slurm tasks each execute eight logical experiments sequentially.  This
# keeps the workflow below qos-high-gpu MaxSubmitJobsPerUser=15.
#SBATCH --array=0-2
#SBATCH --output=logs/ncfm-real-phase1_%A_%a.out
#SBATCH --error=logs/ncfm-real-phase1_%A_%a.err

set -Eeuo pipefail
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1

ROOT="${BENCHMARK_ROOT:-/project/prj-sis01/xiaoyu_xu/med_dd_project/dd_benchmark}"
export NCFM_MATH_ROOT="${NCFM_MATH_ROOT:-$ROOT/research/ncfm_mathematical_analysis}"
PY="/home/xiaoyuxu2/.conda/envs/meddd/bin/python"
cd "$ROOT"
test -x "$PY"
if type module >/dev/null 2>&1; then
  module purge
  module load anaconda3/3.12
  module load cuda/12.6
fi
test -f data/prepared/PathMNIST/statistics.json
test -f data/prepared/COVID/statistics.json
test -f data/prepared/Kvasir/statistics.json
mkdir -p "$NCFM_MATH_ROOT"

: "${ARTIFACT_MANIFEST:?Set ARTIFACT_MANIFEST to the explicit formal artifact manifest}"
test -f "$ARTIFACT_MANIFEST"

export SLURM_STDOUT_PATH="$ROOT/logs/ncfm-real-phase1_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out"
export SLURM_STDERR_PATH="$ROOT/logs/ncfm-real-phase1_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.err"

DATASETS=(PathMNIST COVID Kvasir)
EXPERIMENTS=(e1.1 e1.2 e2.1 e2.2 e3.1 e3.2 e4.1 e4.2)
for (( logical_index=SLURM_ARRAY_TASK_ID; logical_index<24; logical_index+=SLURM_ARRAY_TASK_COUNT )); do
  dataset_index=$((logical_index / 8))
  experiment_index=$((logical_index % 8))
  DATASET="${DATASETS[$dataset_index]}"
  EXP="${EXPERIMENTS[$experiment_index]}"
  RUN_ID="phase1_real_${DATASET,,}_${EXP//./_}_array${SLURM_ARRAY_JOB_ID}_task${SLURM_ARRAY_TASK_ID}_index${logical_index}"
  echo "===== logical_index=$logical_index dataset=$DATASET experiment=$EXP start=$(date -Is) ====="
  "$PY" research/ncfm_medical_analysis/code/real_phase1.py \
    --dataset "$DATASET" \
    --experiment "$EXP" \
    --run-id "$RUN_ID" \
    --num-freqs "${NUM_FREQS:-4096}" \
    --replicas "${REPLICAS:-20}" \
    --max-samples "${MAX_SAMPLES:-1024}" \
    --teacher-id "${TEACHER_ID:-0}" \
    --artifact-manifest "$ARTIFACT_MANIFEST"
done
