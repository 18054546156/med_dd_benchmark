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
#SBATCH --array=0-23
#SBATCH --output=logs/ncfm-real-phase1_%A_%a.out
#SBATCH --error=logs/ncfm-real-phase1_%A_%a.err

set -Eeuo pipefail
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1

ROOT="/project/prj-sis01/xiaoyu_xu/med_dd_project/dd_benchmark"
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

case "$SLURM_ARRAY_TASK_ID" in
  0) DATASET=PathMNIST; EXP=e1.1 ;;  1) DATASET=PathMNIST; EXP=e1.2 ;;
  2) DATASET=PathMNIST; EXP=e2.1 ;;  3) DATASET=PathMNIST; EXP=e2.2 ;;
  4) DATASET=PathMNIST; EXP=e3.1 ;;  5) DATASET=PathMNIST; EXP=e3.2 ;;
  6) DATASET=PathMNIST; EXP=e4.1 ;;  7) DATASET=PathMNIST; EXP=e4.2 ;;
  8) DATASET=COVID;     EXP=e1.1 ;;  9) DATASET=COVID;     EXP=e1.2 ;;
  10) DATASET=COVID;    EXP=e2.1 ;;  11) DATASET=COVID;    EXP=e2.2 ;;
  12) DATASET=COVID;    EXP=e3.1 ;; 13) DATASET=COVID;    EXP=e3.2 ;;
  14) DATASET=COVID;    EXP=e4.1 ;; 15) DATASET=COVID;    EXP=e4.2 ;;
  16) DATASET=Kvasir;   EXP=e1.1 ;; 17) DATASET=Kvasir;   EXP=e1.2 ;;
  18) DATASET=Kvasir;   EXP=e2.1 ;; 19) DATASET=Kvasir;   EXP=e2.2 ;;
  20) DATASET=Kvasir;   EXP=e3.1 ;; 21) DATASET=Kvasir;   EXP=e3.2 ;;
  22) DATASET=Kvasir;   EXP=e4.1 ;; 23) DATASET=Kvasir;   EXP=e4.2 ;;
  *) exit 2 ;;
esac

: "${ARTIFACT_MANIFEST:?Set ARTIFACT_MANIFEST to the explicit formal artifact manifest}"
test -f "$ARTIFACT_MANIFEST"

RUN_ID="phase1_real_${DATASET,,}_${EXP//./_}_array${SLURM_ARRAY_JOB_ID}_task${SLURM_ARRAY_TASK_ID}"
export SLURM_STDOUT_PATH="$ROOT/logs/ncfm-real-phase1_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out"
export SLURM_STDERR_PATH="$ROOT/logs/ncfm-real-phase1_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.err"
ARGS=(
  --dataset "$DATASET"
  --experiment "$EXP"
  --run-id "$RUN_ID"
  --num-freqs "${NUM_FREQS:-4096}"
  --replicas "${REPLICAS:-20}"
  --max-samples "${MAX_SAMPLES:-1024}"
  --teacher-id "${TEACHER_ID:-0}"
  --artifact-manifest "$ARTIFACT_MANIFEST"
)
exec "$PY" research/ncfm_medical_analysis/code/real_phase1.py "${ARGS[@]}"
