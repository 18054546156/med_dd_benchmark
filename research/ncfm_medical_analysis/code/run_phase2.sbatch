#!/usr/bin/env bash
# Phase 2 mechanism diagnostics: QMC, exact importance sampling, and the
# holdout frequency certificate on each real dataset.  The actual three
# condenser variants (QMC, importance, learned_frequency) are submitted by
# scripts/submit_phase2_variants.sh.  This job never replaces the baseline.
#SBATCH --job-name=ncfm-phase2
#SBATCH --partition=gpu-rtx4090
#SBATCH --qos=qos-high-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --array=0-8
#SBATCH --output=logs/ncfm-phase2_%A_%a.out
#SBATCH --error=logs/ncfm-phase2_%A_%a.err

set -Eeuo pipefail
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1

ROOT="${BENCHMARK_ROOT:-/project/prj-sis01/xiaoyu_xu/med_dd_project/dd_benchmark}"
export NCFM_MATH_ROOT="${NCFM_MATH_ROOT:-$ROOT/research/ncfm_mathematical_analysis}"
PY="/home/xiaoyuxu2/.conda/envs/meddd/bin/python"
MANIFEST="${ARTIFACT_MANIFEST:?Set ARTIFACT_MANIFEST to the filled formal artifact manifest}"
cd "$ROOT"
test -x "$PY"
test -f "$MANIFEST"
if type module >/dev/null 2>&1; then
  module purge
  module load anaconda3/3.12
  module load cuda/12.6
fi

case "$SLURM_ARRAY_TASK_ID" in
  0) DATASET=PathMNIST; METHOD=qmc ;;
  1) DATASET=PathMNIST; METHOD=importance ;;
  2) DATASET=PathMNIST; METHOD=certificate ;;
  3) DATASET=COVID;     METHOD=qmc ;;
  4) DATASET=COVID;     METHOD=importance ;;
  5) DATASET=COVID;     METHOD=certificate ;;
  6) DATASET=Kvasir;    METHOD=qmc ;;
  7) DATASET=Kvasir;    METHOD=importance ;;
  8) DATASET=Kvasir;    METHOD=certificate ;;
  *) exit 2 ;;
esac

RUN_ID="phase2_${DATASET,,}_${METHOD}_array${SLURM_ARRAY_JOB_ID}_task${SLURM_ARRAY_TASK_ID}"
OUTPUT="$NCFM_MATH_ROOT/phase2/$RUN_ID/results.json"
exec "$PY" research/ncfm_medical_analysis/code/phase2_analysis.py \
  --dataset "$DATASET" --method "$METHOD" --run-id "$RUN_ID" \
  --artifact-manifest "$MANIFEST" --replicas "${REPLICAS:-20}" \
  --num-freqs "${NUM_FREQS:-4096}" --output "$OUTPUT"
