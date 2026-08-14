#!/usr/bin/env bash
# Evaluate the 20 real NCFM teachers for task difficulty and tuning evidence.
#SBATCH --job-name=ncfm-task-analysis
#SBATCH --partition=gpu-rtx4090
#SBATCH --qos=qos-high-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --array=0-2
#SBATCH --output=logs/ncfm-task-analysis_%A_%a.out
#SBATCH --error=logs/ncfm-task-analysis_%A_%a.err

set -Eeuo pipefail
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
ROOT="/project/prj-sis01/xiaoyu_xu/med_dd_project/dd_benchmark"
PY="/home/xiaoyuxu2/.conda/envs/meddd/bin/python"
: "${ARTIFACT_MANIFEST:?Set ARTIFACT_MANIFEST to the filled formal artifact manifest}"
case "$SLURM_ARRAY_TASK_ID" in
  0) DATASET=PathMNIST ;;
  1) DATASET=COVID ;;
  2) DATASET=Kvasir ;;
  *) exit 2 ;;
esac
cd "$ROOT"
if type module >/dev/null 2>&1; then
  module purge
  module load anaconda3/3.12
  module load cuda/12.6
fi
exec "$PY" research/ncfm_medical_analysis/code/medical_task_analysis.py \
  --root "$ROOT" --dataset "$DATASET" \
  --artifact-manifest "$ARTIFACT_MANIFEST" \
  --output "$ROOT/research/ncfm_medical_analysis/task_analysis/${DATASET}.json"
