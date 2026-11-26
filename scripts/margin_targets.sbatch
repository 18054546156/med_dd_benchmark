#!/usr/bin/env bash
#SBATCH --job-name=ncfm-margin-targets
#SBATCH --partition=gpu-rtx4090
#SBATCH --qos=qos-high-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=logs/margin-targets-%j.out
#SBATCH --error=logs/margin-targets-%j.err

set -Eeuo pipefail
: "${STUDY_ROOT:?}" "${TEACHER_DIR:?}" "${DATA_NPZ:?}"
cd "$STUDY_ROOT"
module purge
export OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
nvidia-smi -L
/home/xiaoyuxu2/.conda/envs/meddd/bin/python scripts/precompute_margin_targets.py \
  --data "$DATA_NPZ" --teacher-dir "$TEACHER_DIR" \
  --out "$STUDY_ROOT/outputs/margin_targets.npz"
