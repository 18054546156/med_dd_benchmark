#!/usr/bin/env bash
#SBATCH --job-name=ncfm-margin-smoke
#SBATCH --partition=gpu-h100
#SBATCH --qos=qos-high-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --output=logs/margin-smoke-%j.out
#SBATCH --error=logs/margin-smoke-%j.err

set -Eeuo pipefail
: "${STUDY_ROOT:?}"
cd "$STUDY_ROOT"
export NCFM_SMOKE=1 NCFM_RUN_TAG="smoke_h100_${SLURM_JOB_ID}"
for id in 0 3 6; do
    SLURM_ARRAY_TASK_ID="$id" bash scripts/margin_run.sbatch
done
