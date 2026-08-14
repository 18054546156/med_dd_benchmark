#!/usr/bin/env bash
# Controlled downstream evaluation. DATASET, METHOD, DATA, and MODEL must be
# supplied explicitly; no old result is selected implicitly.
#SBATCH --job-name=dd-controlled-eval
#SBATCH --partition=gpu-rtx4090
#SBATCH --qos=qos-high-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/dd-controlled-eval_%j.out
#SBATCH --error=logs/dd-controlled-eval_%j.err

set -Eeuo pipefail
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
ROOT="/project/prj-sis01/xiaoyu_xu/med_dd_project/dd_benchmark"
PY="/home/xiaoyuxu2/.conda/envs/meddd/bin/python"
: "${DATASET:?Set DATASET=PathMNIST|COVID|Kvasir}"
: "${METHOD:?Set METHOD=NCFM|HoP}"
: "${DATA:?Set DATA to an explicit synthetic file or directory}"
: "${MODEL:=ConvNet}"
: "${REPEATS:=5}"
: "${RUN_ID:=job_${SLURM_JOB_ID}}"
cd "$ROOT"
test -x "$PY"
test -e "$DATA"
OUT="results/controlled_eval/${METHOD,,}/${DATASET}/${MODEL}/job_${SLURM_JOB_ID}.json"
exec "$PY" research/ncfm_medical_analysis/code/unified_eval_real.py \
  --data "$DATA" --dataset "$DATASET" --architecture "$MODEL" \
  --method "$METHOD" --run-id "$RUN_ID" \
  --repeats "$REPEATS" --output "$OUT"
