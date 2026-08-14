#!/usr/bin/env bash
# Validate the 12 explicit evaluator inputs before allocating GPU jobs.
#SBATCH --job-name=dd-eval-manifest-gate
#SBATCH --partition=cpu-amd9754
#SBATCH --qos=qos-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=logs/dd-eval-manifest-gate_%j.out
#SBATCH --error=logs/dd-eval-manifest-gate_%j.err

set -Eeuo pipefail
ROOT="/project/prj-sis01/xiaoyu_xu/med_dd_project/dd_benchmark"
export NCFM_MATH_ROOT="${NCFM_MATH_ROOT:-$ROOT/research/ncfm_mathematical_analysis}"
PY="/home/xiaoyuxu2/.conda/envs/meddd/bin/python"
MANIFEST="${CONTROLLED_EVAL_MANIFEST:?Set CONTROLLED_EVAL_MANIFEST}"
cd "$ROOT"
exec "$PY" research/ncfm_medical_analysis/code/validate_eval_manifest.py \
  --root "$ROOT" --manifest "$MANIFEST" \
  --output "$NCFM_MATH_ROOT/formal_eval_manifest_validation_${SLURM_JOB_ID}.json"
