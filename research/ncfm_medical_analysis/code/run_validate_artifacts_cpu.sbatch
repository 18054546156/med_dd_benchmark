#!/usr/bin/env bash
# 提交 Phase 1 前的只读 artifact gate；失败时不应提交实验数组。
#SBATCH --job-name=ncfm-artifact-gate
#SBATCH --partition=cpu-amd9754
#SBATCH --qos=qos-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=logs/ncfm-artifact-gate_%j.out
#SBATCH --error=logs/ncfm-artifact-gate_%j.err

set -Eeuo pipefail
ROOT="/project/prj-sis01/xiaoyu_xu/med_dd_project/dd_benchmark"
export NCFM_MATH_ROOT="${NCFM_MATH_ROOT:-$ROOT/research/ncfm_mathematical_analysis}"
PY="/home/xiaoyuxu2/.conda/envs/meddd/bin/python"
: "${ARTIFACT_MANIFEST:?Set ARTIFACT_MANIFEST to a filled formal artifact manifest}"
cd "$ROOT"
exec "$PY" research/ncfm_medical_analysis/code/validate_formal_artifacts.py \
  --root "$ROOT" --manifest "$ARTIFACT_MANIFEST" \
  --output "$NCFM_MATH_ROOT/formal_artifact_validation_${SLURM_JOB_ID}.json"
