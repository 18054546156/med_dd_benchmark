#!/usr/bin/env bash
# One GPU per explicit NCFM/HoP controlled-evaluation entry.
#SBATCH --job-name=dd-unified-eval
#SBATCH --partition=gpu-rtx4090
#SBATCH --qos=qos-high-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-11
#SBATCH --output=logs/dd-unified-eval_%A_%a.out
#SBATCH --error=logs/dd-unified-eval_%A_%a.err

set -Eeuo pipefail
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
ROOT="${BENCHMARK_ROOT:-/project/prj-sis01/xiaoyu_xu/med_dd_project/dd_benchmark}"
PY="/home/xiaoyuxu2/.conda/envs/meddd/bin/python"
MANIFEST="${CONTROLLED_EVAL_MANIFEST:?Set CONTROLLED_EVAL_MANIFEST to an explicit 12-entry manifest}"
cd "$ROOT"
test -x "$PY"
if type module >/dev/null 2>&1; then
  module purge
  module load anaconda3/3.12
  module load cuda/12.6
fi
exec "$PY" research/ncfm_medical_analysis/code/run_eval_manifest.py \
  --root "$ROOT" --manifest "$MANIFEST" --index "$SLURM_ARRAY_TASK_ID" \
  --repeats "${REPEATS:-5}"
