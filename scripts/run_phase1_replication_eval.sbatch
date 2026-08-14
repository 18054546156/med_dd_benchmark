#!/usr/bin/env bash
# Evaluate the 30 explicit E3.1/E4.2 replication artifacts.
#SBATCH --job-name=ncfm-phase1-rep-eval
#SBATCH --partition=gpu-rtx4090
#SBATCH --qos=qos-high-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
# Six tasks process five manifest entries each.
#SBATCH --array=0-5
#SBATCH --output=logs/ncfm-phase1-rep-eval_%A_%a.out
#SBATCH --error=logs/ncfm-phase1-rep-eval_%A_%a.err

set -Eeuo pipefail
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
ROOT="${BENCHMARK_ROOT:-/project/prj-sis01/xiaoyu_xu/med_dd_project/dd_benchmark}"
PY="${PYTHON:-/home/xiaoyuxu2/.conda/envs/meddd/bin/python}"
MANIFEST="${PHASE1_REPLICATION_MANIFEST:?set PHASE1_REPLICATION_MANIFEST}"
cd "$ROOT"
test -x "$PY"
test -f "$MANIFEST"
if type module >/dev/null 2>&1; then
  module purge
  module load anaconda3/3.12
  module load cuda/12.6
fi
ENTRY_COUNT="$($PY -c 'import json,sys; print(len(json.load(open(sys.argv[1]))["evaluations"]))' "$MANIFEST")"
test "$ENTRY_COUNT" -eq 30
for (( index=SLURM_ARRAY_TASK_ID; index<ENTRY_COUNT; index+=SLURM_ARRAY_TASK_COUNT )); do
  echo "===== replication evaluation index=$index start=$(date -Is) ====="
  "$PY" scripts/run_variant_eval.py \
    --root "$ROOT" --manifest "$MANIFEST" --index "$index" \
    --repeats "${REPEATS:-5}"
done
