#!/usr/bin/env bash
# Controlled evaluation of all 3 real Phase 2 NCFM variants.
#SBATCH --job-name=ncfm-p2-eval
#SBATCH --partition=gpu-rtx4090
#SBATCH --qos=qos-high-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=7-00:00:00
# 3 variants x 3 datasets x 5 seeds x 2 architectures = 90 entries.
# The submitter enforces at least five seeds; the formal default is exactly 5.
# Four tasks process 22 or 23 manifest entries each.  Together with the nine
# condenser jobs and follow-up jobs, this also keeps standalone submission at
# or below MaxSubmitJobsPerUser=15.
#SBATCH --array=0-3
#SBATCH --output=logs/ncfm-p2-eval_%A_%a.out
#SBATCH --error=logs/ncfm-p2-eval_%A_%a.err

set -Eeuo pipefail
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
ROOT="${BENCHMARK_ROOT:-/project/prj-sis01/xiaoyu_xu/med_dd_project/dd_benchmark}"
PY="${PYTHON:-/home/xiaoyuxu2/.conda/envs/meddd/bin/python}"
MANIFEST="${PHASE2_VARIANT_MANIFEST:?set PHASE2_VARIANT_MANIFEST}"
cd "$ROOT"
test -x "$PY"
test -f "$MANIFEST"
if type module >/dev/null 2>&1; then
  module purge
  module load anaconda3/3.12
  module load cuda/12.6
fi
ENTRY_COUNT="$($PY -c 'import json,sys; print(len(json.load(open(sys.argv[1]))["evaluations"]))' "$MANIFEST")"
test "$ENTRY_COUNT" -eq 90
for (( index=SLURM_ARRAY_TASK_ID; index<ENTRY_COUNT; index+=SLURM_ARRAY_TASK_COUNT )); do
  echo "===== Phase 2 evaluation index=$index start=$(date -Is) ====="
  "$PY" scripts/run_variant_eval.py \
    --root "$ROOT" --manifest "$MANIFEST" --index "$index" \
    --repeats "${REPEATS:-5}"
done
