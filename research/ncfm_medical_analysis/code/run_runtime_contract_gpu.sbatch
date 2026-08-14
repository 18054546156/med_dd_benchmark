#!/usr/bin/env bash
# GPU preflight for the unified evaluator.  This does not train any model.
#SBATCH --job-name=ncfm-runtime-contract
#SBATCH --partition=gpu-rtx4090
#SBATCH --qos=qos-high-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/ncfm-runtime-contract_%j.out
#SBATCH --error=logs/ncfm-runtime-contract_%j.err

set -Eeuo pipefail
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
ROOT="${BENCHMARK_ROOT:-/project/prj-sis01/xiaoyu_xu/med_dd_project/dd_benchmark}"
export NCFM_MATH_ROOT="${NCFM_MATH_ROOT:-$ROOT/research/ncfm_mathematical_analysis}"
PY="/home/xiaoyuxu2/.conda/envs/meddd/bin/python"
cd "$ROOT"
test -x "$PY"
if type module >/dev/null 2>&1; then
  module purge
  module load anaconda3/3.12
  module load cuda/12.6
fi
"$PY" - <<'PY'
import torch

import matplotlib
import wandb
import yacs

if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available in the allocated GPU job")
print({
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "device_count": torch.cuda.device_count(),
    "device": torch.cuda.get_device_name(0),
    "matplotlib": matplotlib.__version__,
    "wandb": wandb.__version__,
    "yacs": getattr(yacs, "__version__", "imported"),
})
PY
exec "$PY" research/ncfm_medical_analysis/code/runtime_contract.py \
  --root "$ROOT" \
  --output "$NCFM_MATH_ROOT/runtime_contract_${SLURM_JOB_ID}.json"
