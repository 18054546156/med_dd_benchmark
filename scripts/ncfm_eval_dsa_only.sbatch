#!/usr/bin/env bash
# NCFM PathMNIST evaluation - 仅 DSA，无 CutMix
# 用于与 HoP 公平对比
#SBATCH --job-name=NCFM-eval-dsa-only
#SBATCH --partition=gpu-rtx4090
#SBATCH --qos=qos-high-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/ncfm-eval-dsa-only-%j.out
#SBATCH --error=logs/ncfm-eval-dsa-only-%j.err

set -Eeuo pipefail
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

ROOT="/project/prj-sis01/xiaoyu_xu/med_dd_project/dd_benchmark"
PY="/home/xiaoyuxu2/.conda/envs/meddd/bin/python"
CONFIG="$ROOT/configs/ncfm/pathmnist/ipc10_eval_dsa_only.yaml"
LOAD_PATH="$ROOT/results/ncfm_fixed/pathmnist/condense/pathmnist/ipc10/adamw_lr_img_0.0010_numr_reqs4096_factor2_20260813-2210/distilled_data/data_20000.pt"

cd "$ROOT"
export PYTHONPATH="$ROOT/adapted/ncfm:$ROOT/utils${PYTHONPATH:+:$PYTHONPATH}"

test -x "$PY"
test -f "$CONFIG" || { echo "Config not found: $CONFIG" >&2; exit 1; }
test -f "$LOAD_PATH" || { echo "Data not found: $LOAD_PATH" >&2; exit 1; }

echo "===== NCFM PathMNIST evaluation (DSA only, no CutMix) start $(date -Is) ====="
echo "host=$(hostname)"
echo "python=$PY"
echo "config=$CONFIG"
echo "load_path=$LOAD_PATH"
echo "augmentation: DSA only (color_crop_cutout_flip_scale_rotate), no CutMix"
nvidia-smi -L

"$PY" -m torch.distributed.run \
  --standalone \
  --nproc_per_node=4 \
  "$ROOT/adapted/ncfm/evaluation/evaluation_script.py" \
  --config_path "$CONFIG" \
  --run_mode Evaluation \
  --gpu 0,1,2,3 \
  --ipc 10 \
  --load_path "$LOAD_PATH"

echo ""
echo "===== NCFM PathMNIST evaluation (DSA only) complete $(date -Is) ====="
echo ""
echo "Summary:"
echo "  Config: DSA only, no CutMix"
echo "  Purpose: Fair comparison with HoP"
echo "  Previous result (with CutMix): 83.287% ± 0.434"
echo ""
