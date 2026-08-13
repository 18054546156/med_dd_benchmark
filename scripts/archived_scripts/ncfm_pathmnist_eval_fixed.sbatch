#!/usr/bin/env bash
# NCFM PathMNIST evaluation。
# 该作业依赖 condense 作业 25170，只读取本次 20260813-2210 的 data_20000.pt。
# 不会重新执行 pretrain 或 condense，也不会读取旧的 20260813-2137 结果。
#SBATCH --job-name=NCFM-eval-fixed
#SBATCH --partition=gpu-rtx4090
#SBATCH --qos=qos-high-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/ncfm-eval-fixed-%j.out
#SBATCH --error=logs/ncfm-eval-fixed-%j.err

set -Eeuo pipefail
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

ROOT="/project/prj-sis01/xiaoyu_xu/med_dd_project/dd_benchmark"
PY="/home/xiaoyuxu2/.conda/envs/meddd/bin/python"
CONFIG="$ROOT/configs/ncfm/pathmnist/ipc10_full_fixed.yaml"
LOAD_PATH="$ROOT/results/ncfm_fixed/pathmnist/condense/pathmnist/ipc10/adamw_lr_img_0.0010_numr_reqs4096_factor2_20260813-2210/distilled_data/data_20000.pt"

cd "$ROOT"
export PYTHONPATH="$ROOT/adapted/ncfm:$ROOT/utils${PYTHONPATH:+:$PYTHONPATH}"

test -x "$PY"
test -f "$CONFIG"
test -f "$LOAD_PATH"

echo "===== NCFM PathMNIST evaluation start $(date -Is) ====="
echo "host=$(hostname)"
echo "python=$PY"
echo "config=$CONFIG"
echo "load_path=$LOAD_PATH"
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

echo "===== NCFM PathMNIST evaluation complete $(date -Is) ====="
