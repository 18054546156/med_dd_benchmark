#!/usr/bin/env bash
#SBATCH --job-name=NCFM-condense-fixed
#SBATCH --partition=gpu-rtx4090
#SBATCH --qos=qos-high-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/ncfm-condense-fixed-%x-%j.out
#SBATCH --error=logs/ncfm-condense-fixed-%x-%j.err

# 只执行 NCFM condense，不会重新运行 pretrain。
set -Eeuo pipefail
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

ROOT="${BENCHMARK_ROOT:-/project/prj-sis01/xiaoyu_xu/med_dd_project/dd_benchmark}"
PY="${PYTHON:-/home/xiaoyuxu2/.conda/envs/meddd/bin/python}"
CONFIG="$ROOT/configs/ncfm/pathmnist/ipc10_full_fixed.yaml"
PRETRAIN_DIR="$ROOT/pretrained_models/ncfm_fixed/pathmnist"
RESULT_ROOT="$ROOT/results/ncfm_fixed/pathmnist"
NPROC="${NPROC_PER_NODE:-4}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"

cd "$ROOT"
mkdir -p "$ROOT/logs" "$RESULT_ROOT"
export PYTHONPATH="$ROOT/adapted/ncfm:$ROOT/utils${PYTHONPATH:+:$PYTHONPATH}"

test -x "$PY"
test -f "$CONFIG"
test -d "$ROOT/data/prepared/PathMNIST"
INIT_COUNT=$(find "$PRETRAIN_DIR" -maxdepth 1 -type f -name 'premodel*_init.pth.tar' | wc -l)
TRAINED_COUNT=$(find "$PRETRAIN_DIR" -maxdepth 1 -type f -name 'premodel*_trained.pth.tar' | wc -l)
test "$INIT_COUNT" -eq 20
test "$TRAINED_COUNT" -eq 20

echo "===== NCFM PathMNIST condense start $(date -Is) ====="
echo "host=$(hostname)"
echo "python=$PY"
echo "config=$CONFIG"
echo "pretrain_dir=$PRETRAIN_DIR"
echo "result_root=$RESULT_ROOT"
echo "nproc=$NPROC gpu_ids=$GPU_IDS"
nvidia-smi -L

"$PY" -c "import torch; print('torch', torch.__version__); print('cuda', torch.cuda.is_available()); print('gpu_count', torch.cuda.device_count())"

"$PY" -m torch.distributed.run \
  --standalone \
  --nproc_per_node="$NPROC" \
  "$ROOT/adapted/ncfm/condense/condense_script.py" \
  --config_path "$CONFIG" \
  --run_mode Condense \
  --gpu "$GPU_IDS" \
  --ipc 10

SYNTHETIC_PATH=$(find "$RESULT_ROOT" -type f -path '*/distilled_data/data_*.pt' -printf '%T@ %p\n' \
  | sort -n | tail -1 | cut -d' ' -f2-)
test -n "$SYNTHETIC_PATH"
test -f "$SYNTHETIC_PATH"
echo "synthetic_data=$SYNTHETIC_PATH"
echo "===== NCFM PathMNIST condense complete $(date -Is) ====="
