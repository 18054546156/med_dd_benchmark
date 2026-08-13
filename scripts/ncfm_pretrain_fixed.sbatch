#!/usr/bin/env bash
#SBATCH --job-name=NCFM-pretrain-fixed
#SBATCH --partition=gpu-rtx4090
#SBATCH --qos=qos-high-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/ncfm-pretrain-fixed-%x-%j.out
#SBATCH --error=logs/ncfm-pretrain-fixed-%x-%j.err

set -Eeuo pipefail
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

ROOT="${BENCHMARK_ROOT:-/project/prj-sis01/xiaoyu_xu/med_dd_project/dd_benchmark}"
DATASET="${DATASET:?set DATASET to PathMNIST, COVID, or Kvasir}"
PY="/home/xiaoyuxu2/.conda/envs/meddd/bin/python"
NPROC=4
GPU_IDS=0,1,2,3

case "$DATASET" in
  PathMNIST) SLUG=pathmnist ;;
  COVID) SLUG=covid ;;
  Kvasir) SLUG=kvasir ;;
  *) echo "unsupported dataset: $DATASET" >&2; exit 2 ;;
esac

CONFIG="$ROOT/configs/ncfm/$SLUG/ipc10_full_fixed.yaml"
PRETRAIN_DIR="$ROOT/pretrained_models/ncfm_fixed/$SLUG"
cd "$ROOT"
mkdir -p "$ROOT/logs" "$PRETRAIN_DIR"
export PYTHONPATH="$ROOT/adapted/ncfm:$ROOT/utils${PYTHONPATH:+:$PYTHONPATH}"

echo "===== NCFM fixed pretrain start ====="
echo "date=$(date -Is)"
echo "host=$(hostname)"
echo "dataset=$DATASET"
echo "config=$CONFIG"
echo "python=$PY"
nvidia-smi -L

"$PY" -m torch.distributed.run \
  --standalone \
  --nproc_per_node="$NPROC" \
  "$ROOT/adapted/ncfm/pretrain/pretrain_script.py" \
  --config_path "$CONFIG" \
  --run_mode Pretrain \
  --gpu "$GPU_IDS" \
  --ipc 10

INIT_COUNT=$(find "$PRETRAIN_DIR" -maxdepth 1 -type f -name 'premodel*_init.pth.tar' | wc -l)
TRAINED_COUNT=$(find "$PRETRAIN_DIR" -maxdepth 1 -type f -name 'premodel*_trained.pth.tar' | wc -l)
echo "pretrain artifacts: init=$INIT_COUNT trained=$TRAINED_COUNT dir=$PRETRAIN_DIR"
test "$INIT_COUNT" -eq 20
test "$TRAINED_COUNT" -eq 20
echo "===== NCFM fixed pretrain complete date=$(date -Is) ====="
