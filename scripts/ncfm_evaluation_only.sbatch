#!/usr/bin/env bash
# NCFM 已完成 condense 后的 evaluation-only 作业。
# 用法：sbatch --export=DATASET=PathMNIST scripts/ncfm_evaluation_only.sbatch
# 该脚本不会重新执行 pretrain 或 condense。
#SBATCH --job-name=NCFM-eval
#SBATCH --partition=gpu-rtx4090
#SBATCH --qos=qos-high-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/ncfm-eval-%j.out
#SBATCH --error=logs/ncfm-eval-%j.err

set -Eeuo pipefail

ROOT="${BENCHMARK_ROOT:-/project/prj-sis01/xiaoyu_xu/med_dd_project/dd_benchmark}"
DATASET="${DATASET:?请通过 --export=DATASET=PathMNIST 指定数据集}"
PY="${PYTHON:-/home/xiaoyuxu2/.conda/envs/meddd/bin/python}"
NPROC="${NPROC_PER_NODE:-4}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"

case "$DATASET" in
  PathMNIST) SLUG="pathmnist" ;;
  Kvasir) SLUG="kvasir" ;;
  COVID) SLUG="covid" ;;
  *) echo "不支持的数据集: $DATASET" >&2; exit 2 ;;
esac

CONFIG="$ROOT/configs/ncfm/$SLUG/ipc10_full.yaml"
RESULT_ROOT="$ROOT/results/ncfm/$SLUG"
LOG_ROOT="$ROOT/logs"

mkdir -p "$LOG_ROOT"
cd "$ROOT"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTHONPATH="$ROOT/adapted/ncfm:$ROOT/utils${PYTHONPATH:+:$PYTHONPATH}"

SYNTHETIC_PATH="${SYNTHETIC_PATH:-}"
if [[ -z "$SYNTHETIC_PATH" ]]; then
  echo "必须通过 SYNTHETIC_PATH 显式指定本次实验的 data_20000.pt" >&2
  exit 2
fi

test -x "$PY"
test -f "$CONFIG"
test -n "$SYNTHETIC_PATH"
test -f "$SYNTHETIC_PATH"

echo "===== NCFM evaluation-only start $(date -Is) ====="
echo "dataset=$DATASET"
echo "config=$CONFIG"
echo "synthetic_data=$SYNTHETIC_PATH"
echo "python=$PY"
nvidia-smi -L

"$PY" -m torch.distributed.run \
  --standalone \
  --nproc_per_node="$NPROC" \
  "$ROOT/adapted/ncfm/evaluation/evaluation_script.py" \
  --config_path "$CONFIG" \
  --run_mode Evaluation \
  --gpu "$GPU_IDS" \
  --ipc 10 \
  --load_path "$SYNTHETIC_PATH"

echo "===== NCFM evaluation-only complete $(date -Is) ====="
