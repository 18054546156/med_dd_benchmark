#!/usr/bin/env bash
# One Slurm job owns one GPU and generates one quarter of the HoP-TM buffer.
# The algorithm and teacher hyperparameters remain those of buffer_FTD.py.
#SBATCH --job-name=HoP-worker
#SBATCH --partition=gpu-rtx4090
#SBATCH --qos=qos-high-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=7-00:00:00
#SBATCH --output=logs/hop-tm-worker-%x-%j.out
#SBATCH --error=logs/hop-tm-worker-%x-%j.err

set -Eeuo pipefail

ROOT="${BENCHMARK_ROOT:-/project/prj-sis01/xiaoyu_xu/med_dd_project/dd_benchmark}"
DATASET="${DATASET:?set DATASET=COVID or PathMNIST}"
RUN_ID="${RUN_ID:?set a unique RUN_ID}"
WORKER_INDEX="${WORKER_INDEX:?set WORKER_INDEX=0..3}"
PY="${PYTHON:-/home/xiaoyuxu2/.conda/envs/meddd/bin/python}"

case "$DATASET" in
  COVID)
    SLUG=covid
    HOP_DATASET=COVID
    MODEL=ConvNetD5
    LR_TEACHER=0.001
    ;;
  PathMNIST)
    SLUG=pathmnist
    HOP_DATASET=PathMNIST
    MODEL=ConvNet
    LR_TEACHER=0.01
    ;;
  Kvasir)
    SLUG=kvasir
    HOP_DATASET=Kvasir
    MODEL=ConvNetD5
    LR_TEACHER=0.001
    ;;
  *)
    echo "Unsupported DATASET=$DATASET" >&2
    exit 2
    ;;
esac

WORKER_ROOT="$ROOT/buffers/hop_tm_parallel/$SLUG/$RUN_ID/worker_$WORKER_INDEX"
BUFFER_DIR="$WORKER_ROOT/${HOP_DATASET}_NO_ZCA/$MODEL"
LOG_DIR="$ROOT/logs"
HOP_ROOT="$ROOT/adapted/hop_tm"

mkdir -p "$LOG_DIR" "$WORKER_ROOT"
cd "$ROOT"

export PYTHONUNBUFFERED=1
export PYTHONPATH="$HOP_ROOT:$ROOT/utils${PYTHONPATH:+:$PYTHONPATH}"
export WANDB_MODE=offline
export OMP_NUM_THREADS=1

test -x "$PY"
test -d "$ROOT/data/prepared/$DATASET"
if [[ -e "$BUFFER_DIR/replay_buffer_0.pt" ]]; then
  echo "Refusing to overwrite existing worker buffer: $BUFFER_DIR/replay_buffer_0.pt" >&2
  exit 3
fi

echo "===== HoP-TM worker start $(date -Is) ====="
echo "dataset=$DATASET"
echo "run_id=$RUN_ID"
echo "worker_index=$WORKER_INDEX"
echo "host=$(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L

# buffer_FTD.py defaults to GPU 0 only when Slurm did not set a mapping.
# Because this job requests exactly one GPU, CUDA device 0 is local to this job.
"$PY" "$HOP_ROOT/buffer/buffer_FTD.py" \
  --dataset "$HOP_DATASET" \
  --model "$MODEL" \
  --num_experts 25 \
  --train_epochs 100 \
  --save_interval 25 \
  --lr_teacher "$LR_TEACHER" \
  --batch_train 256 \
  --batch_real 256 \
  --dsa True \
  --dsa_strategy color_crop_cutout_flip_scale_rotate \
  --data_path "$ROOT/data/prepared" \
  --buffer_path "$WORKER_ROOT"

test -s "$BUFFER_DIR/replay_buffer_0.pt"
"$PY" - "$BUFFER_DIR/replay_buffer_0.pt" "$WORKER_ROOT/worker_manifest.json" <<'PY'
import json
import sys
from pathlib import Path
import torch

buffer_path, manifest_path = sys.argv[1:]
trajectories = torch.load(buffer_path, map_location="cpu")
if len(trajectories) != 25:
    raise SystemExit(f"expected 25 trajectories, found {len(trajectories)}")
state_counts = {len(trajectory) for trajectory in trajectories}
if state_counts != {101}:
    raise SystemExit(f"expected 101 states per trajectory, found {state_counts}")
Path(manifest_path).write_text(json.dumps({
    "trajectory_count": 25,
    "states_per_trajectory": 101,
    "buffer": str(Path(buffer_path).resolve()),
}, indent=2) + "\n", encoding="utf-8")
print(f"verified worker buffer: trajectories=25 states_per_trajectory=101")
PY

echo "===== HoP-TM worker complete $(date -Is) ====="
