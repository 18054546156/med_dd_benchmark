#!/usr/bin/env bash
# Merge four verified worker buffers, then run the original HoP-TM distill/eval.
#SBATCH --job-name=HoP-finalize
#SBATCH --partition=gpu-rtx4090
#SBATCH --qos=qos-high-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=7-00:00:00
#SBATCH --output=logs/hop-tm-finalize-%x-%j.out
#SBATCH --error=logs/hop-tm-finalize-%x-%j.err

set -Eeuo pipefail

ROOT="${BENCHMARK_ROOT:-/project/prj-sis01/xiaoyu_xu/med_dd_project/dd_benchmark}"
DATASET="${DATASET:?set DATASET=COVID or PathMNIST}"
RUN_ID="${RUN_ID:?set a unique RUN_ID}"
PY="${PYTHON:-/home/xiaoyuxu2/.conda/envs/meddd/bin/python}"

case "$DATASET" in
  COVID)
    SLUG=covid
    HOP_DATASET=COVID
    MODEL=ConvNetD5
    ;;
  PathMNIST)
    SLUG=pathmnist
    HOP_DATASET=PathMNIST
    MODEL=ConvNet
    ;;
  Kvasir)
    SLUG=kvasir
    HOP_DATASET=Kvasir
    MODEL=ConvNetD5
    ;;
  *)
    echo "Unsupported DATASET=$DATASET" >&2
    exit 2
    ;;
esac

HOP_ROOT="$ROOT/adapted/hop_tm"
CONFIG="$ROOT/configs/hop_tm/$SLUG/ipc10_full.yaml"
PARALLEL_ROOT="$ROOT/buffers/hop_tm_parallel/$SLUG/$RUN_ID"
FINAL_BUFFER_DIR="$ROOT/buffers/hop_tm/${HOP_DATASET}_NO_ZCA/$MODEL"
RESULT_DIR="$ROOT/results/hop_tm/$DATASET/ipc10/$RUN_ID"

mkdir -p "$ROOT/logs" "$RESULT_DIR"
cd "$ROOT"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$HOP_ROOT:$ROOT/utils${PYTHONPATH:+:$PYTHONPATH}"
export WANDB_MODE=offline
export OMP_NUM_THREADS=1

test -x "$PY"
test -f "$CONFIG"
test -d "$ROOT/data/prepared/$DATASET"

echo "===== HoP-TM finalize start $(date -Is) ====="
echo "dataset=$DATASET"
echo "run_id=$RUN_ID"
echo "config=$CONFIG"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L

if find "$FINAL_BUFFER_DIR" -maxdepth 1 -type f -name 'replay_buffer_*.pt' 2>/dev/null | grep -q .; then
  echo "Refusing to overwrite an existing official buffer: $FINAL_BUFFER_DIR" >&2
  exit 3
fi

# Verify all four independent Slurm jobs before any merge is performed.
"$PY" - "$PARALLEL_ROOT" "$HOP_DATASET" "$MODEL" "$FINAL_BUFFER_DIR" "$CONFIG" <<'PY'
import json
import os
import sys
from pathlib import Path
import torch

parallel_root, dataset_name, model_name, final_dir, config_path = sys.argv[1:]
all_trajectories = []
worker_files = []
for index in range(4):
    path = Path(parallel_root) / f"worker_{index}" / f"{dataset_name}_NO_ZCA" / model_name / "replay_buffer_0.pt"
    if not path.is_file():
        raise SystemExit(f"missing worker buffer: {path}")
    trajectories = torch.load(path, map_location="cpu")
    if len(trajectories) != 25:
        raise SystemExit(f"{path}: expected 25 trajectories, found {len(trajectories)}")
    for trajectory in trajectories:
        if len(trajectory) != 101:
            raise SystemExit(f"{path}: expected 101 states, found {len(trajectory)}")
    worker_files.append(str(path))
    all_trajectories.extend(trajectories)

if len(all_trajectories) != 100:
    raise SystemExit(f"expected 100 trajectories, found {len(all_trajectories)}")

out = Path(final_dir)
out.mkdir(parents=True, exist_ok=True)
existing = list(out.glob("replay_buffer_*.pt"))
if existing:
    raise SystemExit(f"refusing to overwrite existing files: {existing}")
for index in range(10):
    torch.save(all_trajectories[index * 10:(index + 1) * 10], out / f"replay_buffer_{index}.pt")
(out / "config_used.yaml").write_text(Path(config_path).read_text(encoding="utf-8"), encoding="utf-8")
(out / "manifest.json").write_text(json.dumps({
    "method": "HoP-TM",
    "dataset": dataset_name,
    "model": model_name,
    "trajectory_count": 100,
    "states_per_trajectory": 101,
    "buffer_files": 10,
    "worker_buffers": worker_files,
    "teacher_batch_train": 256,
    "train_epochs": 100,
    "syn_steps": 80,
    "iteration": 10000,
    "high_order": True,
}, indent=2) + "\n", encoding="utf-8")
print("verified and merged: trajectories=100 files=10 states_per_trajectory=101")
PY

echo "===== stage=buffer done $(date -Is) ====="
find "$FINAL_BUFFER_DIR" -maxdepth 1 -type f -name 'replay_buffer_*.pt' -printf '%f %s bytes\n' | sort -V

echo "===== stage=distill start $(date -Is) ====="
"$PY" "$HOP_ROOT/distill/distill_high_order_spl.py" \
  --cfg "$CONFIG" \
  --data_path "$ROOT/data/prepared" \
  --buffer_path "$ROOT/buffers/hop_tm" \
  --save_path "$RESULT_DIR" \
  --num_eval 1 \
  --Iteration 10000
echo "===== stage=distill done $(date -Is) ====="

SYN_DIR=$(find "$RESULT_DIR" -type f -name 'images_best.pt' -printf '%T@ %h\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2- || true)
if [[ -z "$SYN_DIR" ]]; then
  SYN_DIR=$(find "$RESULT_DIR" -type f -name 'images_10000.pt' -printf '%T@ %h\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2- || true)
fi
test -n "$SYN_DIR"

if [[ -f "$SYN_DIR/images_best.pt" ]]; then
  IMAGE_FILE="$SYN_DIR/images_best.pt"
  LABEL_FILE="$SYN_DIR/labels_best.pt"
  LR_FILE="$SYN_DIR/lr_best.pt"
else
  IMAGE_FILE="$SYN_DIR/images_10000.pt"
  LABEL_FILE="$SYN_DIR/labels_10000.pt"
  LR_FILE="$SYN_DIR/lr_10000.pt"
fi
test -f "$IMAGE_FILE"
test -f "$LABEL_FILE"
test -f "$LR_FILE"

echo "synthetic_dir=$SYN_DIR"
echo "===== stage=evaluation start $(date -Is) ====="
"$PY" "$HOP_ROOT/distill/evaluation.py" \
  --dataset "$HOP_DATASET" \
  --model "$MODEL" \
  --eval_mode S \
  --epoch_eval_train 1000 \
  --batch_real 256 \
  --batch_train 128 \
  --dsa True \
  --dsa_strategy color_crop_cutout_flip_scale_rotate \
  --data_path "$ROOT/data/prepared" \
  --data_dir "$IMAGE_FILE" \
  --label_dir "$LABEL_FILE" \
  --lr_dir "$LR_FILE" \
  2>&1 | tee "$RESULT_DIR/evaluation.log"
echo "===== stage=evaluation done $(date -Is) ====="
echo "===== HoP-TM finalize complete $(date -Is) ====="
echo "official_buffer=$FINAL_BUFFER_DIR"
echo "synthetic_dir=$SYN_DIR"
