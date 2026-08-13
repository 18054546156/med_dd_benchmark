#!/usr/bin/env bash
#SBATCH --job-name=HoP-TM-4GPU
#SBATCH --partition=gpu-rtx4090
#SBATCH --qos=qos-high-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=7-00:00:00
#SBATCH --output=logs/hop-tm-4gpu-%x-%j.out
#SBATCH --error=logs/hop-tm-4gpu-%x-%j.err

# HoP-TM 4 GPU 流水线：并行生成 expert trajectory，随后原始 distill 和 DD evaluation。
#
# HoP-TM 官方代码本身不是 DDP，不能用 torchrun 把一个 teacher 强行拆成 4 卡。
# 这里的 4 卡并行方式是：
#   4 个独立单 GPU worker，每个生成 25 个 expert trajectory；
#   4 个 worker 完成后合并为官方格式的 100 条 trajectory；
#   原始 high-order distillation 和作者 evaluation.py 继续单 GPU 执行。
#
# 每个 expert 的官方参数不变：train_epochs=100、expert_epochs=2、
# syn_steps=80、Iteration=10000；仅把独立 expert 任务分片，不改变损失或更新规则。

set -Eeuo pipefail

ROOT="${BENCHMARK_ROOT:-/project/prj-sis01/xiaoyu_xu/med_dd_project/dd_benchmark}"
DATASET="${DATASET:?请通过 --export=DATASET=PathMNIST 或 DATASET=COVID 指定数据集}"
PY="${PYTHON:-/home/xiaoyuxu2/.conda/envs/meddd/bin/python}"
WORKERS=4
EXPERTS_PER_WORKER=25

case "$DATASET" in
  PathMNIST)
    SLUG="pathmnist"
    HOP_DATASET="PathMNIST"
    MODEL="ConvNet"
    LR_TEACHER="0.01"
    ;;
  COVID)
    SLUG="covid"
    HOP_DATASET="COVID"
    MODEL="ConvNetD5"
    LR_TEACHER="0.001"
    ;;
  Kvasir)
    SLUG="kvasir"
    HOP_DATASET="Kvasir"
    MODEL="ConvNetD5"
    LR_TEACHER="0.001"
    ;;
  *)
    echo "不支持的数据集：$DATASET" >&2
    exit 2
    ;;
esac

# Kvasir is 128x128; use a smaller real/teacher batch to stay within one GPU.
if [[ "$DATASET" == "Kvasir" ]]; then
  BATCH_TRAIN=64
  BATCH_REAL=64
else
  BATCH_TRAIN=256
  BATCH_REAL=256
fi

CONFIG="$ROOT/configs/hop_tm/$SLUG/ipc10_full.yaml"
FINAL_BUFFER_DIR="$ROOT/buffers/hop_tm/${HOP_DATASET}_NO_ZCA/$MODEL"
PARALLEL_ROOT="$ROOT/buffers/hop_tm_parallel/${SLUG}/${SLURM_JOB_ID}"
RESULT_DIR="$ROOT/results/hop_tm/$DATASET/ipc10"
LOG_DIR="$ROOT/logs"
HOP_ROOT="$ROOT/adapted/hop_tm"

mkdir -p "$LOG_DIR" "$ROOT/buffers/hop_tm" "$RESULT_DIR" "$PARALLEL_ROOT"
cd "$ROOT"

export PYTHONUNBUFFERED=1
export PYTHONPATH="$HOP_ROOT:$ROOT/utils${PYTHONPATH:+:$PYTHONPATH}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export OMP_NUM_THREADS=1
# srun 的每个 worker 使用独立 bash 进程；显式导出它们需要的变量。
export ROOT PY HOP_DATASET MODEL LR_TEACHER PARALLEL_ROOT BATCH_TRAIN BATCH_REAL

test -x "$PY"
test -f "$CONFIG"
test -d "$ROOT/data/prepared/$DATASET"

echo "===== HoP-TM 4GPU pipeline start $(date -Is) ====="
echo "dataset=$DATASET"
echo "config=$CONFIG"
echo "final_buffer_dir=$FINAL_BUFFER_DIR"
echo "parallel_root=$PARALLEL_ROOT"
echo "python=$PY"
echo "node=$SLURMD_NODENAME"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L

BUFFER_SCRIPT="$HOP_ROOT/buffer/buffer_FTD.py"
DISTILL_SCRIPT="$HOP_ROOT/distill/distill_high_order_spl.py"
EVAL_SCRIPT="$HOP_ROOT/distill/evaluation.py"

# 如果正式 buffer 已存在，拒绝混入新旧轨迹；正式实验必须有明确、完整的一套 buffer。
if [[ -d "$FINAL_BUFFER_DIR" ]] && find "$FINAL_BUFFER_DIR" -maxdepth 1 -type f -name 'replay_buffer_*.pt' | grep -q .; then
  echo "正式 buffer 已存在：$FINAL_BUFFER_DIR；为避免混入旧结果而停止。" >&2
  exit 3
fi

echo "===== stage=buffer start $(date -Is) ====="

# 一个 srun step 包含 4 个 task，每个 task 获得 1 张独立 GPU。
# worker 的 buffer_path 不同，避免并发写同一个文件。
WORKER_LOG_ROOT="$LOG_DIR/hop-tm-4gpu-${DATASET}-${SLURM_JOB_ID}-worker"
export WORKER_LOG_ROOT
srun --exclusive --ntasks="$WORKERS" --ntasks-per-node="$WORKERS" \
  --cpus-per-task=4 --gpus-per-task=1 --gpu-bind=single:1 \
  --job-name="HoP-${SLUG}" \
  bash -c '
  set -Eeuo pipefail
  worker="${SLURM_PROCID}"
  WORKER_ROOT="$PARALLEL_ROOT/worker_$worker"
  mkdir -p "$WORKER_ROOT"
  # Slurm may expose the whole job allocation to each task. HoP-TM is
  # single-GPU per worker, so narrow the visible list before Python starts.
  IFS=',' read -r -a allocated_gpus <<< "${CUDA_VISIBLE_DEVICES:-}"
  local_id="${SLURM_LOCALID:-$worker}"
  if (( ${#allocated_gpus[@]} == 1 )) && [[ -n "${allocated_gpus[0]}" ]]; then
    # Slurm already narrowed this task to one physical GPU.
    export CUDA_VISIBLE_DEVICES="${allocated_gpus[0]}"
  elif [[ -n "${allocated_gpus[$local_id]:-}" ]]; then
    # Fallback for clusters that expose the complete allocation to every task.
    export CUDA_VISIBLE_DEVICES="${allocated_gpus[$local_id]}"
  else
    echo "cannot map worker=$worker local_id=$local_id CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}" >&2
    exit 4
  fi
  export PYTHONUNBUFFERED=1
  export PYTHONPATH="$ROOT/adapted/hop_tm:$ROOT/utils${PYTHONPATH:+:$PYTHONPATH}"
  export WANDB_MODE=offline
  export OMP_NUM_THREADS=1
  echo "worker=$worker host=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
  exec "$PY" "$ROOT/adapted/hop_tm/buffer/buffer_FTD.py" \
    --dataset "$HOP_DATASET" \
    --model "$MODEL" \
    --num_experts 25 \
    --train_epochs 100 \
    --save_interval 25 \
    --lr_teacher "$LR_TEACHER" \
    --batch_train "$BATCH_TRAIN" \
    --batch_real "$BATCH_REAL" \
    --dsa True \
    --dsa_strategy color_crop_cutout_flip_scale_rotate \
    --data_path "$ROOT/data/prepared" \
    --buffer_path "$WORKER_ROOT"
' 2>&1 | tee "${WORKER_LOG_ROOT}.combined.log"

# 把 4 个 25-trajectory 文件合并为官方 distill 读取的 10 个 10-trajectory 文件。
# 每条 trajectory 必须包含初始状态 + 100 个 epoch 状态，共 101 个状态。
"$PY" - "$PARALLEL_ROOT" "$FINAL_BUFFER_DIR" "$HOP_DATASET" "$MODEL" <<'PY'
import os
import sys
import torch

parallel_root, final_dir, dataset_name, model_name = sys.argv[1:]
worker_dirs = [os.path.join(parallel_root, f"worker_{i}") for i in range(4)]
all_trajectories = []

for worker_dir in worker_dirs:
    path = os.path.join(
        worker_dir, f"{dataset_name}_NO_ZCA", model_name, "replay_buffer_0.pt"
    )
    if not os.path.exists(path):
        # COVID 的目录名和模型名由目录遍历兼容取得。
        candidates = []
        for root, _, files in os.walk(worker_dir):
            if "replay_buffer_0.pt" in files:
                candidates.append(os.path.join(root, "replay_buffer_0.pt"))
        if len(candidates) != 1:
            raise SystemExit(f"cannot locate unique worker buffer under {worker_dir}: {candidates}")
        path = candidates[0]
    trajectories = torch.load(path, map_location="cpu")
    if len(trajectories) != 25:
        raise SystemExit(f"{path}: expected 25 trajectories, found {len(trajectories)}")
    for trajectory in trajectories:
        if len(trajectory) != 101:
            raise SystemExit(f"{path}: expected 101 states, found {len(trajectory)}")
    all_trajectories.extend(trajectories)

if len(all_trajectories) != 100:
    raise SystemExit(f"expected 100 trajectories, found {len(all_trajectories)}")

os.makedirs(final_dir, exist_ok=True)
existing = [name for name in os.listdir(final_dir) if name.startswith("replay_buffer_")]
if existing:
    raise SystemExit(f"refusing to overwrite existing files in {final_dir}: {existing}")

for index in range(10):
    chunk = all_trajectories[index * 10:(index + 1) * 10]
    torch.save(chunk, os.path.join(final_dir, f"replay_buffer_{index}.pt"))
print(f"verified and merged trajectories: files=10 total=100 states_per_trajectory=101")
PY

# Keep the canonical HoP-TM buffer path and record the exact run contract.
cp "$CONFIG" "$FINAL_BUFFER_DIR/config_used.yaml"
"$PY" - "$FINAL_BUFFER_DIR" "$DATASET" "$HOP_DATASET" "$MODEL" "$SLURM_JOB_ID" <<'PY'
import json
import sys
from pathlib import Path

out_dir, dataset, hop_dataset, model, job_id = sys.argv[1:]
manifest = {
    "method": "HoP-TM",
    "dataset": dataset,
    "dataset_loader_name": hop_dataset,
    "model": model,
    "slurm_job_id": job_id,
    "num_experts": 100,
    "experts_per_worker": 25,
    "workers": 4,
    "train_epochs": 100,
    "trajectory_states": 101,
    "trajectory_count": 100,
    "buffer_files": 10,
    "buffer_file_trajectories": 10,
    "high_order": True,
    "syn_steps": 80,
    "iteration": 10000,
    "config_file": "config_used.yaml",
}
Path(out_dir, "manifest.json").write_text(
    json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
)
print(f"wrote {Path(out_dir, 'config_used.yaml')}")
print(f"wrote {Path(out_dir, 'manifest.json')}")
PY

echo "===== stage=buffer done $(date -Is) ====="
find "$FINAL_BUFFER_DIR" -maxdepth 1 -type f -name 'replay_buffer_*.pt' -printf '%f %s bytes\n' | sort -V

echo "===== stage=distill start $(date -Is) ====="
# HoP-TM distill 入口没有 DDP；保持作者的单 GPU high-order trajectory matching。
"$PY" "$DISTILL_SCRIPT" \
  --cfg "$CONFIG" \
  --data_path "$ROOT/data/prepared" \
  --buffer_path "$ROOT/buffers/hop_tm" \
  --save_path "$RESULT_DIR" \
  --num_eval 1 \
  --Iteration 10000
echo "===== stage=distill done $(date -Is) ====="

# 选取 best synthetic set；若没有 best，则使用最终 Iteration=10000 的保存结果。
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
# 作者 evaluation.py 固定运行 5 个随机初始化模型，这才是最终 DD evaluation。
"$PY" "$EVAL_SCRIPT" \
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

echo "===== HoP-TM 4GPU pipeline complete $(date -Is) ====="
echo "dataset=$DATASET"
echo "final_buffer_dir=$FINAL_BUFFER_DIR"
echo "synthetic_dir=$SYN_DIR"
