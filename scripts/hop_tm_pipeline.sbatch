#!/usr/bin/env bash
#SBATCH --job-name=HoP-TM-pipeline
#SBATCH --partition=gpu-rtx4090
#SBATCH --qos=qos-high-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=7-00:00:00
#SBATCH --output=logs/hop-tm-%x-%j.out
#SBATCH --error=logs/hop-tm-%x-%j.err

# HoP-TM 正式流水线：expert trajectory -> high-order distillation -> DD evaluation。
#
# 重要：HoP-TM 官方实现不是 DDP 程序，buffer_FTD.py 和 distill_high_order_spl.py
# 都是单进程入口。因此这里申请 1 张 GPU；COVID 和 PathMNIST 可分别提交到
# 不同节点并行运行，但不能把一个任务强行改成 torchrun 多卡。
#
# 官方配置来自 Bian-jh/HoP-TM：
# - PathMNIST: ConvNet, expert=100, train_epochs=100, syn_steps=80,
#   Iteration=10000, lr_img=10, lr_teacher=0.01, max_start_epoch=4。
# - COVID: ConvNetD5, expert=100, train_epochs=100, syn_steps=80,
#   Iteration=10000, lr_img=100, lr_teacher=0.001, max_start_epoch=20。
# - 蒸馏过程 num_eval=1；最终 evaluation.py 按作者 README 固定评估 5 次。

set -Eeuo pipefail

ROOT="${BENCHMARK_ROOT:-/project/prj-sis01/xiaoyu_xu/med_dd_project/dd_benchmark}"
DATASET="${DATASET:?请通过 --export=DATASET=PathMNIST 或 DATASET=COVID 指定数据集}"
PY="${PYTHON:-/home/xiaoyuxu2/.conda/envs/meddd/bin/python}"

case "$DATASET" in
  PathMNIST)
    SLUG="pathmnist"
    HOP_DATASET="PathMNIST"
    MODEL="ConvNet"
    ;;
  COVID)
    SLUG="covid"
    HOP_DATASET="COVID"
    MODEL="ConvNetD5"
    ;;
  Kvasir)
    SLUG="kvasir"
    HOP_DATASET="Kvasir"
    MODEL="ConvNetD5"
    ;;
  *)
    echo "不支持的数据集：$DATASET" >&2
    exit 2
    ;;
esac

CONFIG="$ROOT/configs/hop_tm/$SLUG/ipc10_full.yaml"
BUFFER_DIR="$ROOT/buffers/hop_tm/${HOP_DATASET}_NO_ZCA/$MODEL"
RESULT_DIR="$ROOT/results/hop_tm/$DATASET/ipc10"
LOG_DIR="$ROOT/logs"
HOP_ROOT="$ROOT/adapted/hop_tm"

mkdir -p "$LOG_DIR" "$ROOT/buffers/hop_tm" "$RESULT_DIR"
cd "$ROOT"

export PYTHONUNBUFFERED=1
export PYTHONPATH="$HOP_ROOT:$ROOT/utils${PYTHONPATH:+:$PYTHONPATH}"
# 论文代码使用 W&B 记录实验；HPC 无需联网，offline 不改变训练算法。
export WANDB_MODE="${WANDB_MODE:-offline}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

test -x "$PY"
test -f "$CONFIG"
test -d "$ROOT/data/prepared/$DATASET"

echo "===== HoP-TM pipeline start $(date -Is) ====="
echo "dataset=$DATASET"
echo "config=$CONFIG"
echo "buffer_dir=$BUFFER_DIR"
echo "result_dir=$RESULT_DIR"
echo "python=$PY"
echo "node=$SLURMD_NODENAME"
nvidia-smi -L

BUFFER_SCRIPT="$HOP_ROOT/buffer/buffer_FTD.py"
DISTILL_SCRIPT="$HOP_ROOT/distill/distill_high_order_spl.py"
EVAL_SCRIPT="$HOP_ROOT/distill/evaluation.py"

# buffer_FTD.py 每 10 个 expert 保存一个 replay_buffer 文件。
mapfile -t BUFFER_FILES < <(find "$BUFFER_DIR" -maxdepth 1 -type f -name 'replay_buffer_*.pt' -printf '%f\n' 2>/dev/null | sort -V)
if (( ${#BUFFER_FILES[@]} == 10 )); then
  # 检查 10 个文件合计确实是 100 条轨迹，避免只按文件名误判为完成。
  "$PY" - "$BUFFER_DIR" <<'PY'
import os
import sys
import torch

directory = sys.argv[1]
files = [os.path.join(directory, f) for f in sorted(os.listdir(directory))
         if f.startswith("replay_buffer_") and f.endswith(".pt")]
if len(files) != 10:
    raise SystemExit(f"expected 10 buffer files, found {len(files)}")
total = 0
for path in files:
    trajectories = torch.load(path, map_location="cpu")
    if len(trajectories) != 10:
        raise SystemExit(f"{path}: expected 10 trajectories, found {len(trajectories)}")
    for trajectory in trajectories:
        if len(trajectory) != 101:
            raise SystemExit(f"{path}: expected 101 states per trajectory, found {len(trajectory)}")
    total += len(trajectories)
if total != 100:
    raise SystemExit(f"expected 100 trajectories, found {total}")
print(f"verified expert trajectories: files=10 total={total} states_per_trajectory=101")
PY
  echo "===== stage=buffer skipped: verified existing official trajectories ====="
else
  if (( ${#BUFFER_FILES[@]} != 0 )); then
    echo "发现不完整的 HoP-TM buffer（${#BUFFER_FILES[@]} 个文件），为避免混入不完整轨迹而停止。" >&2
    exit 3
  fi
  echo "===== stage=buffer start $(date -Is) ====="
  "$PY" "$BUFFER_SCRIPT" \
    --dataset "$HOP_DATASET" \
    --model "$MODEL" \
    --num_experts 100 \
    --train_epochs 100 \
    --lr_teacher "$( [[ "$DATASET" == COVID ]] && echo 0.001 || echo 0.01 )" \
    --batch_train 128 \
    --batch_real 256 \
    --dsa True \
    --dsa_strategy color_crop_cutout_flip_scale_rotate \
    --data_path "$ROOT/data/prepared" \
    --buffer_path "$ROOT/buffers/hop_tm"
fi

# argparse 的 zca 是 store_true；正式 no-ZCA 运行不传 --zca。
# 这里检查 buffer 的目录命名，确保 buffer 与 distill 完全一致。
test -d "$BUFFER_DIR"

echo "===== stage=distill start $(date -Is) ====="
"$PY" "$DISTILL_SCRIPT" \
  --cfg "$CONFIG" \
  --data_path "$ROOT/data/prepared" \
  --buffer_path "$ROOT/buffers/hop_tm" \
  --save_path "$RESULT_DIR" \
  --num_eval 1 \
  --Iteration 10000
echo "===== stage=distill done $(date -Is) ====="

# 选取 distill 保存的 best synthetic set；若某次运行没有中间 best，使用最终迭代。
SYN_DIR=$(find "$RESULT_DIR" -type f -name 'images_best.pt' -printf '%T@ %h\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2- || true)
if [[ -z "$SYN_DIR" ]]; then
  SYN_DIR=$(find "$RESULT_DIR" -type f -name 'images_10000.pt' -printf '%T@ %h\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2- || true)
fi
test -n "$SYN_DIR"
test -f "$SYN_DIR/images_best.pt" || test -f "$SYN_DIR/images_10000.pt"

if [[ -f "$SYN_DIR/images_best.pt" ]]; then
  IMAGE_FILE="$SYN_DIR/images_best.pt"
  LABEL_FILE="$SYN_DIR/labels_best.pt"
  LR_FILE="$SYN_DIR/lr_best.pt"
else
  IMAGE_FILE="$SYN_DIR/images_10000.pt"
  LABEL_FILE="$SYN_DIR/labels_10000.pt"
  LR_FILE="$SYN_DIR/lr_10000.pt"
fi
test -f "$LABEL_FILE"
test -f "$LR_FILE"

echo "synthetic_dir=$SYN_DIR"
echo "image_file=$IMAGE_FILE"
echo "label_file=$LABEL_FILE"
echo "lr_file=$LR_FILE"

echo "===== stage=evaluation start $(date -Is) ====="
# 作者 evaluation.py 固定执行 5 个随机初始化 ConvNet，属于 DD evaluation。
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

echo "===== HoP-TM pipeline complete $(date -Is) ====="
echo "dataset=$DATASET"
echo "buffer_dir=$BUFFER_DIR"
echo "synthetic_dir=$SYN_DIR"
