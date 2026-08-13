#!/usr/bin/env bash
#SBATCH --job-name=HoP-distill-existing
#SBATCH --partition=gpu-rtx4090
#SBATCH --qos=qos-high-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/hop-distill-existing-%x-%j.out
#SBATCH --error=logs/hop-distill-existing-%x-%j.err

# 只执行 HoP-TM high-order distill，直接读取已经完成的 buffer。
# 本脚本不会运行 buffer_FTD.py，也不会重新生成 teacher trajectory。
set -Eeuo pipefail
export PYTHONUNBUFFERED=1
export WANDB_MODE=offline
export OMP_NUM_THREADS=1

ROOT="${BENCHMARK_ROOT:-/project/prj-sis01/xiaoyu_xu/med_dd_project/dd_benchmark}"
DATASET="${DATASET:?set DATASET to PathMNIST or COVID}"
PY="/home/xiaoyuxu2/.conda/envs/meddd/bin/python"
DISTILL_SCRIPT="$ROOT/adapted/hop_tm/distill/distill_high_order_spl.py"
EVAL_SCRIPT="$ROOT/adapted/hop_tm/distill/evaluation.py"
ITERATION_OVERRIDE="${ITERATION_OVERRIDE:-10000}"

case "$DATASET" in
  PathMNIST)
    SLUG="pathmnist"
    MODEL="ConvNet"
    CONFIG="${CONFIG_OVERRIDE:-$ROOT/configs/hop_tm/pathmnist/ipc10_full.yaml}"
    BUFFER="$ROOT/buffers/hop_tm/PathMNIST_NO_ZCA/ConvNet"
    RESULT="${RESULT_OVERRIDE:-$ROOT/results/hop_tm_distill_retry/PathMNIST/ipc10}"
    ;;
  COVID)
    SLUG="covid"
    MODEL="ConvNetD5"
    CONFIG="${CONFIG_OVERRIDE:-$ROOT/configs/hop_tm/covid/ipc10_full.yaml}"
    BUFFER="$ROOT/buffers/hop_tm/COVID_NO_ZCA/ConvNetD5"
    RESULT="${RESULT_OVERRIDE:-$ROOT/results/hop_tm_distill_retry/COVID/ipc10}"
    ;;
  *)
    echo "Only PathMNIST and COVID have completed buffers: $DATASET" >&2
    exit 2
    ;;
esac

cd "$ROOT"
mkdir -p "$ROOT/logs" "$RESULT"
export PYTHONPATH="$ROOT/adapted/hop_tm:$ROOT/utils${PYTHONPATH:+:$PYTHONPATH}"

test -x "$PY"
test -f "$CONFIG"
test -f "$BUFFER/manifest.json"
BUFFER_COUNT=$(find "$BUFFER" -maxdepth 1 -type f -name 'replay_buffer_*.pt' | wc -l)
test "$BUFFER_COUNT" -eq 10

echo "===== HoP-TM existing-buffer distill start ====="
echo "date=$(date -Is)"
echo "host=$(hostname)"
echo "dataset=$DATASET"
echo "config=$CONFIG"
echo "buffer=$BUFFER"
echo "buffer_files=$BUFFER_COUNT"
echo "result=$RESULT"
nvidia-smi -L

"$PY" "$DISTILL_SCRIPT" \
  --cfg "$CONFIG" \
  --data_path "$ROOT/data/prepared" \
  --buffer_path "$ROOT/buffers/hop_tm" \
  --save_path "$RESULT" \
  --num_eval 1 \
  --Iteration "$ITERATION_OVERRIDE"

echo "===== HoP-TM existing-buffer distill complete date=$(date -Is) ====="

# 选择最佳合成数据；若没有 best，则使用最终 10000 次迭代结果。
SYN_DIR=$(find "$RESULT" -type f -name 'images_best.pt' -printf '%T@ %h\\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2- || true)
if [[ -z "$SYN_DIR" ]]; then
  SYN_DIR=$(find "$RESULT" -type f -name 'images_10000.pt' -printf '%T@ %h\\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2- || true)
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
echo "===== HoP-TM official DD evaluation start date=$(date -Is) ====="
# 使用 HoP-TM 原始 evaluation.py；该入口评估 5 个随机初始化模型。
"$PY" "$EVAL_SCRIPT" \
  --dataset "$DATASET" \
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
  2>&1 | tee "$RESULT/evaluation.log"
echo "===== HoP-TM official DD evaluation complete date=$(date -Is) ====="
echo "===== HoP-TM existing-buffer distill + evaluation complete date=$(date -Is) ====="
find "$RESULT" -maxdepth 5 -type f -printf '%TY-%Tm-%Td %TH:%TM:%TS %s %p\\n' | sort -r | head -30
