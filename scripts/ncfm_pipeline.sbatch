#!/usr/bin/env bash
#SBATCH --job-name=NCFM-pipeline
#SBATCH --partition=gpu-rtx4090
#SBATCH --qos=qos-high-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=7-00:00:00
#SBATCH --output=logs/ncfm-pipeline-%j.out
#SBATCH --error=logs/ncfm-pipeline-%j.err

# NCFM 单个数据集的完整流水线：pretrain -> condense -> evaluation。
# 提交时通过 --export=DATASET=PathMNIST/Kvasir/COVID 指定数据集。

set -Eeuo pipefail

ROOT="${BENCHMARK_ROOT:-/project/prj-sis01/xiaoyu_xu/med_dd_project/dd_benchmark}"
export NCFM_MATH_ROOT="${NCFM_MATH_ROOT:-$ROOT/research/ncfm_mathematical_analysis}"
DATASET="${DATASET:?必须设置 DATASET，例如 DATASET=PathMNIST}"
PY="${PYTHON:-/home/xiaoyuxu2/.conda/envs/meddd/bin/python}"
NPROC="${NPROC_PER_NODE:-8}"
RUN_ID="${RUN_ID:?必须设置唯一 RUN_ID，例如 RUN_ID=pathmnist-formal-s0-20260814-1}"
if [[ -n "${GPU_IDS:-}" ]]; then
  GPU_IDS="$GPU_IDS"
else
  GPU_IDS=$(seq -s, 0 "$((NPROC - 1))")
fi

case "$DATASET" in
  PathMNIST) SLUG="pathmnist" ;;
  Kvasir)    SLUG="kvasir" ;;
  COVID)     SLUG="covid" ;;
  *) echo "不支持的数据集: $DATASET" >&2; exit 2 ;;
esac

CONFIG="$ROOT/configs/ncfm/$SLUG/ipc10_full.yaml"
# init_script.py expands the YAML value pretrained_models/ncfm to
# pretrained_models/ncfm/<dataset>; keep the validation path identical.
PRETRAIN_DIR="$ROOT/pretrained_models/ncfm/$SLUG/$RUN_ID"
RESULT_ROOT="$ROOT/results/ncfm"
CONDENSE_ROOT="$RESULT_ROOT/condense/$SLUG/ipc10"

mkdir -p "$ROOT/logs" "$ROOT/pretrained_models" "$ROOT/results" "$PRETRAIN_DIR"
cd "$ROOT"

# Batch shells do not inherit the interactive CUDA module state.
if type module >/dev/null 2>&1; then
  module purge
  module load anaconda3/3.12
  module load cuda/12.6
fi

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTHONPATH="$ROOT/adapted/ncfm:$ROOT/utils${PYTHONPATH:+:$PYTHONPATH}"
export NCFM_RUN_ID="$RUN_ID"
export NCFM_PRETRAIN_RUN_DIR="$PRETRAIN_DIR"
export NCFM_EXPERIMENT_VARIANT="baseline"
export NCFM_OBJECTIVE="cf"
export NCFM_FREQUENCY_SAMPLER="mc"
export NCFM_SAMPLING_NET="false"

test -x "$PY"
test -f "$CONFIG"

echo "===== NCFM pipeline start ====="
echo "date=$(date -Is)"
echo "host=$(hostname)"
echo "dataset=$DATASET"
echo "config=$CONFIG"
echo "python=$PY"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L

run_ncfm() {
  local stage="$1"
  shift
  echo "===== stage=$stage start date=$(date -Is) ====="
  "$PY" -m torch.distributed.run \
    --standalone \
    --nproc_per_node="$NPROC" \
    "$@"
  echo "===== stage=$stage done date=$(date -Is) ====="
}

# 1. 用真实训练集预训练 20 个模型。
run_ncfm pretrain \
  "$ROOT/adapted/ncfm/pretrain/pretrain_script.py" \
  --config_path "$CONFIG" \
  --run_mode Pretrain \
  --gpu "$GPU_IDS" \
  --ipc 10

INIT_COUNT=$(find "$PRETRAIN_DIR" -maxdepth 1 -type f -name 'premodel*_init.pth.tar' | wc -l)
TRAINED_COUNT=$(find "$PRETRAIN_DIR" -maxdepth 1 -type f -name 'premodel*_trained.pth.tar' | wc -l)
echo "pretrain artifacts: init=$INIT_COUNT trained=$TRAINED_COUNT dir=$PRETRAIN_DIR"
test "$INIT_COUNT" -ge 20
test "$TRAINED_COUNT" -ge 20

# 2. 使用 20 个预训练模型做 20,000 次 characteristic matching。
run_ncfm condense \
  "$ROOT/adapted/ncfm/condense/condense_script.py" \
  --config_path "$CONFIG" \
  --run_mode Condense \
  --gpu "$GPU_IDS" \
  --ipc 10

# 只允许读取本次 RUN_ID 对应的 artifact；禁止按 mtime 猜旧实验。
mapfile -t SYNTHETIC_MATCHES < <(
  find "$CONDENSE_ROOT" -type f \
    -path "*_${RUN_ID}/distilled_data/data_20000.pt" \
    -print | sort
)
test "${#SYNTHETIC_MATCHES[@]}" -eq 1
SYNTHETIC_PATH="${SYNTHETIC_MATCHES[0]}"
test -f "$SYNTHETIC_PATH"
echo "synthetic_data=$SYNTHETIC_PATH"

# 3. 在固定 test split 上评估合成数据。
run_ncfm evaluation \
  "$ROOT/adapted/ncfm/evaluation/evaluation_script.py" \
  --config_path "$CONFIG" \
  --run_mode Evaluation \
  --gpu "$GPU_IDS" \
  --ipc 10 \
  --load_path "$SYNTHETIC_PATH"

echo "===== NCFM pipeline complete date=$(date -Is) ====="
echo "dataset=$DATASET"
echo "pretrained_dir=$PRETRAIN_DIR"
echo "synthetic_data=$SYNTHETIC_PATH"
echo "run_manifest=$NCFM_MATH_ROOT/runs/ncfm/$SLUG/$RUN_ID/run_manifest.json"

# Register immutable provenance for the formal benchmark artifact gate.
"$PY" "$ROOT/scripts/write_ncfm_run_manifest.py" \
  --root "$ROOT" \
  --dataset "$DATASET" \
  --run-id "$RUN_ID" \
  --config "$CONFIG" \
  --pretrain-dir "$PRETRAIN_DIR" \
  --synthetic "$SYNTHETIC_PATH" \
  --stdout "$ROOT/logs/ncfm-pipeline-${SLURM_JOB_ID}.out" \
  --stderr "$ROOT/logs/ncfm-pipeline-${SLURM_JOB_ID}.err"
