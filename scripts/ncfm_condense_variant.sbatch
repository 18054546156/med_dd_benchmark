#!/usr/bin/env bash
# Condense-only NCFM Phase 2 variant. Reuses one explicit completed teacher
# run, but writes all variant artifacts under a new RUN_ID.
# VARIANT=qmc|importance|learned_frequency|baseline_seed|pixel_mean
#SBATCH --job-name=NCFM-variant
#SBATCH --partition=gpu-rtx4090
#SBATCH --qos=qos-high-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=7-00:00:00
#SBATCH --output=logs/ncfm-variant-%j.out
#SBATCH --error=logs/ncfm-variant-%j.err

set -Eeuo pipefail
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

ROOT="${BENCHMARK_ROOT:-/project/prj-sis01/xiaoyu_xu/med_dd_project/dd_benchmark}"
DATASET="${DATASET:?set DATASET=PathMNIST|COVID|Kvasir}"
VARIANT="${VARIANT:?set VARIANT=qmc|importance|learned_frequency|baseline_seed|pixel_mean}"
RUN_ID="${RUN_ID:?set a unique variant RUN_ID}"
TEACHER_RUN_ID="${TEACHER_RUN_ID:?set the explicit baseline teacher RUN_ID}"
PY="${PYTHON:-/home/xiaoyuxu2/.conda/envs/meddd/bin/python}"
NPROC="${NPROC_PER_NODE:-8}"

case "$DATASET" in
  PathMNIST) SLUG=pathmnist ;;
  COVID) SLUG=covid ;;
  Kvasir) SLUG=kvasir ;;
  *) echo "unsupported DATASET=$DATASET" >&2; exit 2 ;;
esac
case "$VARIANT" in
  qmc|importance|learned_frequency|baseline_seed|pixel_mean) ;;
  *) echo "unsupported VARIANT=$VARIANT" >&2; exit 2 ;;
esac

CONFIG="$ROOT/configs/ncfm/$SLUG/ipc10_full.yaml"
TEACHER_DIR="$ROOT/pretrained_models/ncfm/$SLUG/$TEACHER_RUN_ID"
RESULT_ROOT="$ROOT/results/ncfm"
CONDENSE_ROOT="$RESULT_ROOT/condense/$SLUG/ipc10"

mkdir -p "$ROOT/logs" "$ROOT/results"
cd "$ROOT"
module purge 2>/dev/null || true
module load anaconda3/3.12 2>/dev/null || true
module load cuda/12.6 2>/dev/null || true
export PYTHONPATH="$ROOT/adapted/ncfm:$ROOT/utils${PYTHONPATH:+:$PYTHONPATH}"
export NCFM_RUN_ID="$RUN_ID"
export NCFM_PRETRAIN_RUN_DIR="$TEACHER_DIR"
export NCFM_FREQUENCY_SEED="${FREQUENCY_SEED:-${SEED:-0}}"
export NCFM_CONDENSE_SEED="${SEED:-0}"
export NCFM_EXPERIMENT_VARIANT="$VARIANT"
export NCFM_OBJECTIVE="cf"
if [[ "$VARIANT" == qmc ]]; then
  export NCFM_FREQUENCY_SAMPLER=qmc
elif [[ "$VARIANT" == importance ]]; then
  export NCFM_FREQUENCY_SAMPLER=importance
  export NCFM_IMPORTANCE_MEAN_SHIFT="${IMPORTANCE_MEAN_SHIFT:-0.5}"
elif [[ "$VARIANT" == learned_frequency ]]; then
  export NCFM_SAMPLING_NET=true
  export NCFM_FREQUENCY_SAMPLER=mc
elif [[ "$VARIANT" == pixel_mean ]]; then
  export NCFM_SAMPLING_NET=false
  export NCFM_FREQUENCY_SAMPLER=mc
  export NCFM_OBJECTIVE=pixel_mean
else
  export NCFM_SAMPLING_NET=false
  export NCFM_FREQUENCY_SAMPLER=mc
fi

test -x "$PY"
test -f "$CONFIG"
test -d "$TEACHER_DIR"
test "$(find "$TEACHER_DIR" -maxdepth 1 -name 'premodel*_trained.pth.tar' | wc -l)" -eq 20
nvidia-smi -L

"$PY" -m torch.distributed.run \
  --standalone --nproc_per_node="$NPROC" \
  "$ROOT/adapted/ncfm/condense/condense_script.py" \
  --config_path "$CONFIG" --run_mode Condense \
  --gpu "$(seq -s, 0 $((NPROC - 1)))" --ipc 10

mapfile -t matches < <(find "$CONDENSE_ROOT" -type f \
  -path "*_${RUN_ID}/distilled_data/data_20000.pt" -print | sort)
test "${#matches[@]}" -eq 1
SYNTHETIC_PATH="${matches[0]}"
"$PY" "$ROOT/scripts/write_ncfm_run_manifest.py" \
  --root "$ROOT" --dataset "$DATASET" --run-id "$RUN_ID" \
  --config "$CONFIG" --pretrain-dir "$TEACHER_DIR" \
  --synthetic "$SYNTHETIC_PATH" \
  --stdout "$ROOT/logs/ncfm-variant-${SLURM_JOB_ID}.out" \
  --stderr "$ROOT/logs/ncfm-variant-${SLURM_JOB_ID}.err"
echo "variant=$VARIANT dataset=$DATASET run_id=$RUN_ID synthetic_data=$SYNTHETIC_PATH"
