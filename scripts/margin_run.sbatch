#!/usr/bin/env bash
#SBATCH --job-name=ncfm-margin-study
#SBATCH --partition=gpu-rtx4090
#SBATCH --qos=qos-high-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/margin-%A_%a.out
#SBATCH --error=logs/margin-%A_%a.err

set -Eeuo pipefail
: "${STUDY_ROOT:?}" "${TEACHER_DIR:?}" "${DATA_ROOT:?}"
cd "$STUDY_ROOT"
module purge
export OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
export PYTHONPATH="$STUDY_ROOT/adapted/ncfm:$STUDY_ROOT/utils"
export NCFM_MARGIN_TARGETS="$STUDY_ROOT/outputs/margin_targets.npz"
export NCFM_MARGIN_MODE="$(printf '%s\n' off overall rival | sed -n "$((SLURM_ARRAY_TASK_ID / 3 + 1))p")"
seed="$((SLURM_ARRAY_TASK_ID % 3))"
export NCFM_MARGIN_WEIGHT=100
run_dir="$STUDY_ROOT/outputs/${NCFM_MARGIN_MODE}/seed${seed}"
mkdir -p "$run_dir"
python=/home/xiaoyuxu2/.conda/envs/meddd/bin/python
"$python" scripts/write_margin_config.py --mode "$NCFM_MARGIN_MODE" --seed "$seed" \
    --data-root "$DATA_ROOT" --teacher-dir "$TEACHER_DIR" --out-root "$run_dir"
nvidia-smi -L
"$python" -m torch.distributed.run --standalone --nproc_per_node=1 \
    adapted/ncfm/condense/condense_script.py --config_path "$run_dir/config.yaml" \
    --run_mode Condense --gpu 0 --ipc 10
mapfile -t found < <(find "$run_dir" -type f -path '*/distilled_data/data_20000.pt')
test "${#found[@]}" = 1
"$python" -m torch.distributed.run --standalone --nproc_per_node=1 \
    adapted/ncfm/evaluation/evaluation_script.py --config_path "$run_dir/config.yaml" \
    --run_mode Evaluation --gpu 0 --ipc 10 --val_repeat 5 --load_path "${found[0]}"
