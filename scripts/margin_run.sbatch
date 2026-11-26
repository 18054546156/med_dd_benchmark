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
export OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
export PYTHONPATH="$STUDY_ROOT/adapted/ncfm:$STUDY_ROOT/utils"
export NCFM_MARGIN_TARGETS="$STUDY_ROOT/outputs/margin_targets_logits_v2.npz"
export NCFM_MARGIN_MODE="$(printf '%s\n' off overall rival | sed -n "$((SLURM_ARRAY_TASK_ID / 3 + 1))p")"
seed="$((SLURM_ARRAY_TASK_ID % 3))"
export NCFM_MARGIN_WEIGHT=100
run_dir="$STUDY_ROOT/outputs/${NCFM_RUN_TAG:-formal}/${NCFM_MARGIN_MODE}/seed${seed}"
mkdir -p "$run_dir"
export PYTHONPYCACHEPREFIX="$run_dir/pycache"
python="${PYTHON:-/home/xiaoyuxu2/.conda/envs/hopbench-cu128/bin/python}"
"$python" scripts/write_margin_config.py --mode "$NCFM_MARGIN_MODE" --seed "$seed" \
    --data-root "$DATA_ROOT" --teacher-dir "$TEACHER_DIR" --out-root "$run_dir"
nvidia-smi -L
"$python" -c 'import os,torch; assert torch.cuda.device_count()==1, (os.environ.get("CUDA_VISIBLE_DEVICES"),torch.cuda.device_count()); free,total=torch.cuda.mem_get_info(); print("visible",os.environ.get("CUDA_VISIBLE_DEVICES"),torch.cuda.get_device_name(0),"free_GiB",round(free/2**30,2)); assert free>=18*2**30, "allocated GPU is already occupied"'
"$python" -m torch.distributed.run --standalone --nproc_per_node=1 \
    adapted/ncfm/condense/condense_script.py --config_path "$run_dir/config.yaml" \
    --run_mode Condense --gpu 0 --ipc 10
last_iter=20000
repeats=5
if [[ "${NCFM_SMOKE:-0}" == 1 ]]; then last_iter=2; repeats=1; fi
mapfile -t found < <(find "$run_dir" -type f -path "*/distilled_data/data_${last_iter}.pt")
test "${#found[@]}" = 1
export NCFM_VALIDATION_SELECT=1
export NCFM_METRICS_PATH="$run_dir/evaluation.json"
"$python" -m torch.distributed.run --standalone --nproc_per_node=1 \
    adapted/ncfm/evaluation/evaluation_script.py --config_path "$run_dir/config.yaml" \
    --run_mode Evaluation --gpu 0 --ipc 10 --val_repeat "$repeats" --load_path "${found[0]}"
