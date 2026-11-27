#!/usr/bin/env bash
#SBATCH --job-name=ncfm-fresh
#SBATCH --partition=gpu-h100
#SBATCH --qos=qos-high-gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=18:00:00

# PathMNIST IPC10: 预训练新教师，再蒸馏和验证选模。
set -Eeuo pipefail
: "${CODE_ROOT:?}" "${DATA_ROOT:?}" "${OUTPUT_ROOT:?}"
cd "$CODE_ROOT"
python="${NCFM_PYTHON:-/home/xiaoyuxu2/.conda/envs/hopbench-cu128/bin/python}"
mode="${FRESH_MODE:-smoke}"
test "$mode" = smoke || test "$mode" = full
run="$OUTPUT_ROOT/${mode}-${SLURM_JOB_ID}"
mkdir -p "$OUTPUT_ROOT"
mkdir "$run"
export PYTHONPATH="$PWD/adapted/ncfm:$PWD/utils"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4
export NCFM_VALIDATION_SELECT=1 NCFM_MARGIN_MODE=off
unset NCFM_MARGIN_TARGETS NCFM_MARGIN_WEIGHT
test -x "$python"
sha256sum "$DATA_ROOT/PathMNIST/pathmnist.npz" > "$run/dataset.sha256"
"$python" -c 'import torch; assert torch.cuda.device_count() == 1; free, _ = torch.cuda.mem_get_info(); print(torch.cuda.get_device_name(), free // 2**30); assert free >= 18*2**30'

# 两个模式共享完整数据和模型。短测缩短训练步数并只训练一个教师。
"$python" - "$DATA_ROOT" "$run" "$mode" <<'PY'
import sys
from pathlib import Path

import yaml

data, run = map(Path, sys.argv[1:3])
mode = sys.argv[3]
cfg = yaml.safe_load(Path("configs/ncfm/pathmnist/ipc10_full.yaml").read_text(encoding="utf-8"))
cfg["dataset"]["data_dir"] = str(data.resolve())
cfg["distibution_train"]["workers"] = 4
cfg["save_path"].update(pretrain_dir=str(run / "teachers"), pretrain_dir_is_final=True)
if mode == "smoke":
    cfg["train"].update(model_num=1, pertrain_epochs=2, evaluation_epochs=2, epoch_eval_interval=1)
    cfg["condense"].update(num_premodel=1, niter=2)
for role, seed in [("pretrain", 7341), ("seed7341", 7341), ("seed7342", 7342), ("seed7343", 7343)]:
    cfg["train"]["seed"] = seed
    cfg["save_path"]["save_dir"] = str(run / role)
    (run / f"{role}.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
PY

launch() {
    "$python" -m torch.distributed.run --standalone --nproc_per_node=1 "$@" --gpu 0 --ipc 10
}
last=20000
teachers=20
repeats=5
seeds=(7341 7342 7343)
if [[ "$mode" == smoke ]]; then
    last=2
    teachers=1
    repeats=1
    seeds=(7341)
fi

launch adapted/ncfm/pretrain/pretrain_script.py --config_path "$run/pretrain.yaml"
for ((i=0; i<teachers; i++)); do
    test -s "$run/teachers/premodel${i}_init.pth.tar"
    test -s "$run/teachers/premodel${i}_trained.pth.tar"
done
sha256sum "$run"/teachers/premodel*.pth.tar > "$run/teachers.sha256"
for seed in "${seeds[@]}"; do
    config="$run/seed${seed}.yaml"
    seed_dir="$run/seed${seed}"
    launch adapted/ncfm/condense/condense_script.py --config_path "$config"
    mapfile -t artifacts < <(find "$seed_dir" -type f -path "*/distilled_data/data_${last}.pt")
    test "${#artifacts[@]}" = 1
    sha256sum "${artifacts[0]}" > "$seed_dir/synthetic.sha256"
    export NCFM_METRICS_PATH="$seed_dir/evaluation.json"
    launch adapted/ncfm/evaluation/evaluation_script.py --config_path "$config" \
        --load_path "${artifacts[0]}" --synthetic_normalization once --val_repeat "$repeats"
    test -s "$NCFM_METRICS_PATH"
done
echo "Completed $mode run at $run"
