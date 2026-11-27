#!/usr/bin/env bash
#SBATCH --job-name=ncfm-margin-eval
#SBATCH --partition=gpu-h100
#SBATCH --qos=qos-high-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00

set -Eeuo pipefail
: "${STUDY_ROOT:?}" "${RUN_ROOT:?}"
cd "$RUN_ROOT/code"
export OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD/adapted/ncfm:$PWD/utils"
export NCFM_VALIDATION_SELECT=1
python=/home/xiaoyuxu2/.conda/envs/hopbench-cu128/bin/python
task="$SLURM_ARRAY_TASK_ID"
if (( task < 4 )); then
    if (( task < 2 )); then artifact=26479_0; else artifact=off_seed0; fi
    if (( task % 2 == 0 )); then normalization=raw; else normalization=once; fi
else
    modes=(off overall rival)
    index=$((task - 3))
    artifact="${modes[index/3]}_seed$((index%3))"
    normalization=once
fi
run="$RUN_ROOT/results/${artifact}_${normalization}"
mkdir -p "$run"
export NCFM_METRICS_PATH="$run/evaluation.json"
export PYTHONPYCACHEPREFIX="$run/pycache"
manifest=study_results/margin_20261126/synthetic_sha256.txt
config="study_results/margin_20261126/${artifact}_config.yaml"
if [[ "$artifact" == 26479_0 ]]; then
    config=study_results/margin_20261126/off_seed0_config.yaml
    hash=9c885c049c1b56d6486abc6ad9ca6fa90e6a3875bd48bfab9f698399c7741c9f
    path=/project/prj-sis01/xuxiaoyu/NCFM_Research/results/variants/ncfm/PathMNIST/ConvNetD3_IN_W128/ipc10/seed0/slurm-26479_0/artifacts/condense/pathmnist/ipc10/adamw_lr_img_0.0010_numr_reqs4096_factor2_20260823-0453/distilled_data/data_20000.pt
else
    line=1
    if (( task >= 4 )); then line=$((task - 2)); fi
    read -r hash path < <(sed -n "${line}p" "$manifest")
    path="$STUDY_ROOT/$path"
fi
printf '%s  %s\n' "$hash" "$path" > "$run/synthetic_sha256.txt"
sha256sum --check "$run/synthetic_sha256.txt"

"$python" - "$config" "$run" <<'PY'
from pathlib import Path
import sys
import yaml

source, run = map(Path, sys.argv[1:])
config = yaml.safe_load(source.read_text())
config["train"]["seed"] = 7341
config["save_path"]["save_dir"] = str(run)
(run / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
PY

echo "job=${SLURM_ARRAY_JOB_ID}_${task} artifact=$artifact normalization=$normalization node=$SLURMD_NODENAME"
"$python" -c 'import torch; assert torch.cuda.device_count() == 1; free, _ = torch.cuda.mem_get_info(); print(torch.cuda.get_device_name(0), "free GiB", round(free/2**30, 2)); assert free >= 18*2**30, "allocated GPU is busy"'
"$python" -m torch.distributed.run --standalone --nproc_per_node=1 \
    adapted/ncfm/evaluation/evaluation_script.py --config_path "$run/config.yaml" \
    --gpu 0 --ipc 10 --val_repeat 5 --load_path "$path" \
    --synthetic_normalization "$normalization"
