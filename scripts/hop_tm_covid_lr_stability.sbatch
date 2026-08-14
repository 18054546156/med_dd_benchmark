#!/usr/bin/env bash
# Pre-registered HoP-TM COVID stability screen. Selection uses only whether
# the optimization remains finite; downstream test accuracy is not inspected.
#SBATCH --job-name=HoP-COVID-LR
#SBATCH --partition=gpu-rtx4090
#SBATCH --qos=qos-high-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --array=0-3
#SBATCH --output=logs/hop-covid-lr-%A_%a.out
#SBATCH --error=logs/hop-covid-lr-%A_%a.err

set -Eeuo pipefail
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export WANDB_MODE=offline

ROOT="${BENCHMARK_ROOT:-/project/prj-sis01/xiaoyu_xu/med_dd_project/dd_benchmark}"
MATH_ROOT="${NCFM_MATH_ROOT:-$ROOT/research/ncfm_mathematical_analysis}"
PY="${PYTHON:-/home/xiaoyuxu2/.conda/envs/meddd/bin/python}"
CONFIG="$ROOT/configs/hop_tm/covid/ipc10_full.yaml"
BUFFER_ROOT="${HOP_BUFFER_ROOT:?Set HOP_BUFFER_ROOT to one run-scoped buffer root, for example buffers/hop_tm/covid/<RUN_ID>}"
TAG="${STABILITY_TAG:-hop-covid-lr-${SLURM_ARRAY_JOB_ID}}"
ITERATIONS="${STABILITY_ITERATIONS:-200}"
LR_VALUES=(100 10 1 0.1)
LR_IMG="${LR_VALUES[$SLURM_ARRAY_TASK_ID]}"
RUN_DIR="$MATH_ROOT/hop_stability/$TAG/lr_${LR_IMG}"

mkdir -p "$ROOT/logs" "$RUN_DIR"
cd "$ROOT"
module purge 2>/dev/null || true
module load anaconda3/3.12 2>/dev/null || true
module load cuda/12.6 2>/dev/null || true
export PYTHONPATH="$ROOT/adapted/hop_tm:$ROOT/utils${PYTHONPATH:+:$PYTHONPATH}"

test -x "$PY"
test -f "$CONFIG"
test -d "$BUFFER_ROOT/COVID_NO_ZCA/ConvNetD5"

"$PY" - "$RUN_DIR/contract.json" "$LR_IMG" "$ITERATIONS" "$BUFFER_ROOT" <<'PY'
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

output, lr_img, iterations, buffer_root = sys.argv[1:]
payload = {
    "status": "running",
    "selection_rule": "largest lr_img among candidates that remains finite for every short-run iteration; test accuracy is not used",
    "candidate_lr_img": [100.0, 10.0, 1.0, 0.1],
    "lr_img": float(lr_img),
    "iterations": int(iterations),
    "buffer_root": buffer_root,
    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
    "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
    "started_at": datetime.now(timezone.utc).isoformat(),
}
Path(output).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY

mark_failed() {
  local exit_code=$?
  "$PY" - "$RUN_DIR/contract.json" "$exit_code" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

path = Path(sys.argv[1])
payload = json.loads(path.read_text(encoding="utf-8"))
payload["status"] = "nonfinite_or_failed"
payload["exit_code"] = int(sys.argv[2])
payload["completed_at"] = datetime.now(timezone.utc).isoformat()
path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY
  exit "$exit_code"
}
trap mark_failed ERR

"$PY" "$ROOT/adapted/hop_tm/distill/distill_high_order_spl.py" \
  --cfg "$CONFIG" \
  --data_path "$ROOT/data/prepared" \
  --buffer_path "$BUFFER_ROOT" \
  --save_path "$RUN_DIR/artifacts" \
  --lr_img "$LR_IMG" \
  --Iteration "$ITERATIONS" \
  --num_eval 1

"$PY" - "$RUN_DIR/contract.json" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

path = Path(sys.argv[1])
payload = json.loads(path.read_text(encoding="utf-8"))
payload["status"] = "finite_complete"
payload["completed_at"] = datetime.now(timezone.utc).isoformat()
path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY

trap - ERR

echo "status=finite_complete lr_img=$LR_IMG iterations=$ITERATIONS run_dir=$RUN_DIR"
