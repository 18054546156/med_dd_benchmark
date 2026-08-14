#!/usr/bin/env bash
# 依次执行三种真实医学数据的 CPU 分析。
#SBATCH --job-name=ncfm-medical-all
#SBATCH --partition=cpu-amd9754
#SBATCH --qos=qos-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=logs/ncfm-medical-all_%j.out
#SBATCH --error=logs/ncfm-medical-all_%j.err

set -Eeuo pipefail
export PYTHONUNBUFFERED=1
ROOT="${BENCHMARK_ROOT:-/project/prj-sis01/xiaoyu_xu/med_dd_project/dd_benchmark}"
PY="/home/xiaoyuxu2/.conda/envs/meddd/bin/python"
cd "$ROOT"
test -x "$PY"
for DATASET in PathMNIST COVID Kvasir; do
  "$PY" research/ncfm_medical_analysis/code/medical_dataset_analysis.py \
    --dataset "$DATASET" --data-root data \
    --output "research/ncfm_medical_analysis/dataset_analysis/${DATASET}.json"
done
