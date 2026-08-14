#!/usr/bin/env bash
# 真实医学数据统计与任务特征分析；不训练模型、不使用 toy 数据。
# 输出可用于决定 NCFM/HoP 的分辨率、depth、IPC 和增强敏感性实验。
#SBATCH --job-name=ncfm-medical-analysis
#SBATCH --partition=cpu-amd9754
#SBATCH --qos=qos-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --array=0-2
#SBATCH --output=logs/ncfm-medical-analysis_%A_%a.out
#SBATCH --error=logs/ncfm-medical-analysis_%A_%a.err

set -Eeuo pipefail
export PYTHONUNBUFFERED=1
ROOT="/project/prj-sis01/xiaoyu_xu/med_dd_project/dd_benchmark"
PY="/home/xiaoyuxu2/.conda/envs/meddd/bin/python"
cd "$ROOT"
test -x "$PY"

case "$SLURM_ARRAY_TASK_ID" in
  0) DATASET=PathMNIST ;;
  1) DATASET=COVID ;;
  2) DATASET=Kvasir ;;
  *) exit 2 ;;
esac

exec "$PY" research/ncfm_medical_analysis/code/medical_dataset_analysis.py \
  --dataset "$DATASET" \
  --data-root data \
  --output "research/ncfm_medical_analysis/dataset_analysis/${DATASET}.json"
