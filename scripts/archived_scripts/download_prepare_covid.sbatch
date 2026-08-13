#!/usr/bin/env bash
#SBATCH --job-name=NCFM-COVID-data
#SBATCH --partition=cpu-amd9754
#SBATCH --qos=qos-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/covid-data-%j.out
#SBATCH --error=logs/covid-data-%j.err

# 下载 Kaggle 公共 COVID-19 Radiography Database，解压后生成 benchmark prepared 数据。
# 该数据集包含 COVID、Lung_Opacity、Normal、Viral_Pneumonia 四类。

set -Eeuo pipefail

ROOT="${BENCHMARK_ROOT:-/project/prj-sis01/xiaoyu_xu/med_dd_project/dd_benchmark}"
PY="${PYTHON:-/home/xiaoyuxu2/.conda/envs/meddd/bin/python}"
RAW="$ROOT/data/raw/COVID"
ARCHIVE="$RAW/covid19-radiography-database-v5.zip"
SOURCE="$RAW/source"
URL="https://www.kaggle.com/api/v1/datasets/download/tawsifurrahman/covid19-radiography-database?datasetVersionNumber=5"

mkdir -p "$ROOT/logs" "$RAW" "$SOURCE"
cd "$ROOT"

# -C - 支持断点续传；网络中断后重新提交不会从零开始。
curl -L --fail --retry 30 --retry-all-errors --connect-timeout 60 \
  -C - -o "$ARCHIVE" "$URL"
unzip -tq "$ARCHIVE"
rm -rf "$SOURCE"
mkdir -p "$SOURCE"
unzip -q "$ARCHIVE" -d "$SOURCE"

SOURCE_DIR=$(find "$SOURCE" -type d -name 'COVID' -o -type d -name 'COVID-19_Radiography_Dataset' | head -1)
if [[ -z "$SOURCE_DIR" ]]; then
  echo "无法在 ZIP 中找到 COVID 数据目录" >&2
  find "$SOURCE" -maxdepth 3 -type d -print >&2
  exit 1
fi

"$PY" "$ROOT/scripts/prepare_medical_data.py" \
  --data-root "$ROOT/data" prepare \
  --dataset COVID \
  --source-dir "$SOURCE_DIR" \
  --materialize copy \
  --seed 20260810 \
  --test-ratio 0.2 \
  --val-ratio 0.1 \
  --overwrite

"$PY" "$ROOT/scripts/prepare_medical_data.py" \
  --data-root "$ROOT/data" validate --dataset COVID

echo "COVID prepared 完成: $ROOT/data/prepared/COVID"
find "$ROOT/data/prepared/COVID" -type f | wc -l
du -sh "$ROOT/data/raw/COVID" "$ROOT/data/prepared/COVID"
