#!/usr/bin/env bash
# Build formal manifests only after all six production runs succeed.
# Every RUN_ID is explicit; no filesystem scan or mtime selection is allowed.
#SBATCH --job-name=dd-build-manifests
#SBATCH --partition=cpu-amd9754
#SBATCH --qos=qos-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=logs/dd-build-manifests_%j.out
#SBATCH --error=logs/dd-build-manifests_%j.err

set -Eeuo pipefail
ROOT="${BENCHMARK_ROOT:-/project/prj-sis01/xiaoyu_xu/med_dd_project/dd_benchmark}"
export NCFM_MATH_ROOT="${NCFM_MATH_ROOT:-$ROOT/research/ncfm_mathematical_analysis}"
PY="${PYTHON:-/home/xiaoyuxu2/.conda/envs/meddd/bin/python}"
: "${NCFM_PATHMNIST_RUN_ID:?missing NCFM_PATHMNIST_RUN_ID}"
: "${NCFM_COVID_RUN_ID:?missing NCFM_COVID_RUN_ID}"
: "${NCFM_KVASIR_RUN_ID:?missing NCFM_KVASIR_RUN_ID}"
: "${HOP_PATHMNIST_RUN_ID:?missing HOP_PATHMNIST_RUN_ID}"
: "${HOP_COVID_RUN_ID:?missing HOP_COVID_RUN_ID}"
: "${HOP_KVASIR_RUN_ID:?missing HOP_KVASIR_RUN_ID}"
cd "$ROOT"
test -x "$PY"
exec "$PY" scripts/build_formal_manifests.py \
  --root "$ROOT" \
  --ncfm-pathmnist-run-id "$NCFM_PATHMNIST_RUN_ID" \
  --ncfm-covid-run-id "$NCFM_COVID_RUN_ID" \
  --ncfm-kvasir-run-id "$NCFM_KVASIR_RUN_ID" \
  --hop-pathmnist-run-id "$HOP_PATHMNIST_RUN_ID" \
  --hop-covid-run-id "$HOP_COVID_RUN_ID" \
  --hop-kvasir-run-id "$HOP_KVASIR_RUN_ID"
