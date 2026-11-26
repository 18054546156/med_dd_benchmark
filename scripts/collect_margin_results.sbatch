#!/usr/bin/env bash
#SBATCH --job-name=ncfm-margin-collect
#SBATCH --partition=cpu-amd9754
#SBATCH --qos=qos-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=3G
#SBATCH --time=01:00:00
#SBATCH --output=logs/margin-collect-%j.out
#SBATCH --error=logs/margin-collect-%j.err

set -Eeuo pipefail
: "${STUDY_ROOT:?}" "${PARENT_JOB:?}"
cd "$STUDY_ROOT"
export GIT_TERMINAL_PROMPT=0 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
report="$STUDY_ROOT/outputs/collected_${PARENT_JOB}_${SLURM_JOB_ID}"
accounting_args=()
if [[ -n "${NCFM_ACCOUNTING_FILE:-}" ]]; then
    accounting_args=(--accounting-file "$NCFM_ACCOUNTING_FILE")
fi
/home/xiaoyuxu2/.conda/envs/hopbench-cu128/bin/python scripts/collect_margin_results.py \
    --study-root "$STUDY_ROOT" --job-id "$PARENT_JOB" --run-tag "${NCFM_RUN_TAG:-formal}" \
    --out "$report" --require-complete "${accounting_args[@]}"

repo="$STUDY_ROOT/collector_repo_${SLURM_JOB_ID}"
git clone --single-branch --branch codex/ncfm-margin-study \
    https://github.com/18054546156/med_dd_benchmark.git "$repo"
mkdir -p "$repo/study_results/margin_20261126"
cp "$report"/*.json "$repo/study_results/margin_20261126/"
git -C "$repo" add study_results/margin_20261126
git -C "$repo" -c user.name='xiaoyu xu' -c user.email='937872199@qq.com' \
    commit -m 'Record NCFM angular margin experiment results'
git -C "$repo" push origin HEAD:codex/ncfm-margin-study
