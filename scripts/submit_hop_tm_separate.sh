#!/usr/bin/env bash
# Submit four isolated single-GPU teacher workers and one dependent finalizer.
set -Eeuo pipefail

ROOT="${BENCHMARK_ROOT:-/project/prj-sis01/xiaoyu_xu/med_dd_project/dd_benchmark}"
DATASET="${1:?usage: $0 COVID|PathMNIST}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)-$$}"

case "$DATASET" in
  COVID|PathMNIST|Kvasir) ;;
  *) echo "Unsupported DATASET=$DATASET" >&2; exit 2 ;;
esac

worker_ids=()
for worker in 0 1 2 3; do
  job_id=$(sbatch --parsable \
    --export="ALL,DATASET=$DATASET,RUN_ID=$RUN_ID,WORKER_INDEX=$worker,BENCHMARK_ROOT=$ROOT" \
    "$ROOT/scripts/hop_tm_worker.sbatch")
  worker_ids+=("$job_id")
done

dependency="afterok:${worker_ids[0]}:${worker_ids[1]}:${worker_ids[2]}:${worker_ids[3]}"
final_id=$(sbatch --parsable \
  --dependency="$dependency" \
  --export="ALL,DATASET=$DATASET,RUN_ID=$RUN_ID,BENCHMARK_ROOT=$ROOT" \
  "$ROOT/scripts/hop_tm_finalize.sbatch")

echo "dataset=$DATASET"
echo "run_id=$RUN_ID"
echo "worker_jobs=${worker_ids[*]}"
echo "finalizer_job=$final_id"
