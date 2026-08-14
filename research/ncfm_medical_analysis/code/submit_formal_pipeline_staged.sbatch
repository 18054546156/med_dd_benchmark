#!/usr/bin/env bash
# Stage 1: preflights and six Phase 1 condenser seed sweeps. A dependent
# Stage 2 submitter continues after these jobs finish, keeping submitted jobs
# below the cluster MaxSubmitJobsPerUser=15 limit.
set -Eeuo pipefail

ROOT="${BENCHMARK_ROOT:-/project/prj-sis01/xiaoyu_xu/med_dd_project/dd_benchmark}"
MATH_ROOT="${NCFM_MATH_ROOT:-$ROOT/research/ncfm_mathematical_analysis}"
MANIFEST="${ARTIFACT_MANIFEST:-$ROOT/research/ncfm_medical_analysis/formal_artifact_manifest.json}"
EVAL_MANIFEST="${CONTROLLED_EVAL_MANIFEST:-$ROOT/research/ncfm_medical_analysis/formal_eval_manifest.json}"
PY="${PYTHON:-/home/xiaoyuxu2/.conda/envs/meddd/bin/python}"
cd "$ROOT"
test -f "$MANIFEST"
test -f "$EVAL_MANIFEST"

mapfile -t BASELINE_IDS < <("$PY" - "$MANIFEST" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
for dataset in ("PathMNIST", "COVID", "Kvasir"):
    source = payload.get("source_run_manifests", {}).get(dataset, {}).get("ncfm")
    path = Path(source or "")
    if not path.is_file():
        raise SystemExit(f"missing explicit NCFM run manifest: {source}")
    print(path.parent.name)
PY
)
test "${#BASELINE_IDS[@]}" -eq 3

STATS_JOB="$(sbatch --parsable research/ncfm_medical_analysis/code/run_statistics_cpu.sbatch)"
ANALYSIS_JOB="$(sbatch --parsable --dependency="afterok:$STATS_JOB" \
  research/ncfm_medical_analysis/code/run_dataset_analysis_all.sbatch)"
CONTRACT_JOB="$(sbatch --parsable --dependency="afterok:$STATS_JOB" \
  research/ncfm_medical_analysis/code/run_contract_cpu.sbatch)"
RUNTIME_JOB="$(sbatch --parsable --dependency="afterok:$STATS_JOB:$CONTRACT_JOB" \
  research/ncfm_medical_analysis/code/run_runtime_contract_gpu.sbatch)"

PHASE1_SUBMISSION="$(
  BENCHMARK_ROOT="$ROOT" \
  NCFM_MATH_ROOT="$MATH_ROOT" \
  PHASE1_DEPENDENCY="afterok:$STATS_JOB:$CONTRACT_JOB" \
  NCFM_PATHMNIST_RUN_ID="${BASELINE_IDS[0]}" \
  NCFM_COVID_RUN_ID="${BASELINE_IDS[1]}" \
  NCFM_KVASIR_RUN_ID="${BASELINE_IDS[2]}" \
  bash scripts/submit_phase1_replications.sh
)"
echo "$PHASE1_SUBMISSION"
MERGE_JOB="$(printf '%s\n' "$PHASE1_SUBMISSION" | awk -F= '$1 == "merge" {print $2}')"
test -n "$MERGE_JOB"

STAGE2_JOB="$(sbatch --parsable \
  --job-name=formal-stage2-submit \
  --partition=cpu-amd9754 --qos=qos-normal --cpus-per-task=1 --mem=2G --time=00:20:00 \
  --output=logs/formal-stage2-%j.out --error=logs/formal-stage2-%j.err \
  --dependency="afterok:$ANALYSIS_JOB:$RUNTIME_JOB:$MERGE_JOB" \
  --export="ALL,BENCHMARK_ROOT=$ROOT,NCFM_MATH_ROOT=$MATH_ROOT,ARTIFACT_MANIFEST=$MANIFEST,CONTROLLED_EVAL_MANIFEST=$EVAL_MANIFEST" \
  research/ncfm_medical_analysis/code/formal_stage2_submit.sbatch)"

printf 'stage=1\nstatistics=%s\nanalysis=%s\ncontract=%s\nruntime=%s\nphase1_merge=%s\nstage2_submit=%s\n' \
  "$STATS_JOB" "$ANALYSIS_JOB" "$CONTRACT_JOB" "$RUNTIME_JOB" "$MERGE_JOB" "$STAGE2_JOB"
