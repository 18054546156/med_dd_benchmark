#!/usr/bin/env bash
# Submit the auditable real-data pipeline with Slurm dependencies.
#
# The script starts only from an explicit six-run production manifest, then
# submits the real replication, controlled-evaluation, Phase 1/2, report and
# visualization jobs. It never invents a synthetic path or selects by mtime.
set -Eeuo pipefail

ROOT="${BENCHMARK_ROOT:-/project/prj-sis01/xiaoyu_xu/med_dd_project/dd_benchmark}"
export NCFM_MATH_ROOT="${NCFM_MATH_ROOT:-$ROOT/research/ncfm_mathematical_analysis}"
MANIFEST="${ARTIFACT_MANIFEST:-$ROOT/research/ncfm_medical_analysis/formal_artifact_manifest.json}"
PY="${PYTHON:-/home/xiaoyuxu2/.conda/envs/meddd/bin/python}"
cd "$ROOT"
test -f "$MANIFEST"
test -x "$PY"

mkdir -p logs research/ncfm_medical_analysis/dataset_analysis "$NCFM_MATH_ROOT"

EVAL_MANIFEST="${CONTROLLED_EVAL_MANIFEST:-$ROOT/research/ncfm_medical_analysis/formal_eval_manifest.json}"
test -f "$MANIFEST"
test -f "$EVAL_MANIFEST"

# Derive the three baseline teacher RUN_IDs only from the explicit formal
# manifest, then submit the 30 real E3.1/E4.2 replication runs.  No mtime or
# directory scan is used to choose a teacher.
mapfile -t NCFM_BASELINE_RUN_IDS < <("$PY" - "$MANIFEST" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
for dataset in ("PathMNIST", "COVID", "Kvasir"):
    source = payload.get("source_run_manifests", {}).get(dataset, {}).get("ncfm")
    if not source:
        raise SystemExit(f"missing explicit NCFM source run manifest for {dataset}")
    path = Path(source)
    if not path.is_file():
        raise SystemExit(f"missing NCFM source run manifest: {path}")
    print(path.parent.name)
PY
)
# Statistics are generated first. Artifact/evaluation manifests are validated
# only after this job, because statistics.json is itself a required input.
STATS_JOB="$(sbatch --parsable research/ncfm_medical_analysis/code/run_statistics_cpu.sbatch)"
ANALYSIS_JOB="$(sbatch --parsable --dependency=afterok:${STATS_JOB} research/ncfm_medical_analysis/code/run_dataset_analysis_all.sbatch)"
CONTRACT_JOB="$(sbatch --parsable --dependency=afterok:${STATS_JOB} research/ncfm_medical_analysis/code/run_contract_cpu.sbatch)"
RUNTIME_JOB="$(sbatch --parsable --dependency=afterok:${STATS_JOB}:${CONTRACT_JOB} \
  research/ncfm_medical_analysis/code/run_runtime_contract_gpu.sbatch)"

if (( ${#NCFM_BASELINE_RUN_IDS[@]} != 3 )); then
  echo "expected three explicit NCFM baseline RUN_IDs, got ${#NCFM_BASELINE_RUN_IDS[@]}" >&2
  exit 4
fi
# Phase 1 condensation must not start before train-only statistics and the
# loader contract are verified. Otherwise NCFM can silently use its legacy
# hard-coded fallback statistics.
PHASE1_REPLICATION_SUBMISSION="$(
  BENCHMARK_ROOT="$ROOT" \
  PHASE1_DEPENDENCY="afterok:${STATS_JOB}:${CONTRACT_JOB}" \
  NCFM_PATHMNIST_RUN_ID="${NCFM_BASELINE_RUN_IDS[0]}" \
  NCFM_COVID_RUN_ID="${NCFM_BASELINE_RUN_IDS[1]}" \
  NCFM_KVASIR_RUN_ID="${NCFM_BASELINE_RUN_IDS[2]}" \
  bash scripts/submit_phase1_replications.sh
)"
echo "$PHASE1_REPLICATION_SUBMISSION"
REPLICATION_MERGE_JOB="$(printf '%s\n' "$PHASE1_REPLICATION_SUBMISSION" | awk -F= '$1 == "merge" {print $2}')"
if [[ -z "$REPLICATION_MERGE_JOB" ]]; then
  echo "could not parse Phase 1 replication merge job ID" >&2
  exit 5
fi

# Validate explicit real artifacts before allocating task-analysis/Phase-1
# GPU jobs. This rejects placeholders, legacy outputs and incomplete teachers.
GATE_JOB="$(sbatch --parsable --dependency=afterok:${STATS_JOB}:${ANALYSIS_JOB}:${CONTRACT_JOB}:${RUNTIME_JOB}:${REPLICATION_MERGE_JOB} \
  --export=ALL,NCFM_MATH_ROOT="$NCFM_MATH_ROOT",ARTIFACT_MANIFEST="$MANIFEST" \
  research/ncfm_medical_analysis/code/run_validate_artifacts_cpu.sbatch)"

TASK_ANALYSIS_JOB="$(sbatch --parsable --dependency=afterok:${GATE_JOB} \
  --export=ALL,NCFM_MATH_ROOT="$NCFM_MATH_ROOT",ARTIFACT_MANIFEST="$MANIFEST" \
  research/ncfm_medical_analysis/code/run_task_analysis_gpu.sbatch)"

# Phase 1 is submitted only after data statistics, loader/runtime contracts,
# teacher task analysis and the explicit artifact manifest gate succeed.
PHASE1_JOB="$(sbatch --parsable --dependency=afterok:${GATE_JOB}:${TASK_ANALYSIS_JOB} \
  --export=ALL,NCFM_MATH_ROOT="$NCFM_MATH_ROOT",ARTIFACT_MANIFEST="$MANIFEST" \
  research/ncfm_medical_analysis/code/run_real_phase1.sbatch)"

EVAL_GATE_JOB="$(sbatch --parsable --dependency=afterok:${GATE_JOB} \
  --export=ALL,NCFM_MATH_ROOT="$NCFM_MATH_ROOT",CONTROLLED_EVAL_MANIFEST="$EVAL_MANIFEST" \
  research/ncfm_medical_analysis/code/run_validate_eval_manifest_cpu.sbatch)"
EVAL_JOB="$(sbatch --parsable --dependency=afterok:${EVAL_GATE_JOB} \
  --export=ALL,NCFM_MATH_ROOT="$NCFM_MATH_ROOT",CONTROLLED_EVAL_MANIFEST="$EVAL_MANIFEST" \
  research/ncfm_medical_analysis/code/run_eval_manifest_gpu.sbatch)"

# Phase 2 is a real condenser rerun, not only an estimator simulation.  Submit
# qmc/importance/learned_frequency variants only after the 12 baseline evaluations
# exist.  The helper validates each variant's sampler contract before making
# its own manifest.
PHASE2_TAG="${PHASE2_TAG:-formal-phase2-$(date +%Y%m%d-%H%M%S)}"
PHASE2_VARIANT_SUBMISSION="$(
  BENCHMARK_ROOT="$ROOT" \
  PHASE2_TAG="$PHASE2_TAG" \
  BASELINE_EVAL_DEPENDENCY="afterok:${EVAL_JOB}" \
  NCFM_PATHMNIST_RUN_ID="${NCFM_BASELINE_RUN_IDS[0]}" \
  NCFM_COVID_RUN_ID="${NCFM_BASELINE_RUN_IDS[1]}" \
  NCFM_KVASIR_RUN_ID="${NCFM_BASELINE_RUN_IDS[2]}" \
  bash scripts/submit_phase2_variants.sh
)"
echo "$PHASE2_VARIANT_SUBMISSION"
PHASE2_VARIANT_REPORT_JOB="$(printf '%s\n' "$PHASE2_VARIANT_SUBMISSION" | awk -F= '$1 == "report" {print $2}')"
if [[ -z "$PHASE2_VARIANT_REPORT_JOB" ]]; then
  echo "could not parse Phase 2 variant report job ID" >&2
  exit 5
fi

PHASE2_JOB="$(sbatch --parsable --dependency=afterok:${GATE_JOB} \
  --export=ALL,NCFM_MATH_ROOT="$NCFM_MATH_ROOT",ARTIFACT_MANIFEST="$MANIFEST" \
  research/ncfm_medical_analysis/code/run_phase2.sbatch)"

REPORT_JOB="$(sbatch --parsable --dependency=afterany:${PHASE1_JOB}:${EVAL_JOB}:${PHASE2_JOB}:${PHASE2_VARIANT_REPORT_JOB} \
  --export=ALL,NCFM_MATH_ROOT="$NCFM_MATH_ROOT",ARTIFACT_VALIDATION="$NCFM_MATH_ROOT/formal_artifact_validation_${GATE_JOB}.json",CONTROLLED_EVAL_MANIFEST="$EVAL_MANIFEST",PHASE1_ARRAY_ID="${PHASE1_JOB}",PHASE2_ARRAY_ID="${PHASE2_JOB}",PHASE2_VARIANT_REPORT="$NCFM_MATH_ROOT/phase2_variant_report.json" \
  research/ncfm_medical_analysis/code/generate_formal_report.sbatch)"

TUNING_JOB="$(sbatch --parsable --dependency=afterany:${REPORT_JOB} \
  research/ncfm_medical_analysis/code/generate_tuning_plan.sbatch)"

FIGURE_JOB="$(sbatch --parsable --dependency=afterany:${REPORT_JOB} \
  research/ncfm_medical_analysis/code/generate_visualizations.sbatch)"

printf 'statistics=%s\nanalysis=%s\ncontract=%s\nruntime=%s\ntask_analysis=%s\ngate=%s\nphase1=%s\neval_gate=%s\neval=%s\nphase2_diagnostics=%s\nphase2_variant_report=%s\nreport=%s\ntuning_plan=%s\nfigures=%s\n' \
  "$STATS_JOB" "$ANALYSIS_JOB" "$CONTRACT_JOB" "$RUNTIME_JOB" "$TASK_ANALYSIS_JOB" "$GATE_JOB" "$PHASE1_JOB" "$EVAL_GATE_JOB" "$EVAL_JOB" "$PHASE2_JOB" "$PHASE2_VARIANT_REPORT_JOB" "$REPORT_JOB" "$TUNING_JOB" "$FIGURE_JOB"
