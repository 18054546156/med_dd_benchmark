#!/usr/bin/env bash
# Stage 3: nine real Phase 2 seed sweeps plus estimator diagnostics.
set -Eeuo pipefail

ROOT="${BENCHMARK_ROOT:?}"
MATH_ROOT="${NCFM_MATH_ROOT:?}"
MANIFEST="${ARTIFACT_MANIFEST:?}"
EVAL_MANIFEST="${CONTROLLED_EVAL_MANIFEST:?}"
PY="${PYTHON:-/home/xiaoyuxu2/.conda/envs/meddd/bin/python}"
cd "$ROOT"

mapfile -t BASELINE_IDS < <("$PY" - "$MANIFEST" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
for dataset in ("PathMNIST", "COVID", "Kvasir"):
    path = Path(payload["source_run_manifests"][dataset]["ncfm"])
    if not path.is_file():
        raise SystemExit(f"missing NCFM run manifest: {path}")
    print(path.parent.name)
PY
)
test "${#BASELINE_IDS[@]}" -eq 3

PHASE2_TAG="${PHASE2_TAG:-formal-phase2-$(date +%Y%m%d-%H%M%S)}"
VARIANT_SUBMISSION="$(
  BENCHMARK_ROOT="$ROOT" \
  NCFM_MATH_ROOT="$MATH_ROOT" \
  PHASE2_TAG="$PHASE2_TAG" \
  NCFM_PATHMNIST_RUN_ID="${BASELINE_IDS[0]}" \
  NCFM_COVID_RUN_ID="${BASELINE_IDS[1]}" \
  NCFM_KVASIR_RUN_ID="${BASELINE_IDS[2]}" \
  bash scripts/submit_phase2_variants.sh
)"
echo "$VARIANT_SUBMISSION"
VARIANT_REPORT_JOB="$(printf '%s\n' "$VARIANT_SUBMISSION" | awk -F= '$1 == "report" {print $2}')"
test -n "$VARIANT_REPORT_JOB"

PHASE2_DIAGNOSTIC_JOB="$(sbatch --parsable \
  --export="ALL,BENCHMARK_ROOT=$ROOT,NCFM_MATH_ROOT=$MATH_ROOT,ARTIFACT_MANIFEST=$MANIFEST" \
  research/ncfm_medical_analysis/code/run_phase2.sbatch)"

STAGE4_JOB="$(sbatch --parsable \
  --job-name=formal-stage4-report \
  --partition=cpu-amd9754 --qos=qos-normal --cpus-per-task=1 --mem=2G --time=00:20:00 \
  --output=logs/formal-stage4-%j.out --error=logs/formal-stage4-%j.err \
  --dependency="afterany:$PHASE2_DIAGNOSTIC_JOB:$VARIANT_REPORT_JOB" \
  --export="ALL,BENCHMARK_ROOT=$ROOT,NCFM_MATH_ROOT=$MATH_ROOT,ARTIFACT_MANIFEST=$MANIFEST,CONTROLLED_EVAL_MANIFEST=$EVAL_MANIFEST,PHASE1_JOB_ID=${PHASE1_JOB_ID:?},EVAL_JOB_ID=${EVAL_JOB_ID:?},GATE_JOB_ID=${GATE_JOB_ID:?},PHASE2_DIAGNOSTIC_JOB_ID=$PHASE2_DIAGNOSTIC_JOB,PHASE2_VARIANT_REPORT_JOB_ID=$VARIANT_REPORT_JOB" \
  research/ncfm_medical_analysis/code/formal_stage4_report.sbatch)"

printf 'stage=3\nphase2_tag=%s\nphase2_diagnostics=%s\nphase2_variant_report=%s\nstage4_report=%s\n' \
  "$PHASE2_TAG" "$PHASE2_DIAGNOSTIC_JOB" "$VARIANT_REPORT_JOB" "$STAGE4_JOB"
