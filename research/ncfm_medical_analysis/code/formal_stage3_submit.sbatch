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
  DEFER_FOLLOWUP=1 \
  bash scripts/submit_phase2_variants.sh
)"
echo "$VARIANT_SUBMISSION"
VARIANT_JOBS="$(printf '%s\n' "$VARIANT_SUBMISSION" | awk -F= '$1 == "variant_jobs" {print $2}')"
PHASE2_TAG="$(printf '%s\n' "$VARIANT_SUBMISSION" | awk -F= '$1 == "phase2_tag" {print $2}')"
PHASE2_SEEDS="$(printf '%s\n' "$VARIANT_SUBMISSION" | awk -F= '$1 == "phase2_seeds" {print $2}')"
test -n "$VARIANT_JOBS"
test -n "$PHASE2_TAG"
test -n "$PHASE2_SEEDS"

PHASE2_DIAGNOSTIC_JOB="$(sbatch --parsable \
  --export="ALL,BENCHMARK_ROOT=$ROOT,NCFM_MATH_ROOT=$MATH_ROOT,ARTIFACT_MANIFEST=$MANIFEST" \
  research/ncfm_medical_analysis/code/run_phase2.sbatch)"

VARIANT_DEP="afterok:${VARIANT_JOBS// /:}"
STAGE3B_JOB="$(sbatch --parsable --dependency="$VARIANT_DEP" \
  --export="ALL,BENCHMARK_ROOT=$ROOT,NCFM_MATH_ROOT=$MATH_ROOT,PHASE2_TAG=$PHASE2_TAG,PHASE2_SEEDS=${PHASE2_SEEDS//,/:},ARTIFACT_MANIFEST=$MANIFEST,CONTROLLED_EVAL_MANIFEST=$EVAL_MANIFEST,PHASE1_JOB_ID=${PHASE1_JOB_ID:?},EVAL_JOB_ID=${EVAL_JOB_ID:?},GATE_JOB_ID=${GATE_JOB_ID:?},PHASE2_DIAGNOSTIC_JOB_ID=$PHASE2_DIAGNOSTIC_JOB" \
  research/ncfm_medical_analysis/code/formal_stage3b_submit.sbatch)"

printf 'stage=3\nphase2_tag=%s\nphase2_seeds=%s\nphase2_variant_jobs=%s\nphase2_diagnostics=%s\nstage3b_submit=%s\n' \
  "$PHASE2_TAG" "$PHASE2_SEEDS" "$VARIANT_JOBS" "$PHASE2_DIAGNOSTIC_JOB" "$STAGE3B_JOB"
