#!/usr/bin/env bash
# Continue after the six production jobs, even when one failed.
#
# A successful manifest enters the full formal pipeline. A failed or missing
# manifest produces an explicit insufficient-evidence record instead of
# leaving downstream jobs in DependencyNeverSatisfied.
#SBATCH --job-name=dd-finalize-production
#SBATCH --partition=cpu-amd9754
#SBATCH --qos=qos-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --output=logs/dd-finalize-production_%j.out
#SBATCH --error=logs/dd-finalize-production_%j.err

set -Eeuo pipefail
export PYTHONUNBUFFERED=1

ROOT="${BENCHMARK_ROOT:-/project/prj-sis01/xiaoyu_xu/med_dd_project/dd_benchmark}"
export NCFM_MATH_ROOT="${NCFM_MATH_ROOT:-$ROOT/research/ncfm_mathematical_analysis}"
PY="${PYTHON:-/home/xiaoyuxu2/.conda/envs/meddd/bin/python}"
BUILD_JOB_ID="${BUILD_JOB_ID:-unknown}"
RUN_TAG="${RUN_TAG:-unknown}"
cd "$ROOT"
mkdir -p "$ROOT/logs"

ARTIFACT_MANIFEST="$ROOT/research/ncfm_medical_analysis/formal_artifact_manifest.json"
EVAL_MANIFEST="$ROOT/research/ncfm_medical_analysis/formal_eval_manifest.json"
STATUS_PATH="$NCFM_MATH_ROOT/production_followup_${SLURM_JOB_ID}.json"

if [[ -f "$ARTIFACT_MANIFEST" && -f "$EVAL_MANIFEST" && "$RUN_TAG" != unknown ]]; then
  if grep -Fq "$RUN_TAG" "$ARTIFACT_MANIFEST" && grep -Fq "$RUN_TAG" "$EVAL_MANIFEST"; then
    exec bash research/ncfm_medical_analysis/code/submit_formal_pipeline.sbatch
  fi
fi

"$PY" - "$ROOT" "$BUILD_JOB_ID" "$RUN_TAG" "$STATUS_PATH" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

root = Path(sys.argv[1])
build_job = sys.argv[2]
run_tag = sys.argv[3]
output = Path(sys.argv[4])
payload = {
    "status": "insufficient_evidence",
    "created_at": datetime.now(timezone.utc).isoformat(),
    "reason": "The six production runs did not produce both formal artifact manifests.",
    "build_manifest_job_id": build_job,
    "run_tag": run_tag,
    "artifact_manifest": str(root / "research/ncfm_medical_analysis/formal_artifact_manifest.json"),
    "evaluation_manifest": str(root / "research/ncfm_medical_analysis/formal_eval_manifest.json"),
    "policy": "No Phase 1/Phase 2 claim may be made without explicit real artifacts.",
}
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(payload, sort_keys=True))
PY

# Generate a report even when production failed. The report remains
# insufficient_evidence and is never promoted to a formal conclusion.
"$PY" research/ncfm_medical_analysis/code/generate_formal_report.py \
  --root "$ROOT" \
  --output "$NCFM_MATH_ROOT/formal_report" || true
echo "production_followup=$STATUS_PATH"
