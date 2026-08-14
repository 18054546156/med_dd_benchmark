#!/usr/bin/env python3
"""Create a conservative evidence table from real Phase-1 JSON outputs."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


ORDER = ["e1.1", "e1.2", "e2.1", "e2.2", "e3.1", "e3.2", "e4.1", "e4.2"]


def status_for(exp: str, payload: dict | None) -> tuple[str, str]:
    if payload is None:
        return "insufficient_evidence", "No real-data result JSON was found."
    if payload.get("status") == "failed":
        return "failed", payload.get("error", "run failed")
    if exp == "e2.1" and payload.get("status") == "diagnostic_proxy":
        return "supported", "Held-out-bank proxy measured; not proof of training-bank overfit."
    if exp == "e3.1" and payload.get("status") == "complete":
        return "supported", "Real multi-seed condensation sensitivity was measured; this is not causal proof by itself."
    if exp == "e4.1" and payload.get("status") == "complete":
        return "supported", "Real paired CF-loss/downstream-evaluation association was measured; association is not causal evidence."
    if exp == "e4.2" and payload.get("status") == "complete":
        return "supported", "Real paired feature-space/pixel-space evaluations were measured under the declared evaluator."
    if payload.get("status") in {"insufficient_downstream_pairs", "insufficient_replicates"}:
        return "insufficient_evidence", payload.get("interpretation", "required evidence is missing")
    if exp == "e2.2":
        return "supported", "Real-teacher IS mechanism comparison; released NCFM learned-proposal defect remains unproven."
    if exp in {"e3.2"}:
        return "not_applicable", "No learned-frequency implementation was run in the released NCFM configuration."
    return "supported", "Real-data diagnostic completed; formal defect claim requires the protocol-specific effect test."


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for dataset in ("PathMNIST", "COVID", "Kvasir"):
        for exp in ORDER:
            candidates = sorted(args.runs.glob(f"phase1_real_{dataset.lower()}_{exp.replace('.', '_')}_job*/results.json"))
            if not candidates:
                rows.append({"dataset": dataset, "experiment": exp, "status": "insufficient_evidence",
                             "reason": "No real-data result JSON was found.", "result": None, "result_count": 0})
                continue
            # Preserve every explicit run.  The summary never selects one by
            # mtime or filename; formal adjudication consumes formal_report.
            for candidate in candidates:
                payload = json.loads(candidate.read_text(encoding="utf-8"))
                status, reason = status_for(exp, payload)
                rows.append({"dataset": dataset, "experiment": exp, "status": status, "reason": reason,
                             "result": str(candidate), "result_count": len(candidates)})
    lines = ["# NCFM Medical Phase 1 Evidence Summary", "",
             f"Generated: {datetime.now(timezone.utc).isoformat()}", "",
             "This report never converts missing or toy evidence into a formal defect claim.", "",
             "| Dataset | Experiment | Status | Reason |", "|---|---|---|---|"]
    for row in rows:
        lines.append(f"| {row['dataset']} | {row['experiment']} | {row['status']} | {row['reason']} |")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n\n```json\n" + json.dumps(rows, indent=2) + "\n```\n", encoding="utf-8")
    print(json.dumps({"status": "complete", "output": str(args.output), "rows": len(rows)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
