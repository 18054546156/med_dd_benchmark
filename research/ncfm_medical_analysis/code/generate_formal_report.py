#!/usr/bin/env python3
"""Build a conservative report from real benchmark artifacts only.

The report separates measured observations from causal defect claims. Missing
artifacts remain ``insufficient_evidence`` instead of being filled with legacy
numbers or examples.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from analysis_paths import mathematical_root


DATASETS = ("PathMNIST", "COVID", "Kvasir")
EXPERIMENTS = ("e1.1", "e1.2", "e2.1", "e2.2", "e3.1", "e3.2", "e4.1", "e4.2")
EXPECTED_PHASE1_KEYS = {(dataset, experiment) for dataset in DATASETS for experiment in EXPERIMENTS}
EXPECTED_EVAL_KEYS = {(method, dataset, architecture)
                      for method in ("NCFM", "HoP")
                      for dataset in DATASETS
                      for architecture in ("ConvNet", "ResNet18")}
SOURCE_METHODS = {"NCFM": "NCFM", "HoP": "HoP-TM"}


def read_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def phase1_rows(root: Path, array_id: str | None) -> list[dict]:
    runs = mathematical_root(root) / "runs"
    rows = []
    for dataset in DATASETS:
        for experiment in EXPERIMENTS:
            prefix = f"phase1_real_{dataset.lower()}_{experiment.replace('.', '_')}"
            pattern = f"{prefix}_array{array_id}_task*/results.json" if array_id else ""
            candidates = sorted(path for path in runs.glob(pattern) if path.is_file()) if pattern else []
            if not candidates:
                rows.append({
                    "dataset": dataset,
                    "experiment": experiment,
                    "status": "insufficient_evidence",
                    "path": None,
                    "result": None,
                })
                continue
            # Every run is retained. The report never chooses a result by
            # modification time or filename ordering.
            for path in candidates:
                payload = read_json(path)
                rows.append({
                    "dataset": dataset,
                    "experiment": experiment,
                    "status": payload.get("status") if payload else "failed",
                    "path": str(path),
                    "result": payload,
                })
    return rows


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def evaluation_rows(root: Path, manifest_path: Path | None) -> list[dict]:
    if manifest_path is None or not manifest_path.is_file():
        return []
    manifest = read_json(manifest_path)
    if not manifest or not isinstance(manifest.get("evaluations"), list):
        return []
    rows = []
    for item in manifest["evaluations"]:
        if not isinstance(item, dict) or not item.get("path"):
            continue
        path = Path(str(item["path"]))
        if not path.is_absolute():
            path = (root / path).resolve()
        payload = read_json(path)
        if not payload:
            continue
        expected = {key: item.get(key) for key in ("dataset", "method", "architecture") if item.get(key) is not None}
        actual = {key: payload.get(key) for key in expected}
        if expected != actual:
            raise ValueError(f"Evaluation manifest metadata does not match {path}: expected={expected}, actual={actual}")
        source_value = item.get("source_run_manifest")
        if source_value:
            source_path = Path(str(source_value))
            if not source_path.is_absolute():
                source_path = (root / source_path).resolve()
            source_payload = read_json(source_path)
            if not source_payload:
                raise ValueError(f"Evaluation source manifest is missing or invalid: {source_path}")
            expected_source_method = item.get("source_method", SOURCE_METHODS.get(item.get("method")))
            if (source_payload.get("status") != "complete"
                    or expected_source_method != SOURCE_METHODS.get(item.get("method"))
                    or source_payload.get("method") != expected_source_method
                    or source_payload.get("dataset") != item.get("dataset")):
                raise ValueError(f"Evaluation source manifest identity mismatch: {source_path}")
            declared_synthetic = Path(str(source_payload.get("synthetic", {}).get("path", "")))
            if not declared_synthetic.is_absolute():
                declared_synthetic = (root / declared_synthetic).resolve()
            entry_synthetic = Path(str(item.get("synthetic", "")))
            if not entry_synthetic.is_absolute():
                entry_synthetic = (root / entry_synthetic).resolve()
            if declared_synthetic.resolve() != entry_synthetic.resolve():
                raise ValueError(f"Evaluation source manifest synthetic does not match {path}")
        rows.append({
            "path": str(path),
            "sha256": sha256(path),
            "dataset": payload.get("dataset"),
            "method": payload.get("method"),
            "architecture": payload.get("architecture"),
            "status": payload.get("status", "failed"),
            "test_accuracy": payload.get("test_accuracy"),
            "protocol": payload.get("protocol"),
        })
    return rows


def phase2_rows(root: Path, array_id: str | None) -> list[dict]:
    """Load only the explicitly requested Phase 2 array outputs.

    A broad recursive scan would make a later report depend on stale runs.
    Phase 2 run IDs contain the Slurm array ID, so the caller must provide it
    for a formal report.
    """
    base = mathematical_root(root) / "phase2"
    if not array_id:
        return []
    rows = []
    pattern = f"phase2_*_array{array_id}_task*/results.json"
    for path in sorted(base.glob(pattern)):
        payload = read_json(path)
        if payload is None:
            continue
        rows.append({
            "path": str(path),
            "sha256": sha256(path),
            "dataset": payload.get("dataset"),
            "method": payload.get("method"),
            "status": payload.get("status", "failed"),
            "claim_scope": payload.get("claim_scope"),
        })
    return rows


def evidence_class(experiment: str, status: str) -> tuple[str, str]:
    if status == "failed":
        return "failed", "The real-data run failed; inspect its Slurm stderr."
    if status in {"insufficient_evidence", "insufficient_downstream_pairs", "insufficient_replicates"}:
        return "insufficient_evidence", "Required real artifacts or replicates are missing."
    if experiment == "e3.2":
        return "not_applicable", "The released baseline uses sampling_net=false; learned frequencies require a separate implementation."
    if experiment == "e2.1" and status == "diagnostic_proxy":
        return "supported", "Held-out-bank stability was measured, but the released code does not persist a training bank."
    if experiment == "e2.2" and status == "complete":
        return "supported", "Importance-sampling correction was measured as a mechanism control, not as a released-NCFM defect."
    if status == "complete":
        return "supported", "The declared real-data measurement completed; this is an observation, not an automatic causal defect claim."
    return "insufficient_evidence", "The output status is not sufficient for a formal conclusion."


def defect_adjudication(rows: list[dict]) -> dict:
    """Apply only pre-registered effect rules to completed real outputs."""
    by_exp = {}
    for row in rows:
        by_exp.setdefault(row["experiment"], []).append(row)
    decisions = {}

    e11 = [
        row["result"] for row in by_exp.get("e1.1", [])
        if isinstance(row.get("result"), dict) and row["result"].get("status") == "complete"
    ]
    e11_effect = []
    for payload in e11:
        sweep = payload.get("num_freqs_sweep", {})
        means = []
        for item in sweep.values():
            if not isinstance(item, dict):
                continue
            aggregate = item.get("aggregate", item)
            if isinstance(aggregate, dict) and aggregate.get("mean") is not None:
                means.append(aggregate["mean"])
        if means and min(means) > 0:
            e11_effect.append(max(means) / min(means))
    decisions["e1.1"] = {
        "status": "confirmed" if e11_effect and max(e11_effect) >= 1.20 else "insufficient_evidence",
        "effect": e11_effect,
        "rule": "max/min frequency-sweep mean >= 1.20",
    }

    e12_effect = []
    for payload in by_exp.get("e1.2", []):
        result = payload.get("result") or {}
        if not isinstance(result, dict):
            continue
        if result.get("status") == "complete":
            item = result.get("num_freqs_sweep", {}).get("4096", {})
            if item.get("std_reduction_fraction") is not None:
                e12_effect.append(float(item["std_reduction_fraction"]))
    decisions["e1.2"] = {
        "status": "confirmed" if e12_effect and sum(e12_effect) / len(e12_effect) >= 0.20 else "insufficient_evidence",
        "effect": e12_effect,
        "rule": "mean std reduction at T=4096 >= 0.20",
    }

    e21_effect = []
    for row in by_exp.get("e2.1", []):
        result = row.get("result")
        if not isinstance(result, dict) or result.get("status") != "diagnostic_proxy":
            continue
        paired = result.get("paired_heldout_minus_train", {})
        if paired.get("mean") is None:
            continue
        paired_ci = paired.get("ci95", [None, None])
        e21_effect.append({
            "gap": float(paired["mean"]),
            "paired_gap_ci95": paired_ci,
            "conservative_gap_lower": (
                float(paired_ci[0]) if paired_ci[0] is not None else None
            ),
        })
    # The released NCFM implementation does not persist its training bank.
    # This is a useful stability diagnostic, but it cannot confirm a defect in
    # the released training procedure without an instrumented rerun.
    decisions["e2.1"] = {
        "status": "supported" if e21_effect else "insufficient_evidence",
        "effect": e21_effect,
        "rule": "held-out minus train-bank gap has a positive conservative 95% interval",
        "claim_scope": "diagnostic_proxy_only",
    }

    e22_effect = []
    for row in by_exp.get("e2.2", []):
        result = row.get("result")
        if not isinstance(result, dict) or result.get("status") != "complete":
            continue
        corrected = result.get("corrected", {}).get("mean")
        uncorrected = result.get("uncorrected", {}).get("mean")
        if corrected is not None and uncorrected is not None and uncorrected > 0:
            e22_effect.append(float(corrected / uncorrected))
    # The proposal is synthetic mechanism evidence. The released baseline
    # samples standard Gaussian frequencies directly and exposes no learned
    # proposal density, so this experiment cannot establish a released-NCFM
    # bias or defect by itself.
    decisions["e2.2"] = {
        "status": "supported" if e22_effect else "insufficient_evidence",
        "effect": e22_effect,
        "rule": "corrected absolute error / uncorrected absolute error < 0.80",
        "claim_scope": "mechanism_control_only",
    }

    e31_effect = []
    for row in by_exp.get("e3.1", []):
        result = row.get("result")
        if isinstance(result, dict) and result.get("status") == "complete":
            stats = result.get("test_accuracy", {})
            if stats.get("std") is not None:
                e31_effect.append(float(stats["std"]))
    decisions["e3.1"] = {
        "status": "confirmed" if e31_effect and max(e31_effect) >= 2.0 else "insufficient_evidence",
        "effect": e31_effect,
        "rule": "controlled test-accuracy standard deviation >= 2 percentage points",
    }

    e41_effect = []
    for row in by_exp.get("e4.1", []):
        result = row.get("result")
        if isinstance(result, dict) and result.get("status") == "complete":
            assoc = result.get("association", {})
            pearson = assoc.get("pearson_cf_vs_accuracy")
            spearman = assoc.get("spearman_cf_vs_accuracy")
            if pearson is not None and spearman is not None:
                e41_effect.append({"abs_pearson": abs(float(pearson)), "abs_spearman": abs(float(spearman))})
    decisions["e4.1"] = {
        "status": "confirmed" if e41_effect and all(item["abs_pearson"] < 0.30 and item["abs_spearman"] < 0.30 for item in e41_effect) else "insufficient_evidence",
        "effect": e41_effect,
        "rule": "absolute Pearson and Spearman CF/accuracy association < 0.30",
    }

    e42_effect = []
    for row in by_exp.get("e4.2", []):
        result = row.get("result")
        if isinstance(result, dict) and result.get("status") == "complete":
            delta = result.get("delta_test_accuracy", {}).get("mean")
            if delta is not None:
                e42_effect.append(abs(float(delta)))
    decisions["e4.2"] = {
        "status": "confirmed" if e42_effect and sum(e42_effect) / len(e42_effect) >= 2.0 else "insufficient_evidence",
        "effect": e42_effect,
        "rule": "absolute paired mean test-accuracy delta >= 2 percentage points",
    }
    # The released baseline remains fixed-frequency. A learned-frequency
    # conclusion requires a separately rerun and audited condenser artifact.
    e32_effect = []
    for row in by_exp.get("e3.2", []):
        result = row.get("result")
        if not isinstance(result, dict) or result.get("status") != "complete":
            continue
        delta = result.get("delta_test_accuracy", {}).get("mean")
        if delta is not None:
            e32_effect.append(float(delta))
    decisions["e3.2"] = {
        "status": "confirmed" if e32_effect and abs(sum(e32_effect) / len(e32_effect)) >= 2.0 else "insufficient_evidence",
        "effect": e32_effect,
        "rule": "paired learned-frequency minus fixed-frequency mean test-accuracy delta >= 2 percentage points",
        "claim_scope": "requires a separately rerun learned-frequency condenser; baseline sampling_net=false is preserved",
    }
    return decisions


def build(root: Path, artifact_validation: Path | None, evaluation_manifest: Path | None,
          phase1_array_id: str | None, phase2_array_id: str | None,
          phase2_variant_report: Path | None) -> dict:
    math_root = mathematical_root(root)
    dataset_reports = {}
    for dataset in DATASETS:
        path = root / "research" / "ncfm_medical_analysis" / "dataset_analysis" / f"{dataset}.json"
        stats = root / "data" / "prepared" / dataset / "statistics.json"
        dataset_reports[dataset] = {
            "analysis": read_json(path),
            "analysis_path": str(path) if path.is_file() else None,
            "statistics": read_json(stats),
            "statistics_path": str(stats) if stats.is_file() else None,
        }

    task_analysis = {}
    for dataset in DATASETS:
        path = root / "research" / "ncfm_medical_analysis" / "task_analysis" / f"{dataset}.json"
        task_analysis[dataset] = {
            "analysis": read_json(path),
            "path": str(path) if path.is_file() else None,
        }
    task_analysis_complete = all(
        item["analysis"] is not None
        and item["analysis"].get("status") == "complete"
        and item["analysis"].get("teacher_count") == 20
        for item in task_analysis.values()
    )

    phase1 = phase1_rows(root, phase1_array_id)
    for row in phase1:
        row["evidence_status"], row["evidence_reason"] = evidence_class(row["experiment"], row["status"])

    phase1_keys = {(row["dataset"], row["experiment"]) for row in phase1}
    phase1_statuses = {(row["dataset"], row["experiment"]): row["status"] for row in phase1}
    # Some experiments are deliberately expected to return a scoped
    # non-conclusion (for example E2.1=diagnostic_proxy or E4.1 missing the
    # separately registered pairs).  They must not invalidate independent E1
    # evidence, but an absent/failed task must invalidate the global gate.
    phase1_complete = bool(
        phase1_array_id
        and phase1_keys == EXPECTED_PHASE1_KEYS
        and all(row.get("path") and row.get("status") != "failed" for row in phase1)
    )

    decisions_by_dataset = {}
    for dataset in DATASETS:
        decisions_by_dataset[dataset] = defect_adjudication([row for row in phase1 if row["dataset"] == dataset])
    evaluation_rows_data = evaluation_rows(root, evaluation_manifest)
    phase2 = phase2_rows(root, phase2_array_id)
    variant_report = read_json(phase2_variant_report) if phase2_variant_report else None
    variant_complete = bool(variant_report and variant_report.get("status") == "complete")
    evaluation_keys = {(row.get("method"), row.get("dataset"), row.get("architecture"))
                       for row in evaluation_rows_data}
    evaluation_complete = bool(
        evaluation_keys == EXPECTED_EVAL_KEYS
        and len(evaluation_rows_data) == len(EXPECTED_EVAL_KEYS)
        and all(row.get("status") == "complete" for row in evaluation_rows_data)
    )
    formal_gate_ok = bool(
        artifact_validation
        and read_json(artifact_validation)
        and read_json(artifact_validation).get("status") == "complete"
        and phase1_complete
        and evaluation_complete
        and task_analysis_complete
    )
    overall_complete = formal_gate_ok and variant_complete
    confirmed = [
        f"{dataset}:{experiment}"
        for dataset, decisions in decisions_by_dataset.items()
        for experiment, value in decisions.items()
        if value.get("status") == "confirmed" and overall_complete
    ]
    return {
        "status": "complete" if overall_complete else "insufficient_evidence",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "evidence_policy": {
            "real_data_only": True,
            "toy_or_example_numbers_allowed": False,
            "missing_artifacts_are_insufficient_evidence": True,
            "supported_is_not_equivalent_to_causal_defect_confirmed": True,
        },
        "artifact_validation": {
            "path": str(artifact_validation) if artifact_validation else None,
            "result": read_json(artifact_validation) if artifact_validation else None,
        },
        "datasets": dataset_reports,
        "task_analysis": task_analysis,
        "task_analysis_complete": task_analysis_complete,
        "phase1": phase1,
        "controlled_evaluations": evaluation_rows_data,
        "phase1_completeness": {
            "expected_count": len(EXPECTED_PHASE1_KEYS),
            "observed_count": len(phase1_keys),
            "complete": phase1_complete,
            "missing": sorted(EXPECTED_PHASE1_KEYS - phase1_keys),
            "non_conclusion_statuses": {f"{key[0]}:{key[1]}": status for key, status in phase1_statuses.items()
                                         if status not in {"complete", "not_applicable"}},
        },
        "evaluation_completeness": {
            "expected_count": len(EXPECTED_EVAL_KEYS),
            "observed_count": len(evaluation_keys),
            "complete": evaluation_complete,
            "missing": sorted(EXPECTED_EVAL_KEYS - evaluation_keys),
        },
        "controlled_evaluation_manifest": {
            "path": str(evaluation_manifest) if evaluation_manifest else None,
            "sha256": sha256(evaluation_manifest) if evaluation_manifest and evaluation_manifest.is_file() else None,
        },
        "phase1_array_id": phase1_array_id,
        "phase2_array_id": phase2_array_id,
        "phase2": phase2,
        "phase2_completeness": {
            "expected_count": 9,
            "observed_count": len(phase2),
            "complete": bool(phase2_array_id)
            and len(phase2) == 9
            and all(row.get("status") == "complete" for row in phase2),
        },
        "phase2_variants": {
            "path": str(phase2_variant_report) if phase2_variant_report else None,
            "report": variant_report,
            "complete": variant_complete,
        },
        "mathematical_analysis_root": str(math_root),
        "formal_gate_ok": formal_gate_ok,
        "defect_adjudication": decisions_by_dataset,
        "formal_conclusion": {
            "ncfm_defect_confirmed": bool(confirmed),
            "status": "confirmed" if confirmed else "insufficient_evidence",
            "confirmed_experiments": confirmed,
            "reason": "At least one preregistered effect rule was satisfied and all formal artifact, evaluation, and Phase 2 variant gates passed." if confirmed else "No preregistered effect rule is currently proven with complete formal artifact, evaluation, and Phase 2 variant evidence.",
        },
    }


def markdown(report: dict) -> str:
    lines = [
        "# NCFM Medical Benchmark Formal Report", "",
        f"Generated: {report['created_at']}", "",
        "This report uses real artifacts only. `supported` means that a declared measurement completed; it does not by itself prove a causal NCFM defect.", "",
        "## Dataset Evidence", "", "| Dataset | Statistics | Dataset analysis |", "|---|---|---|",
    ]
    for dataset, item in report["datasets"].items():
        lines.append(f"| {dataset} | {'present' if item['statistics'] else 'missing'} | {'present' if item['analysis'] else 'missing'} |")
    lines += ["", "## Phase 1", "", "| Dataset | Experiment | Run status | Evidence status | Reason |", "|---|---|---|---|---|"]
    for row in report["phase1"]:
        reason = row["evidence_reason"].replace("|", "/")
        lines.append(f"| {row['dataset']} | {row['experiment']} | {row['status']} | {row['evidence_status']} | {reason} |")
    lines += ["", "## Controlled Evaluation", "", "| Method | Dataset | Architecture | Status | Mean test accuracy |", "|---|---|---|---|---|"]
    for row in report["controlled_evaluations"]:
        accuracy = (row.get("test_accuracy") or {}).get("mean") if isinstance(row.get("test_accuracy"), dict) else None
        lines.append(f"| {row.get('method')} | {row.get('dataset')} | {row.get('architecture')} | {row.get('status')} | {accuracy} |")
    lines += ["", "## Phase 2 diagnostics", "", "| Method | Dataset | Status | Claim scope |", "|---|---|---|---|"]
    for row in report["phase2"]:
        lines.append(f"| {row.get('method')} | {row.get('dataset')} | {row.get('status')} | {row.get('claim_scope', '')} |")
    variant = report.get("phase2_variants", {})
    lines += ["", "## Phase 2 real variants", "", f"- status: `{('complete' if variant.get('complete') else 'insufficient_evidence')}`"]
    variant_report = variant.get("report") or {}
    variant_rows = variant_report.get("aggregates") or variant_report.get("results", [])
    for row in variant_rows:
        accuracy = row.get("mean")
        if accuracy is None:
            accuracy = (row.get("test_accuracy") or {}).get("mean")
        delta = row.get("delta_vs_baseline_pp")
        seed_note = f", seeds={row.get('seed_count')}" if row.get("seed_count") is not None else ""
        lines.append(
            f"- {row.get('variant')} / {row.get('dataset')} / {row.get('architecture')}: "
            f"mean test accuracy={accuracy}, delta_vs_baseline_pp={delta}{seed_note}"
        )
    conclusion = report["formal_conclusion"]
    lines += ["", "## Formal Conclusion", "", f"- status: `{conclusion['status']}`", f"- NCFM defect confirmed: `{conclusion['ncfm_defect_confirmed']}`", f"- confirmed experiments: `{', '.join(conclusion.get('confirmed_experiments', [])) or 'none'}`", f"- reason: {conclusion['reason']}", ""]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--artifact-validation", type=Path, default=None)
    parser.add_argument("--evaluation-manifest", type=Path, default=None)
    parser.add_argument("--phase1-array-id", default=None)
    parser.add_argument("--phase2-array-id", default=None)
    parser.add_argument("--phase2-variant-report", type=Path, default=None)
    args = parser.parse_args()
    root = args.root.resolve()
    artifact_validation = args.artifact_validation.resolve() if args.artifact_validation else None
    evaluation_manifest = args.evaluation_manifest.resolve() if args.evaluation_manifest else None
    phase2_variant_report = args.phase2_variant_report.resolve() if args.phase2_variant_report else None
    report = build(root, artifact_validation, evaluation_manifest, args.phase1_array_id,
                   args.phase2_array_id, phase2_variant_report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    args.output.with_suffix(".md").write_text(markdown(report) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "json": str(args.output.with_suffix('.json')), "markdown": str(args.output.with_suffix('.md'))}))
    return 0 if report["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
