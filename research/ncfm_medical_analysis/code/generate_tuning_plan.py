#!/usr/bin/env python3
"""Generate evidence-linked medical tuning hypotheses.

This is deliberately a planning artifact, not a result generator.  It reads
real dataset/task/evaluation JSON files and never invents an accuracy value.
Every recommendation names the source artifact that triggered it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


DATASETS = ("PathMNIST", "COVID", "Kvasir")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path: Path) -> dict | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else None


def source(path: Path, payload: dict | None) -> dict:
    return {
        "path": str(path),
        "sha256": sha256(path) if path.is_file() else None,
        "status": payload.get("status") if payload else "missing",
    }


def plan_dataset(root: Path, dataset: str, formal_report: dict | None) -> dict:
    analysis_path = root / "research" / "ncfm_medical_analysis" / "dataset_analysis" / f"{dataset}.json"
    task_path = root / "research" / "ncfm_medical_analysis" / "task_analysis" / f"{dataset}.json"
    analysis = read(analysis_path)
    task = read(task_path)
    if not analysis or analysis.get("status") != "complete" or not task or task.get("status") != "complete":
        return {
            "status": "insufficient_evidence",
            "dataset": dataset,
            "reason": "Complete real dataset analysis and 20-teacher task analysis are required.",
            "sources": {"dataset_analysis": source(analysis_path, analysis), "task_analysis": source(task_path, task)},
            "recommendations": [],
        }

    recommendations = []
    imbalance = analysis.get("imbalance", {}).get("train", {})
    ratio = imbalance.get("ratio_max_to_min")
    if isinstance(ratio, (int, float)) and ratio >= 3:
        recommendations.append({
            "priority": "high",
            "experiment": "class-balanced-condensation",
            "suggestion": "Compare class-balanced real batches or per-class condensation against the baseline.",
            "reason": f"train max/min class-count ratio={ratio:.3f} >= 3",
        })
    else:
        recommendations.append({
            "priority": "medium",
            "experiment": "sampling-control",
            "suggestion": "Keep the baseline sampler as primary and run a class-balanced sensitivity control.",
            "reason": "The measured train imbalance is not by itself severe enough to replace the baseline.",
        })

    summary = task.get("summary", {})
    train = summary.get("train", {}).get("mean")
    test = summary.get("test", {}).get("mean")
    per_class = summary.get("test", {}).get("per_class", [])
    measured_classes = [
        item for item in per_class
        if isinstance(item, dict) and isinstance(item.get("mean"), (int, float))
    ]
    if measured_classes:
        weakest = sorted(measured_classes, key=lambda item: float(item["mean"]))[:2]
        unstable = [
            item for item in measured_classes
            if isinstance(item.get("std"), (int, float)) and float(item["std"]) >= 2.0
        ]
        if weakest and float(weakest[0]["mean"]) < 60.0:
            recommendations.append({
                "priority": "high",
                "experiment": "class-conditional-ipc",
                "suggestion": "Inspect the weakest test classes and compare class-balanced or class-conditional synthetic allocation before increasing global model depth.",
                "reason": "teacher test per-class mean below 60%: " + ", ".join(
                    f"{item.get('class_name', item.get('class_id'))}={float(item['mean']):.2f}%" for item in weakest
                ),
            })
        if unstable:
            recommendations.append({
                "priority": "medium",
                "experiment": "class-wise-seed-stability",
                "suggestion": "Report class-wise seed variation and test an increased per-class IPC for unstable classes.",
                "reason": "teacher test per-class standard deviation >= 2 percentage points for: " + ", ".join(
                    str(item.get("class_name", item.get("class_id"))) for item in unstable
                ),
            })
    if isinstance(train, (int, float)) and isinstance(test, (int, float)):
        gap = float(train - test)
        if gap >= 10:
            recommendations.append({
                "priority": "high",
                "experiment": "teacher-capacity-and-regularization",
                "suggestion": "Sweep teacher epochs/regularization and compare the configured backbone before interpreting condensation loss.",
                "reason": f"20-teacher mean train-test gap={gap:.3f} percentage points",
            })
        else:
            recommendations.append({
                "priority": "medium",
                "experiment": "backbone-generalization",
                "suggestion": "Retain the baseline teacher and use cross-architecture controlled evaluation as the next sensitivity test.",
                "reason": f"20-teacher mean train-test gap={gap:.3f} percentage points",
            })

    if formal_report:
        evaluations = formal_report.get("controlled_evaluations", [])
        matching = [
            item for item in evaluations
            if item.get("dataset") == dataset and item.get("architecture") == "ConvNet"
            and isinstance((item.get("test_accuracy") or {}).get("mean"), (int, float))
        ]
        if matching and isinstance(test, (int, float)):
            for item in matching:
                method = item.get("method")
                accuracy = float(item["test_accuracy"]["mean"])
                gap = float(test - accuracy)
                if gap >= 10:
                    recommendations.append({
                        "priority": "high",
                        "experiment": "ipc-frequency-sweep",
                        "suggestion": "Sweep IPC and frequency count before changing the medical backbone.",
                        "reason": f"teacher mean test accuracy - {method} controlled ConvNet accuracy={gap:.3f} percentage points",
                    })

    return {
        "status": "complete",
        "dataset": dataset,
        "sources": {"dataset_analysis": source(analysis_path, analysis), "task_analysis": source(task_path, task)},
        "teacher_summary": {"train": train, "test": test, "train_minus_test": (float(train - test) if isinstance(train, (int, float)) and isinstance(test, (int, float)) else None)},
        "recommendations": recommendations,
        "evidence_note": "These are preregistration candidates for the next tuning run, not proof of a causal defect.",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--formal-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    formal_path = args.formal_report if args.formal_report.is_absolute() else root / args.formal_report
    formal = read(formal_path)
    results = [plan_dataset(root, dataset, formal) for dataset in DATASETS]
    status = "complete" if all(item["status"] == "complete" for item in results) else "insufficient_evidence"
    payload = {
        "status": status,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "formal_report": {"path": str(formal_path.resolve()), "sha256": sha256(formal_path) if formal_path.is_file() else None, "status": formal.get("status") if formal else "missing"},
        "datasets": results,
        "policy": "Recommendations are evidence-linked hypotheses; no recommendation is a formal NCFM defect claim.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    lines = ["# Medical Tuning Plan", "", f"Status: `{status}`", ""]
    for item in results:
        lines += [f"## {item['dataset']}", ""]
        for recommendation in item.get("recommendations", []):
            lines.append(f"- [{recommendation['priority']}] {recommendation['experiment']}: {recommendation['suggestion']} ({recommendation['reason']})")
        if not item.get("recommendations"):
            lines.append("- insufficient_evidence")
        lines.append("")
    args.output.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": status, "json": str(args.output.with_suffix('.json')), "markdown": str(args.output.with_suffix('.md'))}))
    return 0 if status == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
