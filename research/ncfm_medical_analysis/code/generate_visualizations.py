#!/usr/bin/env python3
"""Generate figures only from completed real JSON artifacts.

No fallback values are used. A missing input produces a manifest entry with
``insufficient_evidence`` and no fabricated curve.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from analysis_paths import mathematical_root


DATASETS = ("PathMNIST", "COVID", "Kvasir")


def read_json(path: Path):
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def setup_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_dataset_statistics(root: Path, output: Path, plt) -> dict:
    rows = []
    for dataset in DATASETS:
        payload = read_json(root / "research" / "ncfm_medical_analysis" / "dataset_analysis" / f"{dataset}.json")
        if payload:
            rows.append((dataset, payload.get("split_counts", {})))
    if not rows:
        return {"status": "insufficient_evidence", "reason": "No dataset analysis JSON found."}
    fig, ax = plt.subplots(figsize=(8, 5))
    splits = ("train", "val", "test")
    x = list(range(len(rows)))
    width = 0.24
    for offset, split in enumerate(splits):
        ax.bar([v + (offset - 1) * width for v in x], [item[1].get(split, 0) for item in rows], width, label=split)
    ax.set_xticks(x, [item[0] for item in rows])
    ax.set_ylabel("Number of images")
    ax.set_title("Prepared medical dataset split sizes")
    ax.legend()
    fig.tight_layout()
    path = output / "dataset_split_counts.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return {"status": "complete", "path": str(path), "source_count": len(rows)}


def plot_class_balance(root: Path, output: Path, plt) -> dict:
    rows = []
    for dataset in DATASETS:
        payload = read_json(root / "research" / "ncfm_medical_analysis" / "dataset_analysis" / f"{dataset}.json")
        counts = (payload or {}).get("class_counts", {}).get("train") if payload else None
        if isinstance(counts, dict) and counts:
            rows.append((dataset, counts))
    if not rows:
        return {"status": "insufficient_evidence", "reason": "No real train class-count analysis found."}
    fig, axes = plt.subplots(1, len(rows), figsize=(max(8, 4 * len(rows)), 4), squeeze=False)
    for axis, (dataset, counts) in zip(axes[0], rows):
        labels = list(counts)
        axis.bar(labels, [counts[label] for label in labels], color="#12b76a")
        axis.set_title(dataset)
        axis.set_xlabel("Class")
        axis.set_ylabel("Train images")
        axis.tick_params(axis="x", rotation=45)
    fig.suptitle("Real medical training-set class balance")
    fig.tight_layout()
    path = output / "dataset_class_balance.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return {"status": "complete", "path": str(path), "source_count": len(rows)}


def plot_teacher_accuracy(root: Path, output: Path, plt) -> dict:
    rows = []
    for dataset in DATASETS:
        payload = read_json(root / "research" / "ncfm_medical_analysis" / "task_analysis" / f"{dataset}.json")
        if not payload or payload.get("status") != "complete":
            continue
        teachers = payload.get("teachers", [])
        if len(teachers) != 20:
            continue
        values = {
            split: [item.get("splits", {}).get(split, {}).get("accuracy") for item in teachers]
            for split in ("train", "val", "test")
        }
        if all(all(isinstance(value, (int, float)) for value in values[split]) for split in values):
            rows.append((dataset, values))
    if not rows:
        return {"status": "insufficient_evidence", "reason": "No complete 20-teacher task analysis found."}
    fig, axes = plt.subplots(1, len(rows), figsize=(max(8, 4 * len(rows)), 4), squeeze=False)
    for axis, (dataset, values) in zip(axes[0], rows):
        positions = list(range(1, 21))
        for split, color in (("train", "#175cd3"), ("val", "#f79009"), ("test", "#b42318")):
            axis.plot(positions, values[split], marker=".", linewidth=1, label=split, color=color)
        axis.set_title(dataset)
        axis.set_xlabel("Teacher ID")
        axis.set_ylabel("Accuracy (%)")
        axis.set_xticks([1, 5, 10, 15, 20])
        axis.legend()
    fig.suptitle("NCFM teacher task performance on real splits")
    fig.tight_layout()
    path = output / "teacher_accuracy_by_split.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return {"status": "complete", "path": str(path), "source_count": len(rows)}


def plot_teacher_confusion(root: Path, output: Path, plt) -> dict:
    rows = []
    for dataset in DATASETS:
        payload = read_json(root / "research" / "ncfm_medical_analysis" / "task_analysis" / f"{dataset}.json")
        matrix = (payload or {}).get("summary", {}).get("test", {}).get("confusion_matrix_sum") if payload else None
        if payload and payload.get("status") == "complete" and isinstance(matrix, list) and matrix:
            rows.append((dataset, payload.get("class_names", []), np.asarray(matrix, dtype=float)))
    if not rows:
        return {"status": "insufficient_evidence", "reason": "No complete teacher confusion matrices found."}
    fig, axes = plt.subplots(1, len(rows), figsize=(max(8, 4 * len(rows)), 4), squeeze=False)
    for axis, (dataset, class_names, matrix) in zip(axes[0], rows):
        normalized = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1.0)
        image = axis.imshow(normalized, vmin=0.0, vmax=1.0, cmap="Blues")
        labels = [str(name)[:12] for name in class_names] or [str(i) for i in range(len(matrix))]
        axis.set_title(dataset)
        axis.set_xlabel("Predicted")
        axis.set_ylabel("True")
        axis.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
        axis.set_yticks(range(len(labels)), labels)
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    fig.suptitle("Aggregated NCFM teacher test confusion matrices")
    fig.tight_layout()
    path = output / "teacher_test_confusion_matrices.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return {"status": "complete", "path": str(path), "source_count": len(rows)}


def plot_phase1(root: Path, output: Path, plt) -> dict:
    report = read_json(mathematical_root(root) / "formal_report.json")
    if not report:
        return {"status": "insufficient_evidence", "reason": "formal_report.json is missing."}
    decisions = report.get("defect_adjudication", {})
    names, values, colors = [], [], []
    skipped = []
    # The formal report is keyed as dataset -> experiment -> decision.  Keep
    # the experiment-specific units separate in the labels; averaging effects
    # from different experiments would make the chart scientifically wrong.
    dict_effect_key = {
        "e2.1": "conservative_gap_lower",
        "e4.1": "abs_pearson",
    }
    for dataset, dataset_decisions in decisions.items():
        if not isinstance(dataset_decisions, dict):
            skipped.append(dataset)
            continue
        for experiment, decision in dataset_decisions.items():
            if not isinstance(decision, dict):
                skipped.append(f"{dataset}:{experiment}")
                continue
            effect = decision.get("effect", [])
            key = dict_effect_key.get(experiment)
            numeric = []
            for item in effect if isinstance(effect, list) else []:
                if isinstance(item, (int, float)):
                    numeric.append(float(item))
                elif key and isinstance(item, dict) and isinstance(item.get(key), (int, float)):
                    numeric.append(float(item[key]))
            if numeric:
                names.append(f"{dataset}\n{experiment}")
                values.append(sum(numeric) / len(numeric))
                colors.append("#b42318" if decision.get("status") == "confirmed" else "#667085")
            else:
                skipped.append(f"{dataset}:{experiment}")
    if not values:
        return {"status": "insufficient_evidence", "reason": "No numeric Phase 1 effect output is available.", "skipped": skipped}
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(names, values, color=colors)
    ax.set_ylabel("Measured effect (experiment-specific units)")
    ax.set_title("NCFM Phase 1 preregistered effects")
    ax.axhline(0, color="#344054", linewidth=0.8)
    fig.tight_layout()
    path = output / "phase1_effects.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return {"status": "complete", "path": str(path), "experiments": names, "skipped": skipped}


def plot_evaluations(root: Path, output: Path, plt) -> dict:
    # Use the formal manifest as the boundary. A recursive scan of
    # results/controlled_eval would mix old replications, Phase 2 variants and
    # unrelated smoke outputs into the production comparison.
    manifest_path = root / "research" / "ncfm_medical_analysis" / "formal_eval_manifest.json"
    manifest = read_json(manifest_path)
    entries = manifest.get("evaluations", []) if manifest else []
    rows = []
    errors = []
    for entry in entries:
        if not isinstance(entry, dict) or not entry.get("path"):
            errors.append("formal evaluation manifest contains an invalid entry")
            continue
        path = Path(str(entry["path"]))
        if not path.is_absolute():
            path = (root / path).resolve()
        payload = read_json(path)
        expected = {
            key: entry.get(key)
            for key in ("method", "dataset", "architecture")
            if entry.get(key) is not None
        }
        actual = {key: payload.get(key) for key in expected} if payload else {}
        if payload and expected != actual:
            errors.append(f"metadata mismatch: {path}")
            continue
        accuracy = payload.get("test_accuracy", {}).get("mean") if payload else None
        if payload and payload.get("status") == "complete" and isinstance(accuracy, (int, float)):
            rows.append((f"{payload.get('method')}\n{payload.get('dataset')}\n{payload.get('architecture')}", accuracy))
    if not rows:
        return {"status": "insufficient_evidence", "reason": "No completed formal controlled-evaluation JSON found.", "manifest": str(manifest_path), "errors": errors}
    fig, ax = plt.subplots(figsize=(max(8, len(rows) * 1.2), 5))
    ax.bar(range(len(rows)), [item[1] for item in rows], color="#175cd3")
    ax.set_xticks(range(len(rows)), [item[0] for item in rows], rotation=35, ha="right")
    ax.set_ylabel("Mean test accuracy (%)")
    ax.set_title("Controlled evaluation: real synthetic artifacts only")
    fig.tight_layout()
    path = output / "controlled_eval_accuracy.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return {"status": "complete", "path": str(path), "result_count": len(rows), "manifest": str(manifest_path), "errors": errors}


def plot_phase2_variants(root: Path, output: Path, plt) -> dict:
    payload = read_json(mathematical_root(root) / "phase2_variant_report.json")
    if not payload or payload.get("status") != "complete":
        return {"status": "insufficient_evidence", "reason": "Complete Phase 2 variant report is missing."}
    rows = [row for row in payload.get("results", []) if isinstance(row.get("delta_vs_baseline_pp"), (int, float))]
    if not rows:
        return {"status": "insufficient_evidence", "reason": "No paired Phase 2 deltas are available."}
    labels = [f"{row['variant']}\n{row['dataset']}\n{row['architecture']}" for row in rows]
    values = [row["delta_vs_baseline_pp"] for row in rows]
    colors = ["#12b76a" if value >= 0 else "#f04438" for value in values]
    fig, ax = plt.subplots(figsize=(max(10, len(rows) * 0.8), 5))
    ax.bar(range(len(rows)), values, color=colors)
    ax.axhline(0, color="#344054", linewidth=0.8)
    ax.set_xticks(range(len(rows)), labels, rotation=45, ha="right")
    ax.set_ylabel("Delta vs baseline (percentage points)")
    ax.set_title("Phase 2 real NCFM variants: paired controlled evaluation")
    fig.tight_layout()
    path = output / "phase2_variant_deltas.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return {"status": "complete", "path": str(path), "result_count": len(rows)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    try:
        plt = setup_matplotlib()
    except ImportError as exc:
        result = {"status": "failed", "error": f"matplotlib is required: {exc}"}
    else:
        result = {
            "status": "complete",
            "figures": {
                "dataset_statistics": plot_dataset_statistics(root, args.output, plt),
                "class_balance": plot_class_balance(root, args.output, plt),
                "teacher_accuracy": plot_teacher_accuracy(root, args.output, plt),
                "teacher_confusion": plot_teacher_confusion(root, args.output, plt),
                "phase1_effects": plot_phase1(root, args.output, plt),
                "controlled_evaluations": plot_evaluations(root, args.output, plt),
                "phase2_variants": plot_phase2_variants(root, args.output, plt),
            },
            "policy": "No figure is generated from missing, toy, legacy, or hand-entered values.",
        }
        figure_statuses = [
            value.get("status")
            for value in result["figures"].values()
            if isinstance(value, dict)
        ]
        if any(status == "failed" for status in figure_statuses):
            result["status"] = "failed"
        elif any(status == "insufficient_evidence" for status in figure_statuses):
            result["status"] = "insufficient_evidence"
    (args.output / "visualization_manifest.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
