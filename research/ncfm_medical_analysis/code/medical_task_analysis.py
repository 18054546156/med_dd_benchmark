#!/usr/bin/env python3
"""Analyze the medical task using the real NCFM teacher checkpoints.

This is a diagnostic, not a replacement for the controlled synthetic-data
evaluation.  It reports class balance and per-teacher train/val/test behavior
so later tuning can distinguish a hard medical task from a condensation issue.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader


SPECS = {
    "PathMNIST": {"size": 32, "classes": 9, "depth": 3},
    # Formal controlled production uses the same ConvNetD5 backbone as HoP.
    "COVID": {"size": 112, "classes": 4, "depth": 5},
    "Kvasir": {"size": 128, "classes": 8, "depth": 5},
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_teacher_dir(root: Path, dataset: str, manifest_path: Path | None,
                        explicit_dir: Path | None) -> Path:
    if explicit_dir is not None:
        return explicit_dir.resolve()
    if manifest_path is None:
        raise ValueError("teacher directory must come from --teacher-dir or --artifact-manifest")
    manifest = manifest_path if manifest_path.is_absolute() else (root / manifest_path).resolve()
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    entry = payload.get("datasets", payload).get(dataset, {})
    value = entry.get("teacher_dir") if isinstance(entry, dict) else None
    if not value or "<" in str(value) or ">" in str(value):
        raise ValueError(f"manifest has no explicit teacher_dir for {dataset}")
    path = Path(value)
    return path if path.is_absolute() else (root / path).resolve()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def import_components(root: Path):
    shared = root / "utils"
    ncfm = root / "adapted" / "ncfm"
    for path in (shared, ncfm):
        while str(path) in sys.path:
            sys.path.remove(str(path))
    sys.path.insert(0, str(ncfm))
    sys.path.insert(1, str(shared))
    from medical_dataset_utils import load_medical_splits, get_medical_spec, get_class_names
    from utils.utils import define_model
    return load_medical_splits, get_medical_spec, define_model


def load_state(path: Path, device: torch.device) -> dict:
    state = torch.load(path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise ValueError(f"Unsupported checkpoint payload: {path}")
    return {key.removeprefix("module."): value for key, value in state.items()}


def evaluate(model, loader, device, classes: int, mean: torch.Tensor | None = None,
             std: torch.Tensor | None = None) -> dict:
    confusion = torch.zeros(classes, classes, dtype=torch.int64)
    total = correct = 0
    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            images = images.float()
            if mean is not None and std is not None:
                images = (images - mean) / std
            output = model(images.to(device))
            prediction = output.argmax(1).cpu()
            labels = labels.reshape(-1).long().cpu()
            for target, predicted in zip(labels.tolist(), prediction.tolist()):
                if 0 <= target < classes and 0 <= predicted < classes:
                    confusion[target, predicted] += 1
            correct += int((prediction == labels).sum())
            total += len(labels)
    per_class = []
    for index in range(classes):
        count = int(confusion[index].sum())
        per_class.append({
            "class_id": index,
            "count": count,
            "accuracy": (100.0 * float(confusion[index, index]) / count) if count else None,
        })
    return {
        "count": total,
        "accuracy": 100.0 * correct / max(total, 1),
        "confusion_matrix": confusion.tolist(),
        "per_class": per_class,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--dataset", choices=sorted(SPECS), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--teacher-dir", type=Path, default=None,
                        help="Explicit directory containing premodel0..19 checkpoints")
    parser.add_argument("--artifact-manifest", type=Path, default=None,
                        help="Filled formal artifact manifest used to resolve teacher_dir")
    args = parser.parse_args()
    root = args.root.resolve()
    spec = SPECS[args.dataset]
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    set_seed(0)
    load_splits, get_spec, define_model = import_components(root)
    stats_path = root / "data" / "prepared" / args.dataset / "statistics.json"
    if not stats_path.is_file():
        raise FileNotFoundError(stats_path)
    stats = json.loads(stats_path.read_text(encoding="utf-8")).get("statistics", {})
    mean = torch.tensor(stats["mean"], dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor(stats["std"], dtype=torch.float32).view(1, 3, 1, 1)
    splits = load_splits(args.dataset, root / "data", train_skip_normalize=True)
    loaders = {name: DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
               for name, dataset in splits.items()}
    raw_train_counts = torch.bincount(torch.as_tensor(splits["train"].targets).reshape(-1).long(), minlength=spec["classes"])
    teacher_dir = resolve_teacher_dir(root, args.dataset, args.artifact_manifest, args.teacher_dir)
    checkpoint_paths = []
    for teacher_id in range(20):
        path = teacher_dir / f"premodel{teacher_id}_trained.pth.tar"
        if not path.is_file():
            raise FileNotFoundError(path)
        checkpoint_paths.append(path)
    records = []
    for teacher_id, checkpoint in enumerate(checkpoint_paths):
        class Logger:
            def __call__(self, *args, **kwargs):
                return None
        model = define_model(args.dataset.lower(), "instance", "convnet", 3, spec["depth"], 1.0,
                             spec["classes"], Logger(), spec["size"]).to(device)
        missing, unexpected = model.load_state_dict(load_state(checkpoint, device), strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"teacher checkpoint/model mismatch for teacher {teacher_id}: "
                f"missing={list(missing)}, unexpected={list(unexpected)}"
            )
        split_results = {}
        for split_name, loader in loaders.items():
            # train is raw because train_skip_normalize=True; val/test already
            # contain the train-only Normalize transform from the shared loader.
            split_mean = mean if split_name == "train" else None
            split_std = std if split_name == "train" else None
            split_results[split_name] = evaluate(model, loader, device, spec["classes"], split_mean, split_std)
        records.append({
            "teacher_id": teacher_id,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256(checkpoint),
            "missing_keys": list(missing),
            "unexpected_keys": list(unexpected),
            "splits": split_results,
        })
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    class_names = get_class_names(args.dataset) or [str(index) for index in range(spec["classes"])]
    summary = {}
    for split in ("train", "val", "test"):
        values = [record["splits"][split]["accuracy"] for record in records]
        per_class = []
        for class_id in range(spec["classes"]):
            class_values = [
                item["splits"][split]["per_class"][class_id]["accuracy"]
                for item in records
                if item["splits"][split]["per_class"][class_id]["accuracy"] is not None
            ]
            per_class.append({
                "class_id": class_id,
                "class_name": class_names[class_id] if class_id < len(class_names) else str(class_id),
                "mean": float(np.mean(class_values)) if class_values else None,
                "std": float(np.std(class_values, ddof=1)) if len(class_values) >= 2 else None,
                "teacher_count": len(class_values),
            })
        confusion = np.sum(
            np.asarray([record["splits"][split]["confusion_matrix"] for record in records]),
            axis=0,
        )
        summary[split] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "all": values,
            "per_class": per_class,
            "confusion_matrix_sum": confusion.astype(int).tolist(),
        }
    result = {
        "status": "complete",
        "dataset": args.dataset,
        "created_by": str(Path(__file__).resolve()),
        "source_sha256": sha256(Path(__file__).resolve()),
        "runtime": {"python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda, "device": str(device)},
        "spec": {**spec, "statistics": {"path": str(stats_path), "sha256": sha256(stats_path), "mean": stats["mean"], "std": stats["std"]}},
        "train_class_counts": raw_train_counts.tolist(),
        "class_names": class_names,
        "teacher_count": len(records),
        "summary": summary,
        "teachers": records,
        "tuning_notes": [
            "Use teacher train-vs-test gap to separate task/model underfit from condensation error.",
            "Use per-class test accuracy and class counts to decide whether IPC or class-balanced condensation needs a sensitivity run.",
            "Do not use this teacher baseline as synthetic-data accuracy; controlled evaluator remains the comparison metric.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"status": result["status"], "dataset": args.dataset, "output": str(args.output), "test_mean": summary["test"]["mean"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
