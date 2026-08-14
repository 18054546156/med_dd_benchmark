#!/usr/bin/env python3
"""Auditable, model-free analysis of the prepared medical datasets.

The script reads only real prepared data.  It does not train a classifier and
does not turn heuristics into accuracy claims.  Its output is intended to
guide NCFM/HoP configuration choices (resolution, depth, IPC, augmentation)
before expensive condensation runs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image


SPECS = {
    "PathMNIST": {"size": (32, 32), "classes": 9},
    "COVID": {"size": (112, 112), "classes": 4},
    "Kvasir": {"size": (128, 128), "classes": 8},
}
SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def iter_records(root: Path, dataset: str):
    if dataset == "PathMNIST" and (root / "pathmnist.npz").is_file():
        with np.load(root / "pathmnist.npz", allow_pickle=False) as archive:
            keys = {"train": "train_images", "val": "val_images", "test": "test_images"}
            label_keys = {"train": "train_labels", "val": "val_labels", "test": "test_labels"}
            for split, key in keys.items():
                images = archive[key]
                labels = archive[label_keys[split]].reshape(-1)
                for i, (image, label) in enumerate(zip(images, labels)):
                    yield split, str(int(label)), image, f"{key}[{i}]"
        return
    for split in ("train", "val", "test"):
        split_root = root / split
        if not split_root.is_dir():
            raise FileNotFoundError(f"Missing split directory: {split_root}")
        for class_dir in sorted(p for p in split_root.iterdir() if p.is_dir()):
            for path in sorted(p for p in class_dir.rglob("*") if p.is_file() and p.suffix.lower() in SUFFIXES):
                yield split, class_dir.name, path, str(path.relative_to(root))


def image_array(value, size):
    if isinstance(value, Path):
        with Image.open(value) as im:
            return np.asarray(im.convert("RGB").resize(size), dtype=np.float32) / 255.0
    return np.asarray(Image.fromarray(value).convert("RGB").resize(size), dtype=np.float32) / 255.0


def feature_vector(array: np.ndarray) -> np.ndarray:
    gray = array.mean(axis=2)
    gx = np.abs(np.diff(gray, axis=1)).mean()
    gy = np.abs(np.diff(gray, axis=0)).mean()
    return np.array([
        *array.mean(axis=(0, 1)),
        *array.std(axis=(0, 1)),
        float(gray.mean()),
        float(gray.std()),
        float(gx),
        float(gy),
    ], dtype=np.float64)


def analyze(args) -> dict:
    dataset = args.dataset
    root = args.data_root / "prepared" / dataset
    if not root.is_dir():
        raise FileNotFoundError(f"Prepared dataset directory does not exist: {root}")
    size = SPECS[dataset]["size"]
    counts = Counter()
    class_counts = defaultdict(Counter)
    sums = defaultdict(lambda: np.zeros(3, dtype=np.float64))
    squares = defaultdict(lambda: np.zeros(3, dtype=np.float64))
    pixels = Counter()
    feature_values = defaultdict(list)
    duplicate_hashes = defaultdict(list)
    original_modes = Counter()
    original_sizes = Counter()
    sampled = Counter()

    for split, label, value, logical_name in iter_records(root, dataset):
        counts[split] += 1
        class_counts[split][label] += 1
        if isinstance(value, Path):
            file_digest = sha256(value)
            duplicate_hashes[file_digest].append(logical_name)
            with Image.open(value) as im:
                original_modes[im.mode] += 1
                original_sizes[f"{im.width}x{im.height}"] += 1
        array = image_array(value, size)
        flat = array.reshape(-1, 3).astype(np.float64)
        sums[split] += flat.sum(axis=0)
        squares[split] += np.square(flat).sum(axis=0)
        pixels[split] += len(flat)
        if sampled[(split, label)] < args.max_per_class:
            feature_values[(split, label)].append(feature_vector(array))
            sampled[(split, label)] += 1

    def moments(split):
        if not pixels[split]:
            return {"mean": None, "std": None}
        mean = sums[split] / pixels[split]
        std = np.sqrt(np.maximum(squares[split] / pixels[split] - mean * mean, 0.0))
        return {"mean": mean.tolist(), "std": std.tolist()}

    def class_separability(split):
        groups = {label: np.stack(values) for (s, label), values in feature_values.items() if s == split and values}
        if len(groups) < 2:
            return {"status": "insufficient_evidence"}
        centroids = {label: values.mean(axis=0) for label, values in groups.items()}
        within = np.mean([np.linalg.norm(values - centroids[label], axis=1).mean() for label, values in groups.items()])
        between = np.mean([
            np.linalg.norm(a - b)
            for i, a in enumerate(centroids.values())
            for j, b in enumerate(centroids.values()) if j > i
        ])
        return {
            "status": "supported",
            "sampled_classes": sorted(groups),
            "within_distance": float(within),
            "between_centroid_distance": float(between),
            "between_to_within_ratio": float(between / max(within, 1e-12)),
        }

    train_moments = moments("train")
    split_shift = {}
    for split in ("val", "test"):
        if train_moments["mean"] is not None and moments(split)["mean"] is not None:
            split_shift[split] = {
                "mean_l2": float(np.linalg.norm(np.asarray(train_moments["mean"]) - np.asarray(moments(split)["mean"]))),
                "std_l2": float(np.linalg.norm(np.asarray(train_moments["std"]) - np.asarray(moments(split)["std"]))),
            }

    imbalance = {}
    for split, values in class_counts.items():
        numbers = np.asarray(list(values.values()), dtype=np.float64)
        imbalance[split] = {
            "min": int(numbers.min()), "max": int(numbers.max()),
            "ratio_max_to_min": float(numbers.max() / max(numbers.min(), 1)),
            "effective_number": float(numbers.sum() ** 2 / max(np.square(numbers).sum(), 1.0)),
        }

    recommendations = []
    if imbalance.get("train", {}).get("ratio_max_to_min", 1.0) >= 3:
        recommendations.append({"topic": "sampling", "suggestion": "compare class-balanced sampling or per-class condensation batches", "reason": "train imbalance ratio >= 3"})
    else:
        recommendations.append({"topic": "sampling", "suggestion": "keep class-balanced condensation batches as the primary control", "reason": "no severe train imbalance detected"})
    if split_shift.get("test", {}).get("mean_l2", 0.0) > 0.03:
        recommendations.append({"topic": "augmentation", "suggestion": "report color/brightness augmentation ablation and keep test transform fixed", "reason": "train-test color mean shift is non-trivial"})
    ratio = class_separability("train").get("between_to_within_ratio")
    if ratio is not None and ratio < 1.5:
        recommendations.append({"topic": "capacity", "suggestion": "test ConvNet-D3 against D5/ResNet18 and consider IPC>=20 as a sensitivity run", "reason": "handcrafted feature separability ratio < 1.5"})
    else:
        recommendations.append({"topic": "capacity", "suggestion": "retain the original backbone as primary and use cross-architecture evaluation", "reason": "handcrafted feature separability is not low"})

    duplicate_groups = [items for items in duplicate_hashes.values() if len(items) > 1]
    return {
        "status": "complete",
        "dataset": dataset,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "contract": {"size": list(size), "channels": 3, "classes": SPECS[dataset]["classes"]},
        "source": {"pathmnist_npz_sha256": sha256(root / "pathmnist.npz") if (root / "pathmnist.npz").is_file() else None},
        "split_counts": dict(counts),
        "class_counts": {split: dict(values) for split, values in class_counts.items()},
        "imbalance": imbalance,
        "pixel_statistics_raw_01": {split: moments(split) for split in ("train", "val", "test")},
        "split_shift_from_train": split_shift,
        "original_modes": dict(original_modes),
        "original_sizes": dict(original_sizes),
        "duplicate_hash_groups": duplicate_groups,
        "duplicate_file_count": sum(len(items) for items in duplicate_groups),
        "feature_separability_proxy": {split: class_separability(split) for split in ("train", "val", "test")},
        "recommendations": recommendations,
        "evidence_note": "Recommendations are dataset diagnostics, not measured downstream accuracy or proof of an NCFM defect.",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=sorted(SPECS), required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-per-class", type=int, default=500)
    args = parser.parse_args()
    result = analyze(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"status": result["status"], "dataset": args.dataset, "output": str(args.output), "counts": result["split_counts"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
