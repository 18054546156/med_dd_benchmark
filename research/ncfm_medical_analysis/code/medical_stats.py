#!/usr/bin/env python3
"""Create authoritative train-only statistics for prepared ImageFolder/NPZ data."""

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
    "PathMNIST": ((32, 32), 9),
    "COVID": ((112, 112), 4),
    "Kvasir": ((128, 128), 8),
}
SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rgb_pixel_hash(path: Path) -> str:
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        digest = hashlib.sha256()
        digest.update(f"{rgb.width}x{rgb.height}:RGB\0".encode("ascii"))
        digest.update(rgb.tobytes())
    return digest.hexdigest()


def imagefolder(root: Path, size: tuple[int, int]) -> dict:
    counts = Counter()
    classes = defaultdict(Counter)
    modes = Counter()
    sizes = Counter()
    hashes = defaultdict(list)
    total = np.zeros(3, dtype=np.float64)
    squares = np.zeros(3, dtype=np.float64)
    pixels = 0
    for split in ("train", "val", "test"):
        split_root = root / split
        for class_dir in sorted(p for p in split_root.iterdir() if p.is_dir()):
            for path in sorted(p for p in class_dir.rglob("*") if p.is_file() and p.suffix.lower() in SUFFIXES):
                counts[split] += 1
                classes[split][class_dir.name] += 1
                hashes[rgb_pixel_hash(path)].append(str(path.relative_to(root)))
                with Image.open(path) as image:
                    modes[image.mode] += 1
                    sizes[f"{image.width}x{image.height}"] += 1
                    array = np.asarray(
                        image.convert("RGB").resize(size, Image.Resampling.BICUBIC),
                        dtype=np.float64,
                    ) / 255.0
                if split == "train":
                    flat = array.reshape(-1, 3)
                    total += flat.sum(axis=0)
                    squares += np.square(flat).sum(axis=0)
                    pixels += len(flat)
    mean = total / pixels
    std = np.sqrt(np.maximum(squares / pixels - mean * mean, 0.0))
    duplicate_groups = [items for items in hashes.values() if len(items) > 1]
    return {
        "split_counts": dict(counts),
        "class_counts": {split: dict(value) for split, value in classes.items()},
        "mean": mean.tolist(),
        "std": std.tolist(),
        "pixel_range": [0.0, 1.0],
        "resized_size": list(size),
        "statistics_split": "train",
        "pixel_count": pixels,
        "original_mode_counts": dict(modes),
        "original_size_counts": dict(sizes),
        "duplicate_hash_algorithm": "sha256(decoded RGB dimensions + pixel bytes)",
        "duplicate_hash_groups": duplicate_groups,
        "duplicate_file_count": sum(len(group) for group in duplicate_groups),
    }


def pathmnist(root: Path, size: tuple[int, int]) -> dict:
    with np.load(root / "pathmnist.npz", allow_pickle=False) as archive:
        mapping = {"train": "train_images", "val": "val_images", "test": "test_images"}
        counts = {split: int(archive[key].shape[0]) for split, key in mapping.items()}
        classes = {}
        total = np.zeros(3, dtype=np.float64)
        squares = np.zeros(3, dtype=np.float64)
        pixels = 0
        for split, key in mapping.items():
            images = archive[key]
            labels = archive[f"{split}_labels"].reshape(-1)
            classes[split] = {str(int(c)): int((labels == c).sum()) for c in np.unique(labels)}
            if split != "train":
                continue
            for start in range(0, len(images), 512):
                batch = images[start:start + 512]
                resized = np.stack([
                    np.asarray(
                        Image.fromarray(item).resize(size, Image.Resampling.BICUBIC),
                        dtype=np.float64,
                    ) / 255.0
                    for item in batch
                ])
                flat = resized.reshape(-1, 3)
                total += flat.sum(axis=0)
                squares += np.square(flat).sum(axis=0)
                pixels += len(flat)
    mean = total / pixels
    std = np.sqrt(np.maximum(squares / pixels - mean * mean, 0.0))
    return {
        "split_counts": counts,
        "class_counts": classes,
        "mean": mean.tolist(),
        "std": std.tolist(),
        "pixel_range": [0.0, 1.0],
        "resized_size": list(size),
        "statistics_split": "train",
        "pixel_count": pixels,
        "source_sha256": file_hash(root / "pathmnist.npz"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=sorted(SPECS), required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    size, classes = SPECS[args.dataset]
    root = args.data_root / "prepared" / args.dataset
    result = pathmnist(root, size) if args.dataset == "PathMNIST" and (root / "pathmnist.npz").is_file() else imagefolder(root, size)
    duplicate_count = int(result.get("duplicate_file_count", 0))
    payload = {
        "status": "invalid_duplicate_data" if duplicate_count else "complete",
        "dataset": args.dataset,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "data_root": str(root),
        "contract": {"size": list(size), "num_classes": classes, "channels": 3},
        "statistics_space": "raw RGB pixels after resize, values in [0,1]",
        "statistics": result,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "status": payload["status"],
        "dataset": args.dataset,
        "output": str(args.output),
        "mean": result["mean"],
        "std": result["std"],
        "duplicate_file_count": duplicate_count,
    }))
    return 2 if duplicate_count else 0


if __name__ == "__main__":
    raise SystemExit(main())
