#!/usr/bin/env python3
"""Compute train-split channel statistics for the medical benchmark.

The statistics are computed after the same ToTensor + resize operation used by
the benchmark loader. Only the train split is read; val/test are never used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
DATASETS = {
    "PathMNIST": {"size": (32, 32), "npz": "pathmnist.npz"},
    "COVID": {"size": (112, 112)},
    "Kvasir": {"size": (128, 128)},
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def update_stats(image, sums: np.ndarray, squares: np.ndarray) -> int:
    values = image.to(dtype=torch.float64)
    sums += values.sum(dim=(1, 2)).cpu().numpy()
    squares += values.square().sum(dim=(1, 2)).cpu().numpy()
    return image.shape[1] * image.shape[2]


def iter_pathmnist(root: Path):
    npz_path = root / "PathMNIST" / "pathmnist.npz"
    if not npz_path.exists():
        npz_path = root / "pathmnist.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"PathMNIST NPZ not found below {root}")

    with np.load(npz_path, allow_pickle=False) as archive:
        images = archive["train_images"]
        for image in images:
            yield Image.fromarray(image, mode="RGB")


def iter_imagefolder(root: Path, dataset: str):
    train_root = root / dataset / "train"
    if not train_root.is_dir():
        raise FileNotFoundError(f"train directory not found: {train_root}")
    files = sorted(
        path
        for path in train_root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not files:
        raise FileNotFoundError(f"no images found below {train_root}")
    for path in files:
        with Image.open(path) as image:
            yield image.convert("RGB")


def compute_dataset(root: Path, dataset: str) -> dict:
    size = tuple(DATASETS[dataset]["size"])
    iterator = (
        iter_pathmnist(root)
        if dataset == "PathMNIST"
        else iter_imagefolder(root, dataset)
    )
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(
            size,
            interpolation=transforms.InterpolationMode.BICUBIC,
        ),
    ])
    sums = np.zeros(3, dtype=np.float64)
    squares = np.zeros(3, dtype=np.float64)
    pixels = 0
    images = 0
    for image in iterator:
        tensor = transform(image)
        pixels += update_stats(tensor, sums, squares)
        images += 1
    mean = sums / pixels
    std = np.sqrt(np.maximum(squares / pixels - np.square(mean), 0.0))
    return {
        "dataset": dataset,
        "split": "train",
        "resize": list(size),
        "images": images,
        "pixels_per_channel": pixels,
        "mean": [float(value) for value in mean],
        "std": [float(value) for value in std],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    results = {}
    for dataset in DATASETS:
        print(f"computing {dataset} train statistics", flush=True)
        results[dataset] = compute_dataset(args.data_root, dataset)
        print(json.dumps(results[dataset], ensure_ascii=False), flush=True)

    payload = {
        "contract": "train split only; ToTensor then bicubic resize; values in [0, 1]",
        "data_root": str(args.data_root),
        "datasets": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
