#!/usr/bin/env python3
"""Regression check for the NCFM split-specific normalization contract."""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

# Allow direct execution as ``python scripts/validate_ncfm_normalize.py``.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.medical_dataset_utils import MEDICAL_DATASET_SPECS, load_medical_splits


DATASETS = ("PathMNIST", "COVID", "Kvasir")


def make_all_fixtures(root: Path) -> Path:
    for dataset_name in DATASETS:
        dataset_root = root / dataset_name
        for split in ("train", "val", "test"):
            class_dir = dataset_root / split / "class0"
            class_dir.mkdir(parents=True)
            image = np.full((16, 16, 3), 128, dtype=np.uint8)
            Image.fromarray(image).save(class_dir / "sample.png")
    return root


def transform_names(dataset) -> list[str]:
    return [type(item).__name__ for item in dataset.transform.transforms]


def check_dataset(dataset_name: str, data_root: Path) -> None:
    splits = load_medical_splits(dataset_name, data_root, train_skip_normalize=True)
    train_names = transform_names(splits["train"])
    val_names = transform_names(splits["val"])
    test_names = transform_names(splits["test"])
    expected_train = ["ToTensor", "Resize"]
    expected_eval = ["ToTensor", "Resize", "Normalize"]
    assert train_names == expected_train, (dataset_name, train_names)
    assert val_names == expected_eval, (dataset_name, val_names)
    assert test_names == expected_eval, (dataset_name, test_names)
    assert sum(name == "Normalize" for name in val_names) == 1
    assert sum(name == "Normalize" for name in test_names) == 1

    expected_size = tuple(MEDICAL_DATASET_SPECS[dataset_name]["im_size"])
    image, _ = splits["train"][0]
    assert tuple(image.shape) == (3, *expected_size), (dataset_name, image.shape)

    default_splits = load_medical_splits(dataset_name, data_root)
    default_names = transform_names(default_splits["train"])
    assert default_names == expected_eval, (dataset_name, default_names)
    print(f"PASS {dataset_name}: train={train_names}, val/test={expected_eval}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=Path,
        help="Use an existing prepared data root instead of a temporary fixture.",
    )
    args = parser.parse_args()

    if args.data_root:
        for dataset_name in DATASETS:
            check_dataset(dataset_name, args.data_root)
    else:
        with tempfile.TemporaryDirectory(prefix="ncfm-normalize-") as temp_dir:
            data_root = make_all_fixtures(Path(temp_dir))
            for dataset_name in DATASETS:
                check_dataset(dataset_name, data_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
