#!/usr/bin/env python3
"""Validate the shared medical loader contract without training a model."""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from analysis_paths import mathematical_root


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["PathMNIST", "COVID", "Kvasir"])
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    import sys
    sys.path.insert(0, str(root / "utils"))
    from medical_dataset_utils import load_medical_splits, get_medical_spec

    spec = get_medical_spec(args.dataset)
    signature = inspect.signature(load_medical_splits)
    if "train_skip_normalize" in signature.parameters:
        splits = load_medical_splits(args.dataset, root / "data", train_skip_normalize=True)
    elif "skip_normalize" in signature.parameters:
        raise RuntimeError(
            "The shared medical loader does not expose separate train/eval normalization; "
            "formal evaluation requires train_skip_normalize support."
        )
    else:
        raise RuntimeError("Unsupported medical loader signature: missing normalization control")
    result = {"dataset": args.dataset, "spec": spec, "splits": {}}
    for name in ("train", "val", "test"):
        ds = splits[name]
        x, y = next(iter(DataLoader(ds, batch_size=min(16, len(ds)), num_workers=0)))
        result["splits"][name] = {
            "count": len(ds),
            "shape": list(x.shape),
            "label_shape": list(y.shape),
            "min": float(x.min()),
            "max": float(x.max()),
            "mean": float(x.mean()),
            "std": float(x.std()),
        }
        expected = (3, *spec["im_size"])
        assert tuple(x.shape[1:]) == expected, (name, x.shape, expected)
        assert y.ndim == 1, (name, y.shape)
    train_transform = repr(getattr(splits["train"], "transform", None))
    eval_transform = repr(getattr(splits["val"], "transform", None))
    assert "Normalize" not in train_transform, train_transform
    assert "Normalize" in eval_transform, eval_transform
    assert result["splits"]["train"]["min"] >= 0.0
    assert result["splits"]["train"]["max"] <= 1.0
    out = mathematical_root(root) / f"contract_{args.dataset}.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"status": "passed", "output": str(out)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
