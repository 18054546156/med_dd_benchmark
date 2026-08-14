#!/usr/bin/env python3
"""Runtime checks for the real medical loader and unified evaluator.

This is intentionally a preflight check.  It constructs every evaluator
architecture and reads one batch from every prepared split, but it does not
train, select an old result, or report an accuracy.
"""

from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader


DATASETS = ("PathMNIST", "COVID", "Kvasir")
ARCHITECTURES = ("ConvNet", "ResNet18")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve() if args.root else Path(__file__).resolve().parents[3]
    code_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(code_dir))

    from unified_eval_real import SPECS, import_medical_loader, make_model, read_statistics

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    load_splits, get_spec = import_medical_loader(root)
    result = {
        "status": "complete",
        "device": str(device),
        "datasets": {},
        "models": {},
    }
    errors = []

    for dataset in DATASETS:
        try:
            mean, std, stats_meta = read_statistics(root, dataset)
            signature = inspect.signature(load_splits)
            if "train_skip_normalize" not in signature.parameters:
                raise RuntimeError("medical loader must expose train_skip_normalize")
            splits = load_splits(dataset, root / "data", train_skip_normalize=True)
            spec = get_spec(dataset)
            split_result = {"statistics": stats_meta, "spec": spec, "splits": {}}
            for name in ("train", "val", "test"):
                loader = DataLoader(splits[name], batch_size=min(4, len(splits[name])), shuffle=False, num_workers=0)
                images, labels = next(iter(loader))
                # Spell out the expected tuple to keep the check readable and
                # avoid depending on a dataset-specific tensor subclass.
                expected_shape = (3, SPECS[dataset]["size"], SPECS[dataset]["size"])
                if tuple(images.shape[1:]) != expected_shape:
                    raise ValueError(f"{dataset}/{name}: got {tuple(images.shape[1:])}, expected {expected_shape}")
                if labels.ndim != 1:
                    raise ValueError(f"{dataset}/{name}: labels must be 1-D, got {tuple(labels.shape)}")
                split_result["splits"][name] = {
                    "count": len(splits[name]),
                    "shape": list(images.shape),
                    "label_shape": list(labels.shape),
                    "min": float(images.min()),
                    "max": float(images.max()),
                }
            result["datasets"][dataset] = split_result
        except Exception as exc:
            errors.append(f"{dataset}: {exc}")
            result["datasets"][dataset] = {"status": "failed", "error": str(exc)}

        for architecture in ARCHITECTURES:
            key = f"{dataset}/{architecture}"
            try:
                model = make_model(root, dataset, architecture, device)
                dummy = torch.zeros(2, 3, SPECS[dataset]["size"], SPECS[dataset]["size"], device=device)
                with torch.no_grad():
                    output = model(dummy)
                if tuple(output.shape) != (2, SPECS[dataset]["classes"]):
                    raise ValueError(f"output shape {tuple(output.shape)}")
                result["models"][key] = {"status": "passed", "output_shape": list(output.shape)}
                del model, dummy, output
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception as exc:
                errors.append(f"{key}: {exc}")
                result["models"][key] = {"status": "failed", "error": str(exc)}

    if errors:
        result["status"] = "failed"
        result["errors"] = errors
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"status": result["status"], "output": str(args.output), "errors": errors}))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
