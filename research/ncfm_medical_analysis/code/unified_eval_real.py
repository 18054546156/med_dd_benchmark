#!/usr/bin/env python3
"""Reproducible downstream evaluator for real medical DD results.

This evaluator is intentionally independent of each method's evaluator.  It
uses the same test split, raw-image contract, normalization manifest, model
architectures, optimizer, schedule, and seeds for NCFM/HoP outputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


SPECS = {
    "PathMNIST": {"size": 32, "classes": 9, "depth": 3, "ipc": 10},
    "COVID": {"size": 112, "classes": 4, "depth": 5, "ipc": 10},
    "Kvasir": {"size": 128, "classes": 8, "depth": 5, "ipc": 10},
}

PROTOCOL = {
    "version": "controlled-eval-v1",
    "optimizer": "SGD",
    "lr": 0.01,
    "momentum": 0.9,
    "weight_decay": 0.0005,
    "epochs": 1000,
    "batch_size": 256,
    "lr_schedule": {"type": "multistep", "milestones": [500, 750], "gamma": 0.1},
    "augmentation": "none",
    "test_split": "test",
    "default_repeats": 5,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def root_from_file() -> Path:
    return Path(__file__).resolve().parents[3]


def import_medical_loader(root: Path):
    shared_utils = root / "utils"
    if str(shared_utils) not in sys.path:
        sys.path.insert(0, str(shared_utils))
    from medical_dataset_utils import load_medical_splits, get_medical_spec
    return load_medical_splits, get_medical_spec


def import_models(root: Path):
    ncfm_root = root / "adapted" / "ncfm"
    shared_utils = root / "utils"
    # The adapted NCFM package contains a package named ``utils``.  It must
    # precede the shared directory for NCFM's define_model imports.
    for path in (shared_utils, ncfm_root):
        if str(path) in sys.path:
            sys.path.remove(str(path))
    sys.path.insert(0, str(ncfm_root))
    sys.path.insert(1, str(shared_utils))
    from utils.utils import define_model
    import models.resnet as resnet
    return define_model, resnet


def read_statistics(root: Path, dataset: str) -> tuple[torch.Tensor, torch.Tensor, dict]:
    path = root / "data" / "prepared" / dataset / "statistics.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing authoritative statistics: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status", "complete") != "complete":
        raise RuntimeError(
            f"Dataset statistics audit did not pass: {path}; status={payload.get('status')}"
        )
    stats = payload["statistics"]
    if int(stats.get("duplicate_file_count", 0)):
        raise RuntimeError(f"Dataset contains duplicate RGB pixels: {path}")
    mean = torch.tensor(stats["mean"], dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor(stats["std"], dtype=torch.float32).view(1, 3, 1, 1)
    return mean, std, {"path": str(path), "sha256": sha256(path), "mean": stats["mean"], "std": stats["std"]}


def load_synthetic(path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    if path.is_dir():
        image_path, label_path = path / "images_best.pt", path / "labels_best.pt"
        if not image_path.is_file() or not label_path.is_file():
            raise FileNotFoundError(f"Expected images_best.pt and labels_best.pt in {path}")
        images, labels = torch.load(image_path, map_location="cpu"), torch.load(label_path, map_location="cpu")
    else:
        payload = torch.load(path, map_location="cpu")
        if isinstance(payload, dict):
            if "data" in payload and "label" in payload:
                images, labels = payload["data"], payload["label"]
            elif "images" in payload and "labels" in payload:
                images, labels = payload["images"], payload["labels"]
            else:
                raise ValueError(f"Unsupported tensor dictionary keys: {list(payload)}")
        elif isinstance(payload, (tuple, list)) and len(payload) == 2:
            images, labels = payload
        else:
            raise ValueError(f"Unsupported synthetic data type: {type(payload).__name__}")
    if not isinstance(images, torch.Tensor):
        images = torch.stack(list(images))
    if not isinstance(labels, torch.Tensor):
        labels = torch.as_tensor(labels)
    labels = labels.reshape(-1).long()
    images = images.float()
    if images.ndim != 4:
        raise ValueError(f"Synthetic images must be NCHW, got {tuple(images.shape)}")
    if len(images) != len(labels):
        raise ValueError("Synthetic images and labels have different lengths")
    if float(images.min()) < -1e-5 or float(images.max()) > 1.00001:
        raise ValueError("Synthetic images must be raw [0,1] tensors before evaluation")
    return images.clamp(0.0, 1.0), labels


class NullLogger:
    def __call__(self, *args, **kwargs):
        return None


def make_model(root: Path, dataset: str, architecture: str, device: torch.device):
    spec = SPECS[dataset]
    define_model, resnet = import_models(root)
    if architecture == "ConvNet":
        model = define_model(dataset.lower(), "instance", "convnet", 3, spec["depth"], 1.0,
                             spec["classes"], NullLogger(), spec["size"])
    elif architecture == "ResNet18":
        model = resnet.ResNet(dataset.lower(), 18, spec["classes"], norm_type="instance",
                              size=spec["size"], nch=3)
    else:
        raise ValueError(f"Unsupported architecture: {architecture}")
    return model.to(device)


def accuracy(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in loader:
            output = model(images.to(device))
            correct += int((output.argmax(1).cpu() == labels).sum())
            total += len(labels)
    return 100.0 * correct / max(total, 1)


def train_once(images: torch.Tensor, labels: torch.Tensor, test_loader: DataLoader,
               root: Path, dataset: str, architecture: str, seed: int,
               device: torch.device) -> dict:
    set_seed(seed)
    model = make_model(root, dataset, architecture, device)
    loader = DataLoader(TensorDataset(images, labels), batch_size=PROTOCOL["batch_size"],
                        shuffle=True, num_workers=0)
    optimizer = torch.optim.SGD(model.parameters(), lr=PROTOCOL["lr"],
                                momentum=PROTOCOL["momentum"], weight_decay=PROTOCOL["weight_decay"])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=PROTOCOL["lr_schedule"]["milestones"], gamma=PROTOCOL["lr_schedule"]["gamma"]
    )
    criterion = nn.CrossEntropyLoss()
    start = time.time()
    last_train = 0.0
    for _ in range(PROTOCOL["epochs"]):
        model.train()
        correct = total = 0
        for batch, target in loader:
            batch, target = batch.to(device), target.to(device)
            optimizer.zero_grad(set_to_none=True)
            output = model(batch)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            correct += int((output.argmax(1) == target).sum())
            total += len(target)
        last_train = 100.0 * correct / max(total, 1)
        scheduler.step()
    return {"seed": seed, "train_acc": last_train, "test_acc": accuracy(model, test_loader, device),
            "seconds": time.time() - start}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--dataset", choices=sorted(SPECS), required=True)
    parser.add_argument("--architecture", choices=["ConvNet", "ResNet18"], required=True)
    parser.add_argument("--repeats", type=int, default=PROTOCOL["default_repeats"])
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--method", choices=["NCFM", "HoP"], default=None)
    parser.add_argument("--variant", default="baseline",
                        help="Optional method variant label recorded in the result")
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()
    if args.repeats < 5:
        raise ValueError("Formal controlled evaluation requires at least 5 repeats")
    root = root_from_file()
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device if args.device != "auto" else "cpu")
    mean, std, stats_meta = read_statistics(root, args.dataset)
    images, labels = load_synthetic(args.data)
    spec = SPECS[args.dataset]
    if tuple(images.shape[1:]) != (3, spec["size"], spec["size"]):
        raise ValueError(f"Expected synthetic shape [N,3,{spec['size']},{spec['size']}], got {tuple(images.shape)}")
    expected_count = spec["classes"] * spec["ipc"]
    if len(images) != expected_count:
        raise ValueError(
            f"Expected IPC={spec['ipc']} synthetic samples per class "
            f"({expected_count} total), got {len(images)}"
        )
    images = (images - mean) / std
    load_splits, get_spec = import_medical_loader(root)
    # The shared loader leaves train raw for NCFM/HoP pipelines when requested,
    # while val/test retain the authoritative train-only normalization.
    splits = load_splits(args.dataset, root / "data", train_skip_normalize=False)
    test = splits["test"]
    test_loader = DataLoader(test, batch_size=256, shuffle=False, num_workers=0)
    records = [train_once(images, labels, test_loader, root, args.dataset, args.architecture,
                          args.seed + i, device) for i in range(args.repeats)]
    test_acc = [r["test_acc"] for r in records]
    result = {
        "status": "complete",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": PROTOCOL,
        "dataset": args.dataset,
        "architecture": args.architecture,
        "architecture_contract": {
            "name": args.architecture,
            "depth": spec["depth"] if args.architecture == "ConvNet" else 18,
            "norm": "instance",
            "input_shape": [3, spec["size"], spec["size"]],
        },
        "method": args.method,
        "variant": args.variant,
        "run_id": args.run_id,
        "evaluator_source_sha256": sha256(Path(__file__).resolve()),
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": str(device),
        },
        "synthetic_path": str(args.data),
        "synthetic_sha256": sha256(args.data) if args.data.is_file() else None,
        "synthetic_shape": list(images.shape),
        "statistics": stats_meta,
        "test_count": len(test),
        "synthetic_contract": {
            "raw_range_required": [0.0, 1.0],
            "normalization": "train-only statistics.json",
            "labels": "integer class ids",
        },
        "repeats": args.repeats,
        "records": records,
        "test_accuracy": {"mean": float(np.mean(test_acc)), "std": float(np.std(test_acc, ddof=1)),
                          "min": float(np.min(test_acc)), "max": float(np.max(test_acc)), "all": test_acc},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"status": "complete", "output": str(args.output), "mean_test_acc": result["test_accuracy"]["mean"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
