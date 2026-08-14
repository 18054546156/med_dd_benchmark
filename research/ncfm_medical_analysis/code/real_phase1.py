#!/usr/bin/env python3
"""Run auditable NCFM diagnostics on real medical data.

This module deliberately has no toy-data fallback.  It uses the prepared
dataset, an NCFM teacher checkpoint, and (where required) an NCFM synthetic
dataset.  Every run writes its protocol, inputs, seeds, raw measurements and
an evidence status.  A diagnostic can therefore report insufficient evidence
instead of silently turning into a synthetic demonstration.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Subset

from analysis_paths import mathematical_root


DATASETS = {
    "PathMNIST": {
        "key": "pathmnist",
        "size": 32,
        "classes": 9,
        "depth": 3,
        "data_dir": "data/prepared",
    },
    "COVID": {
        "key": "covid",
        "size": 112,
        "classes": 4,
        # Formal controlled production uses the same ConvNetD5 backbone as HoP.
        "depth": 5,
        "data_dir": "data/prepared",
    },
    "Kvasir": {
        "key": "kvasir",
        "size": 128,
        "classes": 8,
        "depth": 5,
        "data_dir": "data/prepared",
    },
}


@dataclass
class RunConfig:
    dataset: str
    experiment: str
    run_id: str
    num_freqs: int = 4096
    replicas: int = 20
    max_samples: int = 1024
    teacher_id: int = 0
    synthetic_data: str | None = None
    seed: int = 0
    device: str = "cuda"
    t_values: tuple[int, ...] = (256, 512, 1024, 2048, 4096, 8192)
    artifact_manifest: str | None = None
    teacher_dir: str | None = None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def json_dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def load_artifact_manifest(root: Path, path_value: str | None, dataset: str) -> tuple[dict, Path | None]:
    """Load only an explicitly supplied artifact manifest.

    A manifest is the boundary between an experiment plan and an actual
    result.  Filesystem ordering and modification time are never used.
    """
    if not path_value:
        return {}, None
    path = Path(path_value)
    if not path.is_absolute():
        path = (root / path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Artifact manifest does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    datasets = payload.get("datasets", payload)
    entry = datasets.get(dataset, {}) if isinstance(datasets, dict) else {}
    if not isinstance(entry, dict):
        raise ValueError(f"Artifact manifest entry for {dataset} must be an object")
    entry = dict(entry)
    entry["_manifest_path"] = str(path)
    entry["_manifest_sha256"] = sha256(path)
    return entry, path


def resolve_artifact_path(root: Path, value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else (root / path).resolve()


def missing_evidence(config: RunConfig, run: Path, reason: str, status: str = "insufficient_evidence") -> dict:
    result = {
        "status": status,
        "experiment": config.experiment,
        "dataset": config.dataset,
        "reason": reason,
        "evidence_policy": "No result is inferred without the required real artifact(s).",
    }
    json_dump(run / "results.json", result)
    return result


def percentile_ci(values: Iterable[float], seed: int = 0) -> dict:
    values = np.asarray(list(values), dtype=np.float64)
    if values.size == 0:
        return {"n": 0, "mean": None, "std": None, "ci95": [None, None]}
    rng = np.random.default_rng(seed)
    boot = rng.choice(values, size=(2000, values.size), replace=True).mean(axis=1)
    return {
        "n": int(values.size),
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1) if values.size > 1 else 0.0),
        "ci95": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))],
    }


def project_root() -> Path:
    # .../research/ncfm_medical_analysis/code/real_phase1.py -> repository root
    return Path(__file__).resolve().parents[3]


def load_stats(root: Path, dataset: str) -> dict:
    path = root / "data" / "prepared" / dataset / "statistics.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing authoritative statistics file: {path}. "
            "Run the medical data audit before Phase 1."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status", "complete") != "complete":
        raise RuntimeError(
            f"Dataset statistics audit did not pass: {path}; status={payload.get('status')}"
        )
    stats = payload.get("statistics", payload)
    if int(stats.get("duplicate_file_count", 0)):
        raise RuntimeError(f"Dataset contains duplicate RGB pixels and is not formal-run eligible: {path}")
    for key in ("mean", "std"):
        if key not in stats or len(stats[key]) != 3:
            raise ValueError(f"Invalid {key} in {path}")
    return {"path": str(path), "sha256": sha256(path), "mean": stats["mean"], "std": stats["std"]}


def import_ncfm(root: Path):
    ncfm = root / "adapted" / "ncfm"
    shared = root / "utils"
    # NCFM has a package named ``utils``; the shared medical helpers are
    # top-level modules under the separate project utils directory.  NCFM
    # must be first so ``utils.utils`` resolves to the adapted implementation.
    for path in (shared, ncfm):
        while str(path) in sys.path:
            sys.path.remove(str(path))
    sys.path.insert(0, str(shared))
    sys.path.insert(0, str(ncfm))
    from utils.utils import define_model
    # NCFM's adapted utils.py imports the shared medical helper as a top-level
    # module.  Importing it as ``utils.medical_dataset_utils`` would resolve
    # against adapted/ncfm/utils and fail because that package is separate.
    from medical_dataset_utils import load_medical_splits
    return define_model, load_medical_splits


def load_splits_with_eval_normalization(load_medical_splits, dataset: str, data_root: Path):
    """Return raw train data and normalized validation/test data.

    ``train_skip_normalize=True`` is intentional: the NCFM data path applies
    augmentation and normalization itself during training.  Feature
    diagnostics normalize the raw train batch explicitly in ``load_features``
    so it matches the loader-normalized validation/test batches exactly.
    """
    parameters = inspect.signature(load_medical_splits).parameters
    if "train_skip_normalize" in parameters:
        return load_medical_splits(dataset, data_root, train_skip_normalize=True)
    if "skip_normalize" in parameters:
        raise RuntimeError(
            "The shared medical loader cannot express raw train plus normalized val/test splits."
        )
    raise RuntimeError("Unsupported medical loader signature: missing normalization control")


def dataset_labels(dataset) -> np.ndarray:
    """Read labels without depending on ImageFolder versus MedMNIST internals."""
    for name in ("targets", "labels"):
        value = getattr(dataset, name, None)
        if value is not None:
            labels = np.asarray(value).reshape(-1)
            if len(labels) == len(dataset):
                return labels.astype(np.int64, copy=False)
    return np.asarray([
        int(torch.as_tensor(dataset[index][1]).reshape(-1)[0])
        for index in range(len(dataset))
    ], dtype=np.int64)


def stratified_indices(
    labels: np.ndarray,
    max_samples: int,
    seed: int,
    expected_classes: int,
) -> tuple[list[int], dict[str, int]]:
    """Select a deterministic, near-equal number of examples from each class."""
    classes = sorted(int(value) for value in np.unique(labels))
    expected = list(range(expected_classes))
    if classes != expected:
        raise ValueError(f"Expected classes {expected}, found {classes}")
    limit = min(int(max_samples), len(labels))
    rng = np.random.default_rng(seed)
    pools = {
        class_id: rng.permutation(np.flatnonzero(labels == class_id)).tolist()
        for class_id in classes
    }
    cursors = {class_id: 0 for class_id in classes}
    selected: list[int] = []
    while len(selected) < limit:
        progressed = False
        for class_id in classes:
            cursor = cursors[class_id]
            if cursor < len(pools[class_id]) and len(selected) < limit:
                selected.append(int(pools[class_id][cursor]))
                cursors[class_id] += 1
                progressed = True
        if not progressed:
            break
    counts = Counter(int(labels[index]) for index in selected)
    return selected, {str(class_id): int(counts[class_id]) for class_id in classes}


def load_features(config: RunConfig, root: Path, split: str) -> tuple[torch.Tensor, torch.Tensor, dict]:
    spec = DATASETS[config.dataset]
    define_model, load_medical_splits = import_ncfm(root)
    stats = load_stats(root, config.dataset)
    splits = load_splits_with_eval_normalization(load_medical_splits, config.dataset, root / "data")
    dataset = splits[split]
    labels_all = dataset_labels(dataset)
    split_offset = {"train": 0, "val": 100_000, "test": 200_000}.get(split, 300_000)
    indices, sampled_class_counts = stratified_indices(
        labels_all,
        config.max_samples,
        config.seed + split_offset,
        spec["classes"],
    )
    loader = DataLoader(Subset(dataset, indices), batch_size=128, shuffle=False, num_workers=0)

    checkpoint_root = resolve_artifact_path(root, config.teacher_dir)
    if checkpoint_root is None:
        checkpoint_root = root / "pretrained_models" / "ncfm" / config.dataset.lower()
    checkpoint = checkpoint_root / f"premodel{config.teacher_id}_trained.pth.tar"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing NCFM teacher checkpoint: {checkpoint}")

    class Logger:
        def __call__(self, *args, **kwargs):
            return None

    device = torch.device(config.device if config.device == "cpu" or torch.cuda.is_available() else "cpu")
    model = define_model(
        config.dataset.lower(), "instance", "convnet", 3, spec["depth"], 1.0,
        spec["classes"], Logger(), spec["size"],
    ).to(device)
    state = torch.load(checkpoint, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if isinstance(state, dict):
        state = {k.removeprefix("module."): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"teacher checkpoint/model mismatch for teacher {config.teacher_id}: "
            f"missing={list(missing)}, unexpected={list(unexpected)}"
        )
    model.eval()

    features, labels = [], []
    mean = torch.tensor(stats["mean"], dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor(stats["std"], dtype=torch.float32).view(1, 3, 1, 1)
    with torch.no_grad():
        for images, target in loader:
            # The formal loader deliberately returns train images in raw
            # [0, 1] form so NCFM can apply its own augmentation path.  Phase
            # 1 compares teacher features, so train/val/test must enter the
            # teacher with the same train-only normalization.
            if split == "train":
                images = (images.float() - mean) / std
            images = images.to(device)
            _, feat = model(images, return_features=True)
            features.append(feat.detach().cpu())
            labels.append(torch.as_tensor(target).reshape(-1).cpu())
    x = torch.cat(features)
    y = torch.cat(labels)
    meta = {
        "split": split,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "checkpoint_missing_keys": list(missing),
        "checkpoint_unexpected_keys": list(unexpected),
        "feature_shape": list(x.shape),
        "sampling": {
            "policy": "deterministic_stratified_round_robin",
            "seed": config.seed + split_offset,
            "selected_count": len(indices),
            "class_counts": sampled_class_counts,
        },
        "stats": stats,
    }
    return x, y, meta


def find_synthetic(config: RunConfig, root: Path) -> Path:
    """Resolve only an explicitly declared artifact.

    Selecting by modification time is unsafe in a benchmark: a legacy result
    can be newer than the formal run or a partial run can hide the intended
    input.  The path is therefore part of the run manifest.
    """
    if not config.synthetic_data:
        raise FileNotFoundError(
            "This experiment requires --synthetic-data pointing to an explicit "
            "data_*.pt artifact; automatic newest-file selection is disabled."
        )
    path = Path(config.synthetic_data)
    if not path.is_absolute():
        path = (root / path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Synthetic artifact does not exist: {path}")
    return path


def load_synthetic_features(config: RunConfig, root: Path) -> tuple[torch.Tensor, torch.Tensor, dict]:
    path = find_synthetic(config, root)
    define_model, _ = import_ncfm(root)
    spec = DATASETS[config.dataset]
    stats = load_stats(root, config.dataset)
    device = torch.device(config.device if config.device == "cpu" or torch.cuda.is_available() else "cpu")
    class Logger:
        def __call__(self, *args, **kwargs):
            return None
    model = define_model(config.dataset.lower(), "instance", "convnet", 3, spec["depth"], 1.0,
                         spec["classes"], Logger(), spec["size"]).to(device)
    checkpoint_root = resolve_artifact_path(root, config.teacher_dir)
    if checkpoint_root is None:
        checkpoint_root = root / "pretrained_models" / "ncfm" / config.dataset.lower()
    checkpoint = checkpoint_root / f"premodel{config.teacher_id}_trained.pth.tar"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing NCFM teacher checkpoint: {checkpoint}")
    state = torch.load(checkpoint, map_location=device)
    state = state.get("state_dict", state) if isinstance(state, dict) else state
    if isinstance(state, dict):
        state = {k.removeprefix("module."): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"teacher checkpoint/model mismatch for teacher {config.teacher_id}: "
            f"missing={list(missing)}, unexpected={list(unexpected)}"
        )
    model.eval()
    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, (tuple, list)) and len(payload) == 2:
        data, labels = payload
    elif isinstance(payload, dict) and {"data", "label"}.issubset(payload):
        data, labels = payload["data"], payload["label"]
    elif isinstance(payload, dict) and {"images", "labels"}.issubset(payload):
        data, labels = payload["images"], payload["labels"]
    else:
        raise ValueError(f"Unsupported synthetic tensor contract in {path}")
    labels = torch.as_tensor(labels).reshape(-1)
    if data.ndim != 4 or labels.ndim != 1:
        raise ValueError(f"Unexpected synthetic tensor contract in {path}: {data.shape}, {labels.shape}")
    if len(data) != len(labels):
        raise ValueError(f"Synthetic data and labels have different lengths in {path}")
    if float(data.min()) < -1e-5 or float(data.max()) > 1.00001:
        raise ValueError(f"Synthetic data must be raw [0,1] tensors before normalization: {path}")
    expected = (3, spec["size"], spec["size"])
    if tuple(data.shape[1:]) != expected:
        raise ValueError(f"Expected synthetic shape [N,{expected[0]},{expected[1]},{expected[2]}], got {tuple(data.shape)}")
    mean = torch.tensor(stats["mean"]).view(1, 3, 1, 1)
    std = torch.tensor(stats["std"]).view(1, 3, 1, 1)
    data = ((data.float().clamp(0, 1) - mean) / std).to(device)
    with torch.no_grad():
        _, features = model(data, return_features=True)
    return features.cpu(), labels.cpu(), {
        "synthetic": str(path),
        "synthetic_sha256": sha256(path),
        "feature_shape": list(features.shape),
        "stats": stats,
    }


def frequency_bank(num_freqs: int, dim: int, seed: int, kind: str = "mc") -> torch.Tensor:
    if kind == "mc":
        generator = torch.Generator().manual_seed(seed)
        return torch.randn(num_freqs, dim, generator=generator)
    if kind == "qmc":
        engine = torch.quasirandom.SobolEngine(dim, scramble=True, seed=seed)
        u = engine.draw(num_freqs).clamp(1e-6, 1 - 1e-6)
        return math.sqrt(2.0) * torch.erfinv(2.0 * u - 1.0)
    raise ValueError(f"Unknown frequency bank kind: {kind}")


def d_omega(real: torch.Tensor, synthetic: torch.Tensor, freqs: torch.Tensor) -> float:
    total = 0.0
    count = 0
    for start in range(0, len(freqs), 512):
        bank = freqs[start:start + 512]
        real_z = bank @ real.T
        syn_z = bank @ synthetic.T
        real_cf = torch.complex(real_z.cos().mean(1), real_z.sin().mean(1))
        syn_cf = torch.complex(syn_z.cos().mean(1), syn_z.sin().mean(1))
        total += float((real_cf - syn_cf).abs().square().sum().item())
        count += len(bank)
    return total / max(count, 1)


def exact_ncfm_cf(real: torch.Tensor, synthetic: torch.Tensor, freqs: torch.Tensor) -> float:
    def parts(x):
        z = freqs @ x.T
        r, i = torch.cos(z).mean(1), torch.sin(z).mean(1)
        return r, i, torch.sqrt(r.square() + i.square())
    rr, ri, rn = parts(real)
    sr, si, sn = parts(synthetic)
    amp = (rn - sn).square()
    phase = (2 * (rn * sn - sr * rr - si * ri)).clamp(min=1e-12)
    return float(torch.sqrt(0.5 * amp + 0.5 * phase).mean().item())


def classwise_metric(
    real: torch.Tensor,
    real_labels: torch.Tensor,
    synthetic: torch.Tensor,
    synthetic_labels: torch.Tensor,
    freqs: torch.Tensor,
    metric,
) -> tuple[float, dict[str, float]]:
    """Apply an NCFM discrepancy per class and aggregate classes equally."""
    real_labels = torch.as_tensor(real_labels).reshape(-1)
    synthetic_labels = torch.as_tensor(synthetic_labels).reshape(-1)
    real_classes = sorted(int(value) for value in torch.unique(real_labels).tolist())
    synthetic_classes = sorted(int(value) for value in torch.unique(synthetic_labels).tolist())
    if real_classes != synthetic_classes:
        raise ValueError(
            f"Real/synthetic class mismatch: real={real_classes}, synthetic={synthetic_classes}"
        )
    per_class = {
        str(class_id): float(metric(
            real[real_labels == class_id],
            synthetic[synthetic_labels == class_id],
            freqs,
        ))
        for class_id in real_classes
    }
    return float(np.mean(list(per_class.values()))), per_class


def classwise_d_omega(
    real: torch.Tensor,
    real_labels: torch.Tensor,
    synthetic: torch.Tensor,
    synthetic_labels: torch.Tensor,
    freqs: torch.Tensor,
) -> tuple[float, dict[str, float]]:
    return classwise_metric(real, real_labels, synthetic, synthetic_labels, freqs, d_omega)


def classwise_exact_ncfm_cf(
    real: torch.Tensor,
    real_labels: torch.Tensor,
    synthetic: torch.Tensor,
    synthetic_labels: torch.Tensor,
    freqs: torch.Tensor,
) -> tuple[float, dict[str, float]]:
    return classwise_metric(
        real, real_labels, synthetic, synthetic_labels, freqs, exact_ncfm_cf
    )


def summarize_per_class(records: list[dict[str, float]], seed: int) -> dict[str, dict]:
    if not records:
        return {}
    classes = sorted(records[0], key=int)
    return {
        class_id: percentile_ci([record[class_id] for record in records], seed)
        for class_id in classes
    }


def write_run(config: RunConfig, root: Path) -> Path:
    run = mathematical_root(root) / "runs" / config.run_id
    run.mkdir(parents=True, exist_ok=True)
    source_path = Path(__file__).resolve()
    slurm_job = os.environ.get("SLURM_JOB_ID")
    array_job = os.environ.get("SLURM_ARRAY_JOB_ID")
    array_task = os.environ.get("SLURM_ARRAY_TASK_ID")
    json_dump(run / "protocol.json", {
        "protocol_version": "phase1-real-v1",
        "created_at": utc_now(),
        "config": asdict(config),
        "source": {"path": str(source_path), "sha256": sha256(source_path)},
        "slurm": {
            "job_id": slurm_job,
            "array_job_id": array_job,
            "array_task_id": array_task,
            "stdout": os.environ.get("SLURM_STDOUT_PATH"),
            "stderr": os.environ.get("SLURM_STDERR_PATH"),
        },
        "dataset_contract": DATASETS[config.dataset],
        "evidence_policy": {
            "real_data_required": True,
            "toy_fallback": False,
            "minimum_replicates": 5,
            "diagnostic_results_are_not_sufficient_for_A_level_claim": True,
        },
        "input_policy": {
            "synthetic_data_explicit": config.synthetic_data is not None,
            "automatic_newest_artifact_selection": False,
        },
    })
    (run / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
    return run


def run_not_implemented_real_variant(config: RunConfig, root: Path, run: Path, variant: str) -> dict:
    """Record a planned variant without pretending the baseline ran it."""
    if variant == "learned_frequency":
        return missing_evidence(
            config, run,
            "The released NCFM baseline has sampling_net=false; a learned-frequency run must be a separately implemented and audited ablation.",
            status="not_applicable",
        )
    return missing_evidence(
        config, run,
        "A real pixel-space condenser with the same teacher, budget, seed and evaluator has not been executed.",
    )


def run_frequency_replicates(config: RunConfig, root: Path, run: Path, kind: str) -> dict:
    real, real_labels, real_meta = load_features(config, root, "train")
    other, other_labels, other_meta = load_synthetic_features(config, root)
    values = []
    sweep = {}
    for num_freqs in config.t_values:
        current = []
        per_class_records = []
        for replica in range(config.replicas):
            freqs = frequency_bank(num_freqs, real.shape[1], config.seed + replica, kind)
            aggregate, per_class = classwise_d_omega(
                real, real_labels, other, other_labels, freqs
            )
            current.append(aggregate)
            per_class_records.append(per_class)
        sweep[str(num_freqs)] = {
            "aggregate": percentile_ci(current, config.seed),
            "per_class": summarize_per_class(per_class_records, config.seed),
        }
        if num_freqs == config.num_freqs:
            values = current
    if not values:
        values = [sweep[str(config.num_freqs)]["aggregate"]["mean"]]
    result = {
        "status": "complete" if config.replicas >= 5 else "insufficient_replicates",
        "experiment": config.experiment,
        "source": "fixed real-train teacher features versus explicit NCFM synthetic features",
        "real_meta": real_meta,
        "comparison_meta": other_meta,
        "frequency_kind": kind,
        "num_freqs": config.num_freqs,
        "num_freqs_sweep": sweep,
        "replica_values": values,
        "statistics": percentile_ci(values, config.seed),
    }
    json_dump(run / "results.json", result)
    return result


def run_mc_qmc_comparison(config: RunConfig, root: Path, run: Path) -> dict:
    """Compare MC and scrambled-QMC on identical real teacher features."""
    real, real_labels, real_meta = load_features(config, root, "train")
    other, other_labels, other_meta = load_synthetic_features(config, root)
    sweep = {}
    for num_freqs in config.t_values:
        mc, qmc = [], []
        mc_per_class, qmc_per_class = [], []
        for replica in range(config.replicas):
            mc_value, mc_classes = classwise_d_omega(
                real, real_labels, other, other_labels,
                frequency_bank(num_freqs, real.shape[1], config.seed + replica, "mc"),
            )
            qmc_value, qmc_classes = classwise_d_omega(
                real, real_labels, other, other_labels,
                frequency_bank(num_freqs, real.shape[1], config.seed + replica, "qmc"),
            )
            mc.append(mc_value)
            qmc.append(qmc_value)
            mc_per_class.append(mc_classes)
            qmc_per_class.append(qmc_classes)
        mc_stats = percentile_ci(mc, config.seed)
        qmc_stats = percentile_ci(qmc, config.seed)
        sweep[str(num_freqs)] = {
            "mc": mc_stats,
            "qmc": qmc_stats,
            "mc_per_class": summarize_per_class(mc_per_class, config.seed),
            "qmc_per_class": summarize_per_class(qmc_per_class, config.seed),
            "std_reduction_fraction": float(
                (mc_stats["std"] - qmc_stats["std"]) / max(mc_stats["std"], 1e-12)
            ),
        }
    result = {
        "status": "complete" if config.replicas >= 5 else "insufficient_replicates",
        "experiment": config.experiment,
        "source": "fixed real-train teacher features versus explicit NCFM synthetic features",
        "comparison": "same feature pair, seed index, frequency count; MC vs scrambled Sobol-QMC",
        "real_meta": real_meta,
        "comparison_meta": other_meta,
        "num_freqs_sweep": sweep,
    }
    json_dump(run / "results.json", result)
    return result


def run_overfitting(config: RunConfig, root: Path, run: Path) -> dict:
    real, real_labels, real_meta = load_features(config, root, "train")
    synthetic, synthetic_labels, synthetic_meta = load_synthetic_features(config, root)
    train_values, heldout_values = [], []
    train_per_class, heldout_per_class = [], []
    for replica in range(config.replicas):
        train_bank = frequency_bank(config.num_freqs, real.shape[1], config.seed + replica, "mc")
        heldout_bank = frequency_bank(config.num_freqs, real.shape[1], config.seed + 10000 + replica, "mc")
        train_value, train_classes = classwise_d_omega(
            real, real_labels, synthetic, synthetic_labels, train_bank
        )
        heldout_value, heldout_classes = classwise_d_omega(
            real, real_labels, synthetic, synthetic_labels, heldout_bank
        )
        train_values.append(train_value)
        heldout_values.append(heldout_value)
        train_per_class.append(train_classes)
        heldout_per_class.append(heldout_classes)
    train_stats, heldout_stats = percentile_ci(train_values, config.seed), percentile_ci(heldout_values, config.seed)
    paired_gaps = [heldout - train for train, heldout in zip(train_values, heldout_values)]
    result = {
        "status": "diagnostic_proxy",
        "experiment": config.experiment,
        "interpretation": "NCFM does not persist a training frequency bank; this is a held-out-bank stability diagnostic, not proof of training-bank overfit.",
        "real_meta": real_meta,
        "synthetic_meta": synthetic_meta,
        "train_bank": train_stats,
        "heldout_bank": heldout_stats,
        "train_bank_per_class": summarize_per_class(train_per_class, config.seed),
        "heldout_bank_per_class": summarize_per_class(heldout_per_class, config.seed),
        "heldout_minus_train": float(heldout_stats["mean"] - train_stats["mean"]),
        "paired_heldout_minus_train": percentile_ci(paired_gaps, config.seed),
    }
    json_dump(run / "results.json", result)
    return result


def run_cf_accuracy(config: RunConfig, root: Path, run: Path) -> dict:
    real, real_labels, real_meta = load_features(config, root, "train")
    synthetic, synthetic_labels, synthetic_meta = load_synthetic_features(config, root)
    records = []
    for replica in range(config.replicas):
        bank = frequency_bank(config.num_freqs, real.shape[1], config.seed + replica, "mc")
        value, per_class = classwise_exact_ncfm_cf(
            real, real_labels, synthetic, synthetic_labels, bank
        )
        records.append({"seed": config.seed + replica, "cf_loss": value, "per_class": per_class})
    values = [r["cf_loss"] for r in records]
    result = {
        "status": "insufficient_downstream_pairs",
        "experiment": config.experiment,
        "interpretation": "CF checkpoints and a common downstream evaluator are required before testing CF/accuracy mismatch.",
        "real_meta": real_meta,
        "synthetic_meta": synthetic_meta,
        "records": records,
        "cf_statistics": percentile_ci(values, config.seed),
    }
    json_dump(run / "results.json", result)
    return result


def load_evaluation_json(
    root: Path,
    value: str,
    *,
    dataset: str | None = None,
    method: str | None = None,
    architecture: str | None = None,
    synthetic: Path | None = None,
) -> tuple[dict, Path]:
    path = resolve_artifact_path(root, value)
    if path is None or not path.is_file():
        raise FileNotFoundError(f"Missing controlled evaluation JSON: {value}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "complete":
        raise ValueError(f"Evaluation is not complete: {path}")
    if "test_accuracy" not in payload:
        raise ValueError(f"Evaluation has no test_accuracy: {path}")
    expected = {
        "dataset": dataset,
        "method": method,
        "architecture": architecture,
    }
    for key, expected_value in expected.items():
        if expected_value is not None and payload.get(key) != expected_value:
            raise ValueError(
                f"Evaluation identity mismatch for {path}: "
                f"{key}={payload.get(key)!r}, expected {expected_value!r}"
            )
    if synthetic is not None:
        declared = resolve_artifact_path(root, payload.get("synthetic_path", ""))
        if declared is None or declared.resolve() != synthetic.resolve():
            raise ValueError(
                f"Evaluation synthetic_path does not match its explicit pair: {path}"
            )
    return payload, path


def run_initialization_sensitivity(config: RunConfig, root: Path, run: Path, artifact: dict) -> dict:
    """Analyze independently condensed real artifacts and paired evaluations."""
    entries = artifact.get("condense_seeds", [])
    if not isinstance(entries, list) or len(entries) < 5:
        return missing_evidence(config, run, "E3.1 requires at least five explicit real condense seed entries.")
    records = []
    real, real_labels, real_meta = load_features(config, root, "train")
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict) or not entry.get("synthetic"):
            return missing_evidence(config, run, f"E3.1 seed entry {index} has no synthetic artifact.")
        seed_config = RunConfig(**{**asdict(config), "synthetic_data": str(resolve_artifact_path(root, entry["synthetic"]))})
        if not Path(seed_config.synthetic_data).is_file():
            return missing_evidence(config, run, f"E3.1 synthetic artifact is missing: {entry['synthetic']}")
        synthetic, synthetic_labels, synthetic_meta = load_synthetic_features(seed_config, root)
        train_bank = frequency_bank(config.num_freqs, real.shape[1], config.seed + index, "mc")
        heldout_bank = frequency_bank(config.num_freqs, real.shape[1], config.seed + 10000 + index, "mc")
        evaluation = {}
        evaluation_path = None
        if not entry.get("evaluation"):
            return missing_evidence(config, run, f"E3.1 seed {index} has no controlled evaluation JSON.")
        evaluation, evaluation_path = load_evaluation_json(
            root,
            entry["evaluation"],
            dataset=config.dataset,
            method="NCFM",
            architecture="ConvNet",
            synthetic=Path(seed_config.synthetic_data),
        )
        train_value, train_per_class = classwise_d_omega(
            real, real_labels, synthetic, synthetic_labels, train_bank
        )
        heldout_value, heldout_per_class = classwise_d_omega(
            real, real_labels, synthetic, synthetic_labels, heldout_bank
        )
        records.append({
            "seed": entry.get("seed", index),
            "synthetic": synthetic_meta,
            "evaluation": {
                "path": str(evaluation_path),
                "sha256": sha256(evaluation_path),
                "mean_test_acc": evaluation["test_accuracy"].get("mean"),
                "architecture": evaluation.get("architecture"),
            },
            "cf_train_bank": train_value,
            "cf_heldout_bank": heldout_value,
            "cf_train_bank_per_class": train_per_class,
            "cf_heldout_bank_per_class": heldout_per_class,
        })
    accuracies = [float(item["evaluation"]["mean_test_acc"]) for item in records]
    result = {
        "status": "complete",
        "experiment": config.experiment,
        "interpretation": "Real multi-seed condensation sensitivity; this measures instability and does not by itself prove a causal defect.",
        "real_meta": real_meta,
        "replicate_count": len(records),
        "records": records,
        "test_accuracy": percentile_ci(accuracies, config.seed),
        "cf_train_bank": percentile_ci([item["cf_train_bank"] for item in records], config.seed),
        "cf_heldout_bank": percentile_ci([item["cf_heldout_bank"] for item in records], config.seed),
        "failure_rate": 0.0,
    }
    result["heldout_minus_train"] = float(result["cf_heldout_bank"]["mean"] - result["cf_train_bank"]["mean"])
    json_dump(run / "results.json", result)
    return result


def run_cf_accuracy_pairs(config: RunConfig, root: Path, run: Path, artifact: dict) -> dict:
    """Pair explicit CF measurements with the same controlled evaluator outputs."""
    pairs = artifact.get("pairs", [])
    if not isinstance(pairs, list) or len(pairs) < 5:
        return missing_evidence(config, run, "E4.1 requires at least five explicit synthetic/evaluation pairs.")
    real, real_labels, real_meta = load_features(config, root, "train")
    records = []
    for index, entry in enumerate(pairs):
        if not isinstance(entry, dict) or not entry.get("synthetic") or not entry.get("evaluation"):
            return missing_evidence(config, run, f"E4.1 pair {index} is incomplete.")
        pair_config = RunConfig(**{**asdict(config), "synthetic_data": str(resolve_artifact_path(root, entry["synthetic"]))})
        expected = resolve_artifact_path(root, entry["synthetic"])
        if expected is None or not expected.is_file():
            return missing_evidence(config, run, f"E4.1 synthetic artifact is missing: {entry['synthetic']}")
        synthetic, synthetic_labels, synthetic_meta = load_synthetic_features(pair_config, root)
        evaluation, evaluation_path = load_evaluation_json(
            root,
            entry["evaluation"],
            dataset=config.dataset,
            method="NCFM",
            architecture="ConvNet",
            synthetic=expected,
        )
        declared = resolve_artifact_path(root, evaluation.get("synthetic_path", ""))
        if declared is None or expected is None or declared.resolve() != expected.resolve():
            raise ValueError(f"E4.1 evaluation does not reference its manifest synthetic artifact: {evaluation_path}")
        classwise_losses = [
            classwise_exact_ncfm_cf(
                real,
                real_labels,
                synthetic,
                synthetic_labels,
                frequency_bank(config.num_freqs, real.shape[1], config.seed + j),
            )
            for j in range(config.replicas)
        ]
        losses = [value for value, _ in classwise_losses]
        records.append({
            "run_id": entry.get("run_id", index),
            "synthetic": synthetic_meta,
            "evaluation": {"path": str(evaluation_path), "sha256": sha256(evaluation_path), "test_accuracy": evaluation["test_accuracy"]},
            "cf": percentile_ci(losses, config.seed),
            "cf_per_class": summarize_per_class(
                [per_class for _, per_class in classwise_losses], config.seed
            ),
        })
    cf = np.asarray([item["cf"]["mean"] for item in records], dtype=np.float64)
    acc = np.asarray([item["evaluation"]["test_accuracy"]["mean"] for item in records], dtype=np.float64)
    def corr(a, b):
        return None if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0 else float(np.corrcoef(a, b)[0, 1])

    def rank(values):
        order = np.argsort(np.argsort(np.asarray(values, dtype=np.float64)))
        return order.astype(np.float64)
    result = {
        "status": "complete",
        "experiment": config.experiment,
        "interpretation": "Real paired CF/accuracy association; association is not causal evidence.",
        "real_meta": real_meta,
        "pair_count": len(records),
        "records": records,
        "association": {
            "pearson_cf_vs_accuracy": corr(cf, acc),
            "spearman_cf_vs_accuracy": corr(rank(cf), rank(acc)),
            "higher_cf_is_worse": True,
        },
    }
    json_dump(run / "results.json", result)
    return result


def run_pixel_feature_pairs(config: RunConfig, root: Path, run: Path, artifact: dict) -> dict:
    """Compare explicitly paired feature-space and pixel-space real artifacts."""
    pairs = artifact.get("pixel_space_pairs", [])
    if not isinstance(pairs, list) or len(pairs) < 5:
        return missing_evidence(config, run, "E4.2 requires at least five explicit feature-space/pixel-space pairs and controlled evaluations.")
    records = []
    for index, entry in enumerate(pairs):
        required = ("feature_evaluation", "pixel_evaluation")
        if not isinstance(entry, dict) or any(not entry.get(key) for key in required):
            return missing_evidence(config, run, f"E4.2 pair {index} is incomplete.")
        feature_synthetic = resolve_artifact_path(root, entry.get("feature_synthetic"))
        pixel_synthetic = resolve_artifact_path(root, entry.get("pixel_synthetic"))
        if feature_synthetic is None or pixel_synthetic is None:
            return missing_evidence(config, run, f"E4.2 pair {index} has no explicit synthetic paths.")
        feature, feature_path = load_evaluation_json(
            root,
            entry["feature_evaluation"],
            dataset=config.dataset,
            method="NCFM",
            architecture="ConvNet",
            synthetic=feature_synthetic,
        )
        pixel, pixel_path = load_evaluation_json(
            root,
            entry["pixel_evaluation"],
            dataset=config.dataset,
            method="NCFM",
            architecture="ConvNet",
            synthetic=pixel_synthetic,
        )
        records.append({
            "seed": entry.get("seed", index),
            "feature_evaluation": {"path": str(feature_path), "sha256": sha256(feature_path), "test_accuracy": feature["test_accuracy"]},
            "pixel_evaluation": {"path": str(pixel_path), "sha256": sha256(pixel_path), "test_accuracy": pixel["test_accuracy"]},
            "delta_mean_test_acc": float(pixel["test_accuracy"]["mean"] - feature["test_accuracy"]["mean"]),
        })
    result = {
        "status": "complete",
        "experiment": config.experiment,
        "interpretation": "Real paired feature-space versus pixel-space comparison under the declared downstream evaluator.",
        "pair_count": len(records),
        "records": records,
        "delta_test_accuracy": percentile_ci([item["delta_mean_test_acc"] for item in records], config.seed),
    }
    json_dump(run / "results.json", result)
    return result


def gaussian_log_density_ratio(freqs: torch.Tensor, mean: torch.Tensor) -> torch.Tensor:
    """log p(w)/q(w) for p=N(0,I), q=N(mean,I)."""
    return -(freqs * mean).sum(dim=1) + 0.5 * mean.square().sum()


def classwise_per_frequency_error(
    real: torch.Tensor,
    real_labels: torch.Tensor,
    other: torch.Tensor,
    other_labels: torch.Tensor,
    freqs: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    real_classes = sorted(int(value) for value in torch.unique(real_labels).tolist())
    other_classes = sorted(int(value) for value in torch.unique(other_labels).tolist())
    if real_classes != other_classes:
        raise ValueError(f"Class mismatch: real={real_classes}, comparison={other_classes}")
    errors = []
    per_class = {}
    for class_id in real_classes:
        class_real = real[real_labels == class_id]
        class_other = other[other_labels == class_id]
        chunks = []
        for start in range(0, len(freqs), 512):
            bank = freqs[start:start + 512]
            real_z = bank @ class_real.T
            other_z = bank @ class_other.T
            real_cf = torch.complex(real_z.cos().mean(1), real_z.sin().mean(1))
            other_cf = torch.complex(other_z.cos().mean(1), other_z.sin().mean(1))
            chunks.append((real_cf - other_cf).abs().square())
        class_errors = torch.cat(chunks)
        errors.append(class_errors)
        per_class[str(class_id)] = class_errors
    return torch.stack(errors).mean(0), per_class


def run_importance_sampling(config: RunConfig, root: Path, run: Path) -> dict:
    """Measure an IS estimator on real teacher features.

    This is a mechanism experiment, not proof that the released NCFM sampler
    is biased: the released implementation samples standard Gaussian
    frequencies directly and does not expose a learned proposal density.
    """
    real, real_labels, real_meta = load_features(config, root, "train")
    other, other_labels, other_meta = load_features(config, root, "val")
    dim = real.shape[1]
    reference_freqs = frequency_bank(config.num_freqs * 4, dim, config.seed + 70000)
    reference, reference_per_class = classwise_d_omega(
        real, real_labels, other, other_labels, reference_freqs
    )
    records = []
    mean = torch.zeros(dim)
    mean[0] = 0.5
    for replica in range(config.replicas):
        generator = torch.Generator().manual_seed(config.seed + replica)
        proposal = torch.randn(config.num_freqs, dim, generator=generator) + mean
        weights = gaussian_log_density_ratio(proposal, mean).exp()
        discrepancy, per_class = classwise_per_frequency_error(
            real, real_labels, other, other_labels, proposal
        )
        uncorrected = float(discrepancy.mean())
        corrected = float((discrepancy * weights).mean())
        records.append({
            "seed": config.seed + replica,
            "uncorrected": uncorrected,
            "corrected": corrected,
            "uncorrected_abs_error": abs(uncorrected - reference),
            "corrected_abs_error": abs(corrected - reference),
            "effective_sample_size": float(weights.sum().square() / weights.square().sum()),
            "uncorrected_per_class": {
                class_id: float(values.mean()) for class_id, values in per_class.items()
            },
            "corrected_per_class": {
                class_id: float((values * weights).mean())
                for class_id, values in per_class.items()
            },
        })
    result = {
        "status": "complete" if config.replicas >= 5 else "insufficient_replicates",
        "experiment": config.experiment,
        "interpretation": "real-teacher mechanism test; does not establish a released NCFM learned-proposal defect",
        "reference": {
            "num_freqs": config.num_freqs * 4,
            "d_omega": reference,
            "per_class": reference_per_class,
        },
        "proposal": {"distribution": "N(mean,I)", "mean_l2": float(mean.norm())},
        "real_meta": real_meta,
        "comparison_meta": other_meta,
        "records": records,
        "uncorrected": percentile_ci([r["uncorrected_abs_error"] for r in records], config.seed),
        "corrected": percentile_ci([r["corrected_abs_error"] for r in records], config.seed),
    }
    json_dump(run / "results.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    parser.add_argument("--experiment", required=True, choices=["e1.1", "e1.2", "e2.1", "e2.2", "e3.1", "e3.2", "e4.1", "e4.2"])
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--num-freqs", type=int, default=4096)
    parser.add_argument("--replicas", type=int, default=20)
    parser.add_argument("--max-samples", type=int, default=1024)
    parser.add_argument("--teacher-id", type=int, default=0)
    parser.add_argument("--synthetic-data", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--t-values", default="256,512,1024,2048,4096,8192")
    parser.add_argument("--artifact-manifest", default=None)
    args = parser.parse_args()
    values = tuple(int(v) for v in args.t_values.split(",") if v.strip())
    config = RunConfig(**{**vars(args), "t_values": values})
    root = project_root()
    # Create the deterministic run location before entering the guarded
    # execution path so an early missing-artifact error is itself recorded.
    run = mathematical_root(root) / "runs" / config.run_id
    run.mkdir(parents=True, exist_ok=True)
    try:
        artifact, _ = load_artifact_manifest(root, config.artifact_manifest, config.dataset)
        if artifact.get("teacher_dir"):
            config.teacher_dir = str(resolve_artifact_path(root, artifact["teacher_dir"]))
        run = write_run(config, root)
        if config.experiment == "e3.1":
            result = run_initialization_sensitivity(config, root, run, artifact)
            result["artifact_manifest"] = artifact
            json_dump(run / "results.json", result)
            print(json.dumps({"run": str(run), "status": result["status"]}, sort_keys=True))
            return 0
        if config.experiment in {"e3.2", "e4.2"}:
            if config.experiment == "e3.2":
                result = run_not_implemented_real_variant(config, root, run, "learned_frequency")
            else:
                result = run_pixel_feature_pairs(config, root, run, artifact)
            result["artifact_manifest"] = artifact
            json_dump(run / "results.json", result)
            print(json.dumps({"run": str(run), "status": result["status"]}, sort_keys=True))
            return 0
        if config.experiment in {"e1.1", "e1.2", "e2.1"}:
            declared = artifact.get("synthetic") or config.synthetic_data
            if not declared:
                result = missing_evidence(config, run, "An explicit formal synthetic artifact is required in --synthetic-data or --artifact-manifest.")
                print(json.dumps({"run": str(run), "status": result["status"]}, sort_keys=True))
                return 0
            resolved = resolve_artifact_path(root, declared)
            if resolved is None or not resolved.is_file() or "<formal-run>" in str(resolved):
                result = missing_evidence(
                    config,
                    run,
                    f"The artifact manifest declares no verified formal synthetic file: {declared}",
                )
                print(json.dumps({"run": str(run), "status": result["status"]}, sort_keys=True))
                return 0
            config.synthetic_data = str(resolved)
        if config.experiment == "e1.1":
            result = run_frequency_replicates(config, root, run, "mc")
        elif config.experiment == "e1.2":
            result = run_mc_qmc_comparison(config, root, run)
        elif config.experiment == "e2.1":
            result = run_overfitting(config, root, run)
        elif config.experiment == "e2.2":
            result = run_importance_sampling(config, root, run)
        elif config.experiment == "e4.1":
            result = run_cf_accuracy_pairs(config, root, run, artifact)
        else:
            raise ValueError(f"Unsupported experiment: {config.experiment}")
        print(json.dumps({"run": str(run), "status": result["status"]}, sort_keys=True))
        return 0
    except Exception as exc:
        json_dump(run / "results.json", {
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "finished_at": utc_now(),
        })
        raise


if __name__ == "__main__":
    raise SystemExit(main())
