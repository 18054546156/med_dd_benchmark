#!/usr/bin/env python3
"""Pair real NCFM CF measurements with controlled downstream evaluations.

Input is an explicit JSON manifest.  No filesystem ordering or newest-file
heuristic is used.  The script refuses to make a formal correlation claim
unless at least five synthetic/evaluation pairs are present.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from real_phase1 import RunConfig, exact_ncfm_cf, frequency_bank, load_features, load_synthetic_features, percentile_ci, project_root


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def rank(values):
    order = np.argsort(np.argsort(np.asarray(values, dtype=np.float64)))
    return order.astype(np.float64)


def pearson(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["PathMNIST", "COVID", "Kvasir"], required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--teacher-id", type=int, default=0)
    parser.add_argument("--replicas", type=int, default=20)
    parser.add_argument("--num-freqs", type=int, default=4096)
    args = parser.parse_args()
    root = project_root()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    pairs = manifest.get("pairs", [])
    real, _, real_meta = load_features(RunConfig(
        dataset=args.dataset, experiment="e4.1", run_id="pair", teacher_id=args.teacher_id,
        device="cuda" if torch.cuda.is_available() else "cpu"), root, "train")
    if len(pairs) < 5:
        result = {
            "status": "insufficient_evidence",
            "reason": "Formal CF/accuracy pairing requires at least five explicit pairs.",
            "pair_count": len(pairs),
            "manifest": str(args.manifest),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result))
        return 0

    records = []
    for item in pairs:
        synthetic = Path(item["synthetic"])
        evaluation = Path(item["evaluation"])
        if not synthetic.is_absolute():
            synthetic = (root / synthetic).resolve()
        if not evaluation.is_absolute():
            evaluation = (root / evaluation).resolve()
        if not synthetic.is_file() or not evaluation.is_file():
            raise FileNotFoundError(f"Missing pair input: {synthetic} or {evaluation}")
        eval_payload = json.loads(evaluation.read_text(encoding="utf-8"))
        declared = Path(eval_payload.get("synthetic_path", ""))
        if declared.name != synthetic.name and str(declared) != str(synthetic):
            raise ValueError(f"Evaluation synthetic_path does not match manifest: {evaluation}")
        config = RunConfig(dataset=args.dataset, experiment="e4.1", run_id=item.get("run_id", synthetic.stem),
                           teacher_id=args.teacher_id, synthetic_data=str(synthetic),
                           replicas=args.replicas, num_freqs=args.num_freqs,
                           device="cuda" if torch.cuda.is_available() else "cpu")
        syn, _, syn_meta = load_synthetic_features(config, root)
        # The empirical CFs are estimated independently from the full real and
        # synthetic sets. Their sample counts are expected to differ for IPC
        # experiments (for example, 10 images per class versus the real train
        # split), so never truncate or require equal lengths here.
        losses = []
        for replica in range(args.replicas):
            freqs = frequency_bank(args.num_freqs, real.shape[1], replica)
            losses.append(exact_ncfm_cf(real, syn, freqs))
        acc = eval_payload.get("test_accuracy", {}).get("mean")
        if acc is None:
            raise ValueError(f"Evaluation JSON has no test_accuracy.mean: {evaluation}")
        records.append({
            "run_id": item.get("run_id", synthetic.stem),
            "synthetic": str(synthetic),
            "synthetic_sha256": sha256(synthetic),
            "evaluation": str(evaluation),
            "evaluation_sha256": sha256(evaluation),
            "cf": percentile_ci(losses, 0),
            "test_accuracy": float(acc),
            "synthetic_meta": syn_meta,
            "real_feature_count": len(real),
            "synthetic_feature_count": len(syn),
        })

    cf = [item["cf"]["mean"] for item in records]
    acc = [item["test_accuracy"] for item in records]
    result = {
        "status": "complete",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "experiment": "E4.1",
        "dataset": args.dataset,
        "manifest": str(args.manifest),
        "manifest_sha256": sha256(args.manifest),
        "real_meta": real_meta,
        "pair_count": len(records),
        "records": records,
        "association": {
            "pearson_cf_vs_accuracy": pearson(cf, acc),
            "spearman_cf_vs_accuracy": pearson(rank(cf), rank(acc)),
            "lower_cf_is_better": True,
            "higher_accuracy_is_better": True,
        },
        "evidence_note": "Association is not causal evidence; report with the explicit pair manifest and controlled evaluator protocol.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"status": result["status"], "pairs": len(records), "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
