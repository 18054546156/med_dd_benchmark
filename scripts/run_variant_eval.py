#!/usr/bin/env python3
"""Run one explicit Phase 2 variant controlled evaluation entry."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    root = args.root.resolve()
    manifest = args.manifest if args.manifest.is_absolute() else root / args.manifest
    entries = json.loads(manifest.read_text(encoding="utf-8"))["evaluations"]
    entry = entries[args.index]
    for key in ("variant", "synthetic", "path"):
        if not entry.get(key) or "<" in str(entry[key]) or ">" in str(entry[key]):
            raise ValueError(f"manifest entry {args.index} has no explicit {key}")
    synthetic = Path(entry["synthetic"])
    output = Path(entry["path"])
    if not synthetic.is_absolute():
        synthetic = root / synthetic
    if not output.is_absolute():
        output = root / output
    if not synthetic.is_file():
        raise FileNotFoundError(synthetic)
    output.parent.mkdir(parents=True, exist_ok=True)
    evaluator = root / "research" / "ncfm_medical_analysis" / "code" / "unified_eval_real.py"
    command = [sys.executable, str(evaluator), "--data", str(synthetic),
               "--dataset", entry["dataset"], "--architecture", entry["architecture"],
               "--method", "NCFM", "--variant", entry["variant"],
               "--run-id", entry.get("run_id", f"{entry['variant']}_{entry['dataset']}_{entry['architecture']}"),
               "--repeats", str(args.repeats), "--device", "cuda", "--output", str(output)]
    print("command=" + " ".join(command), flush=True)
    return subprocess.run(command, cwd=root, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
