#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
from build_current_data_audit import SPECS, build, verify


class CurrentDataAuditTests(unittest.TestCase):
    def prepare(self, root: Path) -> None:
        for dataset, spec in SPECS.items():
            prepared = root / "data" / "prepared" / dataset
            prepared.mkdir(parents=True)
            (prepared / "manifest.json").write_text(
                json.dumps({"dataset": dataset}), encoding="utf-8"
            )
            payload = {
                "status": "complete",
                "contract": {"size": spec["size"], "num_classes": spec["classes"]},
                "statistics": {
                    "statistics_split": "train",
                    "duplicate_file_count": 0,
                    "split_counts": {"train": 10, "val": 2, "test": 3},
                    "mean": [0.1, 0.2, 0.3],
                    "std": [0.4, 0.5, 0.6],
                },
            }
            (prepared / "statistics.json").write_text(json.dumps(payload), encoding="utf-8")

    def test_build_and_verify_exact_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.prepare(root)
            output = root / "audit.json"
            output.write_text(json.dumps(build(root)), encoding="utf-8")
            self.assertEqual(verify(root, output)["status"], "verified")

    def test_verify_rejects_rewritten_statistics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.prepare(root)
            output = root / "audit.json"
            output.write_text(json.dumps(build(root)), encoding="utf-8")
            path = root / "data" / "prepared" / "COVID" / "statistics.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["generated_at"] = "changed"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "statistics"):
                verify(root, output)


if __name__ == "__main__":
    unittest.main()
