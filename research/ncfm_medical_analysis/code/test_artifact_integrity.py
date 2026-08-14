#!/usr/bin/env python3
from __future__ import annotations

import sys
import json
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent))
from artifact_integrity import sha256, verify_run_manifest_integrity


class ArtifactIntegrityTests(unittest.TestCase):
    def make_manifest(self, root: Path) -> dict:
        def file(relative: str, content: str = "data") -> Path:
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
            return path

        records = {}
        for key in ("prepared_manifest", "statistics", "config", "synthetic"):
            path = file(f"records/{key}.bin", key)
            records[key] = {"path": str(path), "sha256": sha256(path)}
        provenance = {}
        for key in ("command", "stdout", "stderr"):
            path = file(f"logs/{key}.log", key)
            provenance[key] = {"path": str(path), "sha256": sha256(path)}
        source = file("source.py", "print('source')\n")
        teacher_dir = root / "teachers"
        init_hashes, trained_hashes = {}, {}
        for index in range(20):
            for kind, hashes in (("init", init_hashes), ("trained", trained_hashes)):
                path = file(f"teachers/premodel{index}_{kind}.pth.tar", f"{index}:{kind}")
                hashes[path.name] = sha256(path)
        return {
            "method": "NCFM",
            **records,
            "provenance": provenance,
            "source_provenance": {"files_sha256": {"source.py": sha256(source)}},
            "pretrained_dir": {
                "path": str(teacher_dir),
                "init_sha256": init_hashes,
                "trained_sha256": trained_hashes,
            },
        }

    def test_complete_manifest_passes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = verify_run_manifest_integrity(root, self.make_manifest(root))
            self.assertEqual(len(result["pretrained_dir"]["trained_sha256"]), 20)

    def test_tampered_artifact_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self.make_manifest(root)
            Path(manifest["synthetic"]["path"]).write_text("tampered", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "synthetic hash mismatch"):
                verify_run_manifest_integrity(root, manifest)

    def test_hop_lr_selection_is_bound_without_test_accuracy(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self.make_manifest(root)
            manifest["method"] = "HoP-TM"
            manifest.pop("pretrained_dir")
            buffer_dir = root / "buffers"
            hashes = {}
            for index in range(10):
                path = buffer_dir / f"replay_buffer_{index}.pt"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(f"buffer:{index}", encoding="utf-8")
                hashes[path.name] = sha256(path)
            manifest["buffer"] = {"path": str(buffer_dir), "trajectory_files": hashes}
            stdout = root / "selection.out"
            stderr = root / "selection.err"
            stdout.write_text("finite\n", encoding="utf-8")
            stderr.write_text("", encoding="utf-8")
            selection_path = root / "selection.json"
            selection = {
                "status": "complete",
                "selection_rule": "largest_finite_short_run",
                "selected_lr_img": 10.0,
                "uses_validation_or_test_accuracy": False,
                "attempts": [{
                    "lr_img": 10.0,
                    "status": "finite_complete",
                    "stdout": {"path": str(stdout), "sha256": sha256(stdout)},
                    "stderr": {"path": str(stderr), "sha256": sha256(stderr)},
                }],
            }
            selection_path.write_text(json.dumps(selection), encoding="utf-8")
            manifest["lr_selection"] = {
                "path": str(selection_path), "sha256": sha256(selection_path)
            }
            manifest["method_contract"] = {
                "lr_img": 10.0, "lr_selection": "largest_finite_short_run"
            }
            result = verify_run_manifest_integrity(root, manifest)
            self.assertEqual(result["lr_selection"]["attempts"][0]["status"], "finite_complete")


if __name__ == "__main__":
    unittest.main()
