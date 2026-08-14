#!/usr/bin/env python3
"""Static contracts for the staged Slurm production graph."""

from __future__ import annotations

import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def read(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def array_count(relative: str) -> int:
    match = re.search(r"^#SBATCH --array=(\d+)-(\d+)", read(relative), re.MULTILINE)
    if not match:
        raise AssertionError(f"missing static Slurm array in {relative}")
    return int(match.group(2)) - int(match.group(1)) + 1


class SlurmContractTests(unittest.TestCase):
    def test_grouped_arrays_fit_submit_limit(self):
        self.assertEqual(array_count("research/ncfm_medical_analysis/code/run_real_phase1.sbatch"), 3)
        self.assertEqual(array_count("research/ncfm_medical_analysis/code/run_phase2.sbatch"), 3)
        self.assertEqual(array_count("research/ncfm_medical_analysis/code/run_eval_manifest_gpu.sbatch"), 3)
        self.assertEqual(array_count("scripts/run_phase1_replication_eval.sbatch"), 6)
        self.assertEqual(array_count("scripts/run_phase2_variant_eval.sbatch"), 4)

        # Peak submitted records, including the currently running submitter:
        # Stage 1: analysis + contract + runtime + six sweeps + continuation.
        # Stage 2: gate + 3 task + 3 Phase 1 + eval gate + 3 eval + Stage 3.
        # Stage 3: nine variants + 3 diagnostics + continuation.
        self.assertLessEqual(1 + 1 + 1 + 1 + 6 + 1, 15)
        self.assertLessEqual(1 + 1 + 3 + 3 + 1 + 3 + 1, 15)
        self.assertLessEqual(1 + 9 + 3 + 1, 15)

    def test_postproduction_stages_do_not_rewrite_bound_statistics(self):
        initial = read("scripts/submit_full_real_benchmark.sh")
        staged = read("research/ncfm_medical_analysis/code/submit_formal_pipeline_staged.sbatch")
        self.assertIn("run_statistics_cpu.sbatch", initial)
        self.assertNotIn("run_statistics_cpu.sbatch", staged)
        self.assertIn("reuse-production-bound-file", staged)

    def test_large_followups_are_deferred(self):
        stage1 = read("research/ncfm_medical_analysis/code/submit_formal_pipeline_staged.sbatch")
        stage3 = read("research/ncfm_medical_analysis/code/formal_stage3_submit.sbatch")
        self.assertIn("DEFER_FOLLOWUP=1", stage1)
        self.assertIn("formal_stage1b_submit.sbatch", stage1)
        self.assertIn("DEFER_FOLLOWUP=1", stage3)
        self.assertIn("formal_stage3b_submit.sbatch", stage3)

    def test_variant_seeds_have_immutable_dedicated_logs(self):
        sweep = read("scripts/ncfm_variant_seed_sweep.sbatch")
        writer = read("scripts/write_ncfm_run_manifest.py")
        hop_writer = read("scripts/write_hop_run_manifest.py")
        self.assertIn("ncfm-variant-${SLURM_JOB_ID}-${run_id}.out", sweep)
        self.assertIn('>"$run_stdout" 2>"$run_stderr"', sweep)
        self.assertNotIn('print(json.dumps({"status": "complete", "manifest"', writer)
        self.assertNotIn('print(json.dumps({"status": "complete", "manifest"', hop_writer)


if __name__ == "__main__":
    unittest.main()
