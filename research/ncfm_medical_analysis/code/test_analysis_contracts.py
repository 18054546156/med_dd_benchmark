#!/usr/bin/env python3
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "scripts"))

from real_phase1 import classwise_d_omega, frequency_bank, stratified_indices
from prepare_medical_data import deduplicate_class_images, rgb_pixel_sha256


class AnalysisContractTests(unittest.TestCase):
    def test_stratified_indices_are_balanced_and_deterministic(self):
        labels = np.repeat(np.arange(3), 10)
        first, counts = stratified_indices(labels, max_samples=8, seed=17, expected_classes=3)
        second, second_counts = stratified_indices(labels, max_samples=8, seed=17, expected_classes=3)
        self.assertEqual(first, second)
        self.assertEqual(counts, second_counts)
        self.assertEqual(sum(counts.values()), 8)
        self.assertLessEqual(max(counts.values()) - min(counts.values()), 1)

    def test_classwise_metric_requires_identical_class_support(self):
        real = torch.tensor([[0.0], [0.1], [3.0], [3.1]])
        real_labels = torch.tensor([0, 0, 1, 1])
        synthetic = torch.tensor([[0.0], [3.0]])
        synthetic_labels = torch.tensor([0, 1])
        bank = frequency_bank(32, 1, seed=4)
        aggregate, per_class = classwise_d_omega(
            real, real_labels, synthetic, synthetic_labels, bank
        )
        self.assertEqual(set(per_class), {"0", "1"})
        self.assertAlmostEqual(aggregate, np.mean(list(per_class.values())), places=12)
        with self.assertRaises(ValueError):
            classwise_d_omega(
                real,
                real_labels,
                synthetic[:1],
                synthetic_labels[:1],
                bank,
            )

    def test_rgb_hash_dedup_removes_cross_class_ambiguity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            red = np.zeros((4, 5, 3), dtype=np.uint8)
            red[..., 0] = 255
            blue = np.zeros((4, 5, 3), dtype=np.uint8)
            blue[..., 2] = 255
            green = np.zeros((4, 5, 3), dtype=np.uint8)
            green[..., 1] = 255
            paths = {
                "a_png": root / "a.png",
                "a_bmp": root / "a.bmp",
                "b_png": root / "b.png",
                "blue": root / "blue.png",
                "green": root / "green.png",
            }
            Image.fromarray(red).save(paths["a_png"])
            Image.fromarray(red).save(paths["a_bmp"])
            Image.fromarray(red).save(paths["b_png"])
            Image.fromarray(blue).save(paths["blue"])
            Image.fromarray(green).save(paths["green"])
            self.assertEqual(rgb_pixel_sha256(paths["a_png"]), rgb_pixel_sha256(paths["a_bmp"]))

            retained, audit = deduplicate_class_images(
                {
                    "class_a": [paths["a_png"], paths["a_bmp"], paths["blue"]],
                    "class_b": [paths["b_png"], paths["green"]],
                },
                root,
            )
            self.assertEqual(retained["class_a"], [paths["blue"]])
            self.assertEqual(retained["class_b"], [paths["green"]])
            self.assertEqual(audit["ambiguous_group_count"], 1)
            self.assertEqual(audit["ambiguous_file_count"], 3)


if __name__ == "__main__":
    unittest.main()
