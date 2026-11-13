"""Focused tests for the two-context STATE pilot data preparation."""

import importlib.util
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import h5py
import numpy as np


path = Path(__file__).with_name("prepare-state-replogle-pilot-v1.py")
spec = importlib.util.spec_from_file_location("pilot", path)
pilot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)


class PilotTests(unittest.TestCase):
    def test_feature_join_requires_matching_symbol_and_ensembl(self):
        a = {"symbols": ["A", "B", "C"], "ids": ["1", "2", "3"]}
        b = {"symbols": ["A", "B", "C"], "ids": ["1", "9", "3"]}
        with patch.object(pilot, "HVG_COUNT", 2):
            self.assertEqual(pilot.shared_axis(a, b), [("A", "1"), ("C", "3")])
        with patch.object(pilot, "HVG_COUNT", 3):
            with self.assertRaises(ValueError):
                pilot.shared_axis(a, b)

    def test_deterministic_context_holdout_excludes_ineligible_genes(self):
        a = {"counts": {f"G{i}": 101 for i in range(50)}}
        b = {"counts": dict(a["counts"])}
        b["counts"]["G0"] = 99
        axis = set(a["counts"])
        first = pilot.select_targets(a, b, axis)
        self.assertEqual(first, pilot.select_targets(a, b, axis))
        sets = [set(first[role]) for role in ("train", "val", "test")]
        self.assertEqual(list(map(len, sets)), [10, 2, 28])
        self.assertFalse(sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
        self.assertNotIn("G0", set.union(*sets))

    def test_sampled_reference_controls_are_disjoint_from_training_controls(self):
        labels = [pilot.CONTROL] + [f"G{i:02}" for i in range(40)]
        codes = np.repeat(np.arange(len(labels)), [40] + [100] * 40)
        target_mask = codes != 0
        meta = {
            "labels": labels, "codes": codes, "batches": np.zeros(len(codes), dtype=int),
            "totals": np.full(len(codes), 100.0), "target_mask": target_mask,
            "control_mask": ~target_mask,
        }
        targets = {"train": labels[1:11], "val": labels[11:13], "test": labels[13:]}
        records = pilot.sample_records(meta, "RPE1", targets, random.Random(1))
        rows = [r["row"] for r in records]
        self.assertEqual(len(rows), len(set(rows)))
        control = [r for r in records if r["target"] == pilot.CONTROL]
        self.assertEqual(sum(r["role"] == "train_control" for r in control), 10)
        self.assertEqual(sum(r["role"] == "reference_control" for r in control), 10)
        self.assertTrue(all(r["role"] == "test" for r in records if r["target"] in targets["test"]))

    def test_chunked_read_preserves_gene_order_and_rejects_bad_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "mini.h5ad"
            with h5py.File(p, "w") as f:
                f.create_dataset("X", data=np.array([[1, 0, 3], [0, 2, 0], [4, 1, 1]], dtype="f4"), chunks=(2, 2))
                v = f.create_group("var")
                v.create_dataset("gene_name", data=np.array(["A", "B", "C"], dtype=h5py.string_dtype()))
                v.create_dataset("ensembl_id", data=np.array(["1", "2", "3"], dtype=h5py.string_dtype()))
            records = [{"row": 0, "library_size": 4.0}, {"row": 2, "library_size": 6.0}]
            actual = pilot.read_selected_counts(p, records, [["C", "3"], ["A", "1"]])
            expected = np.log1p(np.array([[3 / 4, 1 / 4], [1 / 6, 4 / 6]]) * 10000)
            np.testing.assert_allclose(actual, expected, rtol=1e-6)
            with self.assertRaises(ValueError):
                pilot.read_selected_counts(p, [records[0], records[0]], [["A", "1"]])
            with h5py.File(p, "r+") as f:
                f["X"][0, 0] = -1
            with self.assertRaises(ValueError):
                pilot.read_selected_counts(p, records, [["A", "1"]])


if __name__ == "__main__":
    unittest.main()
