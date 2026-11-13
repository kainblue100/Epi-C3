"""Tests for disjoint and batch-aware pilot control-parent assignment."""

import importlib.util
from pathlib import Path
import unittest

import pandas as pd


path = Path(__file__).with_name("freeze-state-pilot-parents-v1.py")
spec = importlib.util.spec_from_file_location("freeze_pilot_parents", path)
freeze = importlib.util.module_from_spec(spec)
spec.loader.exec_module(freeze)


def fixture():
    rows = []
    reference = []
    for batch in ("b1", "b2"):
        for i in range(4):
            rows.append((f"{batch}:c{i}", "RPE1", freeze.CONTROL, batch, "train_control"))
            reference.append((f"{batch}:r{i}", "RPE1", freeze.CONTROL, batch, "reference_control"))
    for target in ("T1", "T2"):
        for i in range(3):
            rows.append((f"{target}:{i}", "RPE1", target, "b1", "test"))
    columns = ["id", "cell_type", "target_gene", "batch_var", "role"]
    return pd.DataFrame(rows, columns=columns).set_index("id"), pd.DataFrame(reference, columns=columns).set_index("id")


class ParentPlanTests(unittest.TestCase):
    def test_unique_disjoint_parents_and_explicit_fallback(self):
        data, reference = fixture()
        result = freeze.build_parent_plan(data, reference, ["T1", "T2"], 3)
        self.assertEqual(result, freeze.build_parent_plan(data, reference, ["T1", "T2"], 3))
        a = set(result["perturbation_parent_pool"])
        b = set(result["predicted_control_parent_pool"])
        c = set(result["reference_control_ids"])
        self.assertFalse(a & b or a & c or b & c)
        self.assertEqual(len(b), len(c))
        self.assertEqual(result["summary"]["cross_batch_parent_assignments"], 2)
        for target in ("T1", "T2"):
            parents = [result["assignments"][f"{target}:{i}"] for i in range(3)]
            self.assertEqual(len(set(parents)), 3)
            self.assertTrue(set(parents).issubset(a))
        self.assertTrue(all(result["assignments"][item] == item for item in b))

    def test_reference_overlap_and_missing_test_cells_are_rejected(self):
        data, reference = fixture()
        reference.index = [data.index[0], *reference.index[1:]]
        with self.assertRaisesRegex(ValueError, "overlap"):
            freeze.build_parent_plan(data, reference, ["T1", "T2"], 3)
        data, reference = fixture()
        with self.assertRaisesRegex(ValueError, "frozen cell count"):
            freeze.build_parent_plan(data, reference, ["T1", "T2"], 4)


if __name__ == "__main__":
    unittest.main()
