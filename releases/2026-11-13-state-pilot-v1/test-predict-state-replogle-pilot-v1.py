"""Tests for enforcing the frozen STATE-pilot parent assignments."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np


path = Path(__file__).with_name("predict-state-replogle-pilot-v1.py")
spec = importlib.util.spec_from_file_location("predict_pilot", path)
predict = importlib.util.module_from_spec(spec)
spec.loader.exec_module(predict)


def make_module():
    strategy = SimpleNamespace(
        use_consecutive_loading=False,
        cache_perturbation_control_pairs=False,
        split_control_mapping={"test": {}},
    )
    dataset = SimpleNamespace(
        cell_barcodes=np.array(["p", "a", "b", "unused"]),
        all_indices=np.arange(4),
        mapping_strategy=strategy,
        _find_split_for_idx=lambda index: "test",
    )
    subset = SimpleNamespace(dataset=dataset, indices=[0, 2, 3])
    return SimpleNamespace(test_datasets=[subset]), strategy


class PredictionMappingTests(unittest.TestCase):
    def test_only_frozen_rows_and_assigned_parents_are_used(self):
        module, strategy = make_module()
        predict.install_parent_mapping(module, {"assignments": {"p": "a", "b": "b"}})
        self.assertEqual(module.test_datasets[0].indices, [0, 2])
        self.assertEqual(strategy.split_control_mapping["test"], {0: [1], 2: [2]})
        self.assertTrue(strategy.cache_perturbation_control_pairs)

    def test_missing_parent_or_missing_test_row_fails(self):
        module, _ = make_module()
        with self.assertRaisesRegex(ValueError, "parent is absent"):
            predict.install_parent_mapping(module, {"assignments": {"p": "missing"}})
        module, _ = make_module()
        with self.assertRaisesRegex(ValueError, "differ"):
            predict.install_parent_mapping(module, {"assignments": {"p": "a", "not-tested": "a"}})


if __name__ == "__main__":
    unittest.main()
