"""Tests for undefined-score handling and paired-condition aggregation."""

import importlib.util
from pathlib import Path
import unittest


path = Path(__file__).with_name("score-state-replogle-pilot-v1.py")
spec = importlib.util.spec_from_file_location("score_pilot", path)
score = importlib.util.module_from_spec(spec)
spec.loader.exec_module(score)


def row(target, real_count, pred_count, pearson=0.5):
    return {
        "perturbation": target,
        "de_nsig_counts_real": real_count,
        "de_nsig_counts_pred": pred_count,
        "discrimination_score_l1": 0.5,
        "pearson_delta": pearson,
        "pr_auc": 0.2,
        "de_spearman_lfc_sig": 0.3,
        "overlap_at_N": 0.4,
    }


class ScoreTests(unittest.TestCase):
    def test_zero_de_is_undefined_and_models_use_common_support(self):
        rows = {
            "a": [row("X", 0, 0), row("Y", 3, 1)],
            "b": [row("X", 0, 0), row("Y", 3, 2, pearson=float("nan"))],
        }
        clean, aggregate = score.summarize(rows, ["X", "Y"], 10)
        self.assertIsNone(clean["a"]["X"]["overlap_at_N"])
        self.assertIsNone(clean["b"]["X"]["pr_auc"])
        self.assertEqual(aggregate["a"]["overlap_at_N"]["n_conditions"], 1)
        self.assertEqual(aggregate["a"]["pearson_delta"]["n_conditions"], 1)
        self.assertEqual(aggregate["b"]["pearson_delta"]["n_conditions"], 1)
        self.assertAlmostEqual(aggregate["a"]["de_count_spearman_all_conditions"]["value"], 1.0)

    def test_constant_de_count_is_undefined_and_real_mismatch_fails(self):
        self.assertIsNone(score.all_condition_spearman([0, 0], [1, 2]))
        self.assertIsNone(score.all_condition_spearman([0, 1], [1, 1]))
        rows = {"a": [row("X", 0, 0)], "b": [row("X", 1, 0)]}
        with self.assertRaisesRegex(ValueError, "Real DE counts differ"):
            score.summarize(rows, ["X"], 10)


if __name__ == "__main__":
    unittest.main()
