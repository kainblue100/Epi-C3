"""Focused arithmetic and input-validation tests for the legacy comparison."""

import unittest

import numpy as np

from compare_legacy_k562_means import (
    summarize,
    target_excluded_discrimination,
    target_excluded_mse,
    target_excluded_pearson,
    target_indices,
)


class MeanComparisonTests(unittest.TestCase):
    def test_target_gene_is_excluded_from_all_metrics(self):
        excluded = np.array([0, 1])
        prediction = np.array([[100.0, 1.0, 3.0], [1.0, 500.0, 3.0]])
        observed = np.array([[-100.0, 1.0, 3.0], [1.0, -500.0, 3.0]])
        np.testing.assert_allclose(target_excluded_mse(prediction, observed, excluded), [0, 0])
        np.testing.assert_allclose(target_excluded_pearson(prediction, observed, excluded), [1, 1])
        np.testing.assert_allclose(target_excluded_discrimination(prediction, observed, excluded), [1, 1])

    def test_constant_effect_is_undefined_not_zero(self):
        predicted = np.zeros((2, 4))
        observed = np.array([[1.0, 2.0, 3.0, 4.0], [4.0, 2.0, 1.0, 3.0]])
        result = target_excluded_pearson(predicted, observed, np.array([0, 1]))
        self.assertTrue(np.isnan(result).all())
        summary = summarize(result[None])
        self.assertIsNone(summary["mean_of_run_macros"])
        self.assertEqual(summary["finite_targets_by_run"], [0])

    def test_pearson_direction(self):
        truth = np.array([[0.0, 2.0, 1.0, 4.0]])
        np.testing.assert_allclose(target_excluded_pearson(truth, truth, np.array([0])), [1])
        np.testing.assert_allclose(target_excluded_pearson(-truth, truth, np.array([0])), [-1])

    def test_discrimination_matches_brute_force_and_averages_ties(self):
        observed = np.array([
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 3.0, 2.0, 0.0],
        ])
        predicted = np.array([
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 3.0, 2.0, 0.0],
            [1.0, 3.0, 2.0, 0.0],
        ])
        excluded = np.array([0, 1, 2])
        expected = []
        for index, target in enumerate(excluded):
            distances = np.sum(np.delete(observed - predicted[index], target, axis=1) ** 2, axis=1)
            correct = distances[index]
            ties = np.isclose(distances, correct, rtol=1e-12, atol=1e-12)
            rank = np.count_nonzero((distances < correct) & ~ties) + (np.count_nonzero(ties) - 1) / 2
            expected.append(1 - rank / len(excluded))
        np.testing.assert_allclose(target_excluded_discrimination(predicted, observed, excluded), expected)
        self.assertEqual(expected[0], 5 / 6)

    def test_invalid_or_mismatched_axes_fail(self):
        with self.assertRaises(ValueError):
            target_indices(["A", "A", "B"], ["A"])
        with self.assertRaises(ValueError):
            target_indices(["A", "B", "C"], ["missing"])
        with self.assertRaises(ValueError):
            target_excluded_mse(np.ones((1, 3)), np.ones((2, 3)), np.array([0, 1]))
        with self.assertRaises(ValueError):
            target_excluded_discrimination(np.ones((1, 3)), np.ones((1, 3)), np.array([0]))


if __name__ == "__main__":
    unittest.main()
