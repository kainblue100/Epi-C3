from __future__ import annotations

import unittest

import numpy as np

import linear_baseline_equations_v1 as baseline


class EquationTests(unittest.TestCase):
    def test_paper_formula_and_stated_objective_are_distinct(self):
        changes = np.array([[7.0, 3.0], [1.0, 4.0]])
        genes = np.array([[2.0], [1.0]])
        perts = np.array([[3.0], [2.0]])
        ridge = 0.1
        formula, formula_bias = baseline.solve_weights(
            changes, genes, perts, ridge, baseline.PAPER_FORMULA
        )
        objective, objective_bias = baseline.solve_weights(
            changes, genes, perts, ridge, baseline.STATED_OBJECTIVE
        )
        expected_formula = (
            genes.T @ changes @ perts
            / ((genes.T @ genes + ridge) * (perts.T @ perts + ridge))
        )
        centered = changes - changes.mean(axis=1, keepdims=True)
        expected_objective = (
            genes.T @ centered @ perts
            / ((genes.T @ genes) * (perts.T @ perts) + ridge)
        )
        np.testing.assert_allclose(formula, expected_formula)
        np.testing.assert_allclose(objective, expected_objective)
        np.testing.assert_allclose(formula_bias, [5.0, 2.5])
        np.testing.assert_allclose(objective_bias, formula_bias)
        self.assertFalse(np.allclose(formula, objective))

    def test_objective_solution_matches_independent_kronecker_solve(self):
        rng = np.random.default_rng(42)
        genes = rng.normal(size=(4, 3))
        perts = rng.normal(size=(5, 2))
        changes = rng.normal(size=(4, 5))
        ridge = 0.27
        weights, bias = baseline.solve_weights(
            changes, genes, perts, ridge, baseline.STATED_OBJECTIVE
        )
        centered = changes - bias[:, None]
        rhs = genes.T @ centered @ perts
        system = np.kron(perts.T @ perts, genes.T @ genes)
        system += ridge * np.eye(system.shape[0])
        expected = np.linalg.solve(system, rhs.reshape(-1, order="F"))
        np.testing.assert_allclose(
            weights, expected.reshape(weights.shape, order="F"), rtol=1e-11, atol=1e-11
        )

    def test_regularization_differs_even_when_intercept_is_zero(self):
        changes = [[2.0, -2.0]]
        genes = [[1.0]]
        perts = [[1.0], [0.25]]
        formula, formula_bias = baseline.solve_weights(
            changes, genes, perts, 0.7, baseline.PAPER_FORMULA
        )
        objective, objective_bias = baseline.solve_weights(
            changes, genes, perts, 0.7, baseline.STATED_OBJECTIVE
        )
        np.testing.assert_array_equal(formula_bias, [0.0])
        np.testing.assert_array_equal(objective_bias, [0.0])
        self.assertFalse(np.allclose(formula, objective))

    def test_objective_handles_rank_deficient_embeddings(self):
        genes = np.array([[1.0, 1.0], [2.0, 2.0], [0.0, 0.0]])
        perts = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
        changes = np.arange(9.0).reshape(3, 3)
        ridge = 0.01
        weights, bias = baseline.solve_weights(
            changes, genes, perts, ridge, baseline.STATED_OBJECTIVE
        )
        centered = changes - bias[:, None]
        system = np.kron(perts.T @ perts, genes.T @ genes)
        system += ridge * np.eye(system.shape[0])
        expected = np.linalg.solve(
            system, (genes.T @ centered @ perts).reshape(-1, order="F")
        )
        np.testing.assert_allclose(
            weights, expected.reshape(weights.shape, order="F"), rtol=1e-9, atol=1e-9
        )

    def test_invalid_regularization_and_dimensions_are_rejected(self):
        for ridge in (0, -1, np.nan, np.inf):
            with self.subTest(ridge=ridge), self.assertRaises(ValueError):
                baseline.solve_weights([[1]], [[1]], [[1]], ridge, baseline.PAPER_FORMULA)
        with self.assertRaisesRegex(ValueError, "axes"):
            baseline.solve_weights([[1, 2]], [[1]], [[1]], 0.1, baseline.PAPER_FORMULA)
        with self.assertRaisesRegex(ValueError, "Unknown"):
            baseline.solve_weights([[1]], [[1]], [[1]], 0.1, "other")

    def test_complex_object_string_and_masked_matrices_are_rejected(self):
        invalid = (
            np.array([[1.0 + 2.0j, 2.0]]),
            np.array([[1.0, 2.0]], dtype=object),
            np.array([["1", "2"]]),
            np.ma.array([[1.0, 2.0]], mask=[[False, True]]),
        )
        for changes in invalid:
            with self.subTest(dtype=changes.dtype), self.assertRaises(ValueError):
                baseline.solve_weights(changes, [[1.0]], [[1.0], [2.0]], 0.1, baseline.PAPER_FORMULA)
        unmasked = np.ma.array([[1.0, 2.0]], mask=False)
        baseline.solve_weights(unmasked, [[1.0]], [[1.0], [2.0]], 0.1, baseline.PAPER_FORMULA)

    def test_masked_arrays_nested_in_sequences_are_rejected(self):
        changes = [np.ma.array([1.0, 2.0], mask=[False, True])]
        with self.assertRaisesRegex(ValueError, "masked"):
            baseline.solve_weights(changes, [[1.0]], [[1.0], [2.0]], 0.1, baseline.PAPER_FORMULA)


class FittingTests(unittest.TestCase):
    def setUp(self):
        self.genes = ["ENSG1", "ENSG2"]
        self.labels = ["A", "A", "B", "B", "control"]
        self.controls = np.ones((5, 2))
        self.treated = self.controls + np.array(
            [[2, 0], [4, 0], [0, 2], [0, 4], [100, 100]], dtype=float
        )
        self.arguments = dict(
            perturbed=self.treated,
            matched_controls=self.controls,
            gene_ids=self.genes,
            labels=self.labels,
            gene_embedding_ids=["ENSG2", "ENSG1"],
            gene_embeddings=np.array([[0, 1], [1, 0]]),
            perturbation_embedding_ids=["B", "A"],
            perturbation_embeddings=np.array([[0, 1], [1, 0]]),
            control_label="control",
            ridge=0.1,
            variant=baseline.STATED_OBJECTIVE,
        )

    def test_matched_means_exclude_control_and_preserve_control_predictions(self):
        model = baseline.fit(**self.arguments)
        self.assertEqual(model.perturbations, ("A", "B"))
        self.assertEqual(model.training_counts, (2, 2))
        control = np.array([[1.0, 2.0], [2.0, 1.0], [5.0, 6.0]])
        actual = model.predict(control, self.genes, ["A", "B", "control"])
        change = np.array([[3.0, 0.0], [0.0, 3.0]])
        bias = change.mean(axis=1)
        expected_offsets = ((change - bias[:, None]) / 1.1).T + bias
        np.testing.assert_allclose(actual[:2], control[:2] + expected_offsets)
        np.testing.assert_array_equal(actual[2], control[2])
        self.assertFalse(model.offsets.flags.writeable)

    def test_exact_axis_is_required_and_no_alias_is_inferred(self):
        model = baseline.fit(**self.arguments)
        with self.assertRaisesRegex(ValueError, "differs"):
            model.predict([[1, 1]], ["ENSG2", "ENSG1"], ["A"])
        bad = dict(self.arguments, gene_ids=["HUMAN_SYMBOL:GENE", "ENSG2"])
        with self.assertRaisesRegex(ValueError, "missing"):
            baseline.fit(**bad)

    def test_missing_duplicate_and_unseen_perturbations_are_rejected(self):
        bad = dict(self.arguments, perturbation_embedding_ids=["A", "A"])
        with self.assertRaisesRegex(ValueError, "unique"):
            baseline.fit(**bad)
        bad = dict(self.arguments, perturbation_embedding_ids=["A", "C"])
        with self.assertRaisesRegex(ValueError, "missing"):
            baseline.fit(**bad)
        model = baseline.fit(**self.arguments)
        with self.assertRaisesRegex(ValueError, "Unseen"):
            model.predict([[1, 1]], self.genes, ["C"])

    def test_nonfinite_negative_and_misaligned_training_inputs_fail(self):
        for value in (np.nan, np.inf, -1.0):
            with self.subTest(value=value):
                treated = self.treated.copy()
                treated[0, 0] = value
                with self.assertRaises(ValueError):
                    baseline.fit(**dict(self.arguments, perturbed=treated))
        with self.assertRaisesRegex(ValueError, "do not match"):
            baseline.fit(**dict(self.arguments, matched_controls=self.controls[:-1]))
        with self.assertRaisesRegex(ValueError, "perturbation"):
            baseline.fit(**dict(self.arguments, labels=["control"] * 5))

    def test_scalar_strings_cannot_be_interpreted_as_identifier_sequences(self):
        invalid = (
            dict(self.arguments, gene_ids="AB", gene_embedding_ids=["A", "B"]),
            dict(self.arguments, labels="AABBB"),
            dict(self.arguments, perturbation_embedding_ids="BA"),
        )
        for arguments in invalid:
            with self.subTest(arguments=arguments), self.assertRaisesRegex(ValueError, "sequence"):
                baseline.fit(**arguments)
        model = baseline.fit(**self.arguments)
        with self.assertRaisesRegex(ValueError, "sequence"):
            model.predict([[1, 1], [1, 1]], self.genes, "AB")

    def test_complex_and_masked_inputs_cannot_reach_fitting_or_prediction(self):
        complex_embeddings = self.arguments["gene_embeddings"].astype(complex)
        complex_embeddings[0, 0] += 1j
        with self.assertRaisesRegex(ValueError, "real numeric"):
            baseline.fit(**dict(self.arguments, gene_embeddings=complex_embeddings))
        model = baseline.fit(**self.arguments)
        masked = np.ma.array([[1.0, 1.0]], mask=[[False, True]])
        with self.assertRaisesRegex(ValueError, "masked"):
            model.predict(masked, self.genes, ["A"])


if __name__ == "__main__":
    unittest.main()
