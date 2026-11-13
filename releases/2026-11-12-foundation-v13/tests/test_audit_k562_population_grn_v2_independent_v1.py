from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


path = Path(__file__).parent.parent / "audit-k562-population-grn-v2-independent-v1.py"
spec = importlib.util.spec_from_file_location("independent_population_audit", path)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def test_target_gene_is_excluded_per_row():
    genes = ["A", "B", "C"]
    targets = ["C", "A"]
    excluded = audit.axes(genes, targets)
    prediction = np.array([[1.0, 2.0, 1000.0], [1000.0, 3.0, 4.0]])
    actual = audit.target_mse(prediction, np.zeros((2, 3)), excluded)
    np.testing.assert_allclose(actual, [2.5, 12.5])


@pytest.mark.parametrize(
    "genes,targets",
    [(["A", "A"], ["A"]), (["A", "B"], ["C"]), (["A", "B"], ["A", "A"])],
)
def test_invalid_axes_fail(genes, targets):
    with pytest.raises(ValueError):
        audit.axes(genes, targets)


def test_neighbor_direction_uses_separate_input_and_reference_controls():
    observed = np.array([3.0, 2.8, 1.02, 0.0])
    reference = np.array([2.0, 3.0, 1.0, 0.0])
    input_control = np.array([1.0, 1.0, 0.0, 0.0])
    prediction = np.array([1.4, 1.2, 0.0, 0.0])
    result = audit.graph_metrics(prediction, observed, reference, input_control, np.array([0, 1, 2]))
    assert result["neighbor_count"] == 3
    assert result["eligible_effect_count"] == 2
    assert result["nonzero_predicted_effect_count"] == 2
    assert result["correct_direction_count"] == 1
    assert result["direction_coverage"] == 1
    assert result["direction_accuracy_all_eligible"] == 0.5


def test_no_eligible_neighbor_has_undefined_direction():
    zeros = np.zeros(3)
    result = audit.graph_metrics(zeros, zeros, zeros, zeros, np.array([0, 1]))
    assert result["direction_coverage"] is None
    assert result["direction_accuracy_given_nonzero"] is None


def test_nonfinite_or_misaligned_values_fail():
    with pytest.raises(ValueError):
        audit.target_mse(np.array([[np.nan, 0.0]]), np.zeros((1, 2)), np.array([0]))
    with pytest.raises(ValueError):
        audit.graph_metrics(np.ones(2), np.ones(2), np.ones(2), np.ones(2), np.array([0, 0]))


def test_score_disagreement_fails():
    with pytest.raises(ValueError, match="differs"):
        audit.check_close(0.1, 0.2, "score")
