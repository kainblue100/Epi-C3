import importlib.util
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location("neighbor_audit", Path(__file__).parent.parent / "audit-grn-neighbors-v1.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def test_neighbor_direction_uses_separate_reference_and_input_controls():
    result = audit.score_neighbors(
        [1.3, 1.0, 1.2], [1.2, 0.8, 1.05], [1.0, 1.1, 1.0], [1.1, 1.0, 1.0], [0, 1, 2],
    )
    assert result["neighbor_count"] == 3
    assert result["neighbor_mse"] == pytest.approx((0.1**2 + 0.2**2 + 0.15**2) / 3)
    assert result["eligible_effect_count"] == 2
    assert result["nonzero_predicted_effect_count"] == 1
    assert result["correct_direction_count"] == 1
    assert result["direction_coverage"] == 0.5
    assert result["direction_accuracy_all_eligible"] == 0.5
    assert result["direction_accuracy_given_nonzero"] == 1.0


def test_no_eligible_or_nonzero_effects_remain_undefined():
    result = audit.score_neighbors([1.0], [1.01], [1.0], [1.0], [0])
    assert result["eligible_effect_count"] == 0
    assert result["direction_coverage"] is None
    assert result["direction_accuracy_given_nonzero"] is None
    with pytest.raises(ValueError, match="distinct"):
        audit.score_neighbors([1.0], [1.0], [1.0], [1.0], [0, 0])


def test_neighbor_index_respects_direction_and_rejects_duplicate_edges(tmp_path):
    edges = tmp_path / "edges.csv"
    edges.write_text("source,target\nA,B\nC,A\n", encoding="utf-8")
    assert audit.neighbor_indices(edges, ["A", "B", "C"], ["A"]) == {"A": [1]}
    edges.write_text("source,target\nA,B\nA,B\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicated"):
        audit.neighbor_indices(edges, ["A", "B", "C"], ["A"])
