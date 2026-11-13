import importlib.util
from pathlib import Path

import numpy as np
import pytest


spec = importlib.util.spec_from_file_location(
    "grn_audit", Path(__file__).resolve().parent.parent / "audit-k562-population-grn-v2.py",
)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def test_excludes_each_target_gene_for_each_seed():
    genes, targets = ["A", "B", "C"], ["A", "C"]
    truth = np.zeros((2, 3))
    prediction = np.array([[[100, 2, 4], [3, 5, 200]], [[300, 6, 8], [7, 9, 400]]])
    assert np.array_equal(audit.recompute(prediction, truth, genes, targets), [[10, 17], [50, 65]])


def test_rejects_missing_or_duplicated_gene_identifiers():
    with pytest.raises(ValueError):
        audit.recompute(np.zeros((1, 1, 2)), np.zeros((1, 2)), ["A", "A"], ["A"])
    with pytest.raises(ValueError):
        audit.recompute(np.zeros((1, 1, 2)), np.zeros((1, 2)), ["A", "B"], ["C"])
