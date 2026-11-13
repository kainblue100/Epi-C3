from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from epi_c3_cell_eval_inputs_v1 import prepare_cell_eval_inputs


def matrix(ids, labels, *, parents=None):
    obs = pd.DataFrame({
        "target": labels,
        "context": ["human"] * len(ids),
        "batch": ["plate"] * len(ids),
    }, index=pd.Index(ids, name="cell"))
    if parents is not None:
        obs["source_control_id"] = parents
    counts = np.tile(np.array([[1.0, 3.0, 0.0]], dtype=np.float32), (len(ids), 1))
    return ad.AnnData(sp.csr_matrix(counts), obs=obs, var=pd.DataFrame(index=["a", "b", "c"]))


def data():
    controls = matrix([f"c{i}" for i in range(8)], ["ctrl"] * 8)
    reference = matrix(["r0", "r1", "t0", "t1"], ["ctrl", "ctrl", "drug", "drug"])
    parents = [f"c{i}" for i in range(8) for _ in range(2)]
    pred = matrix([f"p{i}" for i in range(16)], ["drug"] * 16, parents=parents)
    return reference, controls, {"model": pred, "noeffect": pred.copy()}


def prepare(reference, controls, predictions):
    return prepare_cell_eval_inputs(
        reference, controls, predictions, perturbation_column="target",
        control_label="ctrl", seed=9, minimum_parents=3,
    )


def test_deduplicates_predictions_and_uses_disjoint_controls():
    real, outputs, audit = prepare(*data())
    assert audit["unique_prediction_parents_per_treatment"] == {"drug": 4}
    assert audit["independent_predicted_controls"] == 4
    assert outputs["model"].shape == (8, 3)
    assert outputs["model"].obs_names.equals(outputs["noeffect"].obs_names)
    assert set(outputs["model"].obs_names).isdisjoint(set(real.obs_names))
    parent_ids = set(outputs["model"].obs["source_control_id"]) - {""}
    predicted_control_ids = set(outputs["model"].obs_names) & {f"c{i}" for i in range(8)}
    assert parent_ids.isdisjoint(predicted_control_ids)
    assert np.allclose(np.expm1(real.X.toarray()).sum(axis=1), 10000.0, rtol=1e-6)
    assert np.allclose(np.expm1(outputs["model"].X.toarray()).sum(axis=1), 10000.0, rtol=1e-6)


def test_rejects_overlapping_reference_cells_and_unmatched_panels():
    reference, controls, predictions = data()
    reference.obs_names = ["c0", "r1", "t0", "t1"]
    with pytest.raises(ValueError, match="overlap"):
        prepare(reference, controls, predictions)
    reference, controls, predictions = data()
    predictions["model"] = predictions["model"][:, ["c", "b", "a"]].copy()
    with pytest.raises(ValueError, match="gene axis"):
        prepare(reference, controls, predictions)


def test_rejects_misaligned_arms_and_unknown_parents():
    reference, controls, predictions = data()
    predictions["noeffect"].obs["source_control_id"] = "not-a-control"
    with pytest.raises(ValueError, match="identities differ"):
        prepare(reference, controls, predictions)
    reference, controls, predictions = data()
    predictions["model"].obs.loc["p0", "source_control_id"] = "not-a-control"
    with pytest.raises(ValueError, match="outside"):
        prepare(reference, controls, predictions)


def test_rejects_wrong_batch_fractional_counts_and_too_few_controls():
    reference, controls, predictions = data()
    controls.obs["batch"] = "another"
    with pytest.raises(ValueError, match="batch"):
        prepare(reference, controls, predictions)
    reference, controls, predictions = data()
    predictions["model"].X.data[0] = 0.5
    with pytest.raises(ValueError, match="counts"):
        prepare(reference, controls, predictions)
    reference, controls, predictions = data()
    with pytest.raises(ValueError, match="Too few"):
        prepare(reference, controls[:5].copy(), predictions)
