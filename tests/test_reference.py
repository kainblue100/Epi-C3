from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import torch

from c3vcell.data import CONTROL, write_split
from c3vcell.predict import predict_holdout
from c3vcell.reference import load_reference
from c3vcell.training import train


def test_reference_alignment_uses_a_mask_not_missing_gene_zeros(tmp_path: Path) -> None:
    source = ad.AnnData(
        sp.csr_matrix(np.array([[2, 4, 9], [1, 3, 5]], dtype=np.int32)),
        obs=pd.DataFrame(index=["r0", "r1"]),
        var=pd.DataFrame(index=["G1", "G3", "other"]),
    )
    path = tmp_path / "reference.h5ad"
    source.write_h5ad(path)
    loaded = load_reference(path, ["G0", "G1", "G2", "G3"], max_cells=2, seed=3, min_coverage=0.5)
    assert loaded.measured.tolist() == [0, 1, 0, 1]
    np.testing.assert_array_equal(loaded.matrix.toarray(), [[0, 2, 0, 4], [0, 1, 0, 3]])
    assert loaded.shared_genes == 2
    reversed_axis = load_reference(path, ["G3", "G1"], max_cells=2, seed=3, min_coverage=1)
    assert loaded.source_identity == reversed_axis.source_identity
    with pytest.raises(ValueError, match="below"):
        load_reference(path, ["G0", "G1", "G2", "G3"], max_cells=2, seed=3, min_coverage=0.75)


def test_reference_pretraining_records_and_rejects_heldout_overlap(tmp_path: Path) -> None:
    rng = np.random.default_rng(42)
    labels = [CONTROL] * 30 + sum(([f"G{i}"] * 8 for i in range(4)), [])
    data = ad.AnnData(
        sp.csr_matrix(rng.poisson(3, size=(len(labels), 8)).astype(np.int32) + 1),
        obs=pd.DataFrame(
            {"target_gene": labels, "context": "c", "batch": "b"},
            index=[f"cell-{i}" for i in range(len(labels))],
        ),
        var=pd.DataFrame(index=[f"G{i}" for i in range(8)]),
    )
    split = tmp_path / "split"
    write_split(data, split, control=CONTROL, holdout_fraction=0.5, seed=1)
    reference = ad.read_h5ad(split / "reference.h5ad")[:3, [0, 2, 4, 6]].copy()
    reference_path = tmp_path / "external.h5ad"
    reference.write_h5ad(reference_path)
    checkpoint_path = train(
        split / "train.h5ad", tmp_path / "model", latent_dim=4, hidden_dim=6,
        pretrain_steps=2, flow_steps=2, batch_size=4, seed=3,
        reference_path=reference_path, reference_fraction=0.5, reference_min_coverage=0.5,
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    assert checkpoint["reference"]["shared_genes"] == 4
    assert set(checkpoint["reference"]["cell_ids"]) == set(reference.obs_names)
    with pytest.raises(ValueError, match="overlap"):
        predict_holdout(
            checkpoint_path, split / "reference.h5ad", split / "input_controls.h5ad",
            tmp_path / "prediction.h5ad",
        )


def test_legacy_training_and_external_reference_warn_even_if_inference_ids_differ(
    tmp_path: Path,
) -> None:
    labels = [CONTROL] * 30 + sum(([f"G{i}"] * 8 for i in range(4)), [])
    data = ad.AnnData(
        sp.csr_matrix(np.ones((len(labels), 8), dtype=np.int32)),
        obs=pd.DataFrame(
            {"target_gene": labels, "context": "c", "batch": "b"},
            index=[f"orion-{i:09d}" for i in range(len(labels))],
        ),
        var=pd.DataFrame(index=[f"G{i}" for i in range(8)]),
    )
    split = tmp_path / "split"
    write_split(data, split, control=CONTROL, holdout_fraction=0.5, seed=1)
    for name in ("reference", "input_controls"):
        path = split / f"{name}.h5ad"
        inference_data = ad.read_h5ad(path)
        inference_data.obs_names = [f"{name}-{i}" for i in range(inference_data.n_obs)]
        inference_data.write_h5ad(path)
    external = ad.read_h5ad(split / "train.h5ad")[:3, [0, 2, 4, 6]].copy()
    external.obs_names = [f"orion-{900000000 + i:09d}" for i in range(external.n_obs)]
    external_path = tmp_path / "external.h5ad"
    external.write_h5ad(external_path)
    with pytest.warns(RuntimeWarning, match="assigned by position"):
        checkpoint = train(
            split / "train.h5ad", tmp_path / "model", latent_dim=4, hidden_dim=6,
            pretrain_steps=2, flow_steps=2, batch_size=4, seed=3,
            reference_path=external_path, reference_fraction=0.5, reference_min_coverage=0.5,
        )
    with pytest.warns(RuntimeWarning, match="assigned by position"):
        predict_holdout(
            checkpoint, split / "reference.h5ad", split / "input_controls.h5ad",
            tmp_path / "prediction.h5ad", cells_per_target=2, steps=2,
        )
