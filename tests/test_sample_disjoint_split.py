import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from c3vcell.data import CONTROL, write_sample_disjoint_split


@pytest.fixture
def samples() -> tuple[ad.AnnData, ad.AnnData]:
    genes = pd.DataFrame(index=[f"G{i}" for i in range(8)])

    def make(prefix: str, batches: tuple[str, str], targets: tuple[str, str]) -> ad.AnnData:
        rows = []
        for batch, target in zip(batches, targets):
            rows.extend([{"target_gene": CONTROL, "context": "HCT116", "batch": batch}] * 4)
            rows.extend([{"target_gene": target, "context": "HCT116", "batch": batch}] * 3)
        obs = pd.DataFrame(rows, index=[f"{prefix}-{i}" for i in range(len(rows))])
        values = np.ones((len(rows), len(genes)), dtype=np.int32)
        values[:, 5] = 50 if prefix == "train" else 1
        values[:, 6] = 1 if prefix == "train" else 10_000
        data = ad.AnnData(sp.csr_matrix(values), obs=obs, var=genes.copy())
        data.uns["c3vcell_genome_assembly"] = "GRCh38"
        return data

    return make("train", ("A", "B"), ("G0", "G1")), make(
        "heldout", ("C", "D"), ("G2", "G3")
    )


def test_sample_and_target_split_selects_genes_only_from_training(
    samples: tuple[ad.AnnData, ad.AnnData], tmp_path: Path,
) -> None:
    training, heldout = samples
    out = tmp_path / "split"
    write_sample_disjoint_split(
        training, heldout, out, seed=7, max_genes=6, required_genes=["G4"],
    )
    train = ad.read_h5ad(out / "train.h5ad")
    reference = ad.read_h5ad(out / "reference.h5ad")
    inputs = ad.read_h5ad(out / "input_controls.h5ad")
    manifest = json.loads((out / "split.json").read_text())

    assert train.var_names.tolist() == reference.var_names.tolist() == inputs.var_names.tolist()
    assert set(train.var_names) == {"G0", "G1", "G2", "G3", "G4", "G5"}
    assert "G6" not in train.var_names
    assert set(train.obs_names).isdisjoint(reference.obs_names)
    assert set(train.obs_names).isdisjoint(inputs.obs_names)
    assert set(reference.obs_names).isdisjoint(inputs.obs_names)
    assert set(inputs.obs["target_gene"]) == {CONTROL}
    assert set(train.obs["batch"]) == {"A", "B"}
    assert set(reference.obs["batch"]) == set(inputs.obs["batch"]) == {"C", "D"}
    assert manifest["method"] == "sample-and-target-disjoint-v1"
    assert manifest["train_samples"] == [["HCT116", "A"], ["HCT116", "B"]]
    assert manifest["test_samples"] == [["HCT116", "C"], ["HCT116", "D"]]
    assert manifest["train_targets"] == ["G0", "G1"]
    assert manifest["test_targets"] == ["G2", "G3"]
    assert manifest["train_controls"] == 8
    assert manifest["input_controls"] == 4
    assert manifest["reference_controls"] == 4
    assert manifest["gene_selection"]["required_genes"] == ["G4"]

    repeat = tmp_path / "repeat"
    write_sample_disjoint_split(
        training, heldout, repeat, seed=7, max_genes=6, required_genes=["G4"],
    )
    assert json.loads((repeat / "split.json").read_text()) == manifest
    with pytest.raises(FileExistsError, match="nonempty"):
        write_sample_disjoint_split(training, heldout, out)


def test_sample_split_rejects_target_sample_and_cell_leakage(
    samples: tuple[ad.AnnData, ad.AnnData], tmp_path: Path,
) -> None:
    training, heldout = samples
    overlapping_target = heldout.copy()
    overlapping_target.obs.loc[
        overlapping_target.obs.target_gene == "G2", "target_gene"
    ] = "G0"
    with pytest.raises(ValueError, match="targets overlap"):
        write_sample_disjoint_split(training, overlapping_target, tmp_path / "target")

    overlapping_sample = heldout.copy()
    overlapping_sample.obs.loc[overlapping_sample.obs.batch == "C", "batch"] = "A"
    with pytest.raises(ValueError, match="sample groups overlap"):
        write_sample_disjoint_split(training, overlapping_sample, tmp_path / "sample")

    overlapping_cell = heldout.copy()
    names = overlapping_cell.obs_names.tolist()
    names[0] = training.obs_names[0]
    overlapping_cell.obs_names = names
    with pytest.raises(ValueError, match="cell identifiers overlap"):
        write_sample_disjoint_split(training, overlapping_cell, tmp_path / "cell")


def test_sample_split_requires_controls_in_each_sample(
    samples: tuple[ad.AnnData, ad.AnnData], tmp_path: Path,
) -> None:
    training, heldout = samples
    no_train_controls = training.copy()
    mask = (no_train_controls.obs.batch == "A") & (
        no_train_controls.obs.target_gene == CONTROL
    )
    no_train_controls.obs.loc[mask, "target_gene"] = "G0"
    with pytest.raises(ValueError, match="training sample groups need at least 1"):
        write_sample_disjoint_split(no_train_controls, heldout, tmp_path / "train-controls")

    one_test_control = heldout.copy()
    rows = one_test_control.obs.index[
        (one_test_control.obs.batch == "C") & (one_test_control.obs.target_gene == CONTROL)
    ]
    one_test_control.obs.loc[rows[1:], "target_gene"] = "G2"
    with pytest.raises(ValueError, match="heldout sample groups need at least 2"):
        write_sample_disjoint_split(training, one_test_control, tmp_path / "test-controls")


def test_sample_split_rejects_missing_metadata_and_incompatible_axes(
    samples: tuple[ad.AnnData, ad.AnnData], tmp_path: Path,
) -> None:
    training, heldout = samples
    missing_batch = heldout.copy()
    missing_batch.obs.loc[missing_batch.obs_names[0], "batch"] = None
    with pytest.raises(ValueError, match="non-null 'batch'"):
        write_sample_disjoint_split(training, missing_batch, tmp_path / "missing")

    blank_batch = heldout.copy()
    blank_batch.obs.loc[blank_batch.obs_names[0], "batch"] = " "
    with pytest.raises(ValueError, match="nonempty 'batch'"):
        write_sample_disjoint_split(training, blank_batch, tmp_path / "blank")

    changed_genes = heldout[:, ::-1].copy()
    with pytest.raises(ValueError, match="gene axes"):
        write_sample_disjoint_split(training, changed_genes, tmp_path / "genes")

    changed_mapping = heldout.copy()
    changed_mapping.var["ensembl_ids"] = [f"ENSG{i}" for i in range(changed_mapping.n_vars)]
    with pytest.raises(ValueError, match="Ensembl gene mappings"):
        write_sample_disjoint_split(training, changed_mapping, tmp_path / "mapping")

    changed_assembly = heldout.copy()
    changed_assembly.uns["c3vcell_genome_assembly"] = "GRCh37"
    with pytest.raises(ValueError, match="genome assemblies"):
        write_sample_disjoint_split(training, changed_assembly, tmp_path / "assembly")


def test_sample_split_validates_required_gene_panel(
    samples: tuple[ad.AnnData, ad.AnnData], tmp_path: Path,
) -> None:
    training, heldout = samples
    with pytest.raises(ValueError, match="absent from the source"):
        write_sample_disjoint_split(
            training, heldout, tmp_path / "absent", required_genes=["MISSING"],
        )
    with pytest.raises(ValueError, match="smaller than the number of required genes"):
        write_sample_disjoint_split(training, heldout, tmp_path / "small", max_genes=3)
