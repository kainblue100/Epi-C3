import copy
import json
import subprocess
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import torch
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence

from c3vcell.data import CONTROL, write_split
from c3vcell.calibration import OBJECTIVE, calibrate_effect
from c3vcell.drug import DrugConditioner, predict_drug, train_drug, validate_tahoe
from c3vcell.features import gene_token_mapping
from c3vcell.predict import _resolve_effect_strength, predict_control_baseline
from c3vcell.tahoe import inspect_tahoe, prepare_tahoe, read_samples


@pytest.fixture
def tahoe_files(tmp_path: Path) -> tuple[Path, Path, Path]:
    gene_metadata = tmp_path / "genes.parquet"
    pd.DataFrame({
        "token_id": [3, 4, 5, 6, 7, 8],
        "gene_symbol": ["G0", "G0", "G1", "G2", "G3", "G4"],
        "ensembl_id": ["E0", "E1", "E2", "E3", "E4", "E5"],
    }).to_parquet(gene_metadata)
    sample_rows, cells = [], []
    smiles = ["CCO", "CCN", "CCC", "CCF"]
    for group in range(5):
        drug = "DMSO_TF" if group == 0 else f"Drug{group}"
        dose = 0 if group == 0 else 0.5
        sample = f"S{group}"
        sample_rows.append({
            "sample": sample, "plate": "plate1", "drug": drug,
            "drugname_drugconc": repr([(drug, dose, "uM")]),
        })
        for i in range(30 if group == 0 else 12):
            cells.append({
                "genes": [1, 3, 4, 5, 6, 7, 8],
                "expressions": [-2.0, 1, 2, 1, 2 + group, 1, 3],
                "drug": drug, "sample": sample, "plate": "plate1",
                "cell_line_id": "L1", "BARCODE_SUB_LIB_ID": f"{sample}-{i}",
                "canonical_smiles": None if group == 0 else smiles[group - 1],
            })
    sample_metadata = tmp_path / "samples.parquet"
    pd.DataFrame(sample_rows).to_parquet(sample_metadata)
    shard = tmp_path / "train-00000.parquet"
    pd.DataFrame(cells).to_parquet(shard)
    return gene_metadata, sample_metadata, shard


def _prepared(tahoe_files: tuple[Path, Path, Path], output: Path) -> None:
    genes, samples, shard = tahoe_files
    prepare_tahoe(
        [shard], genes, samples, output,
        drugs=["Drug1", "Drug2", "Drug3", "Drug4"],
        cell_lines=["L1"], plates=["plate1"], max_per_condition=12,
        max_controls_per_group=30, seed=2,
    )


def test_tahoe_counts_cls_dose_and_metadata(
    tahoe_files: tuple[Path, Path, Path], tmp_path: Path,
) -> None:
    genes, samples, shard = tahoe_files
    assert inspect_tahoe([shard]).n_cells.sum() == 78
    assert read_samples(samples)["S1"] == ("Drug1", "plate1", 0.5)
    output = tmp_path / "subset.h5ad"
    _prepared(tahoe_files, output)
    data = ad.read_h5ad(output)
    validate_tahoe(data)
    assert data.shape == (78, 5)
    assert data.var.loc["G0", "ensembl_ids"] == "E0|E1"
    assert np.all(data.X.toarray()[:, 0] == 3)
    assert set(data.obs.loc[data.obs.target_gene == CONTROL, "drug"]) == {"DMSO_TF"}
    assert set(data.obs.loc[data.obs.target_gene != CONTROL, "dose_um"]) == {0.5}
    assert data.obs_names.str.startswith("tahoe-v1-").all()
    assert data.uns["c3vcell_tahoe"]["cell_identity"].startswith("tahoe-v1")
    alternate = tmp_path / "alternate.h5ad"
    prepare_tahoe(
        [shard], genes, samples, alternate, drugs=["Drug3", "Drug4"],
        cell_lines=["L1"], plates=["plate1"], max_per_condition=12,
        max_controls_per_group=30, seed=3,
    )
    alternate_data = ad.read_h5ad(alternate)
    original_by_source = data.obs.reset_index().set_index("source_cell_id")["cell_id"]
    alternate_by_source = alternate_data.obs.reset_index().set_index("source_cell_id")["cell_id"]
    for source_id, cell_id in alternate_by_source.items():
        assert original_by_source[source_id] == cell_id
    with pytest.raises(ValueError, match="distinct"):
        inspect_tahoe([shard, shard])
    bad = pd.read_parquet(samples)
    bad.loc[0, "drugname_drugconc"] = "[('DMSO_TF', 1, 'furlong')]"
    bad_path = tmp_path / "bad-samples.parquet"
    bad.to_parquet(bad_path)
    with pytest.raises(ValueError, match="unsupported"):
        read_samples(bad_path)


def test_smiles_dose_encoder_zero_and_gradient() -> None:
    conditioner = DrugConditioner(6, 8)
    result = conditioner(["CCO", "CCN"], torch.tensor([0.0, 1.0]))
    assert result.shape == (2, 6)
    assert torch.equal(result[0], torch.zeros(6))
    result[1].sum().backward()
    assert conditioner.output[0].weight.grad is not None
    with pytest.raises(ValueError, match="unsupported"):
        conditioner(["C\x01"], torch.tensor([1.0]))


def test_gene_token_mapping_rejects_fractional_ids() -> None:
    metadata = pd.DataFrame({
        "token_id": [3.5], "gene_symbol": ["G"], "ensembl_id": ["E"],
    })
    with pytest.raises(ValueError, match="integers"):
        gene_token_mapping(metadata, token_column="token_id", symbol_column="gene_symbol")


def test_tahoe_panel_selection_uses_counts_only(tmp_path: Path) -> None:
    groups = pd.DataFrame([
        ("DMSO_TF", "S0", "plate1", "L1", 50),
        ("B", "S1", "plate1", "L1", 30),
        ("A", "S2", "plate1", "L1", 30),
        ("C", "S3", "plate1", "L1", 10),
        ("X", "S4", "plate2", "L1", 500),
    ], columns=["drug", "sample", "plate", "cell_line_id", "n_cells"])
    source = tmp_path / "groups.csv"
    groups.to_csv(source, index=False)
    output = tmp_path / "drugs.txt"
    script = Path(__file__).parents[1] / "scripts" / "select-tahoe-drugs.py"
    command = [
        sys.executable, str(script), "--groups", str(source), "--cell-line", "L1",
        "--plate", "plate1", "--drugs", "2", "--min-cells", "20", "--output", str(output),
    ]
    subprocess.run(command, check=True, capture_output=True, text=True)
    assert output.read_text().splitlines() == ["A", "B"]
    assert json.loads(output.with_suffix(".json").read_text())["control_cells"] == 50
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode != 0
    assert "Refusing to overwrite" in result.stderr


def test_tahoe_rejects_invalid_cls_counts_and_missing_matched_controls(
    tahoe_files: tuple[Path, Path, Path], tmp_path: Path,
) -> None:
    genes, samples, shard = tahoe_files
    records = pd.read_parquet(shard)
    records.at[0, "expressions"] = [-1.0, 1, 2, 1, 2, 1, 3]
    invalid = tmp_path / "invalid.parquet"
    records.to_parquet(invalid)
    with pytest.raises(ValueError, match="CLS prefix"):
        prepare_tahoe(
            [invalid], genes, samples, tmp_path / "invalid.h5ad",
            drugs=["Drug1"], cell_lines=["L1"], plates=["plate1"],
            max_per_condition=12, max_controls_per_group=30,
        )
    records = pd.read_parquet(shard)
    records.loc[records.drug == "Drug1", "cell_line_id"] = "L2"
    unmatched = tmp_path / "unmatched.parquet"
    records.to_parquet(unmatched)
    with pytest.raises(ValueError, match="Missing matched DMSO"):
        prepare_tahoe(
            [unmatched], genes, samples, tmp_path / "unmatched.h5ad",
            drugs=["Drug1"], cell_lines=["L1", "L2"], plates=["plate1"],
            max_per_condition=12, max_controls_per_group=30,
        )


def test_tahoe_compound_disjoint_training_and_prediction(
    tahoe_files: tuple[Path, Path, Path], tmp_path: Path,
) -> None:
    source = tmp_path / "subset.h5ad"
    _prepared(tahoe_files, source)
    split = tmp_path / "split"
    write_split(
        ad.read_h5ad(source), split, control=CONTROL, holdout_fraction=0.5, seed=4,
        split_column="compound", require_targets_on_gene_axis=False, max_genes=4,
    )
    train_data = ad.read_h5ad(split / "train.h5ad")
    reference = ad.read_h5ad(split / "reference.h5ad")
    controls = ad.read_h5ad(split / "input_controls.h5ad")
    train_compounds = set(train_data.obs.compound) - {CONTROL}
    ref_compounds = set(reference.obs.compound) - {CONTROL}
    assert train_compounds.isdisjoint(ref_compounds)
    assert set(train_data.obs_names).isdisjoint(reference.obs_names)
    assert set(controls.obs_names).isdisjoint(reference.obs_names)
    checkpoint = train_drug(
        split / "train.h5ad", tmp_path / "model", latent_dim=4, hidden_dim=6,
        pretrain_steps=2, flow_steps=2, batch_size=4, seed=3,
    )
    prediction = tmp_path / "drug-prediction.h5ad"
    predict_drug(
        checkpoint, split / "reference.h5ad", split / "input_controls.h5ad", prediction,
        cells_per_target=6, steps=2, effect_strength=0, seed=3,
    )
    baseline = tmp_path / "baseline.h5ad"
    predict_control_baseline(
        split / "reference.h5ad", split / "input_controls.h5ad", baseline,
        cells_per_target=6, seed=3,
    )
    pred = ad.read_h5ad(prediction)
    base = ad.read_h5ad(baseline)
    assert pred.shape == (12, 4)
    assert (pred.X != base.X).nnz == 0
    assert pred.obs.target_gene.tolist() == base.obs.target_gene.tolist()
    with pytest.raises(ValueError, match="overlap"):
        predict_drug(
            checkpoint, split / "reference.h5ad", split / "reference.h5ad",
            tmp_path / "leaked.h5ad",
        )


def test_tahoe_effect_calibration_holds_out_compounds(
    tahoe_files: tuple[Path, Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "subset.h5ad"
    _prepared(tahoe_files, source)
    data = ad.read_h5ad(source)
    data.obs["target_gene"] = data.obs["target_gene"].astype(str)
    extra_dose = data.obs.index[data.obs["drug"].astype(str) == "Drug1"][:5]
    data.obs.loc[extra_dose, "target_gene"] = "Drug1::0.05uM"
    data.obs.loc[extra_dose, "dose_um"] = 0.05
    data.write_h5ad(source)
    checkpoint_path = train_drug(
        source, tmp_path / "model", latent_dim=4, hidden_dim=6,
        pretrain_steps=2, flow_steps=2, batch_size=4, variance_weight=3.0,
        covariance_weight=0.3, regularization_view="unmasked", seed=3,
    )

    def fake_score(prediction_path: Path, *_args, **kwargs) -> dict:
        assert kwargs["no_target_genes"]
        assert kwargs["stratify_by_batch"]
        strength = ad.read_h5ad(prediction_path).uns["c3vcell_prediction"]["effect_strength"]
        return {"raw_metrics": {"L1": {OBJECTIVE: 1 + (strength - 0.5) ** 2}}}

    monkeypatch.setattr("c3vcell.calibration.score_local", fake_score)
    output = calibrate_effect(
        source, checkpoint_path, tmp_path / "calibration", kind="drug",
        folds=2, strengths=(0.0, 0.5, 1.0), cells_per_target=4, prediction_steps=2,
    )
    record = json.loads(output.read_text(encoding="utf-8"))
    assert record["method"] == "compound-disjoint-training-folds-v1"
    assert record["selected_effect_strength"] == 0.5
    assert set(record["folds"][0]["heldout_targets"]).isdisjoint(
        record["folds"][1]["heldout_targets"]
    )
    for fold in range(2):
        split = tmp_path / "calibration" / f"fold-{fold:02d}" / "split"
        inner = torch.load(
            tmp_path / "calibration" / f"fold-{fold:02d}" / "model" / "model.pt",
            map_location="cpu", weights_only=True,
        )
        assert inner["model_config"]["variance_weight"] == 3.0
        assert inner["model_config"]["covariance_weight"] == 0.3
        assert inner["model_config"]["regularization_view"] == "unmasked"
        assert "relative_mean_std" in inner["representation_diagnostics"]
        training = ad.read_h5ad(split / "train.h5ad")
        reference = ad.read_h5ad(split / "reference.h5ad")
        train_compounds = set(training.obs["compound"].astype(str)) - {CONTROL}
        heldout_compounds = set(reference.obs["compound"].astype(str)) - {CONTROL}
        assert train_compounds.isdisjoint(heldout_compounds)
        for dataset in (training, reference):
            drug = dataset.obs[dataset.obs["drug"].astype(str) == "Drug1"]
            if not drug.empty:
                assert set(drug["dose_um"].astype(float)) == {0.05, 0.5}
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    assert _resolve_effect_strength(
        checkpoint, checkpoint_path, "residual-rounding", None, output,
        calibration_kind="drug", prediction_steps=2,
    ) == 0.5
    with pytest.raises(ValueError, match="Unsupported"):
        _resolve_effect_strength(checkpoint, checkpoint_path, "residual-rounding", None, output)
    record["stratify_by_batch"] = False
    bad = tmp_path / "unstratified-calibration.json"
    bad.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="batch-stratified"):
        _resolve_effect_strength(
            checkpoint, checkpoint_path, "residual-rounding", None, bad,
            calibration_kind="drug", prediction_steps=2,
        )


def test_repeated_smiles_encoding_preserves_outputs_and_gradients() -> None:
    torch.manual_seed(7)
    conditioner = DrugConditioner(latent_dim=4, hidden_dim=6)
    reference = copy.deepcopy(conditioner)
    smiles = ["CCO", "N", "CCO", "C(=O)N", "N", "CCO"]
    doses = torch.tensor([0.05, 0.5, 5.0, 0.1, 0.2, 1.0], requires_grad=True)
    ref_doses = doses.detach().clone().requires_grad_(True)
    actual = conditioner(smiles, doses)

    lengths = torch.tensor([len(value) for value in smiles], dtype=torch.long)
    tokens = torch.zeros((len(smiles), int(lengths.max())), dtype=torch.long)
    for index, value in enumerate(smiles):
        tokens[index, :len(value)] = torch.tensor([ord(char) + 1 for char in value])
    packed = pack_padded_sequence(
        reference.char_embedding(tokens), lengths, batch_first=True, enforce_sorted=False,
    )
    _, state = reference.sequence(packed)
    raw = reference.output(torch.cat((state[-1], torch.log1p(ref_doses)[:, None]), dim=-1))
    expected = F.normalize(raw, dim=-1) * (ref_doses / (1 + ref_doses))[:, None]
    torch.testing.assert_close(actual, expected)
    weights = torch.arange(1, actual.numel() + 1).reshape(actual.shape)
    (actual * weights).sum().backward()
    (expected * weights).sum().backward()
    torch.testing.assert_close(doses.grad, ref_doses.grad)
    for (_, actual_param), (_, expected_param) in zip(
        conditioner.named_parameters(), reference.named_parameters(),
    ):
        torch.testing.assert_close(actual_param.grad, expected_param.grad, rtol=1e-4, atol=1e-7)
