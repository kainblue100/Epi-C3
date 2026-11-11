import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import torch

from c3vcell.data import CONTROL, split_by_target, validate_counts, write_split
from c3vcell.model import C3VCell, ModelConfig
from c3vcell.predict import (
    _read_gene_list, _read_target_requests, _read_targets, _sample_controls, export_vcc_h5ad,
    predict_control_baseline, predict_holdout, predict_targets,
)
from c3vcell.training import (
    _control_groups, _population_shift_groups, _restore_graph, load_model, read_graph,
    representation_diagnostics, train,
)


@pytest.fixture
def dataset() -> ad.AnnData:
    rng = np.random.default_rng(42)
    labels = [CONTROL] * 30 + sum(([f"G{i}"] * 12 for i in range(4)), [])
    counts = rng.poisson(3, size=(len(labels), 16)).astype(np.int32)
    counts[:, 15] += 1
    obs = pd.DataFrame(
        {"target_gene": labels, "context": "HCT116", "batch": "batch1"},
        index=[f"cell-{i}" for i in range(len(labels))],
    )
    var = pd.DataFrame(index=[f"G{i}" for i in range(16)])
    return ad.AnnData(sp.csr_matrix(counts), obs=obs, var=var)


def test_target_and_control_splits_are_disjoint(dataset: ad.AnnData) -> None:
    training, reference, inputs, manifest = split_by_target(
        dataset, holdout_fraction=0.4, seed=7
    )
    assert set(manifest["train_targets"]).isdisjoint(manifest["test_targets"])
    assert len(manifest["test_targets"]) == 2
    assert set(training.obs_names).isdisjoint(reference.obs_names)
    assert set(training.obs_names).isdisjoint(inputs.obs_names)
    assert set(reference.obs_names).isdisjoint(inputs.obs_names)
    assert set(inputs.obs["target_gene"]) == {CONTROL}
    assert set(training.obs["target_gene"]) - {CONTROL} == set(manifest["train_targets"])
    assert set(reference.obs["target_gene"]) - {CONTROL} == set(manifest["test_targets"])


def test_counts_reject_fractional_and_zero_cells(dataset: ad.AnnData) -> None:
    altered = dataset.copy()
    altered.X = altered.X.astype(np.float32)
    altered.X.data[0] = 0.5
    with pytest.raises(ValueError, match="raw integer"):
        validate_counts(altered)
    altered.X = sp.csr_matrix(altered.shape, dtype=np.int32)
    with pytest.raises(ValueError, match="zero total"):
        validate_counts(altered)


def test_model_mask_sampling_and_graph(tmp_path: Path) -> None:
    torch.manual_seed(5)
    model = C3VCell(ModelConfig(n_genes=4, latent_dim=6, hidden_dim=8))
    counts = torch.tensor([[2.0, 0.0, 4.0, 1.0], [1.0, 3.0, 2.0, 2.0]])
    mask = torch.tensor([[1.0, 0.0, 1.0, 1.0]]).repeat(2, 1)
    loss, stats = model.representation_loss(counts, measured=mask)
    assert torch.isfinite(loss) and stats["nb"] >= 0
    loss.backward()
    graph_path = tmp_path / "graph.csv"
    graph_path.write_text("source,target,weight\nG0,G1,1.5\n", encoding="utf-8")
    graph = read_graph(graph_path, ["G0", "G1", "G2", "G3"], torch.device("cpu"))
    assert graph is not None
    result = model.predict_counts(counts, torch.tensor([0, 0]), measured=mask, graph=graph, steps=2)
    assert result.shape == counts.shape
    assert torch.all(result[:, 1] == 0)
    assert torch.all(result >= 0)


def test_restore_graph_checks_high_degree_rows_without_float32_accumulation_error() -> None:
    degree = 2404
    indices = torch.stack((torch.zeros(degree, dtype=torch.long), torch.arange(degree)))
    payload = {
        "indices": indices,
        "values": torch.full((degree,), 1.0 / degree, dtype=torch.float32),
        "shape": [degree, degree],
    }
    remaining = torch.arange(1, degree)
    payload["indices"] = torch.cat((indices, torch.stack((remaining, remaining))), dim=1)
    payload["values"] = torch.cat((payload["values"], torch.ones(degree - 1)))
    graph = _restore_graph(payload, torch.device("cpu"))
    assert graph is not None
    payload["values"][0] += 0.1
    with pytest.raises(ValueError, match="already be normalized"):
        _restore_graph(payload, torch.device("cpu"))


def test_variance_weight_and_collapse_diagnostic() -> None:
    torch.manual_seed(9)
    ordinary = C3VCell(ModelConfig(n_genes=4, latent_dim=4, hidden_dim=6, variance_weight=0.1))
    stronger = C3VCell(ModelConfig(n_genes=4, latent_dim=4, hidden_dim=6, variance_weight=2.0))
    stronger.load_state_dict(ordinary.state_dict())
    counts = torch.tensor([[2.0, 0.0, 4.0, 1.0], [1.0, 3.0, 2.0, 2.0], [5.0, 1.0, 1.0, 1.0]])
    torch.manual_seed(11)
    old_loss, old_stats = ordinary.representation_loss(counts)
    torch.manual_seed(11)
    new_loss, _ = stronger.representation_loss(counts)
    assert float(new_loss - old_loss) == pytest.approx(1.9 * old_stats["variance"], rel=1e-5)
    with pytest.raises(ValueError, match="variance_weight"):
        ModelConfig(n_genes=4, variance_weight=-1)
    with torch.no_grad():
        ordinary.encoder.value.weight.zero_()
        ordinary.encoder.visible.weight.zero_()
    diagnostics = representation_diagnostics(
        ordinary, sp.csr_matrix(counts.numpy()), device=torch.device("cpu"),
        batch_size=2, seed=3,
    )
    assert diagnostics["collapse_warning"] is True
    assert diagnostics["relative_mean_std"] < 0.05


def test_covariance_penalty_unmasked_view_and_rank_warning() -> None:
    torch.manual_seed(19)
    ordinary = C3VCell(ModelConfig(
        n_genes=4, latent_dim=4, hidden_dim=6, regularization_view="unmasked",
    )).double()
    decorrelated = C3VCell(ModelConfig(
        n_genes=4, latent_dim=4, hidden_dim=6,
        covariance_weight=0.3, regularization_view="unmasked",
    )).double()
    decorrelated.load_state_dict(ordinary.state_dict())
    counts = torch.tensor([
        [2.0, 0.0, 4.0, 1.0], [1.0, 3.0, 2.0, 2.0],
        [5.0, 1.0, 1.0, 1.0], [1.0, 2.0, 1.0, 5.0],
    ], dtype=torch.float64)
    torch.manual_seed(21)
    old_loss, old_stats = ordinary.representation_loss(counts)
    torch.manual_seed(21)
    new_loss, stats = decorrelated.representation_loss(counts)
    assert stats["covariance"] > 0
    assert float(new_loss - old_loss) == pytest.approx(0.3 * stats["covariance"], rel=1e-5)
    torch.manual_seed(22)
    _, different_masks = decorrelated.representation_loss(counts)
    assert different_masks["variance"] == pytest.approx(old_stats["variance"])
    assert different_masks["covariance"] == pytest.approx(stats["covariance"])
    new_loss.backward()
    assert torch.isfinite(decorrelated.encoder.value.weight.grad).all()
    with pytest.raises(ValueError, match="at least two"):
        decorrelated.representation_loss(counts[:1])
    with pytest.raises(ValueError, match="covariance_weight"):
        ModelConfig(n_genes=4, covariance_weight=float("nan"))
    with pytest.raises(ValueError, match="regularization_view"):
        ModelConfig(n_genes=4, regularization_view="invalid")

    class RankOne(torch.nn.Module):
        def encode(self, value: torch.Tensor) -> torch.Tensor:
            return value[:, :1] * torch.tensor([[1.0, 2.0, 3.0]])

    matrix = sp.csr_matrix(np.arange(1, 13, dtype=np.float32)[:, None])
    diagnostic = representation_diagnostics(
        RankOne(), matrix, device=torch.device("cpu"), batch_size=4, seed=3,
    )
    assert diagnostic["low_variation_warning"] is False
    assert diagnostic["low_rank_warning"] is True
    assert diagnostic["collapse_warning"] is True
    assert diagnostic["participation_rank"] == pytest.approx(1.0)
    assert diagnostic["first_component_variance_fraction"] == pytest.approx(1.0)


def test_population_shift_uses_matched_training_centroids(dataset: ad.AnnData) -> None:
    torch.manual_seed(4)
    model = C3VCell(ModelConfig(n_genes=16, latent_dim=6, hidden_dim=8))
    matrix = sp.csr_matrix(dataset.X, dtype=np.float32)
    labels = dataset.obs.target_gene.astype(str).to_numpy()
    contexts = dataset.obs.context.astype(str).to_numpy()
    batches = dataset.obs.batch.astype(str).to_numpy()
    exact, by_context = _control_groups(dataset.obs, CONTROL)
    groups, details = _population_shift_groups(
        model, matrix, labels, contexts, batches, exact, by_context,
        control=CONTROL, min_group_cells=5, batch_size=8, device=torch.device("cpu"),
    )
    with torch.no_grad():
        latent = model.encode(torch.tensor(matrix.toarray(), dtype=torch.float32)).numpy()
    expected = latent[labels == "G0"].mean(axis=0) - latent[labels == CONTROL].mean(axis=0)
    np.testing.assert_allclose(groups["G0"][0][1], expected, atol=1e-6)
    assert details["eligible_groups"] == 4
    with pytest.raises(ValueError, match="No context"):
        _population_shift_groups(
            model, matrix, labels, contexts, batches, exact, by_context,
            control=CONTROL, min_group_cells=31, batch_size=8, device=torch.device("cpu"),
        )


def test_population_shift_training_records_configuration(dataset: ad.AnnData, tmp_path: Path) -> None:
    source = tmp_path / "source.h5ad"
    dataset.write_h5ad(source)
    checkpoint_path = train(
        source, tmp_path / "model", latent_dim=6, hidden_dim=8,
        pretrain_steps=2, flow_steps=2, batch_size=8,
        flow_target="population-shift", min_group_cells=5,
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    assert checkpoint["training_parameters"]["flow_target"] == "population-shift"
    assert checkpoint["population_shift_groups"]["eligible_groups"] == 4
    checkpoint["representation_diagnostics"]["collapse_warning"] = True
    flagged = tmp_path / "flagged.pt"
    torch.save(checkpoint, flagged)
    with pytest.warns(RuntimeWarning, match="low-variation or low-rank"):
        load_model(flagged)


def test_residual_rounding_preserves_the_control_when_flow_is_zero() -> None:
    model = C3VCell(ModelConfig(n_genes=4, latent_dim=6, hidden_dim=8))
    for parameter in model.velocity.parameters():
        torch.nn.init.zeros_(parameter)
    control = torch.tensor([[2.0, 0.0, 4.0, 1.0], [1.0, 3.0, 2.0, 2.0]])
    result = model.predict_counts(
        control, torch.tensor([0, 1]), steps=2, sampling="residual-rounding"
    )
    assert torch.equal(result, control.to(torch.int32))


def test_zero_effect_strength_preserves_controls_and_rejects_bad_values() -> None:
    model = C3VCell(ModelConfig(n_genes=4, latent_dim=6, hidden_dim=8))
    control = torch.tensor([[2.0, 0.0, 4.0, 1.0], [1.0, 3.0, 2.0, 2.0]])
    result = model.predict_counts(
        control, torch.tensor([0, 1]), steps=2,
        sampling="residual-rounding", effect_strength=0,
    )
    assert torch.equal(result, control.to(torch.int32))
    for strength in (-0.1, 1.1, float("nan")):
        with pytest.raises(ValueError, match="effect_strength"):
            model.predict_counts(control, torch.tensor([0, 1]), effect_strength=strength)


def test_explicit_target_split_is_exact_and_disjoint(dataset: ad.AnnData) -> None:
    train_data, reference, _, manifest = split_by_target(
        dataset, holdout_targets=["G0", "G3"], seed=1,
    )
    assert manifest["test_targets"] == ["G0", "G3"]
    assert set(train_data.obs.target_gene) == {CONTROL, "G1", "G2"}
    assert set(reference.obs.target_gene) == {CONTROL, "G0", "G3"}
    with pytest.raises(ValueError, match="at least two unique"):
        split_by_target(dataset, holdout_targets=["G0", "G0"])
    with pytest.raises(ValueError, match="leave at least one"):
        split_by_target(dataset, holdout_targets=["G0", "MISSING"])


def test_reference_does_not_include_control_only_contexts(dataset: ad.AnnData) -> None:
    labels = dataset.obs.target_gene.to_numpy()
    contexts = np.full(dataset.n_obs, "a", dtype=object)
    contexts[:15] = "b"
    contexts[np.isin(labels, ["G2", "G3"])] = "b"
    dataset.obs["context"] = contexts
    train_data, reference, inputs, _ = split_by_target(
        dataset, holdout_targets=["G2", "G3"], seed=1,
    )
    assert set(train_data.obs.loc[train_data.obs.target_gene != CONTROL, "context"]) == {"a"}
    assert set(reference.obs.context) == {"b"}
    assert set(inputs.obs.context) == {"b"}


def test_train_predict_end_to_end(dataset: ad.AnnData, tmp_path: Path) -> None:
    split = tmp_path / "split"
    write_split(dataset, split, control=CONTROL, holdout_fraction=0.4, seed=4)
    checkpoint = train(
        split / "train.h5ad", tmp_path / "model", latent_dim=6, hidden_dim=8,
        pretrain_steps=2, flow_steps=2, batch_size=8, seed=4,
    )
    output = tmp_path / "prediction.h5ad"
    predict_holdout(
        checkpoint, split / "reference.h5ad", split / "input_controls.h5ad",
        output, cells_per_target=6, steps=2, seed=4,
    )
    prediction = ad.read_h5ad(output)
    assert prediction.shape == (12, 16)
    assert prediction.obs_names.is_unique
    assert CONTROL not in set(prediction.obs["target_gene"])
    validate_counts(prediction)
    with pytest.raises(FileExistsError):
        predict_holdout(
            checkpoint, split / "reference.h5ad", split / "input_controls.h5ad", output,
        )
    with pytest.raises(ValueError, match="overlap"):
        predict_holdout(
            checkpoint, split / "reference.h5ad", split / "train.h5ad",
            tmp_path / "leaked.h5ad",
        )


def test_gene_selection_uses_training_cells_and_retains_all_targets(
    dataset: ad.AnnData, tmp_path: Path
) -> None:
    out = tmp_path / "selected"
    write_split(
        dataset, out, control=CONTROL, holdout_fraction=0.4, seed=4, max_genes=8
    )
    training = ad.read_h5ad(out / "train.h5ad")
    reference = ad.read_h5ad(out / "reference.h5ad")
    inputs = ad.read_h5ad(out / "input_controls.h5ad")
    assert training.n_vars == 8
    assert training.var_names.tolist() == reference.var_names.tolist() == inputs.var_names.tolist()
    assert {f"G{i}" for i in range(4)} <= set(training.var_names)


def test_gene_selection_retains_prespecified_graph_neighbors(
    dataset: ad.AnnData, tmp_path: Path,
) -> None:
    dataset.X[:, 15] = 1
    out = tmp_path / "selected"
    write_split(
        dataset, out, control=CONTROL, holdout_fraction=0.4, seed=4,
        max_genes=5, required_genes=["G15"],
    )
    train_data = ad.read_h5ad(out / "train.h5ad")
    assert set(train_data.var_names) == {"G0", "G1", "G2", "G3", "G15"}
    for name in ("reference.h5ad", "input_controls.h5ad"):
        assert ad.read_h5ad(out / name).var_names.tolist() == train_data.var_names.tolist()
    selection = json.loads((out / "split.json").read_text(encoding="utf-8"))["gene_selection"]
    assert selection["required_genes"] == ["G15"]
    assert "prespecified required genes" in selection["method"]
    with pytest.raises(ValueError, match="absent from the source"):
        write_split(
            dataset, tmp_path / "missing", control=CONTROL, holdout_fraction=0.4,
            seed=4, max_genes=5, required_genes=["missing"],
        )
    with pytest.raises(ValueError, match="unique nonempty strings"):
        write_split(
            dataset, tmp_path / "duplicate", control=CONTROL, holdout_fraction=0.4,
            seed=4, max_genes=5, required_genes=["G15", "G15"],
        )
    with pytest.raises(ValueError, match="smaller than the number of required genes"):
        write_split(
            dataset, tmp_path / "too-few", control=CONTROL, holdout_fraction=0.4,
            seed=4, max_genes=4, required_genes=["G15"],
        )


def test_control_baseline_uses_only_independent_control_counts(
    dataset: ad.AnnData, tmp_path: Path
) -> None:
    split = tmp_path / "split"
    write_split(dataset, split, control=CONTROL, holdout_fraction=0.4, seed=4)
    controls = ad.read_h5ad(split / "input_controls.h5ad")
    controls.X = controls.X.astype(np.float32)
    controls.write_h5ad(split / "input_controls.h5ad")
    output = tmp_path / "control-baseline.h5ad"
    predict_control_baseline(
        split / "reference.h5ad", split / "input_controls.h5ad",
        output, cells_per_target=6, seed=4,
    )
    prediction = ad.read_h5ad(output)
    controls = ad.read_h5ad(split / "input_controls.h5ad")
    reference = ad.read_h5ad(split / "reference.h5ad")
    assert prediction.shape == (12, 16)
    assert set(prediction.obs["target_gene"]) == set(reference.obs["target_gene"]) - {CONTROL}
    available = {tuple(row) for row in controls.X.toarray()}
    assert all(tuple(row) in available for row in prediction.X.toarray())
    assert prediction.X.dtype.kind in "iu"
    source_rows = controls.obs_names.get_indexer(prediction.obs["source_control_id"])
    assert (source_rows >= 0).all()
    assert (prediction.X != controls.X[source_rows]).nnz == 0
    with pytest.raises(ValueError, match="overlap"):
        predict_control_baseline(
            split / "reference.h5ad", split / "reference.h5ad", tmp_path / "leaked.h5ad",
        )


def test_control_sampling_matches_batch_and_requires_explicit_fallback() -> None:
    reference_context = np.array(["c", "c"])
    reference_batch = np.array(["a", "b"])
    labels = np.array(["G1", "G1"])
    control_context = np.array(["c", "c"])
    control_batch = np.array(["a", "b"])
    rows, batches = _sample_controls(
        reference_context, reference_batch, labels, control_context, control_batch,
        "c", "G1", cells_per_target=20, rng=np.random.default_rng(42),
        allow_unmatched_batch=False,
    )
    assert np.array_equal(control_batch[rows], batches)
    with pytest.raises(ValueError, match="No input controls match"):
        _sample_controls(
            np.array(["c"]), np.array(["b"]), np.array(["G1"]),
            np.array(["c"]), np.array(["a"]), "c", "G1",
            cells_per_target=1, rng=np.random.default_rng(42), allow_unmatched_batch=False,
        )


def test_direct_input_control_sampling_ignores_heldout_expression_batch_and_frequency(
    dataset: ad.AnnData, tmp_path: Path,
) -> None:
    training, reference, controls, _ = split_by_target(dataset, holdout_fraction=0.4, seed=7)
    train_path, reference_path, control_path = (
        tmp_path / "train.h5ad", tmp_path / "reference.h5ad", tmp_path / "controls.h5ad",
    )
    training.write_h5ad(train_path)
    reference.write_h5ad(reference_path)
    controls.write_h5ad(control_path)
    checkpoint = train(
        train_path, tmp_path / "model", latent_dim=4, hidden_dim=6,
        pretrain_steps=2, flow_steps=2, batch_size=4, seed=2,
    )
    selected_rows = reference.obs.reset_index().groupby(
        ["context", "target_gene"], observed=True, sort=False,
    ).head(1)["index"]
    altered = reference[selected_rows.tolist()].copy()
    altered.obs["batch"] = "heldout-only-batch"
    altered.X = sp.csr_matrix(np.full(altered.shape, 2, dtype=np.int32))
    altered_path = tmp_path / "altered-reference.h5ad"
    altered.write_h5ad(altered_path)

    predictions, baselines = [], []
    for name, path in (("original", reference_path), ("altered", altered_path)):
        predicted, baseline = tmp_path / f"{name}-pred.h5ad", tmp_path / f"{name}-baseline.h5ad"
        predict_holdout(
            checkpoint, path, control_path, predicted, cells_per_target=6, seed=19, steps=2,
            sampling="residual-rounding", effect_strength=1, control_sampling="input-controls",
        )
        predict_control_baseline(
            path, control_path, baseline, cells_per_target=6, seed=19,
            control_sampling="input-controls",
        )
        predictions.append(ad.read_h5ad(predicted))
        baselines.append(ad.read_h5ad(baseline))
    assert (predictions[0].X != predictions[1].X).nnz == 0
    for prediction, baseline in zip(predictions, baselines):
        assert prediction.obs["source_control_id"].tolist() == baseline.obs["source_control_id"].tolist()
        assert prediction.uns["c3vcell_prediction"]["control_sampling"] == "input-controls"
    assert predictions[0].obs["source_control_id"].tolist() == predictions[1].obs["source_control_id"].tolist()


def test_target_requests_match_holdout_inference_without_reference(
    dataset: ad.AnnData, tmp_path: Path,
) -> None:
    from c3vcell.calibration import OBJECTIVE, training_identity
    from c3vcell.integrity import file_sha256

    training, reference, controls, manifest = split_by_target(dataset, holdout_fraction=0.4, seed=7)
    train_path, reference_path, control_path = (
        tmp_path / "train.h5ad", tmp_path / "reference.h5ad", tmp_path / "controls.h5ad",
    )
    training.write_h5ad(train_path)
    reference.write_h5ad(reference_path)
    controls.write_h5ad(control_path)
    checkpoint_path = train(
        train_path, tmp_path / "model", latent_dim=4, hidden_dim=6,
        pretrain_steps=2, flow_steps=2, batch_size=4, seed=2,
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    calibration = tmp_path / "calibration.json"
    calibration.write_text(json.dumps({
        "method": "target-disjoint-training-folds-v1",
        "objective": OBJECTIVE,
        "sampling": "residual-rounding",
        "control_sampling": "input-controls",
        "prediction_steps": 2,
        "cells_per_target": 6,
        "selected_effect_strength": 0.5,
        "training_identity": training_identity(
            checkpoint["training_cell_ids"], checkpoint["training_targets"],
        ),
        "checkpoint_sha256": file_sha256(checkpoint_path),
    }), encoding="utf-8")
    requests = tmp_path / "requests.csv"
    pd.DataFrame({
        "context": ["HCT116"] * len(manifest["test_targets"]),
        "target_gene": manifest["test_targets"],
    }).to_csv(requests, index=False)
    holdout_path, targets_path = tmp_path / "holdout.h5ad", tmp_path / "targets.h5ad"
    common = dict(
        cells_per_target=6, steps=2, seed=19,
        sampling="residual-rounding", calibration_path=calibration,
    )
    predict_holdout(
        checkpoint_path, reference_path, control_path, holdout_path,
        control_sampling="input-controls", **common,
    )
    predict_targets(checkpoint_path, requests, control_path, targets_path, **common)
    holdout, prediction = ad.read_h5ad(holdout_path), ad.read_h5ad(targets_path)
    assert (holdout.X != prediction.X).nnz == 0
    pd.testing.assert_frame_equal(holdout.obs, prediction.obs)
    assert prediction.uns["c3vcell_prediction"]["request_source_sha256"] == file_sha256(requests)
    assert prediction.uns["c3vcell_prediction"]["input_controls_sha256"] == file_sha256(control_path)
    assert prediction.uns["c3vcell_prediction"]["checkpoint_sha256"] == file_sha256(checkpoint_path)
    assert prediction.uns["c3vcell_prediction"]["calibration_sha256"] == file_sha256(calibration)
    assert prediction.uns["c3vcell_prediction"]["effect_strength"] == 0.5
    with pytest.raises(ValueError, match="cells per target"):
        predict_targets(
            checkpoint_path, requests, control_path, tmp_path / "bad-calibration.h5ad",
            **{**common, "cells_per_target": 5},
        )
    with pytest.raises(FileExistsError):
        predict_targets(checkpoint_path, requests, control_path, targets_path)


def test_target_requests_reject_invalid_or_leaking_inputs(
    dataset: ad.AnnData, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    training, reference, controls, manifest = split_by_target(dataset, holdout_fraction=0.4, seed=7)
    train_path, control_path = tmp_path / "train.h5ad", tmp_path / "controls.h5ad"
    training.write_h5ad(train_path)
    controls.write_h5ad(control_path)
    checkpoint = train(
        train_path, tmp_path / "model", latent_dim=4, hidden_dim=6,
        pretrain_steps=2, flow_steps=2, batch_size=4, seed=2,
    )
    requests = tmp_path / "requests.csv"
    valid_target = manifest["test_targets"][0]
    invalid = [
        ("context,other\nHCT116,G0\n", "context and target_gene"),
        ("context,target_gene\n", "unique, nonempty"),
        (f"context,target_gene\nHCT116,{valid_target}\nHCT116,{valid_target}\n", "unique, nonempty"),
        ("context,target_gene\nHCT116, \n", "unique, nonempty"),
        (f"context,target_gene\nHCT116,{CONTROL}\n", "control label"),
        (f"context,target_gene\nHCT116,{manifest['train_targets'][0]}\n", "appeared during training"),
        ("context,target_gene\nHCT116,MISSING\n", "absent from the model gene axis"),
        (f"context,target_gene\nother,{valid_target}\n", "Missing input controls"),
    ]
    for text, error in invalid:
        requests.write_text(text, encoding="utf-8")
        with pytest.raises(ValueError, match=error):
            predict_targets(checkpoint, requests, control_path, tmp_path / "invalid.h5ad")
    requests.write_text(f"context,target_gene\nHCT116,{valid_target}\n", encoding="utf-8")
    assert _read_target_requests(requests) == [("HCT116", valid_target)]
    with pytest.raises(ValueError, match="overlap"):
        predict_targets(checkpoint, requests, train_path, tmp_path / "leaked.h5ad")
    changed = controls.copy()
    changed.var_names = changed.var_names[::-1]
    changed_path = tmp_path / "changed-axis.h5ad"
    changed.write_h5ad(changed_path)
    with pytest.raises(ValueError, match="gene axes"):
        predict_targets(checkpoint, requests, changed_path, tmp_path / "changed.h5ad")

    from c3vcell.integrity import file_sha256
    hashes = {}

    def drifting_hash(path: Path) -> str:
        hashes[path] = hashes.get(path, 0) + 1
        return "changed" if path == requests and hashes[path] == 2 else file_sha256(path)

    monkeypatch.setattr("c3vcell.predict.file_sha256", drifting_hash)
    with pytest.raises(RuntimeError, match="changed while"):
        predict_targets(checkpoint, requests, control_path, tmp_path / "drift.h5ad")


def test_predict_targets_cli_accepts_metadata_only_request() -> None:
    from c3vcell.cli import build_parser

    args = build_parser().parse_args([
        "predict-targets", "--checkpoint", "model.pt", "--targets", "requests.csv",
        "--controls", "controls.h5ad", "--output", "prediction.h5ad",
        "--sampling", "residual-rounding", "--effect-calibration", "calibration.json",
        "--cells-per-target", "400",
    ])
    assert args.targets == Path("requests.csv")
    assert args.effect_calibration == Path("calibration.json")
    assert args.cells_per_target == 400
    assert not hasattr(args, "reference")


def test_rejects_missing_training_targets(dataset: ad.AnnData) -> None:
    dataset.var_names = [f"OTHER{i}" for i in range(dataset.n_vars)]
    with pytest.raises(ValueError, match="absent"):
        split_by_target(dataset)


def test_official_csv_headers_and_cell_count(tmp_path: Path) -> None:
    genes = [f"G{i}" for i in range(18_533)]
    gene_file = tmp_path / "gene_names.csv"
    pd.DataFrame({"gene_name": genes}).to_csv(gene_file, index=False)
    target_file = tmp_path / "pert_counts.csv"
    pd.DataFrame({"target_gene": genes[:300], "n_cells": 400}).to_csv(target_file, index=False)
    assert _read_gene_list(gene_file) == genes
    assert _read_targets(target_file) == genes[:300]
    pd.DataFrame({"target_gene": genes[:300], "n_cells": 399}).to_csv(target_file, index=False)
    with pytest.raises(ValueError, match="400"):
        _read_targets(target_file)


def test_vcc_export_sparse_fallback_and_control_leakage(
    dataset: ad.AnnData, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.h5ad"
    dataset.write_h5ad(source)
    checkpoint = train(
        source, tmp_path / "model", latent_dim=4, hidden_dim=6,
        pretrain_steps=2, flow_steps=2, batch_size=4, seed=3,
    )
    genes = dataset.var_names.tolist() + ["MISSING"]
    monkeypatch.setattr("c3vcell.predict._read_gene_list", lambda _: genes)
    monkeypatch.setattr("c3vcell.predict._read_targets", lambda _: ["MISSING"])
    monkeypatch.setattr("c3vcell.predict._check_disk_reserve", lambda _: None)
    controls = {}
    for context in ("A", "B", "C"):
        counts = sp.hstack(
            [dataset.X[:4], sp.csr_matrix((4, 1), dtype=np.int32)], format="csr",
        )
        obs = pd.DataFrame(
            {"target_gene": [CONTROL] * 4, "context": [context] * 4},
            index=[f"{context}-independent-{i}" for i in range(4)],
        )
        path = tmp_path / f"controls-{context}.h5ad"
        ad.AnnData(counts, obs=obs, var=pd.DataFrame(index=genes)).write_h5ad(path)
        controls[context] = path
    output = tmp_path / "export.h5ad"
    export_vcc_h5ad(
        checkpoint, controls, tmp_path / "genes.csv", tmp_path / "targets.csv",
        output, allow_missing_target=True, sampling="residual-rounding", seed=3,
    )
    result = ad.read_h5ad(output)
    assert result.shape == (1200, len(genes))
    assert set(result.obs["context"]) == {"A", "B", "C"}
    assert set(result.obs["target_gene"]) == {"MISSING"}
    assert sp.isspmatrix_csr(result.X)
    control_rows = {tuple(row) for row in dataset.X[:4].toarray()}
    assert all(tuple(row[:-1]) in control_rows and row[-1] == 0 for row in result.X.toarray())
    assert output.with_suffix(".h5ad.json").is_file()

    leaked = ad.read_h5ad(controls["A"])
    leaked.obs_names = [dataset.obs_names[0], *leaked.obs_names[1:]]
    leaked.write_h5ad(controls["A"])
    with pytest.raises(ValueError, match="overlap training cells"):
        export_vcc_h5ad(
            checkpoint, controls, tmp_path / "genes.csv", tmp_path / "targets.csv",
            tmp_path / "leaked.h5ad", allow_missing_target=True,
        )
