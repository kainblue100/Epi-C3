import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import torch

from c3vcell.data import CONTROL, split_by_target
from c3vcell.ensemble import _check_alignment, _check_members, predict_ensemble
from c3vcell.training import train


@pytest.fixture
def ensemble_inputs(tmp_path: Path) -> tuple[list[Path], Path, Path]:
    rng = np.random.default_rng(42)
    labels = [CONTROL] * 30 + sum(([f"G{i}"] * 12 for i in range(4)), [])
    counts = rng.poisson(3, size=(len(labels), 16)).astype(np.int32)
    counts[:, 15] += 1
    data = ad.AnnData(
        sp.csr_matrix(counts),
        obs=pd.DataFrame(
            {"target_gene": labels, "context": "HCT116", "batch": "batch1"},
            index=[f"cell-{i}" for i in range(len(labels))],
        ),
        var=pd.DataFrame(index=[f"G{i}" for i in range(16)]),
    )
    training, reference, controls, _ = split_by_target(data, holdout_fraction=0.4, seed=7)
    train_path = tmp_path / "train.h5ad"
    reference_path = tmp_path / "reference.h5ad"
    controls_path = tmp_path / "controls.h5ad"
    training.write_h5ad(train_path)
    reference.write_h5ad(reference_path)
    controls.write_h5ad(controls_path)
    checkpoints = [
        train(
            train_path, tmp_path / f"model-{seed}", latent_dim=4, hidden_dim=6,
            pretrain_steps=2, flow_steps=2, batch_size=4, seed=seed,
        )
        for seed in (2, 3)
    ]
    return checkpoints, reference_path, controls_path


def test_ensemble_balanced_mixture_and_pseudobulk_disagreement(
    ensemble_inputs: tuple[list[Path], Path, Path], tmp_path: Path,
) -> None:
    checkpoints, reference, controls = ensemble_inputs
    prediction = tmp_path / "mixture.h5ad"
    disagreement = tmp_path / "disagreement.h5ad"
    predict_ensemble(
        checkpoints, reference, controls, prediction, disagreement,
        cells_per_target=7, seed=5, steps=2,
    )
    result = ad.read_h5ad(prediction)
    summary = ad.read_h5ad(disagreement)
    assert result.n_obs == 14 and result.obs_names.is_unique
    for _, group in result.obs.groupby(["context", "target_gene"], observed=True):
        sizes = group["ensemble_member"].value_counts()
        assert sorted(sizes.tolist()) == [3, 4]
    means = np.stack(
        [summary.layers["member_0_mean"], summary.layers["member_1_mean"]], axis=1,
    )
    np.testing.assert_allclose(summary.X, means.mean(axis=1), rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(
        summary.layers["between_model_std"], means.std(axis=1, ddof=1),
        rtol=1e-6, atol=1e-7,
    )
    manifest = json.loads(prediction.with_suffix(".h5ad.ensemble.json").read_text())
    assert manifest["disagreement_is_calibrated"] is False
    assert [member["training_seed"] for member in manifest["checkpoints"]] == [2, 3]
    with pytest.raises(FileExistsError, match="overwrite"):
        predict_ensemble(checkpoints, reference, controls, prediction, disagreement)


def test_zero_effect_uses_identical_controls_across_models(
    ensemble_inputs: tuple[list[Path], Path, Path], tmp_path: Path,
) -> None:
    checkpoints, reference, controls_path = ensemble_inputs
    prediction = tmp_path / "zero.h5ad"
    disagreement = tmp_path / "zero-disagreement.h5ad"
    predict_ensemble(
        checkpoints, reference, controls_path, prediction, disagreement,
        cells_per_target=6, seed=6, steps=2, effect_strength=0,
    )
    result = ad.read_h5ad(prediction)
    controls = ad.read_h5ad(controls_path)
    expected_rows = controls.obs_names.get_indexer(result.obs["source_control_id"])
    assert np.all(expected_rows >= 0)
    assert (result.X != controls.X[expected_rows]).nnz == 0
    summary = ad.read_h5ad(disagreement)
    assert np.count_nonzero(summary.layers["between_model_std"]) == 0


def test_ensemble_input_control_sampling_ignores_heldout_batch_and_frequency(
    ensemble_inputs: tuple[list[Path], Path, Path], tmp_path: Path,
) -> None:
    checkpoints, reference_path, controls = ensemble_inputs
    reference = ad.read_h5ad(reference_path)
    selected = reference.obs.reset_index().groupby(
        ["context", "target_gene"], observed=True, sort=False,
    ).head(1)["index"]
    altered = reference[selected.tolist()].copy()
    altered.obs["batch"] = "heldout-only-batch"
    altered.X = sp.csr_matrix(np.full(altered.shape, 2, dtype=np.int32))
    altered_path = tmp_path / "altered-reference.h5ad"
    altered.write_h5ad(altered_path)
    outputs = []
    for label, path in (("original", reference_path), ("altered", altered_path)):
        output, disagreement = tmp_path / f"{label}.h5ad", tmp_path / f"{label}-disagreement.h5ad"
        predict_ensemble(
            checkpoints, path, controls, output, disagreement, cells_per_target=6, seed=8,
            steps=2, control_sampling="input-controls",
        )
        result = ad.read_h5ad(output)
        summary = ad.read_h5ad(disagreement)
        manifest = json.loads(output.with_suffix(".h5ad.ensemble.json").read_text())
        assert result.uns["c3vcell_ensemble"]["control_sampling"] == "input-controls"
        assert manifest["control_sampling"] == "input-controls"
        outputs.append((result, summary))
    assert outputs[0][0].obs.equals(outputs[1][0].obs)
    assert (outputs[0][0].X != outputs[1][0].X).nnz == 0
    np.testing.assert_array_equal(outputs[0][1].X, outputs[1][1].X)
    np.testing.assert_array_equal(
        outputs[0][1].layers["between_model_std"], outputs[1][1].layers["between_model_std"],
    )


def test_ensemble_rejects_duplicate_or_mismatched_training_provenance(
    ensemble_inputs: tuple[list[Path], Path, Path], tmp_path: Path,
) -> None:
    checkpoints, reference, controls = ensemble_inputs
    with pytest.raises(ValueError, match="distinct"):
        predict_ensemble(
            [checkpoints[0], checkpoints[0]], reference, controls,
            tmp_path / "duplicate.h5ad", tmp_path / "duplicate-summary.h5ad",
        )
    changed = torch.load(checkpoints[1], map_location="cpu", weights_only=True)
    changed["training_cell_ids"][0] = "different-training-cell"
    altered = tmp_path / "altered.pt"
    torch.save(changed, altered)
    with pytest.raises(ValueError, match="same genes, training cells"):
        predict_ensemble(
            [checkpoints[0], altered], reference, controls,
            tmp_path / "altered.h5ad", tmp_path / "altered-summary.h5ad",
        )
    changed = torch.load(checkpoints[1], map_location="cpu", weights_only=True)
    changed["training_source_sha256"] = "different-counts"
    altered_counts = tmp_path / "altered-counts.pt"
    torch.save(changed, altered_counts)
    with pytest.raises(ValueError, match="training counts"):
        _check_members([checkpoints[0], altered_counts])
    legacy = []
    for index, path in enumerate(checkpoints):
        state = torch.load(path, map_location="cpu", weights_only=True)
        del state["training_source_sha256"]
        old = tmp_path / f"legacy-{index}.pt"
        torch.save(state, old)
        legacy.append(old)
    with pytest.warns(RuntimeWarning, match="matching cell IDs"):
        _check_members(legacy)
    with pytest.raises(ValueError, match="training counts"):
        _check_members([checkpoints[0], legacy[1]])
    compatible = torch.load(checkpoints[1], map_location="cpu", weights_only=True)
    for key in ("variance_weight", "covariance_weight", "regularization_view"):
        del compatible["model_config"][key]
        del compatible["training_parameters"][key]
    legacy_defaults = tmp_path / "legacy-defaults.pt"
    torch.save(compatible, legacy_defaults)
    _check_members([checkpoints[0], legacy_defaults])
    compatible["model_config"]["covariance_weight"] = 0.1
    incompatible = tmp_path / "different-covariance.pt"
    torch.save(compatible, incompatible)
    with pytest.raises(ValueError, match="model configuration"):
        _check_members([checkpoints[0], incompatible])


def test_alignment_detects_different_control_identity() -> None:
    first = ad.AnnData(
        sp.csr_matrix([[1]]),
        obs=pd.DataFrame(
            {
                "context": ["A"], "target_gene": ["G0"], "batch": ["b"],
                "source_control_id": ["control-1"],
            },
            index=["pred-0"],
        ),
        var=pd.DataFrame(index=["G0"]),
    )
    second = first.copy()
    second.obs["source_control_id"] = "control-2"
    with pytest.raises(ValueError, match="identical cells and controls"):
        _check_alignment([first, second])
