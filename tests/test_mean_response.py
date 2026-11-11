from pathlib import Path
import json
import sys

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import torch

from c3vcell.cli import build_parser, main
from c3vcell.data import CONTROL, read_h5ad_subset
from c3vcell.mean_response import _conditional_logfold, fit_mean_response, predict_mean_response
from c3vcell.model import C3VCell, ModelConfig
from c3vcell.predict import predict_control_baseline
from c3vcell.training import train


@pytest.fixture
def mean_inputs(tmp_path: Path) -> dict[str, Path]:
    genes = pd.DataFrame(index=["G0", "G1", "G2"])

    def write(name: str, counts: list[list[int]], labels: list[str], prefix: str) -> Path:
        obs = pd.DataFrame(
            {"target_gene": labels, "context": "A", "batch": "b"},
            index=[f"{prefix}-{i}" for i in range(len(labels))],
        )
        path = tmp_path / f"{name}.h5ad"
        ad.AnnData(sp.csr_matrix(counts), obs=obs, var=genes.copy()).write_h5ad(path)
        return path

    train = write(
        "train", [[10, 10, 0]] * 10 + [[15, 5, 0]] * 10 + [[5, 15, 0]] * 2,
        [CONTROL] * 10 + ["G0"] * 10 + ["G1"] * 2, "train",
    )
    reference = write(
        "reference", [[10, 10, 0]] * 4 + [[3, 17, 0]] * 4,
        [CONTROL] * 4 + ["G2"] * 4, "reference",
    )
    controls = write("controls", [[10, 10, 0]] * 6, [CONTROL] * 6, "input")
    return {"train": train, "reference": reference, "controls": controls}


def test_mean_response_weights_targets_equally_and_zero_preserves_controls(
    mean_inputs: dict[str, Path], tmp_path: Path,
) -> None:
    profile = tmp_path / "profile.h5ad"
    fit_mean_response(mean_inputs["train"], profile, min_group_cells=2)
    fitted = ad.read_h5ad(profile)
    np.testing.assert_allclose(fitted.X, 0, atol=1e-6)
    assert fitted.obs["training_targets"].tolist() == [2]
    output = tmp_path / "prediction.h5ad"
    predict_mean_response(
        profile, mean_inputs["reference"], mean_inputs["controls"], output,
        cells_per_target=8, effect_strength=0, seed=3,
    )
    predicted = ad.read_h5ad(output)
    controls = ad.read_h5ad(mean_inputs["controls"])
    source = controls.obs_names.get_indexer(predicted.obs["source_control_id"])
    assert np.all(source >= 0)
    assert (predicted.X != controls.X[source]).nnz == 0
    assert set(predicted.obs["target_gene"]) == {"G2"}
    assert predicted.X.dtype.kind == "i"
    with pytest.raises(FileExistsError):
        fit_mean_response(mean_inputs["train"], profile, min_group_cells=2)
    with pytest.raises(FileExistsError):
        predict_mean_response(profile, mean_inputs["reference"], mean_inputs["controls"], output)


def test_mean_response_uses_the_training_perturbation_effect(
    mean_inputs: dict[str, Path], tmp_path: Path,
) -> None:
    training = ad.read_h5ad(mean_inputs["train"])
    training.X[-2:, :] = sp.csr_matrix([[15, 5, 0], [15, 5, 0]])
    training.write_h5ad(mean_inputs["train"])
    profile = tmp_path / "profile.h5ad"
    fit_mean_response(mean_inputs["train"], profile, min_group_cells=2)
    output = tmp_path / "prediction.h5ad"
    predict_mean_response(
        profile, mean_inputs["reference"], mean_inputs["controls"], output,
        cells_per_target=20, seed=4,
    )
    predicted = ad.read_h5ad(output)
    assert np.asarray(predicted.X[:, 0].mean()).item() > 14
    assert np.asarray(predicted.X[:, 1].mean()).item() < 6
    assert predicted.X[:, 2].nnz == 0


def test_mean_response_fits_and_applies_batch_specific_profiles(
    mean_inputs: dict[str, Path], tmp_path: Path,
) -> None:
    training = ad.read_h5ad(mean_inputs["train"])
    labels = training.obs["target_gene"].astype(str).to_numpy()
    training.X = sp.csr_matrix(
        [[10, 10, 0] if label == CONTROL else [15, 5, 0] for label in labels]
    )
    second = training.copy()
    second.obs_names = [f"second-{name}" for name in second.obs_names]
    second.obs["batch"] = "b2"
    second.X = sp.csr_matrix(
        [[10, 10, 0] if label == CONTROL else [5, 15, 0] for label in labels]
    )
    ad.concat([training, second]).write_h5ad(mean_inputs["train"])
    for key in ("reference", "controls"):
        first = ad.read_h5ad(mean_inputs[key])
        second = first.copy()
        second.obs_names = [f"second-{name}" for name in second.obs_names]
        second.obs["batch"] = "b2"
        ad.concat([first, second]).write_h5ad(mean_inputs[key])
    profile, output = tmp_path / "profile.h5ad", tmp_path / "prediction.h5ad"
    fit_mean_response(mean_inputs["train"], profile, min_group_cells=2)
    predict_mean_response(
        profile, mean_inputs["reference"], mean_inputs["controls"], output,
        cells_per_target=80, seed=7,
    )
    prediction = ad.read_h5ad(output)
    for batch, expect_greater in (("b", True), ("b2", False)):
        subset = prediction[prediction.obs["batch"] == batch]
        assert subset.n_obs > 0
        gene0, gene1 = np.asarray(subset.X.mean(axis=0)).ravel()[:2]
        assert (gene0 > gene1) == expect_greater
        source = ad.read_h5ad(mean_inputs["controls"]).obs.loc[
            subset.obs["source_control_id"], "batch"
        ].astype(str)
        assert set(source) == {batch}


@pytest.mark.parametrize("cells_per_target", [4, 8])
def test_input_control_sampling_ignores_heldout_expression_batch_and_frequency(
    mean_inputs: dict[str, Path], tmp_path: Path, cells_per_target: int,
) -> None:
    controls = ad.read_h5ad(mean_inputs["controls"])
    controls.X = sp.csr_matrix([[10 + i, 10, 0] for i in range(controls.n_obs)])
    controls.write_h5ad(mean_inputs["controls"])
    reference = ad.read_h5ad(mean_inputs["reference"])
    altered = reference[[0, 1, 4]].copy()
    altered.obs["batch"] = "heldout-only-batch"
    altered.X = sp.csr_matrix([[1, 2, 0], [3, 4, 0], [50, 1, 0]])
    altered_path = tmp_path / "altered-reference.h5ad"
    altered.write_h5ad(altered_path)

    profile = tmp_path / "profile.h5ad"
    fit_mean_response(mean_inputs["train"], profile, min_group_cells=2)
    predictions = {}
    for label, reference_path in (("original", mean_inputs["reference"]), ("altered", altered_path)):
        for kind in ("baseline", "mean"):
            output = tmp_path / f"{label}-{kind}.h5ad"
            options = {"cells_per_target": cells_per_target, "seed": 17, "control_sampling": "input-controls"}
            if kind == "baseline":
                predict_control_baseline(reference_path, mean_inputs["controls"], output, **options)
            else:
                predict_mean_response(profile, reference_path, mean_inputs["controls"], output, **options)
            predictions[label, kind] = ad.read_h5ad(output)

    for kind in ("baseline", "mean"):
        first, second = predictions["original", kind], predictions["altered", kind]
        assert first.obs["source_control_id"].tolist() == second.obs["source_control_id"].tolist()
        assert (first.X != second.X).nnz == 0
        assert set(first.obs["batch"]) == {"b"}
    sources = predictions["original", "baseline"].obs["source_control_id"].tolist()
    assert sources == predictions["original", "mean"].obs["source_control_id"].tolist()
    assert (len(set(sources)) == cells_per_target) is (cells_per_target <= controls.n_obs)
    assert predictions["original", "mean"].uns["c3vcell_mean_response_prediction"][
        "control_sampling"
    ] == "input-controls"

    controls.obs["batch"] = "unprofiled-batch"
    unprofiled = tmp_path / "unprofiled-controls.h5ad"
    controls.write_h5ad(unprofiled)
    with pytest.raises(ValueError, match="Missing mean-response profile"):
        predict_mean_response(
            profile, mean_inputs["reference"], unprofiled, tmp_path / "unprofiled.h5ad",
            cells_per_target=cells_per_target, control_sampling="input-controls",
        )


def test_underpopulated_target_is_recorded(
    mean_inputs: dict[str, Path], tmp_path: Path,
) -> None:
    profile = tmp_path / "profile.h5ad"
    fit_mean_response(mean_inputs["train"], profile, min_group_cells=3)
    fitted = ad.read_h5ad(profile)
    assert fitted.obs["training_targets"].tolist() == [1]
    assert list(fitted.uns["c3vcell_mean_response"]["targets_without_eligible_groups"]) == ["G1"]
    assert fitted.X[0, 0] > 0


def test_conditional_logfold_is_zero_at_its_anchor() -> None:
    torch.manual_seed(19)
    model = C3VCell(ModelConfig(n_genes=3, latent_dim=4, hidden_dim=6)).eval()
    counts = torch.tensor([[10.0, 5.0, 1.0], [3.0, 7.0, 2.0]])
    indices = torch.tensor([0, 0])
    anchor = model.target_embedding(torch.tensor([0]))
    same = _conditional_logfold(model, counts, indices, anchor, steps=3, graph=None)
    other = _conditional_logfold(
        model, counts, torch.tensor([1, 1]), anchor, steps=3, graph=None,
    )
    torch.testing.assert_close(same, torch.zeros_like(same), atol=1e-7, rtol=0)
    assert torch.max(torch.abs(other)).item() > 1e-6


def test_shared_residual_requires_matching_training_and_zero_reproduces_mean(
    mean_inputs: dict[str, Path], tmp_path: Path,
) -> None:
    checkpoint = train(
        mean_inputs["train"], tmp_path / "model", latent_dim=4, hidden_dim=6,
        pretrain_steps=2, flow_steps=2, batch_size=4, seed=2,
    )
    profile = tmp_path / "profile.h5ad"
    fit_mean_response(mean_inputs["train"], profile, min_group_cells=2)
    average, zero, hybrid = (tmp_path / f"{name}.h5ad" for name in ("average", "zero", "hybrid"))
    predict_mean_response(
        profile, mean_inputs["reference"], mean_inputs["controls"], average,
        cells_per_target=16, seed=9,
    )
    predict_mean_response(
        profile, mean_inputs["reference"], mean_inputs["controls"], zero,
        cells_per_target=16, seed=9, checkpoint_path=checkpoint, conditional_strength=0,
    )
    predict_mean_response(
        profile, mean_inputs["reference"], mean_inputs["controls"], hybrid,
        cells_per_target=16, seed=9, checkpoint_path=checkpoint, conditional_strength=1,
    )
    original, at_zero, combined = (ad.read_h5ad(path) for path in (average, zero, hybrid))
    assert (original.X != at_zero.X).nnz == 0
    for column in ("target_gene", "context", "batch", "source_control_id"):
        assert original.obs[column].astype(str).tolist() == at_zero.obs[column].astype(str).tolist()
    assert combined.shape == original.shape
    assert combined.X.dtype.kind == "i" and (combined.X.data >= 0).all()
    info = combined.uns["c3vcell_mean_response_prediction"]
    assert info["conditional_strength"] == 1
    assert info["conditional_anchor"] == "mean training-target embedding"

    changed = ad.read_h5ad(mean_inputs["train"])
    changed.obs_names = [f"changed-{name}" for name in changed.obs_names]
    changed_path = tmp_path / "changed-train.h5ad"
    changed.write_h5ad(changed_path)
    changed_checkpoint = train(
        changed_path, tmp_path / "changed-model", latent_dim=4, hidden_dim=6,
        pretrain_steps=2, flow_steps=2, batch_size=4, seed=2,
    )
    with pytest.raises(ValueError, match="training cells differ"):
        predict_mean_response(
            profile, mean_inputs["reference"], mean_inputs["controls"],
            tmp_path / "mismatch.h5ad", checkpoint_path=changed_checkpoint,
            conditional_strength=1,
        )


def test_mean_response_rejects_leakage_and_unmatched_groups(
    mean_inputs: dict[str, Path], tmp_path: Path,
) -> None:
    profile = tmp_path / "profile.h5ad"
    fit_mean_response(mean_inputs["train"], profile, min_group_cells=2)
    reference = ad.read_h5ad(mean_inputs["reference"])
    reference.obs["target_gene"] = reference.obs["target_gene"].astype(str).replace({"G2": "G0"})
    seen = tmp_path / "seen.h5ad"
    reference.write_h5ad(seen)
    with pytest.raises(ValueError, match="appeared during training"):
        predict_mean_response(profile, seen, mean_inputs["controls"], tmp_path / "seen-pred.h5ad")

    controls = ad.read_h5ad(mean_inputs["controls"])
    controls.obs_names = ["train-0", *controls.obs_names[1:]]
    leaked = tmp_path / "leaked.h5ad"
    controls.write_h5ad(leaked)
    with pytest.raises(ValueError, match="overlap"):
        predict_mean_response(
            profile, mean_inputs["reference"], leaked, tmp_path / "leaked-pred.h5ad",
        )

    reference = ad.read_h5ad(mean_inputs["reference"])
    controls = ad.read_h5ad(mean_inputs["controls"])
    reference.obs["batch"] = "new"
    controls.obs["batch"] = "new"
    ref_new, ctrl_new = tmp_path / "ref-new.h5ad", tmp_path / "ctrl-new.h5ad"
    reference.write_h5ad(ref_new)
    controls.write_h5ad(ctrl_new)
    with pytest.raises(ValueError, match="Missing mean-response profile"):
        predict_mean_response(profile, ref_new, ctrl_new, tmp_path / "missing.h5ad")
    output = tmp_path / "fallback.h5ad"
    predict_mean_response(
        profile, ref_new, ctrl_new, output, cells_per_target=5, allow_unmatched_batch=True,
    )
    assert list(ad.read_h5ad(output).uns["c3vcell_mean_response_prediction"]["profile_fallback_groups"]) == ["A/new"]


def test_prepare_cli_accepts_explicit_holdout_targets() -> None:
    args = build_parser().parse_args([
        "prepare", "--input", "source.h5ad", "--outdir", "split",
        "--include-targets", "G0", "G1", "G2", "--holdout-targets", "G1", "G2",
    ])
    assert args.include_targets == ["G0", "G1", "G2"]
    assert args.holdout_targets == ["G1", "G2"]


def test_shared_residual_cli_defaults_to_no_conditional_effect() -> None:
    args = build_parser().parse_args([
        "predict-shared-residual", "--profile", "profile.h5ad", "--checkpoint", "model.pt",
        "--reference", "reference.h5ad", "--controls", "controls.h5ad",
        "--output", "prediction.h5ad",
    ])
    assert args.conditional_strength is None
    assert args.conditional_calibration is None


def test_prepare_cli_uses_explicit_holdout_and_rejects_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    labels = [CONTROL] * 12 + sum(([f"G{i}"] * 4 for i in range(4)), [])
    obs = pd.DataFrame(
        {"target_gene": labels, "context": "A", "batch": "b"},
        index=[f"cell-{i}" for i in range(len(labels))],
    )
    dataset = ad.AnnData(
        sp.csr_matrix(np.ones((len(labels), 4), dtype=np.int32)),
        obs=obs, var=pd.DataFrame(index=[f"G{i}" for i in range(4)]),
    )
    source, split = tmp_path / "source.h5ad", tmp_path / "split"
    dataset.write_h5ad(source)

    def run(outdir: Path, targets: list[str]) -> None:
        monkeypatch.setattr(sys, "argv", [
            "c3-vcell", "prepare", "--input", str(source), "--outdir", str(outdir),
            "--include-targets", "G0", "G1", "G2", "--holdout-targets", *targets,
        ])
        main()

    run(split, ["G1", "G2"])
    manifest = json.loads((split / "split.json").read_text())
    assert manifest["test_targets"] == ["G1", "G2"]
    assert manifest["train_targets"] == ["G0"]
    with pytest.raises(ValueError, match="Holdout targets must be present"):
        run(tmp_path / "missing", ["G1", "MISSING"])
    with pytest.raises(ValueError, match="unique"):
        read_h5ad_subset(
            source, label_column="target_gene", control=CONTROL, max_per_target=4,
            max_controls=12, max_targets=0, seed=0, include_targets=["G0", "G0"],
        )
    with pytest.raises(ValueError, match="absent from the source"):
        read_h5ad_subset(
            source, label_column="target_gene", control=CONTROL, max_per_target=4,
            max_controls=12, max_targets=0, seed=0, include_targets=["G0", "MISSING"],
        )
