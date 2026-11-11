import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import torch

from c3vcell.calibration import CALIBRATION_OBJECTIVES, OBJECTIVE
from c3vcell.cli import build_parser
from c3vcell.data import CONTROL, write_split
from c3vcell.diagnostics import diagnose_checkpoint
from c3vcell.integrity import file_sha256
from c3vcell.mean_response import fit_mean_response, predict_mean_response
from c3vcell.predict import predict_control_baseline
from c3vcell.shared_calibration import _select, _select_with_null, calibrate_shared_residual
from c3vcell.training import train


@pytest.fixture
def shared_inputs(tmp_path: Path) -> dict[str, Path]:
    rng = np.random.default_rng(42)
    labels = [CONTROL] * 60 + sum(([f"G{i}"] * 8 for i in range(8)), [])
    data = ad.AnnData(
        sp.csr_matrix(rng.poisson(3, size=(len(labels), 10)).astype(np.int32) + 1),
        obs=pd.DataFrame(
            {"target_gene": labels, "context": "c", "batch": "b"},
            index=[f"cell-{i}" for i in range(len(labels))],
        ),
        var=pd.DataFrame(index=[f"G{i}" for i in range(10)]),
    )
    split = tmp_path / "outer"
    write_split(
        data, split, control=CONTROL, holdout_fraction=0.25, seed=4,
        holdout_targets=["G6", "G7"],
    )
    source = split / "train.h5ad"
    checkpoint = train(
        source, tmp_path / "model", latent_dim=4, hidden_dim=6,
        pretrain_steps=2, flow_steps=2, batch_size=4, variance_weight=2,
        covariance_weight=0.25, regularization_view="unmasked", seed=3,
    )
    profile = tmp_path / "profile.h5ad"
    fit_mean_response(source, profile)
    return {
        "source": source, "checkpoint": checkpoint, "profile": profile,
        "reference": split / "reference.h5ad", "controls": split / "input_controls.h5ad",
    }


def test_shared_selection_requires_consistent_fold_improvement() -> None:
    candidates = [0.0, 0.5, 1.0]
    assert _select(candidates, [[1.0, 0.8, 0.9], [1.0, 0.9, 0.95]], 0.02)[0] == 0.5
    assert _select(candidates, [[1.0, 0.5, 0.9], [1.0, 1.01, 1.0]], 0.02)[0] == 0
    assert _select(candidates, [[1.0, 0.99, 1.0], [1.0, 0.99, 1.0]], 0.02)[0] == 0


def test_null_selection_requires_consistent_improvement() -> None:
    candidates = [0.0, 0.5]
    values = [[1.0, 0.8], [1.0, 0.9]]
    assert _select_with_null(candidates, values, [1.1, 1.2], 0.02, 1.0)[:2] == (1.0, 0.5)
    assert _select_with_null(candidates, values, [0.7, 1.2], 0.02, 1.0)[:2] == (0.0, 0.0)
    assert _select_with_null(candidates, values, [0.8, 0.9], 0.02, 1.0)[:2] == (0.0, 0.0)
    with pytest.raises(ValueError, match="finite no-response objective"):
        _select_with_null(candidates, values, [1.0, float("nan")], 0.02, 1.0)


def test_null_calibration_cli_is_opt_in() -> None:
    required = [
        "calibrate-shared-residual", "--source", "train.h5ad", "--checkpoint", "model.pt",
        "--profile", "profile.h5ad", "--outdir", "calibration",
    ]
    assert not build_parser().parse_args(required).include_null_control
    assert build_parser().parse_args([*required, "--include-null-control"]).include_null_control
    assert build_parser().parse_args(required).objective == OBJECTIVE
    assert build_parser().parse_args([
        *required, "--objective", "expr_mse_unbiased_capped",
    ]).objective == "expr_mse_unbiased_capped"


@pytest.mark.parametrize(
    ("shared_loss", "null_loss", "expected_shared", "expected_conditional", "control_sampling"),
    [(1.2, 1.0, 0.0, 0.0, "input-controls"), (0.8, 1.3, 1.0, 0.5, "reference-batch")],
)
@pytest.mark.parametrize("objective", CALIBRATION_OBJECTIVES)
def test_optional_null_calibration_is_nested_and_applied(
    shared_inputs: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    shared_loss: float, null_loss: float, expected_shared: float, expected_conditional: float,
    control_sampling: str, objective: str,
) -> None:
    def fake_score(prediction_path: Path, *_args, **kwargs) -> dict:
        assert kwargs["metrics"] == (objective,)
        if prediction_path.name == "null-control.h5ad":
            value = null_loss
        else:
            strength = ad.read_h5ad(prediction_path).uns["c3vcell_mean_response_prediction"][
                "conditional_strength"
            ]
            value = shared_loss + (strength - 0.5) ** 2
        return {"raw_metrics": {"c": {objective: value}}}

    monkeypatch.setattr("c3vcell.shared_calibration.score_local", fake_score)
    directory = tmp_path / "null-calibration"
    calibration = calibrate_shared_residual(
        shared_inputs["source"], shared_inputs["checkpoint"], shared_inputs["profile"],
        directory, folds=2, strengths=(0, 0.5, 1), include_no_response=True,
        cells_per_target=4, prediction_steps=2, control_sampling=control_sampling,
        objective=objective,
    )
    record = json.loads(calibration.read_text(encoding="utf-8"))
    assert record["format_version"] == 2
    assert record["selected_shared_strength"] == expected_shared
    assert record["selected_conditional_strength"] == expected_conditional
    assert record["mean_null_objective"] == null_loss
    assert record["control_sampling"] == control_sampling
    assert record["objective"] == objective
    if objective != OBJECTIVE:
        assert "not a scored challenge metric" in record["objective_note"]
    for fold in range(2):
        fold_dir = directory / f"fold-{fold:02d}"
        null = ad.read_h5ad(fold_dir / "null-control.h5ad")
        shared = ad.read_h5ad(fold_dir / "strength-00.h5ad")
        assert null.obs["source_control_id"].tolist() == shared.obs["source_control_id"].tolist()
    output = tmp_path / "calibrated-null.h5ad"
    predict_mean_response(
        shared_inputs["profile"], shared_inputs["reference"], shared_inputs["controls"], output,
        checkpoint_path=shared_inputs["checkpoint"], conditional_calibration_path=calibration,
        cells_per_target=4, steps=2, seed=7, control_sampling=control_sampling,
    )
    predicted = ad.read_h5ad(output)
    info = predicted.uns["c3vcell_mean_response_prediction"]
    assert info["effect_strength"] == expected_shared
    assert info["conditional_strength"] == expected_conditional
    if expected_shared == 0:
        assert info["note"].startswith("Unmodified input controls")
        baseline_path = tmp_path / "control-baseline.h5ad"
        predict_control_baseline(
            shared_inputs["reference"], shared_inputs["controls"], baseline_path,
            cells_per_target=4, seed=7, control_sampling=control_sampling,
        )
        baseline = ad.read_h5ad(baseline_path)
        assert (predicted.X != baseline.X).nnz == 0
        assert predicted.obs["source_control_id"].tolist() == baseline.obs["source_control_id"].tolist()
        for key in ("reference", "controls"):
            altered = ad.read_h5ad(shared_inputs[key])
            altered.obs["batch"] = "new-batch"
            altered.write_h5ad(tmp_path / f"new-{key}.h5ad")
        predict_mean_response(
            shared_inputs["profile"], tmp_path / "new-reference.h5ad",
            tmp_path / "new-controls.h5ad", tmp_path / "new-batch-null.h5ad",
            checkpoint_path=shared_inputs["checkpoint"], conditional_calibration_path=calibration,
            cells_per_target=4, steps=2, control_sampling=control_sampling,
        )
    with pytest.raises(ValueError, match="prediction settings differ"):
        predict_mean_response(
            shared_inputs["profile"], shared_inputs["reference"], shared_inputs["controls"],
            tmp_path / "wrong-shared-strength.h5ad", checkpoint_path=shared_inputs["checkpoint"],
            conditional_calibration_path=calibration, effect_strength=0.5, cells_per_target=4, steps=2,
            control_sampling=control_sampling,
        )
    with pytest.raises(ValueError, match="prediction settings differ"):
        predict_mean_response(
            shared_inputs["profile"], shared_inputs["reference"], shared_inputs["controls"],
            tmp_path / "wrong-sampling.h5ad", checkpoint_path=shared_inputs["checkpoint"],
            conditional_calibration_path=calibration, cells_per_target=4, steps=2,
            control_sampling=("reference-batch" if control_sampling == "input-controls" else "input-controls"),
        )
    record["selected_shared_strength"] = 0.5
    tampered = tmp_path / "tampered-null.json"
    tampered.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="selection is inconsistent"):
        predict_mean_response(
            shared_inputs["profile"], shared_inputs["reference"], shared_inputs["controls"],
            tmp_path / "tampered-null.h5ad", checkpoint_path=shared_inputs["checkpoint"],
            conditional_calibration_path=tampered, cells_per_target=4, steps=2,
            control_sampling=control_sampling,
        )


def test_undefined_null_objective_fails_without_selection(
    shared_inputs: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "c3vcell.shared_calibration.score_local",
        lambda prediction, *_args, **_kwargs: {
            "raw_metrics": {"c": {OBJECTIVE: None if prediction.name == "null-control.h5ad" else 1.0}}
        },
    )
    directory = tmp_path / "undefined-null"
    with pytest.raises(ValueError, match="Undefined no-response calibration objective"):
        calibrate_shared_residual(
            shared_inputs["source"], shared_inputs["checkpoint"], shared_inputs["profile"],
            directory, folds=2, strengths=(0, 1), include_no_response=True,
            cells_per_target=4, prediction_steps=2,
        )
    assert (directory / "calibration-error.json").exists()
    assert not (directory / "shared-residual-calibration.json").exists()


def test_shared_calibration_is_nested_and_bound_to_artifacts(
    shared_inputs: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_score(prediction_path: Path, *_args, **_kwargs) -> dict:
        strength = ad.read_h5ad(prediction_path).uns["c3vcell_mean_response_prediction"][
            "conditional_strength"
        ]
        return {"raw_metrics": {"c": {OBJECTIVE: 1 + (strength - 0.5) ** 2}}}

    monkeypatch.setattr("c3vcell.shared_calibration.score_local", fake_score)
    directory = tmp_path / "calibration"
    calibration = calibrate_shared_residual(
        shared_inputs["source"], shared_inputs["checkpoint"], shared_inputs["profile"],
        directory, folds=2, strengths=(0, 0.5, 1), cells_per_target=4, prediction_steps=2,
    )
    record = json.loads(calibration.read_text(encoding="utf-8"))
    plan = json.loads((directory / "calibration-plan.json").read_text(encoding="utf-8"))
    checkpoint = torch.load(shared_inputs["checkpoint"], map_location="cpu", weights_only=True)
    assert checkpoint["training_source_sha256"] == file_sha256(shared_inputs["source"])
    assert record["selected_conditional_strength"] == 0.5
    assert plan["fold_targets"] == [fold["heldout_targets"] for fold in record["folds"]]
    assert set(sum(plan["fold_targets"], [])) == set(checkpoint["training_targets"])
    assert set(plan["fold_targets"][0]).isdisjoint(plan["fold_targets"][1])
    assert not set(sum(plan["fold_targets"], [])).intersection({"G6", "G7"})
    for fold in range(2):
        fold_dir = directory / f"fold-{fold:02d}"
        manifest = json.loads((fold_dir / "split" / "split.json").read_text(encoding="utf-8"))
        fold_checkpoint = torch.load(fold_dir / "model" / "model.pt", map_location="cpu", weights_only=True)
        assert fold_checkpoint["model_config"]["variance_weight"] == 2
        assert fold_checkpoint["model_config"]["covariance_weight"] == 0.25
        assert fold_checkpoint["model_config"]["regularization_view"] == "unmasked"
        assert "relative_mean_std" in fold_checkpoint["representation_diagnostics"]
        assert set(manifest["train_targets"]).isdisjoint(manifest["test_targets"])
        assert set(manifest["train_cell_ids"]).isdisjoint(manifest["test_cell_ids"])
        assert set(manifest["train_cell_ids"]).isdisjoint(manifest["input_control_cell_ids"])
        assert set(manifest["test_cell_ids"]).isdisjoint(manifest["input_control_cell_ids"])
        predictions = [ad.read_h5ad(fold_dir / f"strength-{i:02d}.h5ad") for i in range(3)]
        for prediction in predictions[1:]:
            assert prediction.obs["source_control_id"].tolist() == predictions[0].obs[
                "source_control_id"
            ].tolist()
    output = tmp_path / "calibrated.h5ad"
    predict_mean_response(
        shared_inputs["profile"], shared_inputs["reference"], shared_inputs["controls"], output,
        checkpoint_path=shared_inputs["checkpoint"], conditional_calibration_path=calibration,
        cells_per_target=4, steps=2,
    )
    info = ad.read_h5ad(output).uns["c3vcell_mean_response_prediction"]
    assert info["conditional_strength"] == 0.5
    assert info["conditional_calibration_sha256"] == file_sha256(calibration)
    legacy_record = dict(record)
    del legacy_record["control_sampling"]
    legacy_calibration = tmp_path / "legacy-calibration.json"
    legacy_calibration.write_text(json.dumps(legacy_record), encoding="utf-8")
    predict_mean_response(
        shared_inputs["profile"], shared_inputs["reference"], shared_inputs["controls"],
        tmp_path / "legacy-calibration-prediction.h5ad", checkpoint_path=shared_inputs["checkpoint"],
        conditional_calibration_path=legacy_calibration, cells_per_target=4, steps=2,
    )
    with pytest.raises(ValueError, match="prediction settings differ"):
        predict_mean_response(
            shared_inputs["profile"], shared_inputs["reference"], shared_inputs["controls"],
            tmp_path / "legacy-wrong-sampling.h5ad", checkpoint_path=shared_inputs["checkpoint"],
            conditional_calibration_path=legacy_calibration, cells_per_target=4, steps=2,
            control_sampling="input-controls",
        )
    with pytest.raises(ValueError, match="prediction settings differ"):
        predict_mean_response(
            shared_inputs["profile"], shared_inputs["reference"], shared_inputs["controls"],
            tmp_path / "wrong-settings.h5ad", checkpoint_path=shared_inputs["checkpoint"],
            conditional_calibration_path=calibration, cells_per_target=5, steps=2,
        )
    record["selected_conditional_strength"] = 1.0
    tampered = tmp_path / "tampered.json"
    tampered.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="selection is inconsistent"):
        predict_mean_response(
            shared_inputs["profile"], shared_inputs["reference"], shared_inputs["controls"],
            tmp_path / "tampered.h5ad", checkpoint_path=shared_inputs["checkpoint"],
            conditional_calibration_path=tampered, cells_per_target=4, steps=2,
        )


def test_shared_calibration_rejects_changed_counts_and_legacy_checkpoints(
    shared_inputs: dict[str, Path], tmp_path: Path,
) -> None:
    altered = ad.read_h5ad(shared_inputs["source"])
    altered.X.data[0] += 1
    changed = tmp_path / "changed.h5ad"
    altered.write_h5ad(changed)
    changed_profile = tmp_path / "changed-profile.h5ad"
    fit_mean_response(changed, changed_profile)
    diagnostic_path = tmp_path / "diagnostics.json"
    diagnose_checkpoint(
        shared_inputs["checkpoint"], shared_inputs["source"], diagnostic_path,
        max_cells=16, batch_size=4, seed=3,
    )
    diagnostic = json.loads(diagnostic_path.read_text(encoding="utf-8"))
    assert diagnostic["checkpoint_has_source_fingerprint"] is True
    assert diagnostic["sampled_cells"] == 16
    with pytest.raises(ValueError, match="training-count file"):
        diagnose_checkpoint(shared_inputs["checkpoint"], changed, tmp_path / "bad-diagnostics.json")
    with pytest.raises(ValueError, match="training count files differ"):
        predict_mean_response(
            changed_profile, shared_inputs["reference"], shared_inputs["controls"],
            tmp_path / "changed-prediction.h5ad", checkpoint_path=shared_inputs["checkpoint"],
        )
    with pytest.raises(ValueError, match="training-count file"):
        calibrate_shared_residual(
            changed, shared_inputs["checkpoint"], changed_profile, tmp_path / "bad-source", folds=2,
        )
    checkpoint = torch.load(shared_inputs["checkpoint"], map_location="cpu", weights_only=True)
    del checkpoint["training_source_sha256"]
    legacy = tmp_path / "legacy.pt"
    torch.save(checkpoint, legacy)
    with pytest.raises(ValueError, match="Retrain"):
        calibrate_shared_residual(
            shared_inputs["source"], legacy, shared_inputs["profile"], tmp_path / "legacy", folds=2,
        )
    with pytest.warns(RuntimeWarning, match="matching cell IDs"):
        predict_mean_response(
            shared_inputs["profile"], shared_inputs["reference"], shared_inputs["controls"],
            tmp_path / "legacy-prediction.h5ad", checkpoint_path=legacy, cells_per_target=2,
        )


def test_undefined_objective_fails_without_selecting_strength(
    shared_inputs: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "c3vcell.shared_calibration.score_local",
        lambda *_args, **_kwargs: {"raw_metrics": {"c": {OBJECTIVE: None}}},
    )
    directory = tmp_path / "undefined"
    with pytest.raises(ValueError, match="Undefined calibration objective"):
        calibrate_shared_residual(
            shared_inputs["source"], shared_inputs["checkpoint"], shared_inputs["profile"],
            directory, folds=2, strengths=(0, 1), cells_per_target=4, prediction_steps=2,
        )
    assert (directory / "calibration-plan.json").exists()
    assert (directory / "calibration-error.json").exists()
    assert not (directory / "shared-residual-calibration.json").exists()


def test_legacy_variance_weight_is_inherited(
    shared_inputs: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = torch.load(shared_inputs["checkpoint"], map_location="cpu", weights_only=True)
    del checkpoint["model_config"]["variance_weight"]
    del checkpoint["training_parameters"]["variance_weight"]
    del checkpoint["model_config"]["covariance_weight"]
    del checkpoint["training_parameters"]["covariance_weight"]
    del checkpoint["model_config"]["regularization_view"]
    del checkpoint["training_parameters"]["regularization_view"]
    legacy = tmp_path / "legacy-config.pt"
    torch.save(checkpoint, legacy)
    monkeypatch.setattr(
        "c3vcell.shared_calibration.score_local",
        lambda *_args, **_kwargs: {"raw_metrics": {"c": {OBJECTIVE: 1.0}}},
    )
    directory = tmp_path / "legacy-calibration"
    calibrate_shared_residual(
        shared_inputs["source"], legacy, shared_inputs["profile"], directory,
        folds=2, strengths=(0, 1), cells_per_target=4, prediction_steps=2,
    )
    for fold in range(2):
        inner = torch.load(
            directory / f"fold-{fold:02d}" / "model" / "model.pt",
            map_location="cpu", weights_only=True,
        )
        assert inner["model_config"]["variance_weight"] == 0.1
        assert inner["model_config"]["covariance_weight"] == 0.0
        assert inner["model_config"]["regularization_view"] == "masked"
