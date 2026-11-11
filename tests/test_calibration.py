import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import torch

from c3vcell.calibration import (
    CALIBRATION_OBJECTIVES, OBJECTIVE, _select_effect_strength, calibrate_effect, training_identity,
)
from c3vcell.cli import build_parser
from c3vcell.data import CONTROL
from c3vcell.integrity import file_sha256
from c3vcell.predict import _resolve_effect_strength, export_vcc_h5ad
from c3vcell.training import train


def test_training_identity_is_order_independent_and_rejects_duplicates() -> None:
    assert training_identity(["c2", "c1"], ["G1", "G0"]) == training_identity(
        ["c1", "c2"], ["G0", "G1"]
    )
    with pytest.raises(ValueError, match="unique"):
        training_identity(["c1", "c1"], ["G1"])


def test_strict_effect_selection_chooses_best_feasible_strength() -> None:
    candidates = [0.0, 0.25, 0.5]
    folds = [[1.0, 0.8, 0.5], [1.0, 0.9, 1.01]]
    assert _select_effect_strength(candidates, folds, 0.02, False)[0] == 0.5
    selected, means = _select_effect_strength(candidates, folds, 0.02, True)
    assert selected == 0.25
    assert means == pytest.approx([1.0, 0.85, 0.755])
    assert _select_effect_strength(candidates, [[1, 0.99, 1], [1, 0.99, 1]], 0.02, True)[0] == 0
    with pytest.raises(ValueError, match="finite scores"):
        _select_effect_strength(candidates, [[1, float("nan"), 1], [1, 1, 1]], 0.02, True)


def test_genetic_calibration_cli_accepts_input_control_sampling(tmp_path: Path) -> None:
    args = build_parser().parse_args([
        "calibrate-effect", "--source", "train.h5ad", "--checkpoint", "model.pt",
        "--outdir", "calibration", "--control-sampling", "input-controls",
    ])
    assert args.control_sampling == "input-controls"
    assert args.objective == OBJECTIVE
    alternate = build_parser().parse_args([
        "calibrate-effect", "--source", "train.h5ad", "--checkpoint", "model.pt",
        "--outdir", "calibration", "--objective", "expr_mse_unbiased_capped",
    ])
    assert alternate.objective == "expr_mse_unbiased_capped"
    strict = build_parser().parse_args([
        "calibrate-effect", "--source", "train.h5ad", "--checkpoint", "model.pt",
        "--outdir", "calibration", "--require-every-fold-improvement",
    ])
    assert strict.require_every_fold_improvement
    with pytest.raises(ValueError, match="Drug calibration currently requires"):
        calibrate_effect(
            tmp_path / "unused.h5ad", tmp_path / "unused.pt", tmp_path / "out",
            kind="drug", control_sampling="input-controls",
        )


@pytest.mark.parametrize("control_sampling", ["reference-batch", "input-controls"])
@pytest.mark.parametrize("objective", CALIBRATION_OBJECTIVES)
def test_effect_calibration_uses_disjoint_training_folds_and_binds_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, control_sampling: str, objective: str,
) -> None:
    rng = np.random.default_rng(42)
    labels = [CONTROL] * 30 + sum(([f"G{i}"] * 8 for i in range(4)), [])
    matrix = rng.poisson(3, size=(len(labels), 10)).astype(np.int32) + 1
    data = ad.AnnData(
        sp.csr_matrix(matrix),
        obs=pd.DataFrame(
            {"target_gene": labels, "context": "c", "batch": "b"},
            index=[f"cell-{i}" for i in range(len(labels))],
        ),
        var=pd.DataFrame(index=[f"G{i}" for i in range(10)]),
    )
    source = tmp_path / "source.h5ad"
    data.write_h5ad(source)
    checkpoint_path = train(
        source, tmp_path / "model", latent_dim=4, hidden_dim=6,
        pretrain_steps=2, flow_steps=2, batch_size=4, variance_weight=0.7,
        covariance_weight=0.2, regularization_view="unmasked", seed=2,
    )

    def fake_score(prediction_path: Path, *_args, **kwargs) -> dict:
        assert kwargs["metrics"] == (objective,)
        strength = ad.read_h5ad(prediction_path).uns["c3vcell_prediction"]["effect_strength"]
        return {"raw_metrics": {"c": {objective: 1 + (strength - 0.5) ** 2}}}

    monkeypatch.setattr("c3vcell.calibration.score_local", fake_score)
    output = calibrate_effect(
        source, checkpoint_path, tmp_path / "calibration", folds=2,
        strengths=(0.0, 0.5, 1.0), cells_per_target=4, prediction_steps=2,
        control_sampling=control_sampling, objective=objective,
    )
    record = json.loads(output.read_text(encoding="utf-8"))
    assert record["selected_effect_strength"] == 0.5
    assert record["control_sampling"] == control_sampling
    assert record["objective"] == objective
    plan = tmp_path / "calibration" / "calibration-plan.json"
    assert record["plan_sha256"] == file_sha256(plan)
    assert json.loads(plan.read_text(encoding="utf-8"))["fold_targets"] == [
        fold["heldout_targets"] for fold in record["folds"]
    ]
    if objective != OBJECTIVE:
        assert "not a scored challenge metric" in record["objective_note"]
    assert set(record["folds"][0]["heldout_targets"]).isdisjoint(
        record["folds"][1]["heldout_targets"]
    )
    assert set(sum((fold["heldout_targets"] for fold in record["folds"]), [])) == {
        "G0", "G1", "G2", "G3",
    }
    for fold in range(2):
        inner = torch.load(
            tmp_path / "calibration" / f"fold-{fold:02d}" / "model" / "model.pt",
            map_location="cpu", weights_only=True,
        )
        assert inner["model_config"]["variance_weight"] == 0.7
        assert inner["model_config"]["covariance_weight"] == 0.2
        assert inner["model_config"]["regularization_view"] == "unmasked"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    assert _resolve_effect_strength(
        checkpoint, checkpoint_path, "residual-rounding", None, output, prediction_steps=2,
        control_sampling=control_sampling, cells_per_target=4,
    ) == 0.5
    with pytest.raises(ValueError, match="cells per target"):
        _resolve_effect_strength(
            checkpoint, checkpoint_path, "residual-rounding", None, output, prediction_steps=2,
            control_sampling=control_sampling, cells_per_target=5,
        )
    other_sampling = "input-controls" if control_sampling == "reference-batch" else "reference-batch"
    with pytest.raises(ValueError, match="control sampling mode"):
        _resolve_effect_strength(
            checkpoint, checkpoint_path, "residual-rounding", None, output,
            prediction_steps=2, control_sampling=other_sampling,
        )
    if control_sampling == "reference-batch":
        legacy_record = dict(record)
        del legacy_record["control_sampling"]
        legacy = tmp_path / "legacy-calibration.json"
        legacy.write_text(json.dumps(legacy_record), encoding="utf-8")
        assert _resolve_effect_strength(
            checkpoint, checkpoint_path, "residual-rounding", None, legacy, prediction_steps=2,
        ) == 0.5
        monkeypatch.setattr("c3vcell.predict._read_gene_list", lambda _: data.var_names.tolist())
        monkeypatch.setattr("c3vcell.predict._read_targets", lambda _: ["G5"])
        with pytest.raises(ValueError, match="control sampling mode"):
            export_vcc_h5ad(
                checkpoint_path, {c: tmp_path / f"{c}.h5ad" for c in ("A", "B", "C")},
                tmp_path / "genes.csv", tmp_path / "targets.csv", tmp_path / "export.h5ad",
                sampling="residual-rounding", calibration_path=legacy, steps=2,
            )
    with pytest.raises(ValueError, match="solver steps"):
        _resolve_effect_strength(
            checkpoint, checkpoint_path, "residual-rounding", None, output, prediction_steps=3,
            control_sampling=control_sampling,
        )
    with pytest.raises(ValueError, match="residual-rounding"):
        _resolve_effect_strength(
            checkpoint, checkpoint_path, "negative-binomial", None, output,
            control_sampling=control_sampling,
        )
    altered = checkpoint.copy()
    altered["training_cell_ids"] = altered["training_cell_ids"][:-1]
    with pytest.raises(ValueError, match="training cells"):
        _resolve_effect_strength(
            altered, checkpoint_path, "residual-rounding", None, output,
            control_sampling=control_sampling,
        )
    copy = tmp_path / "different.pt"
    copy.write_bytes(checkpoint_path.read_bytes() + b"x")
    with pytest.raises(ValueError, match="exact checkpoint"):
        _resolve_effect_strength(
            checkpoint, copy, "residual-rounding", None, output,
            control_sampling=control_sampling,
        )


def test_effect_calibration_rejects_an_unsupported_objective_before_io(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unsupported calibration objective"):
        calibrate_effect(
            tmp_path / "unused.h5ad", tmp_path / "unused.pt", tmp_path / "out",
            objective="invented-metric",
        )
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize(("strict", "expected"), [(False, 0.5), (True, 0.0)])
def test_effect_calibration_can_require_improvement_in_every_fold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, strict: bool, expected: float,
) -> None:
    labels = [CONTROL] * 24 + sum(([f"G{i}"] * 4 for i in range(4)), [])
    data = ad.AnnData(
        sp.csr_matrix(np.full((len(labels), 8), 3, dtype=np.int32)),
        obs=pd.DataFrame(
            {"target_gene": labels, "context": "c", "batch": "b"},
            index=[f"cell-{i}" for i in range(len(labels))],
        ),
        var=pd.DataFrame(index=[f"G{i}" for i in range(8)]),
    )
    source = tmp_path / "source.h5ad"
    data.write_h5ad(source)
    checkpoint = train(
        source, tmp_path / "model", latent_dim=4, hidden_dim=6,
        pretrain_steps=2, flow_steps=2, batch_size=4,
    )

    def fake_score(prediction_path: Path, *_args, **_kwargs) -> dict:
        strength = ad.read_h5ad(prediction_path).uns["c3vcell_prediction"]["effect_strength"]
        value = 1.0 if strength == 0 else (0.8 if prediction_path.parent.name == "fold-00" else 1.1)
        return {"raw_metrics": {"c": {OBJECTIVE: value}}}

    monkeypatch.setattr("c3vcell.calibration.score_local", fake_score)
    output = calibrate_effect(
        source, checkpoint, tmp_path / "calibration", folds=2, strengths=(0.0, 0.5),
        cells_per_target=4, prediction_steps=2, require_every_fold_improvement=strict,
    )
    record = json.loads(output.read_text(encoding="utf-8"))
    assert record["selected_effect_strength"] == expected
    assert record["require_every_fold_improvement"] is strict


def test_effect_calibration_records_best_feasible_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    labels = [CONTROL] * 24 + sum(([f"G{i}"] * 4 for i in range(4)), [])
    data = ad.AnnData(
        sp.csr_matrix(np.full((len(labels), 8), 3, dtype=np.int32)),
        obs=pd.DataFrame(
            {"target_gene": labels, "context": "c", "batch": "b"},
            index=[f"cell-{i}" for i in range(len(labels))],
        ),
        var=pd.DataFrame(index=[f"G{i}" for i in range(8)]),
    )
    source = tmp_path / "source.h5ad"
    data.write_h5ad(source)
    checkpoint = train(
        source, tmp_path / "model", latent_dim=4, hidden_dim=6,
        pretrain_steps=2, flow_steps=2, batch_size=4,
    )

    def fake_score(prediction_path: Path, *_args, **_kwargs) -> dict:
        strength = ad.read_h5ad(prediction_path).uns["c3vcell_prediction"]["effect_strength"]
        values = (
            {0.0: 1.0, 0.25: 0.8, 0.5: 0.5}
            if prediction_path.parent.name == "fold-00"
            else {0.0: 1.0, 0.25: 0.9, 0.5: 1.01}
        )
        return {"raw_metrics": {"c": {OBJECTIVE: values[strength]}}}

    monkeypatch.setattr("c3vcell.calibration.score_local", fake_score)
    output = calibrate_effect(
        source, checkpoint, tmp_path / "calibration", folds=2, strengths=(0, 0.25, 0.5),
        cells_per_target=4, prediction_steps=2, require_every_fold_improvement=True,
    )
    record = json.loads(output.read_text(encoding="utf-8"))
    plan = json.loads((output.parent / "calibration-plan.json").read_text(encoding="utf-8"))
    assert record["selected_effect_strength"] == 0.25
    assert record["selection_rule"] == plan["selection_rule"]
    assert "every fold" in record["selection_rule"]


def test_effect_calibration_rejects_changed_plan(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    labels = [CONTROL] * 24 + sum(([f"G{i}"] * 4 for i in range(4)), [])
    data = ad.AnnData(
        sp.csr_matrix(np.full((len(labels), 8), 3, dtype=np.int32)),
        obs=pd.DataFrame(
            {"target_gene": labels, "context": "c", "batch": "b"},
            index=[f"cell-{i}" for i in range(len(labels))],
        ),
        var=pd.DataFrame(index=[f"G{i}" for i in range(8)]),
    )
    source = tmp_path / "source.h5ad"
    data.write_h5ad(source)
    checkpoint = train(
        source, tmp_path / "model", latent_dim=4, hidden_dim=6,
        pretrain_steps=2, flow_steps=2, batch_size=4,
    )
    out = tmp_path / "calibration"

    def fake_score(_prediction_path: Path, *_args, **_kwargs) -> dict:
        (out / "calibration-plan.json").write_text("{}", encoding="utf-8")
        return {"raw_metrics": {"c": {OBJECTIVE: 1.0}}}

    monkeypatch.setattr("c3vcell.calibration.score_local", fake_score)
    with pytest.raises(RuntimeError, match="calibration input changed"):
        calibrate_effect(
            source, checkpoint, out, folds=2, strengths=(0.0, 0.5),
            cells_per_target=4, prediction_steps=2,
        )
    assert not (out / "effect-calibration.json").exists()


def test_effect_calibration_rejects_external_reference_leakage_and_changes(tmp_path: Path) -> None:
    labels = [CONTROL] * 24 + sum(([f"G{i}"] * 4 for i in range(4)), [])
    data = ad.AnnData(
        sp.csr_matrix(np.full((len(labels), 8), 3, dtype=np.int32)),
        obs=pd.DataFrame(
            {"target_gene": labels, "context": "c", "batch": "b"},
            index=[f"cell-{i}" for i in range(len(labels))],
        ),
        var=pd.DataFrame(index=[f"G{i}" for i in range(8)]),
    )
    source = tmp_path / "source.h5ad"
    data.write_h5ad(source)
    external = data[:3].copy()
    reference = tmp_path / "external.h5ad"
    external.write_h5ad(reference)
    checkpoint = train(
        source, tmp_path / "model", latent_dim=4, hidden_dim=6,
        pretrain_steps=2, flow_steps=2, batch_size=4, reference_path=reference,
    )
    with pytest.raises(ValueError, match="overlaps calibration source"):
        calibrate_effect(source, checkpoint, tmp_path / "calibration", folds=2)
    external.X.data[0] += 1
    external.write_h5ad(reference)
    with pytest.raises(ValueError, match="Reference data changed"):
        calibrate_effect(source, checkpoint, tmp_path / "calibration2", folds=2)
