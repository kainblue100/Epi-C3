import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from c3vcell.data import CONTROL
from c3vcell.ensemble import predict_ensemble
from c3vcell.training import train
from c3vcell.uncertainty import apply_intervals, calibrate_intervals, evaluate_intervals


@pytest.fixture(params=["reference-batch", "input-controls"])
def interval_inputs(tmp_path: Path, request: pytest.FixtureRequest) -> dict[str, Path]:
    rng = np.random.default_rng(42)
    labels = [CONTROL] * 60 + sum(([f"G{i}"] * 10 for i in range(14)), [])
    counts = rng.poisson(3, size=(len(labels), 20)).astype(np.int32)
    counts[:, 19] += 1
    data = ad.AnnData(
        sp.csr_matrix(counts),
        obs=pd.DataFrame(
            {"target_gene": labels, "context": np.tile(["C", "D"], len(labels) // 2), "batch": "b"},
            index=[f"cell-{i}" for i in range(len(labels))],
        ),
        var=pd.DataFrame(index=[f"G{i}" for i in range(20)]),
    )
    train_rows = list(range(20)) + list(range(60, 80))
    cal_rows = list(range(20, 30)) + list(range(80, 180))
    eval_rows = list(range(40, 50)) + list(range(180, 200))
    paths = {}
    for name, rows in (
        ("train", train_rows), ("cal_reference", cal_rows),
        ("cal_controls", range(30, 40)), ("eval_reference", eval_rows),
        ("eval_controls", range(50, 60)),
    ):
        path = tmp_path / f"{name}.h5ad"
        data[list(rows)].copy().write_h5ad(path)
        paths[name] = path
    checkpoints = [
        train(
            paths["train"], tmp_path / f"model-{seed}", latent_dim=4, hidden_dim=6,
            pretrain_steps=2, flow_steps=2, batch_size=4, seed=seed,
        )
        for seed in (2, 3)
    ]
    for phase in ("cal", "eval"):
        prediction = tmp_path / f"{phase}_prediction.h5ad"
        disagreement = tmp_path / f"{phase}_disagreement.h5ad"
        predict_ensemble(
            checkpoints, paths[f"{phase}_reference"], paths[f"{phase}_controls"],
            prediction, disagreement, cells_per_target=6, seed=5, steps=2,
            control_sampling=request.param,
        )
        paths[f"{phase}_prediction"] = prediction
        paths[f"{phase}_disagreement"] = disagreement
    return paths


def _calibrate(paths: dict[str, Path], output: Path, **kwargs: float) -> None:
    calibrate_intervals(
        paths["cal_prediction"], paths["cal_disagreement"], paths["cal_reference"],
        paths["cal_controls"], output, **kwargs,
    )


def test_target_block_conformal_calibration_and_coverage(
    interval_inputs: dict[str, Path], tmp_path: Path,
) -> None:
    paths = interval_inputs
    calibration = tmp_path / "calibration.json"
    intervals = tmp_path / "intervals.h5ad"
    coverage = tmp_path / "coverage.json"
    _calibrate(paths, calibration, alpha=0.1, scale_floor=1)
    metadata = json.loads(calibration.read_text())
    assert metadata["n_calibration_targets"] == 10
    assert metadata["quantile_rank"] == 10
    assert metadata["quantile"] == max(metadata["calibration_scores"].values())
    mode = json.loads(paths["eval_prediction"].with_suffix(".h5ad.ensemble.json").read_text())[
        "control_sampling"
    ]
    assert metadata["settings"]["control_sampling"] == mode
    wrong = dict(metadata)
    wrong["settings"] = dict(metadata["settings"])
    wrong["settings"]["control_sampling"] = (
        "input-controls" if mode == "reference-batch" else "reference-batch"
    )
    mismatch = tmp_path / "mismatched-sampling.json"
    mismatch.write_text(json.dumps(wrong), encoding="utf-8")
    with pytest.raises(ValueError, match="inference settings"):
        apply_intervals(
            mismatch, paths["eval_prediction"], paths["eval_disagreement"],
            paths["eval_controls"], tmp_path / "mismatched-intervals.h5ad",
        )
    if mode == "reference-batch":
        legacy = dict(metadata)
        legacy["settings"] = dict(metadata["settings"])
        del legacy["settings"]["control_sampling"]
        legacy_path = tmp_path / "legacy-calibration.json"
        legacy_path.write_text(json.dumps(legacy), encoding="utf-8")
        apply_intervals(
            legacy_path, paths["eval_prediction"], paths["eval_disagreement"],
            paths["eval_controls"], tmp_path / "legacy-intervals.h5ad",
        )
    apply_intervals(
        calibration, paths["eval_prediction"], paths["eval_disagreement"],
        paths["eval_controls"], intervals,
    )
    data = ad.read_h5ad(intervals)
    assert data.shape == (4, 20)
    assert set(data.obs["context"]) == {"C", "D"}
    assert np.all(data.layers["lower"] >= 0)
    assert np.all(data.layers["upper"] >= data.layers["lower"])
    evaluate_intervals(
        calibration, paths["eval_prediction"], paths["eval_disagreement"],
        paths["eval_controls"], intervals, paths["eval_reference"], coverage,
    )
    result = json.loads(coverage.read_text())
    assert result["n_evaluation_targets"] == 2
    assert result["nominal_joint_coverage"] == pytest.approx(0.9)
    assert 0 <= result["empirical_target_joint_coverage"] <= 1
    assert 0 <= result["empirical_gene_fraction_covered"] <= 1


def test_intervals_reject_leakage_and_insufficient_calibration(
    interval_inputs: dict[str, Path], tmp_path: Path,
) -> None:
    paths = interval_inputs
    with pytest.raises(ValueError, match="calibration targets are needed"):
        _calibrate(paths, tmp_path / "too-few.json", alpha=0.01)
    with pytest.raises(ValueError, match="reference does not match"):
        calibrate_intervals(
            paths["cal_prediction"], paths["cal_disagreement"], paths["eval_reference"],
            paths["cal_controls"], tmp_path / "wrong-reference.json",
        )
    calibration = tmp_path / "calibration.json"
    _calibrate(paths, calibration)
    with pytest.raises(ValueError, match="targets overlap interval calibration"):
        apply_intervals(
            calibration, paths["cal_prediction"], paths["cal_disagreement"],
            paths["cal_controls"], tmp_path / "leaked.h5ad",
        )


def test_interval_tampering_is_rejected(
    interval_inputs: dict[str, Path], tmp_path: Path,
) -> None:
    paths = interval_inputs
    calibration = tmp_path / "calibration.json"
    intervals = tmp_path / "intervals.h5ad"
    _calibrate(paths, calibration)
    apply_intervals(
        calibration, paths["eval_prediction"], paths["eval_disagreement"],
        paths["eval_controls"], intervals,
    )
    altered = ad.read_h5ad(intervals)
    altered.layers["upper"][0, 0] += 1
    altered.write_h5ad(intervals)
    with pytest.raises(ValueError, match="values do not match"):
        evaluate_intervals(
            calibration, paths["eval_prediction"], paths["eval_disagreement"],
            paths["eval_controls"], intervals, paths["eval_reference"],
            tmp_path / "coverage.json",
        )
