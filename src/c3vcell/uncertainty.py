"""Split-conformal intervals for observed target-level pseudobulk means."""

from __future__ import annotations

import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

from .calibration import file_sha256
from .data import validate_counts
from .ensemble import _check_members
from .identity import check_cell_id_schemes
from .predict import _validate_control_sampling


@dataclass
class _Inputs:
    prediction: ad.AnnData
    disagreement: ad.AnnData
    manifest: dict
    controls: ad.AnnData
    checkpoints: list[dict]
    groups: list[tuple[str, str]]
    center: np.ndarray
    spread: np.ndarray


def _read_inputs(
    prediction_path: Path, disagreement_path: Path, controls_path: Path,
) -> _Inputs:
    manifest_path = prediction_path.with_suffix(prediction_path.suffix + ".ensemble.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format_version") != 1:
        raise ValueError("Unsupported ensemble manifest")
    for path, key in (
        (prediction_path, "prediction_sha256"),
        (disagreement_path, "disagreement_sha256"),
        (controls_path, "controls_sha256"),
    ):
        if file_sha256(path) != manifest.get(key):
            raise ValueError(f"Ensemble provenance does not match {path}")
    entries = manifest["checkpoints"]
    paths = [Path(entry["path"]) for entry in entries]
    digests, seeds = _check_members(paths)
    if digests != [entry["sha256"] for entry in entries] or seeds != [
        entry["training_seed"] for entry in entries
    ]:
        raise ValueError("Ensemble checkpoint provenance has changed")
    checkpoints = [torch.load(path, map_location="cpu", weights_only=True) for path in paths]
    prediction = ad.read_h5ad(prediction_path)
    disagreement = ad.read_h5ad(disagreement_path)
    controls = ad.read_h5ad(controls_path)
    validate_counts(prediction)
    validate_counts(controls)
    if (
        not prediction.var_names.equals(disagreement.var_names)
        or not prediction.var_names.equals(controls.var_names)
        or prediction.var_names.tolist() != checkpoints[0]["genes"]
    ):
        raise ValueError("Ensemble, disagreement, controls, and checkpoint gene axes must match")
    for name, dataset in (("prediction", prediction), ("disagreement", disagreement)):
        if not {"context", "target_gene"}.issubset(dataset.obs.columns):
            raise ValueError(f"{name} requires context and target_gene labels")
        if dataset.obs[["context", "target_gene"]].isna().any().any():
            raise ValueError(f"{name} contains missing group labels")
    if "source_control_id" not in prediction.obs or prediction.obs["source_control_id"].isna().any():
        raise ValueError("Ensemble predictions require source control identifiers")
    if "target_gene" not in controls.obs or set(controls.obs["target_gene"].astype(str)) != {
        checkpoints[0]["control"]
    }:
        raise ValueError("Input controls must contain only the checkpoint's control label")
    if not set(prediction.obs["source_control_id"].astype(str)) <= set(controls.obs_names):
        raise ValueError("Ensemble predictions refer to unknown input controls")
    if "between_model_std" not in disagreement.layers:
        raise ValueError("Ensemble disagreement is missing its standard deviation layer")
    pred_context = prediction.obs["context"].astype(str).to_numpy()
    pred_target = prediction.obs["target_gene"].astype(str).to_numpy()
    groups = sorted(set(zip(pred_context, pred_target)))
    dis_groups = list(zip(
        disagreement.obs["context"].astype(str), disagreement.obs["target_gene"].astype(str),
    ))
    if len(set(dis_groups)) != len(dis_groups) or set(groups) != set(dis_groups):
        raise ValueError("Ensemble prediction and disagreement groups differ")
    training_ids = set(checkpoints[0]["training_cell_ids"])
    reference_ids = set((checkpoints[0].get("reference") or {}).get("cell_ids", []))
    check_cell_id_schemes(training_ids, reference_ids, controls.obs_names)
    training_ids.update(reference_ids)
    if training_ids.intersection(controls.obs_names):
        raise ValueError("Input controls overlap training cells")
    if set(pred_target).intersection(checkpoints[0]["training_targets"]):
        raise ValueError("Predicted targets overlap model training targets")
    lookup = {group: index for index, group in enumerate(dis_groups)}
    spread = np.asarray(disagreement.layers["between_model_std"])[
        [lookup[group] for group in groups]
    ].astype(np.float64)
    if not np.isfinite(spread).all() or np.any(spread < 0):
        raise ValueError("Model disagreement must be finite and nonnegative")
    matrix = sp.csr_matrix(prediction.X, dtype=np.float64)
    center = []
    for context, target in groups:
        rows = np.flatnonzero((pred_context == context) & (pred_target == target))
        if len(rows) != manifest["cells_per_target"]:
            raise ValueError("Ensemble target groups must have the recorded number of cells")
        center.append(np.asarray(matrix[rows].mean(axis=0)).ravel())
    return _Inputs(
        prediction, disagreement, manifest, controls, checkpoints, groups,
        np.stack(center).astype(np.float64), spread,
    )


def _truth(inputs: _Inputs, reference_path: Path) -> tuple[np.ndarray, ad.AnnData]:
    if file_sha256(reference_path) != inputs.manifest["reference_sha256"]:
        raise ValueError("Held-out reference does not match the ensemble manifest")
    reference = ad.read_h5ad(reference_path)
    validate_counts(reference)
    if not reference.var_names.equals(inputs.prediction.var_names):
        raise ValueError("Held-out reference gene axis does not match the ensemble")
    if not {"context", "target_gene"}.issubset(reference.obs.columns):
        raise ValueError("Held-out reference requires context and target_gene labels")
    if reference.obs[["context", "target_gene"]].isna().any().any():
        raise ValueError("Held-out reference has missing group labels")
    training_ids = set(inputs.checkpoints[0]["training_cell_ids"])
    reference_ids = set((inputs.checkpoints[0].get("reference") or {}).get("cell_ids", []))
    check_cell_id_schemes(
        training_ids, reference_ids, reference.obs_names, inputs.controls.obs_names,
    )
    training_ids.update(reference_ids)
    if training_ids.intersection(reference.obs_names):
        raise ValueError("Held-out reference overlaps training cells")
    if set(reference.obs_names).intersection(inputs.controls.obs_names):
        raise ValueError("Held-out reference overlaps input controls")
    context = reference.obs["context"].astype(str).to_numpy()
    target = reference.obs["target_gene"].astype(str).to_numpy()
    control = inputs.checkpoints[0]["control"]
    groups = set(zip(context[target != control], target[target != control]))
    if groups != set(inputs.groups):
        raise ValueError("Held-out reference and ensemble target groups differ")
    matrix = sp.csr_matrix(reference.X, dtype=np.float64)
    means = []
    for group_context, group_target in inputs.groups:
        rows = np.flatnonzero((context == group_context) & (target == group_target))
        if len(rows) < 2:
            raise ValueError("Each reference target/context needs at least two cells")
        means.append(np.asarray(matrix[rows].mean(axis=0)).ravel())
    return np.stack(means).astype(np.float64), reference


def _settings(manifest: dict) -> dict:
    settings = {
        key: manifest[key]
        for key in (
            "cells_per_target", "seed", "steps", "sampling", "effect_strength",
            "allow_unmatched_batch",
        )
    }
    settings["control_sampling"] = manifest.get("control_sampling", "reference-batch")
    _validate_control_sampling(settings["control_sampling"])
    return settings


def _write_json_new(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", prefix=".c3-uncertainty-", dir=path.parent, delete=False,
    ) as stream:
        temp = Path(stream.name)
        json.dump(data, stream, indent=2)
        stream.write("\n")
    try:
        os.link(temp, path)
    finally:
        temp.unlink()


def calibrate_intervals(
    prediction_path: Path,
    disagreement_path: Path,
    reference_path: Path,
    controls_path: Path,
    output_path: Path,
    *,
    alpha: float = 0.1,
    scale_floor: float = 1.0,
) -> None:
    if not math.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must be finite and between zero and one")
    if not math.isfinite(scale_floor) or scale_floor <= 0:
        raise ValueError("scale_floor must be finite and positive")
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite {output_path}")
    inputs = _read_inputs(prediction_path, disagreement_path, controls_path)
    truth, reference = _truth(inputs, reference_path)
    scale = np.maximum(inputs.spread, scale_floor)
    residual = np.abs(truth - inputs.center) / scale
    targets = sorted({target for _, target in inputs.groups})
    scores = np.asarray([
        np.max(residual[[i for i, (_, name) in enumerate(inputs.groups) if name == target]])
        for target in targets
    ])
    n = len(scores)
    rank = math.ceil((n + 1) * (1 - alpha))
    if rank > n:
        raise ValueError(
            f"At least {math.ceil(1 / alpha) - 1} calibration targets are needed "
            "for a finite interval at this alpha"
        )
    quantile = float(np.partition(scores, rank - 1)[rank - 1])
    output = {
        "format_version": 1,
        "kind": "target-block-split-conformal-empirical-pseudobulk",
        "alpha": alpha,
        "scale_floor": scale_floor,
        "n_calibration_targets": n,
        "quantile_rank": rank,
        "quantile": quantile,
        "calibration_targets": targets,
        "calibration_scores": dict(zip(targets, scores.tolist())),
        "calibration_reference_sha256": file_sha256(reference_path),
        "calibration_reference_cell_ids": reference.obs_names.astype(str).tolist(),
        "calibration_control_cell_ids": inputs.controls.obs_names.astype(str).tolist(),
        "checkpoint_sha256": [entry["sha256"] for entry in inputs.manifest["checkpoints"]],
        "settings": _settings(inputs.manifest),
        "gene_names": inputs.prediction.var_names.astype(str).tolist(),
        "assumption": (
            "Target-level residual scores must be exchangeable between calibration and future "
            "targets. Intervals cover observed finite-sample pseudobulk means jointly across "
            "genes/contexts, not latent biological means or individual cells."
        ),
    }
    _write_json_new(output_path, output)


def _check_calibration(calibration: dict, inputs: _Inputs) -> None:
    if (
        calibration.get("format_version") != 1
        or calibration.get("kind") != "target-block-split-conformal-empirical-pseudobulk"
    ):
        raise ValueError("Unsupported interval calibration")
    calibration_settings = dict(calibration["settings"])
    calibration_settings.setdefault("control_sampling", "reference-batch")
    if calibration["checkpoint_sha256"] != [
        entry["sha256"] for entry in inputs.manifest["checkpoints"]
    ] or calibration_settings != _settings(inputs.manifest):
        raise ValueError("Interval calibration does not match the ensemble and inference settings")
    if calibration["gene_names"] != inputs.prediction.var_names.tolist():
        raise ValueError("Interval calibration gene axis differs")
    targets = calibration["calibration_targets"]
    scores = calibration["calibration_scores"]
    alpha = calibration["alpha"]
    n = len(targets)
    if (
        len(set(targets)) != n or set(scores) != set(targets)
        or calibration["n_calibration_targets"] != n
        or not math.isfinite(alpha) or not 0 < alpha < 1
    ):
        raise ValueError("Interval calibration target scores or alpha are invalid")
    rank = math.ceil((n + 1) * (1 - alpha))
    values = np.asarray([scores[target] for target in targets], dtype=np.float64)
    if (
        rank > n or calibration["quantile_rank"] != rank
        or not np.isfinite(values).all() or np.any(values < 0)
        or calibration["quantile"] != float(np.partition(values, rank - 1)[rank - 1])
    ):
        raise ValueError("Interval calibration quantile is inconsistent with its target scores")
    if set(targets).intersection(inputs.checkpoints[0]["training_targets"]):
        raise ValueError("Interval calibration targets overlap model training targets")
    if {target for _, target in inputs.groups}.intersection(calibration["calibration_targets"]):
        raise ValueError("Evaluation targets overlap interval calibration targets")
    calibration_ids = set(calibration["calibration_reference_cell_ids"])
    calibration_control_ids = set(calibration["calibration_control_cell_ids"])
    training_ids = set(inputs.checkpoints[0]["training_cell_ids"])
    reference_ids = set((inputs.checkpoints[0].get("reference") or {}).get("cell_ids", []))
    check_cell_id_schemes(
        calibration_ids, calibration_control_ids, training_ids, reference_ids,
        inputs.controls.obs_names,
    )
    calibration_ids.update(calibration_control_ids)
    training_ids.update(reference_ids)
    if calibration_ids.intersection(training_ids):
        raise ValueError("Interval calibration cells overlap model training cells")
    if calibration_ids.intersection(inputs.controls.obs_names):
        raise ValueError("Evaluation controls overlap interval calibration cells")
    if (
        not math.isfinite(calibration["quantile"]) or calibration["quantile"] < 0
        or not math.isfinite(calibration["scale_floor"]) or calibration["scale_floor"] <= 0
    ):
        raise ValueError("Interval calibration contains invalid numeric values")


def _bounds(calibration: dict, inputs: _Inputs) -> tuple[np.ndarray, np.ndarray]:
    radius = calibration["quantile"] * np.maximum(inputs.spread, calibration["scale_floor"])
    lower = np.maximum(0, np.nextafter(inputs.center - radius, -np.inf))
    upper = np.nextafter(inputs.center + radius, np.inf)
    if not np.isfinite(lower).all() or not np.isfinite(upper).all():
        raise ValueError("Calibrated interval bounds are not finite")
    return lower, upper


def apply_intervals(
    calibration_path: Path,
    prediction_path: Path,
    disagreement_path: Path,
    controls_path: Path,
    output_path: Path,
) -> None:
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite {output_path}")
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    inputs = _read_inputs(prediction_path, disagreement_path, controls_path)
    _check_calibration(calibration, inputs)
    lower, upper = _bounds(calibration, inputs)
    intervals = ad.AnnData(
        X=inputs.center,
        obs=pd.DataFrame(
            [{"context": context, "target_gene": target} for context, target in inputs.groups],
            index=[f"group-{i:06d}" for i in range(len(inputs.groups))],
        ),
        var=inputs.prediction.var.copy(),
    )
    intervals.layers["lower"] = lower
    intervals.layers["upper"] = upper
    intervals.uns["c3vcell_interval"] = {
        "calibration_sha256": file_sha256(calibration_path),
        "prediction_sha256": file_sha256(prediction_path),
        "disagreement_sha256": file_sha256(disagreement_path),
        "alpha": calibration["alpha"],
        "assumption": calibration["assumption"],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=".c3-interval-", suffix=".h5ad", dir=output_path.parent, delete=False,
    ) as stream:
        temp = Path(stream.name)
    try:
        intervals.write_h5ad(temp, compression="gzip")
        os.link(temp, output_path)
    finally:
        temp.unlink()


def evaluate_intervals(
    calibration_path: Path,
    prediction_path: Path,
    disagreement_path: Path,
    controls_path: Path,
    interval_path: Path,
    reference_path: Path,
    output_path: Path,
) -> None:
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite {output_path}")
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    inputs = _read_inputs(prediction_path, disagreement_path, controls_path)
    _check_calibration(calibration, inputs)
    truth, reference = _truth(inputs, reference_path)
    calibration_ids = set(calibration["calibration_reference_cell_ids"])
    calibration_control_ids = set(calibration["calibration_control_cell_ids"])
    check_cell_id_schemes(calibration_ids, calibration_control_ids, reference.obs_names)
    calibration_ids.update(calibration_control_ids)
    if calibration_ids.intersection(reference.obs_names):
        raise ValueError("Evaluation reference overlaps interval calibration cells")
    intervals = ad.read_h5ad(interval_path)
    provenance = intervals.uns.get("c3vcell_interval", {})
    if (
        provenance.get("calibration_sha256") != file_sha256(calibration_path)
        or provenance.get("prediction_sha256") != file_sha256(prediction_path)
        or provenance.get("disagreement_sha256") != file_sha256(disagreement_path)
    ):
        raise ValueError("Intervals do not match the supplied calibration and ensemble")
    interval_groups = list(zip(
        intervals.obs["context"].astype(str), intervals.obs["target_gene"].astype(str),
    ))
    if interval_groups != inputs.groups or not intervals.var_names.equals(inputs.prediction.var_names):
        raise ValueError("Interval groups or gene axis differ from the evaluation reference")
    lower = np.asarray(intervals.layers["lower"], dtype=np.float64)
    upper = np.asarray(intervals.layers["upper"], dtype=np.float64)
    if not np.isfinite(lower).all() or not np.isfinite(upper).all() or np.any(lower > upper):
        raise ValueError("Intervals contain invalid bounds")
    expected_lower, expected_upper = _bounds(calibration, inputs)
    if (
        not np.array_equal(intervals.X, inputs.center)
        or not np.array_equal(lower, expected_lower)
        or not np.array_equal(upper, expected_upper)
    ):
        raise ValueError("Interval values do not match the calibrated ensemble")
    covered = (truth >= lower) & (truth <= upper)
    targets = sorted({target for _, target in inputs.groups})
    joint = {
        target: bool(np.all(covered[[i for i, (_, name) in enumerate(inputs.groups) if name == target]]))
        for target in targets
    }
    _write_json_new(output_path, {
        "nominal_joint_coverage": 1 - calibration["alpha"],
        "empirical_target_joint_coverage": float(np.mean(list(joint.values()))),
        "empirical_gene_fraction_covered": float(np.mean(covered)),
        "n_evaluation_targets": len(targets),
        "target_joint_coverage": joint,
        "mean_interval_width": float(np.mean(upper - lower)),
        "median_interval_width": float(np.median(upper - lower)),
        "reference_sha256": file_sha256(reference_path),
        "interval_sha256": file_sha256(interval_path),
        "assumption": calibration["assumption"],
        "interpretation": (
            "Coverage is descriptive for these finite-sample pseudobulk means; a small target "
            "set cannot establish future coverage or exchangeability."
        ),
    })
