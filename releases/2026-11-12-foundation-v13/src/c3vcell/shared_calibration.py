"""Calibrate shared-response residuals using target-disjoint training folds."""

from __future__ import annotations

import json
import math
from pathlib import Path

import anndata as ad
import numpy as np
import torch

from .calibration import (
    CALIBRATION_OBJECTIVES, OBJECTIVE, _parameters, _training_source,
    calibration_objective_note, training_identity,
)
from .data import write_split
from .identity import check_cell_id_schemes
from .integrity import file_sha256
from .mean_response import fit_mean_response, predict_mean_response
from .predict import ControlSampling, _validate_control_sampling, predict_control_baseline
from .reference import load_reference
from .scoring import score_local
from .training import train

METHOD = "target-disjoint-shared-residual-v1"
METHOD_WITH_NULL = "target-disjoint-shared-residual-with-null-v2"


def _candidates(strengths: tuple[float, ...] | list[float]) -> list[float]:
    candidates = list(map(float, strengths))
    if (
        len(candidates) < 2
        or not all(math.isfinite(value) and 0 <= value <= 1 for value in candidates)
        or len(set(candidates)) != len(candidates)
        or 0.0 not in candidates
    ):
        raise ValueError("At least two unique finite conditional strengths in [0, 1] must include zero")
    return sorted(candidates)


def _select(
    candidates: list[float], fold_values: list[list[float]], min_relative_improvement: float,
) -> tuple[float, list[float]]:
    if not math.isfinite(min_relative_improvement) or not 0 <= min_relative_improvement < 1:
        raise ValueError("min_relative_improvement must be in [0, 1)")
    values = np.asarray(fold_values, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] != len(candidates):
        raise ValueError("Each of at least two folds must have a value for every strength")
    if not np.isfinite(values).all():
        raise ValueError("Calibration objective values must be finite")
    means = values.mean(axis=0)
    baseline_index = candidates.index(0.0)
    best_index = int(np.argmin(means))
    required = min_relative_improvement * max(abs(float(means[baseline_index])), 1e-8)
    improvements = values[:, baseline_index] - values[:, best_index]
    selected = 0.0
    if float(means[baseline_index] - means[best_index]) > required and np.all(improvements > 0):
        selected = candidates[best_index]
    return selected, means.tolist()


def _select_with_null(
    candidates: list[float],
    fold_values: list[list[float]],
    null_values: list[float],
    min_relative_improvement: float,
    shared_strength: float,
) -> tuple[float, float, list[float], float]:
    selected, means = _select(candidates, fold_values, min_relative_improvement)
    null = np.asarray(null_values, dtype=np.float64)
    if null.shape != (len(fold_values),) or not np.isfinite(null).all():
        raise ValueError("Each fold needs a finite no-response objective")
    values = np.asarray(fold_values, dtype=np.float64)[:, candidates.index(selected)]
    null_mean = float(null.mean())
    improvement = null_mean - float(values.mean())
    required = min_relative_improvement * max(abs(null_mean), 1e-8)
    if improvement > required and np.all(null > values):
        return shared_strength, selected, means, null_mean
    return 0.0, 0.0, means, null_mean


def _write_json_new(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def calibrate_shared_residual(
    source_path: Path,
    checkpoint_path: Path,
    profile_path: Path,
    outdir: Path,
    *,
    folds: int = 3,
    strengths: tuple[float, ...] = (0.0, 0.25, 0.5, 1.0),
    min_relative_improvement: float = 0.02,
    shared_strength: float = 1.0,
    include_no_response: bool = False,
    control_sampling: ControlSampling = "reference-batch",
    cells_per_target: int = 400,
    prediction_steps: int = 8,
    seed: int = 0,
    device_name: str = "cpu",
    threads: int = 2,
    objective: str = OBJECTIVE,
) -> Path:
    _validate_control_sampling(control_sampling)
    objective_note = calibration_objective_note(objective)
    if outdir.exists() and any(outdir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty directory: {outdir}")
    if folds < 2 or min(cells_per_target, prediction_steps, threads) < 1:
        raise ValueError("Use at least two folds and positive prediction sizes, steps, and threads")
    if not isinstance(include_no_response, bool):
        raise ValueError("include_no_response must be a boolean")
    if not math.isfinite(shared_strength) or not 0 <= shared_strength <= 1:
        raise ValueError("shared_strength must be finite and in [0, 1]")
    if not math.isfinite(min_relative_improvement) or not 0 <= min_relative_improvement < 1:
        raise ValueError("min_relative_improvement must be in [0, 1)")
    candidates = _candidates(strengths)
    checkpoint_digest = file_sha256(checkpoint_path)
    profile_digest = file_sha256(profile_path)
    source_digest = file_sha256(source_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("format_version") != 1 or checkpoint.get("kind") == "tahoe-drug-smiles-dose":
        raise ValueError("Shared-residual calibration requires a genetic checkpoint")
    if checkpoint.get("training_source_sha256") is None:
        raise ValueError("Retrain the checkpoint to record its training-count fingerprint")
    if checkpoint["training_source_sha256"] != source_digest:
        raise ValueError("Calibration source does not match the checkpoint training-count file")
    config = checkpoint["model_config"]
    if config.get("mask_probability") != 0.2 or config.get("teacher_decay") != 0.99:
        raise ValueError("Shared-residual calibration requires the default representation configuration")
    parameters = _parameters(checkpoint, {})
    profile = ad.read_h5ad(profile_path)
    metadata = profile.uns.get("c3vcell_mean_response", {})
    if (
        metadata.get("format_version") != 1
        or metadata.get("kind") != "training-target-uniform-batch-mean-logfold"
    ):
        raise ValueError("Unsupported mean-response profile")
    if str(metadata["source_sha256"]) != source_digest:
        raise ValueError("Mean-response profile was fitted on a different training-count file")
    if (
        profile.var_names.tolist() != checkpoint["genes"]
        or set(metadata["training_cell_ids"].astype(str)) != set(checkpoint["training_cell_ids"])
        or set(metadata["training_targets"].astype(str)) != set(checkpoint["training_targets"])
        or str(metadata["control"]) != checkpoint["control"]
    ):
        raise ValueError("Mean-response profile and checkpoint training identities differ")
    profile_parameters = {
        "min_group_cells": int(metadata["min_group_cells"]),
        "pseudocount": float(metadata["pseudocount"]),
        "logfold_limit": float(metadata["logfold_limit"]),
    }
    data = _training_source(
        source_path, checkpoint, group_column="target_gene", checkpoint_groups="training_targets",
    )
    if data.var_names.tolist() != checkpoint["genes"]:
        raise ValueError("Calibration source must have the exact checkpoint gene axis")
    if data.n_obs != len(checkpoint["training_cell_ids"]):
        raise ValueError("Calibration source and checkpoint training cells differ")
    reference_metadata = checkpoint.get("reference")
    if reference_metadata is not None:
        reference = load_reference(
            Path(reference_metadata["path"]), checkpoint["genes"],
            max_cells=reference_metadata["max_cells"], seed=reference_metadata["seed"],
            min_coverage=reference_metadata["min_coverage"],
        )
        if reference.source_identity != reference_metadata["source_identity"]:
            raise ValueError("Reference data changed since the final checkpoint was trained")
        check_cell_id_schemes(reference.cell_ids, data.obs_names)
        if set(reference.cell_ids).intersection(data.obs_names):
            raise ValueError("External representation reference overlaps calibration source cells")
        parameters.update({
            "reference_path": Path(reference_metadata["path"]),
            "reference_fraction": reference_metadata["fraction"],
            "reference_max_cells": reference_metadata["max_cells"],
            "reference_seed": reference_metadata["seed"],
            "reference_min_coverage": reference_metadata["min_coverage"],
            "expected_reference_identity": reference_metadata["source_identity"],
        })
    targets = np.asarray(sorted(checkpoint["training_targets"]), dtype=object)
    if len(targets) < 2 * folds:
        raise ValueError("Each calibration fold needs at least two distinct training targets")
    target_folds = np.array_split(np.random.default_rng(seed).permutation(targets), folds)
    identity = training_identity(checkpoint["training_cell_ids"], checkpoint["training_targets"])
    settings = {
        "method": METHOD,
        "objective": objective,
        "objective_note": objective_note + " Undefined values stop calibration.",
        "checkpoint_sha256": checkpoint_digest,
        "profile_sha256": profile_digest,
        "training_source_sha256": source_digest,
        "training_identity": identity,
        "strengths": candidates,
        "fold_targets": [sorted(fold.tolist()) for fold in target_folds],
        "min_relative_improvement": min_relative_improvement,
        "selection_rule": "minimum mean objective, improve every fold, exceed relative threshold",
        "shared_strength": shared_strength,
        "cells_per_target": cells_per_target,
        "prediction_steps": prediction_steps,
        "allow_unmatched_batch": parameters["allow_unmatched_batch"],
        "control_sampling": control_sampling,
        "seed": seed,
        "training_parameters": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in parameters.items()
        },
        "model_config": config,
        "profile_parameters": profile_parameters,
        "external_reference_identity": (
            reference_metadata["source_identity"] if reference_metadata else None
        ),
        "epigenomic_manifest_sha256": (
            checkpoint["epigenetics"]["manifest_sha256"]
            if checkpoint.get("epigenetics") is not None else None
        ),
        "reuses_checkpoint_graph": checkpoint.get("graph") is not None,
    }
    if include_no_response:
        settings.update({
            "method": METHOD_WITH_NULL,
            "include_no_response": True,
            "selection_rule": (
                "select residual against shared, then require the selected response "
                "to improve every fold and exceed the relative threshold against no response"
            ),
        })
    outdir.mkdir(parents=True, exist_ok=True)
    _write_json_new(outdir / "calibration-plan.json", settings)
    records = []
    for fold, heldout in enumerate(target_folds):
        directory = outdir / f"fold-{fold:02d}"
        split_dir = directory / "split"
        write_split(
            data, split_dir, control=checkpoint["control"], holdout_fraction=1 / folds,
            seed=seed + fold, max_genes=0, holdout_targets=heldout.tolist(),
        )
        inner_checkpoint = train(
            split_dir / "train.h5ad", directory / "model", control=checkpoint["control"],
            latent_dim=config["latent_dim"], hidden_dim=config["hidden_dim"],
            variance_weight=config.get("variance_weight", 0.1),
            covariance_weight=config.get("covariance_weight", 0.0),
            regularization_view=config.get("regularization_view", "masked"),
            _prior_checkpoint=checkpoint if (
                checkpoint.get("graph") is not None or checkpoint.get("epigenetics") is not None
            ) else None,
            seed=seed + fold, device_name=device_name, **parameters,
        )
        inner_profile = directory / "profile.h5ad"
        fit_mean_response(
            split_dir / "train.h5ad", inner_profile, control=checkpoint["control"],
            **profile_parameters,
        )
        values = []
        for index, candidate in enumerate(candidates):
            name = f"strength-{index:02d}"
            prediction = directory / f"{name}.h5ad"
            predict_mean_response(
                inner_profile, split_dir / "reference.h5ad", split_dir / "input_controls.h5ad",
                prediction, cells_per_target=cells_per_target, seed=seed + 10_000 + fold,
                effect_strength=shared_strength, allow_unmatched_batch=parameters["allow_unmatched_batch"],
                control_sampling=control_sampling,
                device_name=device_name, checkpoint_path=inner_checkpoint,
                conditional_strength=candidate, steps=prediction_steps,
            )
            result = score_local(
                prediction, split_dir / "reference.h5ad", directory / f"{name}-score",
                control=checkpoint["control"], threads=threads, metrics=(objective,),
            )
            context_values = [scores[objective] for scores in result["raw_metrics"].values()]
            if not context_values or any(value is None or not math.isfinite(value) for value in context_values):
                failure = {
                    "fold": fold, "conditional_strength": candidate,
                    "objective": objective, "raw_metrics": result["raw_metrics"],
                    "reason": "The specified expression objective is undefined.",
                }
                _write_json_new(outdir / "calibration-error.json", failure)
                raise ValueError(f"Undefined calibration objective in fold {fold}, strength {candidate}")
            values.append(float(np.mean(context_values)))
        record = {
            "fold": fold,
            "heldout_targets": sorted(heldout.tolist()),
            "inner_checkpoint_sha256": file_sha256(inner_checkpoint),
            "inner_profile_sha256": file_sha256(inner_profile),
            "values": values,
        }
        if include_no_response:
            null_prediction = directory / "null-control.h5ad"
            predict_control_baseline(
                split_dir / "reference.h5ad", split_dir / "input_controls.h5ad", null_prediction,
                control=checkpoint["control"], cells_per_target=cells_per_target,
                seed=seed + 10_000 + fold, allow_unmatched_batch=parameters["allow_unmatched_batch"],
                control_sampling=control_sampling,
            )
            null_result = score_local(
                null_prediction, split_dir / "reference.h5ad", directory / "null-control-score",
                control=checkpoint["control"], threads=threads, metrics=(objective,),
            )
            null_values = [scores[objective] for scores in null_result["raw_metrics"].values()]
            if not null_values or any(
                value is None or not math.isfinite(value) for value in null_values
            ):
                _write_json_new(outdir / "calibration-error.json", {
                    "fold": fold, "candidate": "no-response-control", "objective": objective,
                    "raw_metrics": null_result["raw_metrics"],
                    "reason": "The specified expression objective is undefined.",
                })
                raise ValueError(f"Undefined no-response calibration objective in fold {fold}")
            record["null_objective"] = float(np.mean(null_values))
        records.append(record)
    fold_values = [record["values"] for record in records]
    if include_no_response:
        selected_shared, selected, means, null_mean = _select_with_null(
            candidates, fold_values, [record["null_objective"] for record in records],
            min_relative_improvement, shared_strength,
        )
    else:
        selected, means = _select(candidates, fold_values, min_relative_improvement)
    if (
        file_sha256(source_path) != source_digest
        or file_sha256(checkpoint_path) != checkpoint_digest
        or file_sha256(profile_path) != profile_digest
    ):
        raise RuntimeError("A calibration input changed while fitting")
    if reference_metadata is not None:
        current_reference = load_reference(
            Path(reference_metadata["path"]), checkpoint["genes"],
            max_cells=reference_metadata["max_cells"], seed=reference_metadata["seed"],
            min_coverage=reference_metadata["min_coverage"],
        )
        if current_reference.source_identity != reference_metadata["source_identity"]:
            raise RuntimeError("External representation reference changed while calibrating")
    result = {
        **settings,
        "format_version": 2 if include_no_response else 1,
        "folds": records,
        "mean_objective_by_strength": means,
        "selected_conditional_strength": selected,
    }
    if include_no_response:
        result.update({
            "selected_shared_strength": selected_shared,
            "mean_null_objective": null_mean,
        })
    output = outdir / "shared-residual-calibration.json"
    _write_json_new(output, result)
    if include_no_response:
        print(
            f"Selected shared strength {selected_shared} and conditional strength {selected}; "
            f"calibration saved to {output}", flush=True,
        )
    else:
        print(f"Selected conditional strength {selected}; calibration saved to {output}", flush=True)
    return output


def resolve_shared_strength(
    path: Path,
    checkpoint_path: Path,
    checkpoint: dict,
    profile_path: Path,
    profile_metadata: dict,
    *,
    cells_per_target: int,
    shared_strength: float,
    prediction_steps: int,
    allow_unmatched_batch: bool,
    control_sampling: ControlSampling = "reference-batch",
) -> tuple[float, float]:
    _validate_control_sampling(control_sampling)
    result = json.loads(path.read_text(encoding="utf-8"))
    kind = (result.get("format_version"), result.get("method"))
    if kind not in ((1, METHOD), (2, METHOD_WITH_NULL)) or result.get("objective") not in CALIBRATION_OBJECTIVES:
        raise ValueError("Unsupported shared-residual calibration")
    if result["checkpoint_sha256"] != file_sha256(checkpoint_path):
        raise ValueError("Shared-residual calibration requires the exact checkpoint")
    if result["profile_sha256"] != file_sha256(profile_path):
        raise ValueError("Shared-residual calibration requires the exact mean-response profile")
    if (
        result["training_source_sha256"] != checkpoint.get("training_source_sha256")
        or result["training_source_sha256"] != str(profile_metadata["source_sha256"])
        or result["training_identity"] != training_identity(
            checkpoint["training_cell_ids"], checkpoint["training_targets"]
        )
    ):
        raise ValueError("Shared-residual calibration training identity differs")
    if (
        result["prediction_steps"] != prediction_steps
        or result["cells_per_target"] != cells_per_target
        or result["shared_strength"] != shared_strength
        or result["allow_unmatched_batch"] != allow_unmatched_batch
        or result.get("control_sampling", "reference-batch") != control_sampling
    ):
        raise ValueError("Shared-residual calibration prediction settings differ")
    candidates = _candidates(result["strengths"])
    records = result["folds"]
    if len(records) != len(result["fold_targets"]):
        raise ValueError("Shared-residual calibration fold count differs")
    targets = []
    for record, planned in zip(records, result["fold_targets"]):
        if record["heldout_targets"] != planned or len(planned) < 2:
            raise ValueError("Shared-residual calibration folds differ from the plan")
        targets.extend(planned)
    if len(targets) != len(set(targets)) or set(targets) != set(checkpoint["training_targets"]):
        raise ValueError("Shared-residual calibration targets do not partition training targets")
    fold_values = [record["values"] for record in records]
    if kind == (2, METHOD_WITH_NULL):
        try:
            null_values = [float(record["null_objective"]) for record in records]
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Shared-residual calibration is missing no-response fold scores") from exc
        selected_shared, selected, means, null_mean = _select_with_null(
            candidates, fold_values, null_values, result["min_relative_improvement"], shared_strength,
        )
        if (
            selected_shared != result["selected_shared_strength"]
            or null_mean != result["mean_null_objective"]
        ):
            raise ValueError("Shared-residual calibration selection is inconsistent with its fold scores")
    else:
        selected, means = _select(
            candidates, fold_values, result["min_relative_improvement"],
        )
        selected_shared = shared_strength
    if selected != result["selected_conditional_strength"] or means != result["mean_objective_by_strength"]:
        raise ValueError("Shared-residual calibration selection is inconsistent with its fold scores")
    return selected_shared, selected
