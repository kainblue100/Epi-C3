"""Fit prediction shrinkage on perturbation-disjoint training folds."""

from __future__ import annotations

import hashlib
import json
import math
import warnings
from pathlib import Path
from typing import Literal

import anndata as ad
import numpy as np
import scipy.sparse as sp
import torch

from .data import validate_counts, write_split
from .identity import check_cell_id_schemes
from .integrity import file_sha256
from .predict import ControlSampling, _validate_control_sampling, predict_holdout
from .reference import load_reference
from .scoring import score_local
from .training import train

OBJECTIVE = "expr_mse_unbiased_capped_norm"
CALIBRATION_OBJECTIVES = (OBJECTIVE, "expr_mse_unbiased_capped")


def calibration_objective_note(objective: str) -> str:
    if objective == OBJECTIVE:
        return (
            "Mean of the VCC preset normalized expression metric across contexts and "
            "perturbation-disjoint folds; not a calibrated six-metric or official leaderboard score."
        )
    if objective == "expr_mse_unbiased_capped":
        return (
            "Mean of the VCC preset unnormalized capped expression-error numerator across "
            "contexts and perturbation-disjoint folds. This diagnostic can be negative, depends "
            "on the panel, and is not a scored challenge metric or official leaderboard score."
        )
    raise ValueError(f"Unsupported calibration objective: {objective}")


def training_identity(cell_ids: list[str], targets: list[str]) -> str:
    if len(set(cell_ids)) != len(cell_ids) or len(set(targets)) != len(targets):
        raise ValueError("Training cell and target identifiers must be unique")
    payload = json.dumps(
        {"cell_ids": sorted(cell_ids), "targets": sorted(targets)},
        sort_keys=True, ensure_ascii=False, separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _training_source(
    path: Path, checkpoint: dict, *, group_column: str, checkpoint_groups: str,
) -> ad.AnnData:
    source_digest = checkpoint.get("training_source_sha256")
    if source_digest is None:
        warnings.warn(
            "The checkpoint predates training-file fingerprints; matching cell IDs "
            "cannot verify the training counts.",
            RuntimeWarning, stacklevel=2,
        )
    elif file_sha256(path) != source_digest:
        raise ValueError("Calibration source does not match the checkpoint training-count file")
    source = ad.read_h5ad(path, backed="r")
    try:
        if not source.obs_names.is_unique or not source.var_names.is_unique:
            raise ValueError("Calibration source requires unique cell and gene identifiers")
        requested = checkpoint["training_cell_ids"]
        if len(set(requested)) != len(requested):
            raise ValueError("Checkpoint contains duplicate training cells")
        check_cell_id_schemes(requested, source.obs_names)
        positions = source.obs_names.get_indexer(requested)
        if np.any(positions < 0):
            raise ValueError("Calibration source is missing checkpoint training cells")
        positions = np.sort(positions)
        data = ad.AnnData(
            X=sp.csr_matrix(source.X[positions, :]),
            obs=source.obs.iloc[positions].copy(),
            var=source.var.copy(),
            uns=dict(source.uns),
        )
    finally:
        source.file.close()
    validate_counts(data)
    if not set(checkpoint["genes"]) <= set(data.var_names):
        raise ValueError("Calibration source is missing checkpoint genes")
    for column in ("target_gene", group_column):
        if column not in data.obs or data.obs[column].isna().any():
            raise ValueError(f"Calibration source is missing {column!r} labels")
    source_targets = sorted(set(data.obs["target_gene"].astype(str)) - {checkpoint["control"]})
    if source_targets != sorted(checkpoint["training_targets"]):
        raise ValueError("Calibration source targets differ from the checkpoint")
    source_groups = sorted(set(data.obs[group_column].astype(str)) - {checkpoint["control"]})
    if source_groups != sorted(checkpoint[checkpoint_groups]):
        raise ValueError("Calibration source perturbation groups differ from the checkpoint")
    if source_digest is not None and file_sha256(path) != source_digest:
        raise RuntimeError("Calibration source changed while it was being read")
    return data


def _parameters(checkpoint: dict, overrides: dict) -> dict:
    saved = checkpoint.get("training_parameters")
    defaults = {"allow_unmatched_batch": False}
    result = {}
    for key in ("pretrain_steps", "flow_steps", "batch_size", "learning_rate"):
        requested = overrides.get(key)
        existing = saved.get(key) if saved is not None else None
        if requested is not None and existing is not None and requested != existing:
            raise ValueError(f"Calibration {key} must match the final checkpoint")
        result[key] = existing if requested is None else requested
        if result[key] is None:
            raise ValueError(f"Older checkpoints require an explicit calibration {key}")
    result["allow_unmatched_batch"] = (
        saved.get("allow_unmatched_batch", False) if saved is not None
        else defaults["allow_unmatched_batch"]
    )
    result["flow_target"] = saved.get("flow_target", "paired") if saved is not None else "paired"
    result["min_group_cells"] = saved.get("min_group_cells", 5) if saved is not None else 5
    return result


def _select_effect_strength(
    candidates: list[float],
    fold_values: list[list[float]],
    min_relative_improvement: float,
    require_every_fold_improvement: bool,
) -> tuple[float, list[float]]:
    values = np.asarray(fold_values, dtype=np.float64)
    if (
        len(candidates) < 2 or len(set(candidates)) != len(candidates)
        or 0.0 not in candidates
        or not all(math.isfinite(value) and 0 <= value <= 1 for value in candidates)
        or values.ndim != 2 or values.shape[0] < 2 or values.shape[1] != len(candidates)
        or not np.isfinite(values).all()
    ):
        raise ValueError("Calibration needs finite scores for each candidate in at least two folds")
    if not math.isfinite(min_relative_improvement) or not 0 <= min_relative_improvement < 1:
        raise ValueError("min_relative_improvement must be in [0, 1)")
    if not isinstance(require_every_fold_improvement, bool):
        raise ValueError("require_every_fold_improvement must be a boolean")
    means = values.mean(axis=0)
    baseline_index = candidates.index(0.0)
    baseline = float(means[baseline_index])
    required = min_relative_improvement * max(abs(baseline), 1e-8)
    eligible = [
        index for index, candidate in enumerate(candidates)
        if candidate != 0.0
        and baseline - means[index] > required
        and (
            not require_every_fold_improvement
            or np.all(values[:, index] < values[:, baseline_index])
        )
    ]
    if not eligible:
        return 0.0, means.tolist()
    best = min(eligible, key=lambda index: (means[index], candidates[index]))
    return candidates[best], means.tolist()


def calibrate_effect(
    source_path: Path,
    checkpoint_path: Path,
    outdir: Path,
    *,
    folds: int = 3,
    strengths: tuple[float, ...] = (0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0),
    min_relative_improvement: float = 0.02,
    pretrain_steps: int | None = None,
    flow_steps: int | None = None,
    batch_size: int | None = None,
    learning_rate: float | None = None,
    cells_per_target: int = 400,
    prediction_steps: int = 16,
    seed: int = 0,
    device_name: str = "cpu",
    threads: int = 2,
    kind: Literal["genetic", "drug"] = "genetic",
    control_sampling: ControlSampling = "reference-batch",
    objective: str = OBJECTIVE,
    require_every_fold_improvement: bool = False,
) -> Path:
    _validate_control_sampling(control_sampling)
    objective_note = calibration_objective_note(objective)
    if outdir.exists() and any(outdir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty directory: {outdir}")
    if folds < 2:
        raise ValueError("At least two disjoint validation folds are required")
    if kind not in {"genetic", "drug"}:
        raise ValueError("Calibration kind must be genetic or drug")
    if not isinstance(require_every_fold_improvement, bool):
        raise ValueError("require_every_fold_improvement must be a boolean")
    if kind == "drug" and control_sampling != "reference-batch":
        raise ValueError("Drug calibration currently requires reference-batch sampling")
    if cells_per_target < 1 or prediction_steps < 1 or threads < 1:
        raise ValueError("Prediction sizes, steps, and threads must be positive")
    if not math.isfinite(min_relative_improvement) or not 0 <= min_relative_improvement < 1:
        raise ValueError("min_relative_improvement must be in [0, 1)")
    candidates = sorted(set(map(float, strengths)))
    if (
        len(candidates) != len(strengths)
        or 0.0 not in candidates
        or not all(math.isfinite(value) and 0 <= value <= 1 for value in candidates)
    ):
        raise ValueError("Unique finite strengths must lie in [0, 1] and include zero")

    checkpoint_digest = file_sha256(checkpoint_path)
    source_digest = file_sha256(source_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("format_version") != 1:
        raise ValueError("Unsupported model checkpoint")
    if (checkpoint.get("kind") == "tahoe-drug-smiles-dose") != (kind == "drug"):
        raise ValueError("Calibration kind does not match the checkpoint")
    if kind == "drug" and (
        checkpoint.get("graph") is not None or checkpoint.get("epigenetics") is not None
    ):
        raise ValueError("Drug calibration does not support graph or epigenomic priors")
    config = checkpoint["model_config"]
    if config.get("mask_probability") != 0.2 or config.get("teacher_decay") != 0.99:
        raise ValueError("Effect calibration requires the default representation configuration")
    parameters = _parameters(checkpoint, {
        "pretrain_steps": pretrain_steps,
        "flow_steps": flow_steps,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
    })
    group_column = "target_gene" if kind == "genetic" else "compound"
    group_key = "training_targets" if kind == "genetic" else "training_compounds"
    data = _training_source(
        source_path, checkpoint, group_column=group_column, checkpoint_groups=group_key,
    )
    if (
        checkpoint.get("graph") is not None or checkpoint.get("epigenetics") is not None
    ) and data.var_names.tolist() != checkpoint["genes"]:
        raise ValueError("Prior-conditioned calibration requires the exact checkpoint gene axis")
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
            raise ValueError(
                "External representation reference overlaps calibration source cells"
            )
        parameters.update({
            "reference_path": Path(reference_metadata["path"]),
            "reference_fraction": reference_metadata["fraction"],
            "reference_max_cells": reference_metadata["max_cells"],
            "reference_seed": reference_metadata["seed"],
            "reference_min_coverage": reference_metadata["min_coverage"],
            "expected_reference_identity": reference_metadata["source_identity"],
        })
    targets = np.asarray(sorted(checkpoint[group_key]), dtype=object)
    if len(targets) < 2 * folds:
        raise ValueError("Each validation fold needs at least two distinct training targets")
    rng = np.random.default_rng(seed)
    target_folds = np.array_split(rng.permutation(targets), folds)
    identity = training_identity(checkpoint["training_cell_ids"], checkpoint[group_key])
    outdir.mkdir(parents=True, exist_ok=True)
    plan_path = outdir / "calibration-plan.json"
    plan = {
        "kind": kind,
        "checkpoint_sha256": checkpoint_digest,
        "training_source_sha256": source_digest,
        "training_identity": identity,
        "objective": objective,
        "objective_note": objective_note,
        "strengths": candidates,
        "fold_targets": [sorted(fold.tolist()) for fold in target_folds],
        "min_relative_improvement": min_relative_improvement,
        "require_every_fold_improvement": require_every_fold_improvement,
        "selection_rule": (
            "lowest mean among candidates improving every fold and meeting the threshold"
            if require_every_fold_improvement
            else "lowest mean meeting the threshold"
        ),
        "control_sampling": control_sampling,
        "cells_per_target": cells_per_target,
        "prediction_steps": prediction_steps,
        "seed": seed,
    }
    with plan_path.open("x", encoding="utf-8") as stream:
        json.dump(plan, stream, indent=2, allow_nan=False)
        stream.write("\n")
    plan_digest = file_sha256(plan_path)
    fold_records = []
    for fold, heldout in enumerate(target_folds):
        directory = outdir / f"fold-{fold:02d}"
        split_dir = directory / "split"
        write_split(
            data, split_dir, control=checkpoint["control"],
            holdout_fraction=1 / folds, seed=seed + fold,
            max_genes=len(checkpoint["genes"]), holdout_targets=heldout.tolist(),
            split_column=group_column, require_targets_on_gene_axis=(kind == "genetic"),
        )
        if kind == "genetic":
            inner_checkpoint = train(
                split_dir / "train.h5ad", directory / "model", control=checkpoint["control"],
                latent_dim=config["latent_dim"], hidden_dim=config["hidden_dim"],
                variance_weight=config.get("variance_weight", 0.1),
                covariance_weight=config.get("covariance_weight", 0.0),
                regularization_view=config.get("regularization_view", "masked"),
                _prior_checkpoint=checkpoint if (
                    checkpoint.get("graph") is not None
                    or checkpoint.get("epigenetics") is not None
                ) else None,
                seed=seed + fold, device_name=device_name, **parameters,
            )
        else:
            from .drug import train_drug

            drug_parameters = {
                key: value for key, value in parameters.items()
                if key not in {"allow_unmatched_batch", "flow_target", "min_group_cells"}
            }
            inner_checkpoint = train_drug(
                split_dir / "train.h5ad", directory / "model",
                latent_dim=config["latent_dim"], hidden_dim=config["hidden_dim"],
                variance_weight=config.get("variance_weight", 0.1),
                covariance_weight=config.get("covariance_weight", 0.0),
                regularization_view=config.get("regularization_view", "masked"),
                seed=seed + fold, device_name=device_name, **drug_parameters,
            )
        values = []
        for candidate in candidates:
            name = f"strength-{candidate:.6f}".rstrip("0").rstrip(".")
            pred_path = directory / f"{name}.h5ad"
            if kind == "genetic":
                predict_holdout(
                    inner_checkpoint, split_dir / "reference.h5ad",
                    split_dir / "input_controls.h5ad", pred_path,
                    cells_per_target=cells_per_target, device_name=device_name,
                    seed=seed + 10_000 + fold, steps=prediction_steps,
                    sampling="residual-rounding", effect_strength=candidate,
                    allow_unmatched_batch=parameters["allow_unmatched_batch"],
                    control_sampling=control_sampling,
                )
            else:
                from .drug import predict_drug

                predict_drug(
                    inner_checkpoint, split_dir / "reference.h5ad",
                    split_dir / "input_controls.h5ad", pred_path,
                    cells_per_target=cells_per_target, device_name=device_name,
                    seed=seed + 10_000 + fold, steps=prediction_steps,
                    sampling="residual-rounding", effect_strength=candidate,
                )
            result = score_local(
                pred_path, split_dir / "reference.h5ad", directory / f"{name}-score",
                control=checkpoint["control"], threads=threads, metrics=(objective,),
                no_target_genes=(kind == "drug"),
                stratify_by_batch=(kind == "drug"),
            )
            context_values = [
                scores[objective] for scores in result["raw_metrics"].values()
            ]
            if not context_values or any(
                value is None or not math.isfinite(value) for value in context_values
            ):
                raise ValueError(f"Undefined calibration objective in fold {fold}, strength {candidate}")
            values.append(float(np.mean(context_values)))
        fold_records.append({
            "fold": fold, "heldout_targets": sorted(heldout.tolist()),
            "values": values,
        })

    selected, means = _select_effect_strength(
        candidates, [record["values"] for record in fold_records],
        min_relative_improvement, require_every_fold_improvement,
    )
    baseline = means[candidates.index(0.0)]
    if (
        file_sha256(source_path) != source_digest
        or file_sha256(checkpoint_path) != checkpoint_digest
        or file_sha256(plan_path) != plan_digest
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
        "method": (
            "target-disjoint-training-folds-v1" if kind == "genetic"
            else "compound-disjoint-training-folds-v1"
        ),
        "checkpoint_sha256": checkpoint_digest,
        "training_source_sha256": source_digest,
        "plan_sha256": plan_digest,
        "training_identity": identity,
        "source": str(source_path),
        "objective": objective,
        "objective_note": objective_note,
        "no_target_genes": kind == "drug",
        "stratify_by_batch": kind == "drug",
        "sampling": "residual-rounding",
        "control_sampling": control_sampling,
        "strengths": candidates,
        "folds": fold_records,
        "mean_objective_by_strength": means,
        "baseline_objective": baseline,
        "min_relative_improvement": min_relative_improvement,
        "require_every_fold_improvement": require_every_fold_improvement,
        "selection_rule": plan["selection_rule"],
        "selected_effect_strength": selected,
        "cells_per_target": cells_per_target,
        "prediction_steps": prediction_steps,
        "seed": seed,
        "training_parameters": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in parameters.items()
        },
        "external_reference_identity": (
            reference_metadata["source_identity"] if reference_metadata else None
        ),
    }
    output = outdir / "effect-calibration.json"
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"Selected effect strength {selected}; calibration saved to {output}", flush=True)
    return output
