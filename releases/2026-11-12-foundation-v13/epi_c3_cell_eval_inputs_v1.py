"""Prepare independent-control, per-cell predictions for official cell-eval."""

from __future__ import annotations

import hashlib
from typing import Mapping

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp


def _validate_counts(adata: ad.AnnData, name: str) -> None:
    if not adata.obs_names.is_unique or not adata.var_names.is_unique or not adata.n_obs or not adata.n_vars:
        raise ValueError(f"{name} must have nonempty, unique cell and gene axes")
    matrix = sp.csr_matrix(adata.X)
    if (
        not np.isfinite(matrix.data).all() or np.any(matrix.data < 0)
        or np.any(matrix.data != np.floor(matrix.data))
        or np.any(np.asarray(matrix.sum(axis=1)).ravel() <= 0)
    ):
        raise ValueError(f"{name} must contain finite, nonnegative per-cell counts")


def _log_normalize(adata: ad.AnnData) -> ad.AnnData:
    result = adata.copy()
    matrix = sp.csr_matrix(result.X, dtype=np.float64)
    totals = np.asarray(matrix.sum(axis=1)).ravel()
    matrix = matrix.multiply((10_000.0 / totals)[:, None]).tocsr()
    matrix.data = np.log1p(matrix.data).astype(np.float32)
    result.X = matrix.astype(np.float32)
    result.uns["log1p"] = {"base": None}
    return result


def prepare_cell_eval_inputs(
    reference: ad.AnnData,
    controls: ad.AnnData,
    predictions: Mapping[str, ad.AnnData],
    *,
    perturbation_column: str,
    control_label: str,
    source_control_column: str = "source_control_id",
    context_columns: tuple[str, ...] = ("context", "batch"),
    seed: int = 0,
    minimum_parents: int = 3,
) -> tuple[ad.AnnData, dict[str, ad.AnnData], dict]:
    if not predictions or not perturbation_column or not control_label or minimum_parents < 2:
        raise ValueError("Prediction arms, perturbation labels and a minimum parent count are required")
    _validate_counts(reference, "reference")
    _validate_counts(controls, "input controls")
    for arm, predicted in predictions.items():
        if not isinstance(arm, str) or not arm:
            raise ValueError("Prediction arms need nonempty names")
        _validate_counts(predicted, f"{arm} predictions")
        if not predicted.var_names.equals(reference.var_names):
            raise ValueError(f"{arm} predictions have a different ordered gene axis")
        if set(predicted.obs_names).intersection(reference.obs_names):
            raise ValueError(f"{arm} prediction identities overlap reference cells")
    if not controls.var_names.equals(reference.var_names):
        raise ValueError("Input controls have a different ordered gene axis")
    if set(reference.obs_names).intersection(controls.obs_names):
        raise ValueError("Reference and input-control cell identities overlap")
    for name, dataset in [("reference", reference), ("controls", controls), *predictions.items()]:
        for column in (perturbation_column, *context_columns):
            if column not in dataset.obs or dataset.obs[column].isna().any():
                raise ValueError(f"{name} lacks complete {column!r} metadata")
    for column in context_columns:
        values = {str(value) for dataset in (reference, controls, *predictions.values())
                  for value in dataset.obs[column].unique()}
        if len(values) != 1:
            raise ValueError(f"All inputs must have one matching {column!r}")
    if set(controls.obs[perturbation_column].astype(str)) != {control_label}:
        raise ValueError("Input-control observations must all have the control label")
    ref_labels = set(reference.obs[perturbation_column].astype(str))
    if control_label not in ref_labels:
        raise ValueError("Reference observations must include independent controls")
    treatments = ref_labels - {control_label}
    if not treatments:
        raise ValueError("Reference observations need at least one perturbation")

    ordered_ids = sorted(
        map(str, controls.obs_names),
        key=lambda item: hashlib.sha256(f"{seed}\0{item}".encode("utf-8")).digest(),
    )
    cut = len(ordered_ids) // 2
    parent_ids, predicted_control_ids = set(ordered_ids[:cut]), set(ordered_ids[cut:])
    if len(parent_ids) < minimum_parents or len(predicted_control_ids) < minimum_parents:
        raise ValueError("Too few independent input controls to split parents from predicted controls")

    anchor = next(iter(predictions.values()))
    if source_control_column not in anchor.obs or anchor.obs[source_control_column].isna().any():
        raise ValueError("Each predicted cell needs its source-control identity")
    selected_by_treatment: dict[str, list[str]] = {}
    source_ids = anchor.obs[source_control_column].astype(str)
    labels = anchor.obs[perturbation_column].astype(str)
    if not set(source_ids).issubset(set(map(str, controls.obs_names))):
        raise ValueError("Predictions refer to a cell outside the input-control set")
    if set(labels) != treatments:
        raise ValueError("Predicted and reference treatment labels do not match")
    for treatment in sorted(treatments):
        candidates = pd.DataFrame({
            "row": anchor.obs_names.astype(str),
            "source": source_ids.to_numpy(),
            "treatment": labels.to_numpy(),
        })
        candidates = candidates[
            (candidates["treatment"] == treatment) & candidates["source"].isin(parent_ids)
        ].sort_values(["source", "row"])
        chosen = candidates.drop_duplicates("source")
        if len(chosen) < minimum_parents:
            raise ValueError(f"Treatment {treatment!r} has too few independent prediction parents")
        selected_by_treatment[treatment] = chosen["row"].tolist()

    selected = [row for treatment in sorted(treatments) for row in selected_by_treatment[treatment]]
    anchor_metadata = anchor.obs.loc[selected, [perturbation_column, source_control_column]]
    ordered_control_rows = [item for item in ordered_ids if item in predicted_control_ids]
    predicted_controls = controls[ordered_control_rows].copy()
    predicted_controls.obs[source_control_column] = ""
    outputs = {}
    for arm, predicted in predictions.items():
        if not predicted.obs_names.equals(anchor.obs_names) or source_control_column not in predicted.obs:
            raise ValueError(f"{arm} prediction cells are not aligned with the other arms")
        metadata = predicted.obs.loc[selected, [perturbation_column, source_control_column]]
        if not metadata.astype(str).equals(anchor_metadata.astype(str)):
            raise ValueError(f"{arm} prediction identities differ from the other arms")
        combined = ad.concat(
            [predicted[selected].copy(), predicted_controls], axis=0, join="outer", merge="same",
        )
        if not combined.obs_names.is_unique:
            raise ValueError(f"{arm} prediction cells overlap the independent predicted controls")
        outputs[arm] = _log_normalize(combined)
    audit = {
        "normalization": "log1p(counts / per-cell library size * 10000), applied identically",
        "treatments": sorted(treatments),
        "reference_cells": reference.n_obs,
        "input_controls": controls.n_obs,
        "parent_control_pool": len(parent_ids),
        "independent_predicted_controls": len(predicted_control_ids),
        "unique_prediction_parents_per_treatment": {
            treatment: len(selected_by_treatment[treatment]) for treatment in sorted(treatments)
        },
        "predictions_selected": len(selected),
        "selection_seed": seed,
        "caveats": [
            "One prediction per source control and treatment prevents repeated parents being counted as independent cells.",
            "Predicted controls and prediction parents are disjoint, and both are disjoint from reference cells.",
            "Predictions across different treatments can share parents and are not independent biological replicates.",
        ],
    }
    return _log_normalize(reference), outputs, audit
