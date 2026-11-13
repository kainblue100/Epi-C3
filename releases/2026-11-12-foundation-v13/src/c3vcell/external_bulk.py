"""Exploratory cross-study comparison against replicated bulk TPM."""

from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import spearmanr

from .data import CONTROL, validate_counts
from .integrity import file_sha256
from .predict import _contexts, _labels


def _mean_panel_cpm(matrix: sp.spmatrix) -> np.ndarray:
    counts = sp.csr_matrix(matrix, dtype=np.float64)
    totals = np.asarray(counts.sum(axis=1)).ravel()
    if not np.isfinite(totals).all() or np.any(totals <= 0):
        raise ValueError("Every cell must have positive counts on the common gene panel")
    return np.asarray(counts.multiply((1_000_000.0 / totals)[:, None]).mean(axis=0)).ravel()


def _correlation(left: np.ndarray, right: np.ndarray, method: str) -> float | None:
    if len(left) < 2 or np.all(left == left[0]) or np.all(right == right[0]):
        return None
    result = np.corrcoef(left, right)[0, 1] if method == "pearson" else spearmanr(left, right).statistic
    if not np.isfinite(result):
        raise ValueError(f"{method} correlation is non-finite for nonconstant inputs")
    return float(result)


def _panel_metrics(predicted: np.ndarray, observed: np.ndarray) -> dict:
    if len(predicted) == 0:
        return {"n_genes": 0, "undefined_reason": "No genes meet the panel criteria"}
    squared_error = float(np.mean((predicted - observed) ** 2))
    baseline_squared_error = float(np.mean(observed ** 2))
    directional = (predicted != 0) & (observed != 0)
    return {
        "n_genes": len(predicted),
        "pearson_lfc": _correlation(predicted, observed, "pearson"),
        "spearman_lfc": _correlation(predicted, observed, "spearman"),
        "mean_absolute_lfc_error": float(np.mean(np.abs(predicted - observed))),
        "mean_squared_lfc_error": squared_error,
        "zero_response_mean_squared_lfc_error": baseline_squared_error,
        "relative_mse_improvement_vs_zero": (
            1.0 - squared_error / baseline_squared_error if baseline_squared_error > 0 else None
        ),
        "predicted_nonzero_fraction": float(np.mean(predicted != 0)),
        "direction_comparable_genes": int(np.count_nonzero(directional)),
        "direction_agreement_among_comparable": (
            float(np.mean(np.sign(predicted[directional]) == np.sign(observed[directional])))
            if directional.any() else None
        ),
        "correlation_note": (
            "Null indicates fewer than two genes or a constant observed or predicted vector."
        ),
    }


def compare_bulk_tpm(
    prediction_path: Path,
    controls_path: Path,
    bulk_path: Path,
    output_path: Path,
    *,
    target: str,
    context: str,
    control_samples: list[str],
    perturbed_samples: list[str],
    minimum_tpm: float = 1.0,
    pseudocount: float = 1.0,
    control_label: str = CONTROL,
) -> None:
    """Compare gene-wise population LFCs; this is not a single-cell VCC score."""
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite {output_path}")
    if (
        not target or not context or minimum_tpm < 0 or not np.isfinite(minimum_tpm)
        or pseudocount <= 0 or not np.isfinite(pseudocount)
    ):
        raise ValueError("Target, context and finite nonnegative thresholds are required")
    samples = [*control_samples, *perturbed_samples]
    if (
        len(control_samples) < 2 or len(perturbed_samples) < 2
        or any(not sample for sample in samples) or len(set(samples)) != len(samples)
    ):
        raise ValueError("Require at least two unique, disjoint samples in each bulk group")
    inputs = {
        "prediction_sha256": prediction_path,
        "controls_sha256": controls_path,
        "bulk_sha256": bulk_path,
    }
    digests = {name: file_sha256(path) for name, path in inputs.items()}
    prediction = ad.read_h5ad(prediction_path)
    controls = ad.read_h5ad(controls_path)
    validate_counts(prediction)
    validate_counts(controls)
    if prediction.var_names.tolist() != controls.var_names.tolist():
        raise ValueError("Prediction and controls must have identical ordered genes")
    if set(_labels(prediction.obs)) != {target} or set(_contexts(prediction.obs)) != {context}:
        raise ValueError("Prediction must contain exactly the requested target and context")
    if set(_labels(controls.obs)) != {control_label}:
        raise ValueError("Input controls must contain only control cells")
    metadata = prediction.uns.get("c3vcell_prediction", {})
    if metadata.get("control_sampling") != "input-controls":
        raise ValueError("Prediction must use input-controls sampling")
    if metadata.get("input_controls_sha256") != digests["controls_sha256"]:
        raise ValueError("Prediction was not bound to this exact control file")
    if "source_control_id" not in prediction.obs or prediction.obs["source_control_id"].isna().any():
        raise ValueError("Predictions need their source_control_id for paired comparison")
    source_rows = controls.obs_names.get_indexer(prediction.obs["source_control_id"].astype(str))
    if np.any(source_rows < 0):
        raise ValueError("A predicted cell references a missing source control")
    if np.any(_contexts(controls.obs)[source_rows] != context):
        raise ValueError("Source controls do not match the prediction context")

    bulk = pd.read_csv(bulk_path, sep="\t", dtype={"Symbol": str}, keep_default_na=False)
    if "Symbol" not in bulk.columns or not set(samples) <= set(bulk.columns):
        raise ValueError("Bulk data are missing Symbol or requested sample columns")
    if bulk["Symbol"].eq("").any() or bulk["Symbol"].duplicated().any():
        raise ValueError("Bulk gene symbols must be unique and nonempty")
    bulk = bulk.set_index("Symbol")
    values = bulk[samples].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(values.to_numpy()).all() or (values.to_numpy() < 0).any():
        raise ValueError("Bulk TPM values must be finite and nonnegative")
    shared = [gene for gene in prediction.var_names.astype(str) if gene in values.index]
    if target not in shared or len(shared) < 3:
        raise ValueError("The target and at least two other genes must be shared with the bulk data")
    common = [gene for gene in shared if gene != target]
    indices = prediction.var_names.get_indexer(common)
    bulk_common = values.loc[common]
    bulk_totals = bulk_common.sum(axis=0)
    if (bulk_totals <= 0).any() or not np.isfinite(bulk_totals.to_numpy()).all():
        raise ValueError("Every bulk sample must have positive TPM on the common gene panel")
    bulk_cpm = bulk_common.divide(bulk_totals, axis=1) * 1_000_000.0
    observed_control = bulk_cpm[control_samples].mean(axis=1).to_numpy()
    observed_perturbed = bulk_cpm[perturbed_samples].mean(axis=1).to_numpy()
    observed_lfc = np.log2(observed_perturbed + pseudocount) - np.log2(observed_control + pseudocount)
    pred_mean = _mean_panel_cpm(sp.csr_matrix(prediction.X)[:, indices])
    control_mean = _mean_panel_cpm(sp.csr_matrix(controls.X)[source_rows][:, indices])
    predicted_lfc = np.log2(pred_mean + pseudocount) - np.log2(control_mean + pseudocount)
    raw_tpm = bulk_common[[*control_samples, *perturbed_samples]]
    expressed = np.maximum(
        raw_tpm[control_samples].mean(axis=1).to_numpy(),
        raw_tpm[perturbed_samples].mean(axis=1).to_numpy(),
    ) >= minimum_tpm
    panels = {
        "all_common_nontarget": np.ones(len(common), dtype=bool),
        "expressed_common_nontarget": expressed,
    }
    if any(file_sha256(path) != digests[name] for name, path in inputs.items()):
        raise RuntimeError("Comparison input changed while it was being read")
    result = {
        "kind": "exploratory-cross-study-bulk-tpm-comparison",
        "not_a_vcc_score": True,
        "target": target,
        "context": context,
        "control_samples": control_samples,
        "perturbed_samples": perturbed_samples,
        "n_prediction_cells": prediction.n_obs,
        "n_shared_genes_in_normalization": len(common),
        "minimum_original_tpm": minimum_tpm,
        "pseudocount_panel_cpm": pseudocount,
        "inputs_sha256": digests,
        "method": (
            "Exclude the perturbed gene, normalize each bulk sample and each predicted or "
            "matched source-control cell to one million on the same shared gene panel, and "
            "compare log2 fold changes of group mean normalized values."
        ),
        "limitations": [
            "Bulk TPM and single-cell counts are different assays; gene-length normalization, "
            "cell composition, perturbation mode and study conditions can differ.",
            "Gene-wise correlations do not constitute independent gene replicates or VCC scores.",
            "Single-target results cannot establish generalization across perturbations.",
            "Low-expression filtering uses the external bulk measurements.",
        ],
        "panels": {
            name: _panel_metrics(predicted_lfc[mask], observed_lfc[mask])
            for name, mask in panels.items()
        },
    }
    serialized = json.dumps(result, indent=2, allow_nan=False) + "\n"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as stream:
        stream.write(serialized)
