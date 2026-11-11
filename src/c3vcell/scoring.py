"""Invoke the official cell-eval2 VCC 2026 preset on local held-out data."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from .data import validate_counts
from .identity import check_cell_id_schemes
from .integrity import file_sha256


SCORED_METRICS = (
    "pds_cosine",
    "expr_mse_unbiased_capped_norm",
    "de_wilcoxon_direction_fidelity_yield_raw",
    "de_wilcoxon_direction_reach_raw",
    "de_wilcoxon_sig_jaccard",
    "de_wilcoxon_lfc_nmae",
)
DIAGNOSTIC_METRICS = ("expr_mse_unbiased_capped", "expr_distance_unbiased")
AVAILABLE_METRICS = SCORED_METRICS + DIAGNOSTIC_METRICS


def _run(arguments: list[str]) -> None:
    print("Running:", " ".join(arguments), flush=True)
    subprocess.run(arguments, check=True)


def _find_scorer() -> str:
    executable = "cell-eval2.exe" if os.name == "nt" else "cell-eval2"
    adjacent = Path(sys.executable).parent / executable
    if adjacent.is_file() and os.access(adjacent, os.X_OK):
        return str(adjacent)
    scorer = shutil.which("cell-eval2")
    if scorer is None:
        raise RuntimeError("cell-eval2 is not installed in the active environment or PATH")
    return scorer


def _read_raw_metrics(
    path: Path, metrics: tuple[str, ...] = SCORED_METRICS
) -> tuple[dict[str, float | None], list[str]]:
    frame = pd.read_csv(path)
    if "statistic" not in frame or set(metrics) - set(frame.columns):
        raise ValueError(f"Official aggregate is missing expected metric columns: {path}")
    mean = frame.loc[frame["statistic"] == "mean"]
    if len(mean) != 1:
        raise ValueError(f"Official aggregate must contain exactly one mean row: {path}")
    result: dict[str, float | None] = {}
    undefined = []
    for metric in metrics:
        value = float(mean.iloc[0][metric])
        if np.isfinite(value):
            result[metric] = value
        else:
            result[metric] = None
            undefined.append(metric)
    return result, undefined


def score_local(
    prediction_path: Path,
    reference_path: Path,
    outdir: Path,
    *,
    control: str = "non-targeting",
    backend: str = "pdex",
    calibrate: bool = False,
    threads: int = 2,
    metrics: tuple[str, ...] = SCORED_METRICS,
    no_target_genes: bool = False,
    stratify_by_batch: bool = False,
) -> dict:
    if threads < 1:
        raise ValueError("threads must be positive")
    if not metrics or len(set(metrics)) != len(metrics) or not set(metrics) <= set(AVAILABLE_METRICS):
        raise ValueError("Metrics must be a nonempty, unique subset of supported VCC and diagnostic metrics")
    if calibrate and metrics != SCORED_METRICS:
        raise ValueError("Six-metric score calibration requires all six metrics")
    scorer = _find_scorer()
    if outdir.exists() and any(outdir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty directory: {outdir}")
    prediction_sha256 = file_sha256(prediction_path)
    reference_sha256 = file_sha256(reference_path)
    prediction = ad.read_h5ad(prediction_path)
    reference = ad.read_h5ad(reference_path)
    for name, dataset in (("prediction", prediction), ("reference", reference)):
        validate_counts(dataset, allow_zero_cells=True)
        if "target_gene" not in dataset.obs or dataset.obs["target_gene"].isna().any():
            raise ValueError(f"{name} needs a non-null obs['target_gene'] column")
        if not dataset.obs_names.is_unique:
            raise ValueError(f"{name} has duplicate cell identifiers")
    if (prediction.obs["target_gene"].astype(str) == control).any():
        raise ValueError("Prediction must contain perturbations only, without control rows")
    if "source_control_id" in prediction.obs:
        if prediction.obs["source_control_id"].isna().any():
            raise ValueError("Prediction source control IDs cannot be missing")
        input_controls = prediction.obs["source_control_id"].astype(str)
        check_cell_id_schemes(input_controls, reference.obs_names)
        if set(input_controls) & set(reference.obs_names.astype(str)):
            raise ValueError("Prediction source controls overlap scoring reference cells")
    if "context" not in prediction.obs or "context" not in reference.obs:
        raise ValueError("Both files need obs['context']")
    if prediction.obs["context"].isna().any() or reference.obs["context"].isna().any():
        raise ValueError("Context labels cannot be missing")
    if stratify_by_batch:
        for name, dataset in (("prediction", prediction), ("reference", reference)):
            if "batch" not in dataset.obs or dataset.obs["batch"].isna().any():
                raise ValueError(f"{name} needs non-null batch labels for stratified scoring")
            dataset.obs["context"] = [
                json.dumps([context, batch], ensure_ascii=False, separators=(",", ":"))
                for context, batch in zip(
                    dataset.obs["context"].astype(str), dataset.obs["batch"].astype(str)
                )
            ]
    pred_contexts = set(prediction.obs["context"].astype(str))
    ref_contexts = set(reference.obs["context"].astype(str))
    if pred_contexts != ref_contexts:
        raise ValueError("Prediction and reference contexts differ")
    if prediction.var_names.tolist() != reference.var_names.tolist():
        raise ValueError("Prediction and reference gene axes differ")
    totals = np.asarray(prediction.X.sum(axis=1)).ravel()
    if np.any(totals > 1_000_000):
        raise ValueError("Prediction cells cannot exceed 1,000,000 total counts")
    groups = pd.DataFrame({
        "context": prediction.obs["context"].astype(str).to_numpy(),
        "target": prediction.obs["target_gene"].astype(str).to_numpy(),
        "total": totals,
    })
    if (groups.groupby(["context", "target"], observed=True)["total"].sum() == 0).any():
        raise ValueError("An entire context/perturbation prediction group has zero counts")
    if not no_target_genes:
        missing = sorted(
            set(prediction.obs["target_gene"].astype(str)) - set(prediction.var_names.astype(str))
        )
        if missing:
            raise ValueError(f"Scoring target genes are absent from the gene axis: {missing[:20]}")
    outdir.mkdir(parents=True, exist_ok=True)
    config = [
        "--preset", "vcc2026",
        "--pert-col", "target_gene",
        "--control", control,
        "--set", f"de.backend={backend}",
        "--set", "device=cpu",
        "--set", f"num_threads={threads}",
        "--set", f"gather_threads={threads}",
    ]
    if metrics != SCORED_METRICS:
        config.extend(["--set", f"metrics={json.dumps(list(metrics))}"])
    if no_target_genes:
        config.append("--no-target-genes")
    context_paths = {}
    raw_metrics = {}
    undefined_metrics = {}
    for index, context in enumerate(sorted(pred_contexts)):
        context_dir = outdir / f"context-{index:03d}"
        context_paths[context] = context_dir.name
        context_dir.mkdir()
        pred_file = context_dir / "prediction.h5ad"
        real_file = context_dir / "reference.h5ad"
        pred_context = prediction[prediction.obs["context"].astype(str) == context]
        real_context = reference[reference.obs["context"].astype(str) == context]
        real_labels = real_context.obs["target_gene"].astype(str)
        pred_targets = set(pred_context.obs["target_gene"].astype(str))
        real_targets = set(real_labels) - {control}
        if pred_targets != real_targets:
            raise ValueError(f"Prediction and reference targets differ in context {context!r}")
        control_cells = real_context[real_labels == control]
        if not control_cells.n_obs:
            raise ValueError(f"Reference controls are missing for context {context!r}")
        # cell-eval2 requires matching label sets even though vcc2026 uses real controls.
        score_input = ad.concat(
            [pred_context, control_cells], axis=0, join="inner", index_unique="-"
        )
        score_input.write_h5ad(pred_file)
        real_context.write_h5ad(real_file)
        run_dir = context_dir / "run"
        _run([scorer, "run", "-ap", str(pred_file), "-ar", str(real_file), *config, "-o", str(run_dir)])
        raw_metrics[context], undefined = _read_raw_metrics(run_dir / "agg_results.csv", metrics)
        if undefined:
            undefined_metrics[context] = undefined
            print(
                f"WARNING: undefined metrics for context {context!r}: {undefined}. "
                "The local sample may be too small to define these scores.",
                flush=True,
            )
            if calibrate:
                raise ValueError(f"Cannot calibrate undefined metrics for context {context!r}")
        if calibrate:
            baseline_dir = context_dir / "baseline"
            baseline_pred = context_dir / "baseline.h5ad"
            _run([
                scorer, "baseline", "-ar", str(real_file), *config,
                "--save-pred", str(baseline_pred), "-o", str(baseline_dir),
            ])
            bundle_dir = context_dir / "local-reference-bundle"
            _run([
                scorer, "prep-real-bundle", "--real", str(real_file),
                "--baseline", str(baseline_pred), *config, "-o", str(bundle_dir),
            ])
            _run([
                scorer, "score", "--user-agg", str(run_dir / "agg_results.csv"),
                "--real-bundle", str(bundle_dir), "-o", str(context_dir / "local_scaled_scores.csv"),
            ])
    if file_sha256(prediction_path) != prediction_sha256 or file_sha256(reference_path) != reference_sha256:
        raise RuntimeError("Scoring inputs changed during evaluation")
    metadata = {
        "implementation": "cell-eval2",
        "required_version": "0.22.0",
        "preset": "vcc2026",
        "backend": backend,
        "metrics": list(metrics),
        "diagnostic_metrics": [metric for metric in metrics if metric in DIAGNOSTIC_METRICS],
        "no_target_genes": no_target_genes,
        "stratify_by_batch": stratify_by_batch,
        "threads": threads,
        "prediction_sha256": prediction_sha256,
        "reference_sha256": reference_sha256,
        "contexts": sorted(pred_contexts),
        "context_directories": context_paths,
        "raw_metrics": raw_metrics,
        "undefined_metrics": undefined_metrics,
        "local_reference_calibration": calibrate,
        "official_leaderboard_score": False,
        "evaluation_controls": (
            "Reference controls were appended only to the scorer input; "
            "the submitted prediction file was not modified."
        ),
        "notes": (
            "Scored raw metrics use the official preset; diagnostic metrics are not scored "
            "challenge metrics. Local reference anchors, if requested, are estimated from this "
            "held-out dataset and are not the challenge's private anchors. "
            "When no_target_genes is enabled, no genetic target is excluded; "
            "the result is not a VCC genetic-perturbation score."
        ),
    }
    (outdir / "scoring-provenance.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return metadata


def score_single_context_observable(
    prediction_path: Path,
    reference_path: Path,
    outdir: Path,
    *,
    context: str,
    threads: int = 2,
    scorer: Callable[..., dict] = score_local,
) -> dict:
    """Keep a nonpositive expression denominator explicitly undefined."""
    denominator = scorer(
        prediction_path, reference_path, outdir.with_name(outdir.name + "-denominator"),
        threads=threads, metrics=("expr_distance_unbiased",),
    )
    values = {
        key: scores["expr_distance_unbiased"]
        for key, scores in denominator["raw_metrics"].items()
    }
    if set(values) != {context} or values[context] is None or not np.isfinite(values[context]):
        raise ValueError(f"The {context} expression denominator diagnostic is unavailable")
    undefined_expression = values[context] <= 0
    expression_metric = "expr_mse_unbiased_capped_norm"
    metrics = tuple(
        metric for metric in SCORED_METRICS
        if not undefined_expression or metric != expression_metric
    )
    if undefined_expression:
        print(
            f"WARNING: {context}'s expression denominator is nonpositive; the normalized "
            "expression metric will be recorded as undefined.", flush=True,
        )
    scored = scorer(prediction_path, reference_path, outdir, threads=threads, metrics=metrics)
    if undefined_expression:
        scored["raw_metrics"][context][expression_metric] = None
        scored["undefined_metrics"].setdefault(context, []).append(expression_metric)
    result = {
        "raw_metrics": scored["raw_metrics"],
        "undefined_metrics": scored["undefined_metrics"],
        "expression_denominator_diagnostic": values,
        "normalized_expression_status": (
            "undefined: nonpositive real-effect denominator" if undefined_expression else "defined"
        ),
        "official_leaderboard_score": False,
    }
    with (outdir / "evaluation-summary.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return result
