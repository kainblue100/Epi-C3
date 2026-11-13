"""Deterministic evidence review for repeated HCT116 network diagnostics."""

from __future__ import annotations

import json
import math
from pathlib import Path

import anndata as ad
import pandas as pd
import torch

from .data import CONTROL
from .epigenetics import NETWORK_FEATURES, load_epigenetic_bundle
from .identity import check_cell_id_schemes
from .integrity import file_sha256
from .scoring import SCORED_METRICS

METRIC_DIRECTIONS = {
    "pds_cosine": 1,
    "expr_mse_unbiased_capped_norm": -1,
    "de_wilcoxon_direction_fidelity_yield_raw": 1,
    "de_wilcoxon_direction_reach_raw": 1,
    "de_wilcoxon_sig_jaccard": 1,
    "de_wilcoxon_lfc_nmae": -1,
}
CELLFORGE_REVISION = "6585ff93e452a8a0f2ad6ed8144ab6d998a3b400"


def _read_json(path: Path) -> dict:
    digest = file_sha256(path)
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict) or file_sha256(path) != digest:
        raise ValueError(f"Invalid or changing JSON artifact: {path}")
    return value


def _require_hash(path: Path, expected: str) -> None:
    if not isinstance(expected, str) or file_sha256(path) != expected:
        raise ValueError(f"Artifact fingerprint mismatch: {path}")


def _check_scores(scores: dict, undefined: dict | None = None) -> None:
    if not isinstance(scores, dict) or not scores:
        raise ValueError("A score needs at least one context")
    if undefined is not None and (not isinstance(undefined, dict) or set(undefined) - set(scores)):
        raise ValueError("Undefined metrics contain unexpected contexts")
    for context, metrics in scores.items():
        if not isinstance(context, str) or not context or not isinstance(metrics, dict):
            raise ValueError("Invalid scoring context")
        if set(metrics) != set(SCORED_METRICS):
            raise ValueError(f"Incorrect VCC metric set for context {context}")
        missing = []
        for name, value in metrics.items():
            if value is None:
                missing.append(name)
            elif isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"Invalid metric value for {context}: {name}")
        if undefined is not None:
            recorded = undefined.get(context, [])
            if (
                not isinstance(recorded, list) or not all(isinstance(item, str) for item in recorded)
                or len(recorded) != len(set(recorded))
                or set(recorded) != set(missing)
            ):
                raise ValueError(f"Undefined-metric record does not match scores for {context}")


def compare_scores(candidate: dict, baselines: dict[str, dict]) -> dict:
    """Compare raw metrics without interpreting undefined values as zero or proof."""
    _check_scores(candidate)
    for scores in baselines.values():
        _check_scores(scores)
        if set(scores) != set(candidate):
            raise ValueError("Candidate and baseline contexts differ")
    comparisons = {}
    for context, metrics in candidate.items():
        comparisons[context] = {}
        for name, direction in METRIC_DIRECTIONS.items():
            value = metrics[name]
            row = {"candidate": value, "direction": "higher" if direction == 1 else "lower"}
            for label, scores in baselines.items():
                baseline = scores[context][name]
                row[label] = baseline
                row[f"improvement_vs_{label}"] = (
                    direction * (value - baseline) if value is not None and baseline is not None else None
                )
            comparisons[context][name] = row
    return comparisons


def _check_score_provenance(
    path: Path, *, prediction_sha256: str, reference_sha256: str,
    raw_metrics: dict, undefined_metrics: dict | None,
) -> None:
    metadata = _read_json(path)
    if (
        metadata.get("implementation") != "cell-eval2"
        or metadata.get("required_version") != "0.22.0"
        or metadata.get("preset") != "vcc2026"
        or metadata.get("metrics") != list(SCORED_METRICS)
        or metadata.get("no_target_genes") is not False
        or metadata.get("official_leaderboard_score") is not False
        or metadata.get("local_reference_calibration") is not False
        or metadata.get("prediction_sha256") != prediction_sha256
        or metadata.get("reference_sha256") != reference_sha256
        or metadata.get("raw_metrics") != raw_metrics
        or (undefined_metrics is not None and metadata.get("undefined_metrics") != undefined_metrics)
    ):
        raise ValueError(f"Scoring provenance does not match the experiment: {path}")
    _check_scores(raw_metrics, metadata.get("undefined_metrics"))


def _coverage(bundle: Path, heldout: list[str], data: ad.AnnData) -> dict:
    _, masks, metadata = load_epigenetic_bundle(bundle, data)
    genes = data.var_names.astype(str).tolist()
    indices = {gene: index for index, gene in enumerate(genes)}
    if not set(heldout) <= set(indices):
        raise ValueError("A held-out target is absent from the epigenetic gene axis")
    graph = pd.read_csv(bundle / "graph.csv", keep_default_na=False)
    features = ("promoter_methylation", "linked_element_methylation")
    columns = {name: NETWORK_FEATURES.index(name) for name in features}
    per_target = {}
    for target in heldout:
        neighbors = set(graph.loc[graph["source"] == target, "target"])
        reachable = neighbors | {target}
        per_target[target] = {
            "outgoing_neighbors": len(neighbors),
            **{
                f"{name}_measured_in_target_or_neighbors": sum(
                    int(masks[indices[gene], column]) for gene in reachable
                )
                for name, column in columns.items()
            },
        }
    return {
        "measured_genes": {
            name: int(masks[:, column].sum()) for name, column in columns.items()
        },
        "per_heldout_target": per_target,
        "note": (
            "A measured mask indicates an observed CpG in a region, not read depth or causal "
            "evidence. Neighbor counts follow the graph's source-to-target message convention."
        ),
        "manifest_sha256": metadata["manifest_sha256"],
    }


def _validate_split(
    train: ad.AnnData, reference_path: Path, controls_path: Path,
    training: list[str], heldout: list[str],
) -> dict:
    reference = ad.read_h5ad(reference_path, backed="r")
    controls = ad.read_h5ad(controls_path, backed="r")
    try:
        datasets = (train, reference, controls)
        if any(
            not data.obs_names.is_unique or not data.var_names.is_unique
            or "target_gene" not in data.obs or data.obs["target_gene"].isna().any()
            for data in datasets
        ):
            raise ValueError("Split cells, genes, or perturbation labels are invalid")
        if any(data.var_names.tolist() != train.var_names.tolist() for data in datasets[1:]):
            raise ValueError("Split gene axes differ")
        ids = [set(data.obs_names.astype(str)) for data in datasets]
        check_cell_id_schemes(*ids)
        if ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2]:
            raise ValueError("Training, evaluation, and input-control cells overlap")
        labels = [set(data.obs["target_gene"].astype(str)) for data in datasets]
        if labels != [set(training) | {CONTROL}, set(heldout) | {CONTROL}, {CONTROL}]:
            raise ValueError("Split perturbation labels disagree with the experiment plan")
        return {
            "training_cells": train.n_obs,
            "reference_cells": reference.n_obs,
            "input_control_cells": controls.n_obs,
            "genes": train.n_vars,
            "input_control_ids": ids[2],
            "reference_ids": ids[1],
        }
    finally:
        reference.file.close()
        controls.file.close()


def _validate_prediction(
    path: Path, *, genes: list[str], heldout: list[str], cells_per_target: int, split: dict,
) -> None:
    data = ad.read_h5ad(path, backed="r")
    try:
        if (
            not data.obs_names.is_unique or data.var_names.tolist() != genes
            or "target_gene" not in data.obs or data.obs["target_gene"].isna().any()
            or "source_control_id" not in data.obs or data.obs["source_control_id"].isna().any()
        ):
            raise ValueError(f"Prediction metadata or gene axis is invalid: {path}")
        counts = data.obs["target_gene"].astype(str).value_counts().to_dict()
        if counts != {target: cells_per_target for target in heldout}:
            raise ValueError(f"Prediction targets or cell counts disagree with the plan: {path}")
        input_ids = set(data.obs["source_control_id"].astype(str))
        if not input_ids <= split["input_control_ids"] or input_ids & split["reference_ids"]:
            raise ValueError(f"Prediction controls disagree with the held-out split: {path}")
    finally:
        data.file.close()


def review_network_diagnostic(
    run: Path, parent_run: Path, parent_diagnostic: Path, spec: Path, *,
    source_archive: Path | None = None, program: Path | None = None,
) -> dict:
    """Verify a repeated, uncalibrated diagnostic and prevent validation claims."""
    plan_digest = file_sha256(run / "diagnostic-plan.json")
    result_digest = file_sha256(run / "result.json")
    plan = _read_json(run / "diagnostic-plan.json")
    parent = _read_json(parent_run / "experiment-plan.json")
    previous_plan = _read_json(parent_diagnostic / "diagnostic-plan.json")
    previous_result = _read_json(parent_diagnostic / "result.json")
    result = _read_json(run / "result.json")
    paths = {
        "parent_plan_sha256": parent_run / "experiment-plan.json",
        "previous_diagnostic_plan_sha256": parent_diagnostic / "diagnostic-plan.json",
        "previous_results_sha256": parent_diagnostic / "result.json",
        "network_spec_sha256": spec,
        "source_sha256": parent_run / "split" / "train.h5ad",
        "reference_sha256": parent_run / "split" / "reference.h5ad",
        "controls_sha256": parent_run / "split" / "input_controls.h5ad",
    }
    for key, path in paths.items():
        _require_hash(path, plan.get(key))
    if plan.get("source_archive_sha256") is not None:
        if source_archive is None:
            raise ValueError("A recorded source archive must be supplied for review")
        _require_hash(source_archive, plan["source_archive_sha256"])
    if program is not None:
        _require_hash(program, plan.get("program_sha256"))
    calibration_status = previous_plan.get("calibration_status")
    cells_per_target = plan.get("prediction_cells_per_target")
    if (
        plan.get("arms") != ["real-network", "shuffled-network"]
        or plan.get("effect_strength") != 1.0
        or not isinstance(cells_per_target, int) or isinstance(cells_per_target, bool)
        or cells_per_target < 1
        or previous_plan.get("parent_plan_sha256") != plan["parent_plan_sha256"]
        or previous_plan.get("source_sha256") != plan["source_sha256"]
        or previous_plan.get("reference_sha256") != plan["reference_sha256"]
        or previous_plan.get("controls_sha256") != plan["controls_sha256"]
        or previous_plan.get("diagnostic_strength") != 1.0
        or not isinstance(calibration_status, str) or not calibration_status.startswith("failed:")
        or "nonpositive" not in calibration_status
    ):
        raise ValueError("This review only supports the recorded uncalibrated repeated diagnostic")
    copied = result.get("previous_diagnostic", {})
    if (
        copied.get("result_sha256") != plan["previous_results_sha256"]
        or copied.get("control_baseline") != previous_result.get("control-baseline")
        or copied.get("no_prior") != previous_result.get("no-prior")
        or copied.get("v1_prior") != previous_result.get("hct116-epigenome")
    ):
        raise ValueError("Copied baseline results differ from the previous diagnostic")
    heldout, training = parent.get("heldout_targets"), parent.get("training_targets")
    if (
        not isinstance(heldout, list) or not heldout
        or not all(isinstance(target, str) and target for target in heldout)
        or len(set(heldout)) != len(heldout)
        or not isinstance(training, list) or not training
        or not all(isinstance(target, str) and target for target in training)
        or len(set(training)) != len(training)
        or CONTROL in heldout or CONTROL in training or set(heldout) & set(training)
    ):
        raise ValueError("The parent experiment does not have disjoint target lists")
    baseline_prediction = parent_run / "control-baseline.h5ad"
    baseline_score = _read_json(parent_run / "control-baseline-score" / "scoring-provenance.json")
    _require_hash(baseline_prediction, baseline_score.get("prediction_sha256"))
    baseline = previous_result["control-baseline"]
    _check_score_provenance(
        parent_run / "control-baseline-score" / "scoring-provenance.json",
        prediction_sha256=baseline_score["prediction_sha256"], reference_sha256=plan["reference_sha256"],
        raw_metrics=baseline, undefined_metrics=None,
    )
    no_prior = previous_result["no-prior"]
    _require_hash(parent_run / "no-prior" / "model" / "model.pt", no_prior.get("model_sha256"))
    _require_hash(
        parent_diagnostic / "no-prior" / "diagnostic-strength-1.h5ad",
        no_prior.get("prediction_sha256"),
    )
    _check_score_provenance(
        parent_diagnostic / "no-prior" / "score" / "scoring-provenance.json",
        prediction_sha256=no_prior["prediction_sha256"], reference_sha256=plan["reference_sha256"],
        raw_metrics=no_prior["raw_metrics"], undefined_metrics=no_prior["undefined_metrics"],
    )
    data = ad.read_h5ad(paths["source_sha256"], backed="r")
    try:
        split = _validate_split(data, paths["reference_sha256"], paths["controls_sha256"], training, heldout)
        genes = data.var_names.astype(str).tolist()
        coverage = _coverage(run / "network", heldout, data)
        _, _, shuffled_metadata = load_epigenetic_bundle(run / "shuffled-network", data)
    finally:
        data.file.close()
    real_metadata = _read_json(run / "network" / "manifest.json")
    if (
        real_metadata.get("specification_sha256") != plan["network_spec_sha256"]
        or shuffled_metadata.get("control", {}).get("reference_manifest_sha256") != coverage["manifest_sha256"]
        or shuffled_metadata.get("control", {}).get("seed") != plan.get("shuffle_seed")
    ):
        raise ValueError("The real and shuffled epigenomic bundles do not match the plan")
    for arm, bundle in (("real-network", "network"), ("shuffled-network", "shuffled-network")):
        entry = result.get(arm, {})
        model = run / arm / "model" / "model.pt"
        prediction = run / arm / "diagnostic-strength-1.h5ad"
        _require_hash(model, entry.get("model_sha256"))
        _require_hash(prediction, entry.get("prediction_sha256"))
        if _read_json(run / arm / "result.json") != entry:
            raise ValueError(f"Per-arm result disagrees with the aggregate: {arm}")
        checkpoint = torch.load(model, map_location="cpu", weights_only=True)
        epigenetics = checkpoint.get("epigenetics")
        if (
            checkpoint.get("training_source_sha256") != plan["source_sha256"]
            or checkpoint.get("training_targets") != training
            or not isinstance(epigenetics, dict)
            or epigenetics.get("manifest_sha256") != file_sha256(run / bundle / "manifest.json")
        ):
            raise ValueError(f"Checkpoint provenance disagrees with the experiment: {arm}")
        _validate_prediction(
            prediction, genes=genes, heldout=heldout,
            cells_per_target=cells_per_target, split=split,
        )
        _check_score_provenance(
            run / arm / "score" / "scoring-provenance.json",
            prediction_sha256=entry["prediction_sha256"], reference_sha256=plan["reference_sha256"],
            raw_metrics=entry["raw_metrics"], undefined_metrics=entry["undefined_metrics"],
        )
    for key, path in paths.items():
        _require_hash(path, plan[key])
    _require_hash(run / "diagnostic-plan.json", plan_digest)
    _require_hash(run / "result.json", result_digest)
    if source_archive is not None and plan.get("source_archive_sha256") is not None:
        _require_hash(source_archive, plan["source_archive_sha256"])
    if program is not None:
        _require_hash(program, plan["program_sha256"])
    comparisons = compare_scores(
        result["real-network"]["raw_metrics"],
        {
            "no_response": baseline,
            "no_prior": no_prior["raw_metrics"],
            "shuffled_network": result["shuffled-network"]["raw_metrics"],
        },
    )
    return {
        "format": "c3vcell-network-research-review-v1",
        "method_reference": {
            "url": "https://github.com/gersteinlab/CellForge",
            "revision": CELLFORGE_REVISION,
            "adaptation": "task analysis, explicit alternatives, deterministic verification, evidence-gated iteration",
        },
        "reviewer": {
            "type": "deterministic artifact and metric checks",
            "independent_expert_review_performed": False,
        },
        "input_sha256": {key: plan[key] for key in paths},
        "diagnostic_plan_sha256": plan_digest,
        "result_sha256": result_digest,
        "source_archive_verified": plan.get("source_archive_sha256") is not None,
        "program_verified": program is not None,
        "scope": {
            "type": "exploratory",
            "holdout_previously_scored": True,
            "calibration_status": previous_plan["calibration_status"],
            "effect_strength": plan["effect_strength"],
            "official_leaderboard_score": False,
            "training_targets": len(training),
            "heldout_targets": len(heldout),
            "split_sizes": {
                key: value for key, value in split.items() if not key.endswith("_ids")
            },
        },
        "methylation_coverage": coverage,
        "comparisons": comparisons,
        "gates": {
            "artifact_integrity": "passed",
            "independent_holdout": "blocked",
            "defined_inner_calibration": "blocked",
            "official_challenge_score": "not performed",
            "validated_improvement_claim": "blocked",
        },
        "limitations": [
            "The same outer holdout has been scored in earlier experiments.",
            "The inner calibration objective was undefined; fixed effect strength is uncalibrated.",
            "A shuffled network also randomizes observed feature values; it does not isolate graph topology.",
            "Reference bulk assays are not paired perturbation measurements or causal edges.",
            "Positive score differences in one exploratory split are not independent validation.",
        ],
    }
