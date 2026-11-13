"""Audit matched pilot predictions and score them with cell-eval."""

import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path

import anndata as ad
import numpy as np
from scipy.stats import spearmanr


MODELS = ("state", "perturb_mean", "context_mean")
CONTROL = "non-targeting"
PER_TARGET_METRICS = (
    "discrimination_score_l1",
    "pearson_delta",
    "pr_auc",
    "de_spearman_lfc_sig",
    "overlap_at_N",
)


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def finite(value):
    if value is None:
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def all_condition_spearman(real, pred):
    if len(real) != len(pred) or len(real) < 2:
        return None
    if len(set(real)) < 2 or len(set(pred)) < 2:
        return None
    return finite(spearmanr(real, pred).statistic)


def summarize(per_model_rows, targets, n_features):
    clean = {}
    real_counts = {}
    for model, rows in per_model_rows.items():
        by_target = {str(row["perturbation"]): row for row in rows}
        if len(by_target) != len(rows) or set(by_target) != set(targets):
            raise ValueError(f"{model} has missing or repeated metric conditions")
        clean[model] = {}
        for target in targets:
            row = by_target[target]
            true_count = row.get("de_nsig_counts_real")
            pred_count = row.get("de_nsig_counts_pred")
            if true_count is None or pred_count is None:
                raise ValueError("DE count metric is missing")
            if int(true_count) != true_count or int(pred_count) != pred_count:
                raise ValueError("DE count is not an integer")
            true_count = int(true_count)
            pred_count = int(pred_count)
            if not (0 <= true_count <= n_features and 0 <= pred_count <= n_features):
                raise ValueError("DE count is outside the feature axis")
            if target in real_counts and real_counts[target] != true_count:
                raise ValueError("Real DE counts differ between model arms")
            real_counts[target] = true_count
            values = {metric: finite(row.get(metric)) for metric in PER_TARGET_METRICS}
            if true_count == 0:
                values["overlap_at_N"] = None
            if true_count == 0 or true_count == n_features:
                values["pr_auc"] = None
            if true_count < 2:
                values["de_spearman_lfc_sig"] = None
            values["de_nsig_counts_real"] = true_count
            values["de_nsig_counts_pred"] = pred_count
            clean[model][target] = values

    aggregate = {model: {} for model in per_model_rows}
    for metric in PER_TARGET_METRICS:
        common = [
            target for target in targets
            if all(clean[model][target][metric] is not None for model in per_model_rows)
        ]
        for model in per_model_rows:
            values = [clean[model][target][metric] for target in common]
            aggregate[model][metric] = {
                "value": float(np.mean(values)) if values else None,
                "n_conditions": len(values),
            }
    for model in per_model_rows:
        aggregate[model]["de_count_spearman_all_conditions"] = {
            "value": all_condition_spearman(
                [clean[model][target]["de_nsig_counts_real"] for target in targets],
                [clean[model][target]["de_nsig_counts_pred"] for target in targets],
            ),
            "n_conditions": len(targets),
        }
    return clean, aggregate


def check_arm(pred, real, manifest, plan):
    if not pred.var_names.equals(real.var_names) or pred.n_vars != 2000:
        raise ValueError("Real and predicted feature axes differ")
    if not pred.obs_names.is_unique or not real.obs_names.is_unique:
        raise ValueError("Duplicate output cell identities")
    if not np.isfinite(pred.X).all() or not np.isfinite(real.X).all():
        raise ValueError("Nonfinite output expression")
    expected_counts = {target: plan["cells_per_target"] for target in plan["test_targets"]}
    expected_counts[CONTROL] = len(plan["predicted_control_parent_pool"])
    if Counter(pred.obs["target_gene"].astype(str)) != expected_counts:
        raise ValueError("Predicted condition sizes differ from the frozen plan")
    if Counter(real.obs["target_gene"].astype(str)) != expected_counts:
        raise ValueError("Real condition sizes differ from predicted condition sizes")
    sources = pred.obs["pert_cell_barcode"].astype(str).tolist()
    parents = pred.obs["ctrl_cell_barcode"].astype(str).tolist()
    labels = pred.obs["target_gene"].astype(str).tolist()
    if set(sources) != set(plan["assignments"]):
        raise ValueError("Predicted rows differ from the frozen parent plan")
    if any(plan["assignments"][source] != parent for source, parent in zip(sources, parents, strict=True)):
        raise ValueError("Predicted parent assignment differs from the frozen plan")
    for target in plan["test_targets"]:
        target_parents = [parent for label, parent in zip(labels, parents, strict=True) if label == target]
        if len(target_parents) != len(set(target_parents)):
            raise ValueError("A perturbation has repeated parents")
        if not set(target_parents).issubset(plan["perturbation_parent_pool"]):
            raise ValueError("A perturbation used a control-only parent")
        pred_ids = {source for label, source in zip(labels, sources, strict=True) if label == target}
        real_ids = set(real.obs_names[real.obs["target_gene"].astype(str) == target].astype(str))
        if pred_ids != real_ids:
            raise ValueError("Real and predicted perturbation cells differ")
    control_parents = {parent for label, parent in zip(labels, parents, strict=True) if label == CONTROL}
    if control_parents != set(plan["predicted_control_parent_pool"]):
        raise ValueError("Predicted controls do not match the frozen pool")
    reference_ids = set(real.obs_names[real.obs["target_gene"].astype(str) == CONTROL].astype(str))
    if reference_ids != set(plan["reference_control_ids"]):
        raise ValueError("Real controls do not match the independent reference pool")
    if control_parents & set(plan["perturbation_parent_pool"]) or reference_ids & set(sources):
        raise ValueError("Predicted or reference control pools overlap")
    for _, group in pred.obs.groupby("set_index", observed=True):
        if group["target_gene"].nunique() != 1 or len(group) > 8:
            raise ValueError("A prediction set mixes conditions or exceeds the set length")
    unique_counts = []
    for target in plan["test_targets"]:
        values = np.asarray(pred.X[pred.obs["target_gene"].astype(str).to_numpy() == target])
        unique_counts.append(len(np.unique(values, axis=0)))
    return {
        "minimum_unique_prediction_vectors_per_condition": min(unique_counts),
        "maximum_unique_prediction_vectors_per_condition": max(unique_counts),
        "conditions_with_one_unique_prediction_vector": sum(n == 1 for n in unique_counts),
        "clipped_below_zero": manifest["clip_below_zero"],
        "clipped_above_fourteen": manifest["clip_above_fourteen"],
    }


def check_same_reference(first_pred, first_real, pred, real):
    cols = ("target_gene", "cell_type", "batch_var", "pert_cell_barcode", "ctrl_cell_barcode", "set_index")
    if not first_pred.var_names.equals(pred.var_names):
        raise ValueError("Model prediction feature axes differ")
    for col in cols:
        if first_pred.obs[col].astype(str).tolist() != pred.obs[col].astype(str).tolist():
            raise ValueError(f"Model prediction identities differ in {col}")
    if first_real.obs_names.astype(str).tolist() != real.obs_names.astype(str).tolist():
        raise ValueError("Real-cell identities differ between models")
    if first_real.obs["target_gene"].astype(str).tolist() != real.obs["target_gene"].astype(str).tolist():
        raise ValueError("Real-cell condition labels differ between models")
    if not np.array_equal(first_real.X, real.X):
        raise ValueError("Real expression differs between models")


def evaluate(prediction_root, parent_plan_path, output):
    from cell_eval import MetricsEvaluator
    from cell_eval.metrics import metrics_registry
    from cell_eval.utils import guess_is_lognorm

    if output.exists():
        raise FileExistsError(output)
    plan = json.loads(parent_plan_path.read_text(encoding="utf-8"))
    plan_hash = sha256(parent_plan_path)
    arms = {}
    audits = {}
    first_pred = first_real = None
    for model in MODELS:
        directory = prediction_root / model
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        if manifest["model"] != model or manifest["parent_plan_sha256"] != plan_hash:
            raise ValueError("Prediction manifest does not match model or frozen plan")
        pred_path = directory / "predicted.h5ad"
        real_path = directory / "real.h5ad"
        if sha256(pred_path) != manifest["predicted_h5ad_sha256"]:
            raise ValueError("Prediction file hash mismatch")
        if sha256(real_path) != manifest["real_h5ad_sha256"]:
            raise ValueError("Reference file hash mismatch")
        pred = ad.read_h5ad(pred_path)
        real = ad.read_h5ad(real_path)
        if not guess_is_lognorm(pred) or not guess_is_lognorm(real):
            raise ValueError("cell-eval would renormalize an already frozen input")
        audits[model] = check_arm(pred, real, manifest, plan)
        if first_pred is None:
            first_pred, first_real = pred, real
        else:
            check_same_reference(first_pred, first_real, pred, real)
        arms[model] = (pred, real, manifest)

    output.mkdir(mode=0o700, parents=True)
    keep = set(PER_TARGET_METRICS) | {"de_nsig_counts"}
    skip = sorted(set(metrics_registry.list_metrics()) - keep)
    rows = {}
    for model, (pred, real, _) in arms.items():
        print(f"Scoring {model}", flush=True)
        evaluator = MetricsEvaluator(
            adata_pred=pred, adata_real=real, control_pert=CONTROL,
            pert_col="target_gene", num_threads=1, outdir=str(output / model),
            pdex_kwargs={"is_log1p": True, "geometric_mean": True, "epsilon": 0.0},
        )
        result, _ = evaluator.compute(
            profile="full", skip_metrics=skip, write_csv=True, break_on_error=True
        )
        rows[model] = result.to_dicts()
    per_target, aggregate = summarize(rows, plan["test_targets"], 2000)
    report = {
        "format": "epi-c3-state-replogle-pilot-scores-v1",
        "scope": "Two cell lines, one split, 40 selected perturbations; not the paper benchmark",
        "parent_plan_sha256": plan_hash,
        "input_manifest_sha256": plan["input_manifest_sha256"],
        "cell_eval_version": importlib.metadata.version("cell-eval"),
        "pdex_version": importlib.metadata.version("pdex"),
        "conditions": len(plan["test_targets"]),
        "cells_per_condition": plan["cells_per_target"],
        "hvg_count": 2000,
        "control_counts": {
            "perturbation_parent_pool": len(plan["perturbation_parent_pool"]),
            "predicted_control": len(plan["predicted_control_parent_pool"]),
            "independent_real_control": len(plan["reference_control_ids"]),
        },
        "cross_batch_parent_assignments": plan["summary"]["cross_batch_parent_assignments"],
        "audit": audits,
        "aggregate": aggregate,
        "per_target": per_target,
        "metric_notes": {
            "discrimination_score_l1": "cell-eval 0.8.2 raw rank score, not the paper-normalized PDS",
            "pearson_delta": "cell-eval correlation of mean log-expression effects; target gene is not masked",
            "pr_auc": "average precision for real FDR<0.05 labels ranked by predicted FDR; undefined with one class",
            "de_spearman_lfc_sig": "Spearman over real significant DE genes; undefined below two genes or for constant/nonfinite ranks",
            "overlap_at_N": "cell-eval significant-gene overlap; excluded when real N is zero",
            "de_count_spearman_all_conditions": "Spearman across all 28 real and predicted DE counts, including zeros; undefined for a constant vector",
            "de_settings": "pdex 0.3.0 Mann-Whitney U / BH FDR<0.05; is_log1p=True, geometric_mean=True, epsilon=0",
            "aggregation": "mean across conditions finite in every model for each per-condition metric",
        },
        "limitations": [
            "STATE was trained from scratch for 512 CPU steps with one small architecture, without optimization or an uncertainty estimate.",
            "STATE set outputs are coupled by attention; predicted cells are not independent biological replicates.",
            "The same input control parents can recur across perturbations, though not within a perturbation.",
            "Predicted and real controls are independent of each other and disjoint from treatment parents; batch is not modeled as a DE covariate.",
            "Only STATE, PerturbMean and ContextMean were run; Epi-C3 models and other paper baselines were not evaluated.",
            "This pilot uses a different feature selection, sample subset and split from the paper; do not mix it with author scores.",
            "Existing Epi-C3 foundation pretraining has not been audited for overlap with this pilot.",
        ],
    }
    with (output / "summary.json").open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({"aggregate": aggregate, "audit": audits}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--parent-plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    evaluate(args.prediction_root, args.parent_plan, args.output)


if __name__ == "__main__":
    main()
