"""Independently audit the saved two-context STATE pilot metrics."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy.stats import false_discovery_control, mannwhitneyu, spearmanr
from sklearn.metrics import average_precision_score


MODELS = ("state", "perturb_mean", "context_mean")
CONTROL = "non-targeting"


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require_close(name, actual, expected, tolerance=2e-6):
    if pd.isna(actual) and pd.isna(expected):
        return 0.0
    if not np.isfinite(actual) or not np.isfinite(expected):
        raise ValueError(f"{name}: mismatched nonfinite values")
    difference = abs(float(actual) - float(expected))
    if difference > tolerance:
        raise ValueError(f"{name}: absolute difference {difference} exceeds {tolerance}")
    return difference


def finite_spearman(real, pred):
    if len(real) < 2 or np.unique(real).size < 2 or np.unique(pred).size < 2:
        return None
    value = float(spearmanr(real, pred).statistic)
    return value if math.isfinite(value) else None


def load_de(path, targets, n_features):
    frame = pd.read_csv(path)
    if frame.duplicated(["target", "feature"]).any():
        raise ValueError(f"{path.name}: duplicate target-feature rows")
    groups = {target: part.set_index("feature") for target, part in frame.groupby("target")}
    if set(groups) != set(targets):
        raise ValueError(f"{path.name}: target set mismatch")
    for target, part in groups.items():
        if len(part) != n_features or not part.index.is_unique:
            raise ValueError(f"{path.name}: wrong feature count for {target}")
        pvalues = part["p_value"].to_numpy()
        fdr = part["fdr"].to_numpy()
        if not np.isfinite(pvalues).all() or not np.isfinite(fdr).all():
            raise ValueError(f"{path.name}: nonfinite p-value or FDR")
        if not np.allclose(false_discovery_control(pvalues), fdr, atol=1e-12, rtol=1e-10):
            raise ValueError(f"{path.name}: BH correction mismatch for {target}")
    return groups


def audit(root, predictions, output):
    summary_path = root / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary["format"] != "epi-c3-state-replogle-pilot-scores-v1":
        raise ValueError("Unexpected summary format")
    targets = sorted(summary["per_target"]["state"])
    if len(targets) != 28 or summary["hvg_count"] != 2000:
        raise ValueError("Unexpected pilot dimensions")
    if any(set(summary["per_target"][model]) != set(targets) for model in MODELS):
        raise ValueError("Model target sets differ")

    reference_de_hashes = [digest(root / model / "real_de.csv") for model in MODELS]
    reference_h5ad_hashes = [digest(predictions / model / "real.h5ad") for model in MODELS]
    if len(set(reference_de_hashes)) != 1 or len(set(reference_h5ad_hashes)) != 1:
        raise ValueError("Real reference data differ between models")
    real_de = load_de(root / MODELS[0] / "real_de.csv", targets, summary["hvg_count"])

    all_pred_de = {}
    audit_stats = {}
    first_reference = None
    for model in MODELS:
        pred = ad.read_h5ad(predictions / model / "predicted.h5ad")
        real = ad.read_h5ad(predictions / model / "real.h5ad")
        if not pred.var_names.equals(real.var_names) or real.n_vars != 2000:
            raise ValueError("Feature axes differ")
        if first_reference is None:
            first_reference = real
        elif not np.array_equal(real.X, first_reference.X):
            raise ValueError("Reference values differ")
        pred_labels = pred.obs["target_gene"].astype(str).to_numpy()
        real_labels = real.obs["target_gene"].astype(str).to_numpy()
        if set(pred_labels) != set(targets) | {CONTROL} or set(real_labels) != set(pred_labels):
            raise ValueError("Prediction and reference conditions differ")
        if any(np.sum(pred_labels == target) != 100 or np.sum(real_labels == target) != 100
               for target in targets):
            raise ValueError("Incorrect per-target sample size")
        if np.sum(pred_labels == CONTROL) != 280 or np.sum(real_labels == CONTROL) != 280:
            raise ValueError("Incorrect independent-control sample size")

        pred_mean = {t: np.asarray(pred.X[pred_labels == t]).mean(axis=0) for t in [CONTROL, *targets]}
        real_mean = {t: np.asarray(real.X[real_labels == t]).mean(axis=0) for t in [CONTROL, *targets]}
        pred_effects = np.stack([pred_mean[t] - pred_mean[CONTROL] for t in targets])
        real_effects = np.stack([real_mean[t] - real_mean[CONTROL] for t in targets])
        feature_names = np.asarray(real.var_names.astype(str))
        rows = pd.read_csv(root / model / "results.csv").set_index("perturbation")
        if set(rows.index) != set(targets) or not rows.index.is_unique:
            raise ValueError("Saved metric conditions differ")
        pred_de = load_de(root / model / "pred_de.csv", targets, summary["hvg_count"])
        all_pred_de[model] = pred_de
        max_error = {name: 0.0 for name in (
            "pearson_delta", "discrimination_score_l1", "pr_auc",
            "de_spearman_lfc_sig", "overlap_at_N", "mann_whitney_p_value",
        )}
        tie_boundary_count = 0
        tie_boundary_mismatch = 0
        for i, target in enumerate(targets):
            row = rows.loc[target]
            true_effect, estimated_effect = real_effects[i], pred_effects[i]
            pearson = np.corrcoef(true_effect, estimated_effect)[0, 1]
            max_error["pearson_delta"] = max(
                max_error["pearson_delta"],
                require_close("pearson_delta", pearson, row["pearson_delta"]),
            )
            include = feature_names != target
            distances = np.abs(
                real_effects[:, include].astype(np.float64)
                - estimated_effect[include].astype(np.float64)
            ).sum(axis=1)
            rank = int(np.flatnonzero(np.argsort(distances) == i)[0])
            value = 1 - rank / len(targets)
            max_error["discrimination_score_l1"] = max(
                max_error["discrimination_score_l1"],
                require_close("discrimination_score_l1", value, row["discrimination_score_l1"]),
            )

            actual = real_de[target]
            estimated = pred_de[target].reindex(actual.index)
            if estimated["target"].isna().any():
                raise ValueError("DE feature axes differ")
            real_sig = actual["fdr"].to_numpy() < 0.05
            pred_sig = estimated["fdr"].to_numpy() < 0.05
            n_real, n_pred = int(real_sig.sum()), int(pred_sig.sum())
            if n_real != row["de_nsig_counts_real"] or n_pred != row["de_nsig_counts_pred"]:
                raise ValueError("DE significant-gene count mismatch")
            if n_real != summary["per_target"][model][target]["de_nsig_counts_real"]:
                raise ValueError("Summary real DE count mismatch")
            if n_pred != summary["per_target"][model][target]["de_nsig_counts_pred"]:
                raise ValueError("Summary predicted DE count mismatch")
            if 0 < n_real < len(actual):
                q = np.clip(estimated["fdr"].to_numpy(), 1e-10, 1.0)
                ap = average_precision_score(real_sig, -np.log10(q))
                max_error["pr_auc"] = max(
                    max_error["pr_auc"], require_close("pr_auc", ap, row["pr_auc"], 1e-10)
                )
            if n_real >= 2:
                lfc = finite_spearman(
                    actual["log2_fold_change"].to_numpy()[real_sig],
                    estimated["log2_fold_change"].to_numpy()[real_sig],
                )
                expected = row["de_spearman_lfc_sig"]
                if lfc is not None:
                    max_error["de_spearman_lfc_sig"] = max(
                        max_error["de_spearman_lfc_sig"],
                        require_close("de_spearman_lfc_sig", lfc, expected, 1e-10),
                    )
                elif not pd.isna(expected):
                    raise ValueError("Undefined logFC Spearman mismatch")
            if n_real:
                real_genes = set(actual.index[real_sig])
                significant_pred = estimated.loc[pred_sig].copy()
                significant_pred["absolute_lfc"] = significant_pred["log2_fold_change"].abs()
                significant_pred = significant_pred.sort_values(
                    "absolute_lfc", ascending=False, kind="stable"
                )
                tied = (
                    len(significant_pred) > n_real
                    and significant_pred["absolute_lfc"].iloc[n_real - 1]
                    == significant_pred["absolute_lfc"].iloc[n_real]
                )
                overlap = len(real_genes.intersection(significant_pred.index[:n_real])) / n_real
                difference = abs(overlap - float(row["overlap_at_N"]))
                if tied:
                    tie_boundary_count += 1
                    tie_boundary_mismatch += int(difference > 1e-10)
                else:
                    max_error["overlap_at_N"] = max(
                        max_error["overlap_at_N"],
                        require_close("overlap_at_N", overlap, row["overlap_at_N"], 1e-10),
                    )

            if i < 2:
                for feature_index in (0, 100, 999):
                    feature = feature_names[feature_index]
                    for matrix, labels, frame in (
                        (real.X, real_labels, actual),
                        (pred.X, pred_labels, estimated),
                    ):
                        p = mannwhitneyu(
                            np.asarray(matrix[labels == target, feature_index]),
                            np.asarray(matrix[labels == CONTROL, feature_index]),
                            alternative="two-sided", method="asymptotic",
                        ).pvalue
                        max_error["mann_whitney_p_value"] = max(
                            max_error["mann_whitney_p_value"],
                            require_close("mann_whitney_p_value", p, frame.loc[feature, "p_value"], 1e-7),
                        )

        for metric, item in summary["aggregate"][model].items():
            if metric == "de_count_spearman_all_conditions":
                actual = [summary["per_target"][model][t]["de_nsig_counts_real"] for t in targets]
                estimated = [summary["per_target"][model][t]["de_nsig_counts_pred"] for t in targets]
                value = finite_spearman(actual, estimated)
                if value is None:
                    if item["value"] is not None:
                        raise ValueError("Undefined count Spearman mismatch")
                else:
                    require_close(metric, value, item["value"], 1e-12)
                continue
            shared = [
                t for t in targets
                if all(summary["per_target"][m][t][metric] is not None for m in MODELS)
            ]
            if len(shared) != item["n_conditions"]:
                raise ValueError("Aggregate common-support count mismatch")
            value = float(np.mean([summary["per_target"][model][t][metric] for t in shared]))
            require_close(metric, value, item["value"], 1e-12)
        audit_stats[model] = {
            "maximum_absolute_metric_differences": max_error,
            "overlap_cutoff_ties": tie_boundary_count,
            "overlap_cutoff_tie_mismatches": tie_boundary_mismatch,
            "infinite_predicted_lfc": int(sum(
                np.isinf(pred_de[t]["log2_fold_change"].to_numpy()).sum() for t in targets
            )),
        }

    finite_common = {model: [] for model in MODELS}
    affected_conditions = 0
    removed_gene_occurrences = 0
    real_infinite = int(sum(
        np.isinf(real_de[t]["log2_fold_change"].to_numpy()).sum() for t in targets
    ))
    for target in targets:
        real = real_de[target]
        sig = real["fdr"].to_numpy() < 0.05
        real_lfc = real["log2_fold_change"].to_numpy()[sig]
        genes = real.index[sig]
        predicted = {
            model: all_pred_de[model][target].reindex(genes)["log2_fold_change"].to_numpy()
            for model in MODELS
        }
        keep = np.isfinite(real_lfc)
        for values in predicted.values():
            keep &= np.isfinite(values)
        removed = int((~keep).sum())
        affected_conditions += int(removed > 0)
        removed_gene_occurrences += removed
        values = {model: finite_spearman(real_lfc[keep], predicted[model][keep]) for model in MODELS}
        if all(value is not None for value in values.values()):
            for model, value in values.items():
                finite_common[model].append(value)

    report = {
        "format": "epi-c3-state-pilot-independent-metric-audit-v1",
        "status": "complete",
        "source_summary_sha256": digest(summary_path),
        "reference_h5ad_sha256": reference_h5ad_hashes[0],
        "reference_de_sha256": reference_de_hashes[0],
        "conditions": len(targets),
        "features": summary["hvg_count"],
        "bh_corrected_groups_per_model": len(targets) * 2,
        "mann_whitney_spot_checks_per_model": 12,
        "infinite_real_lfc": real_infinite,
        "model_audits": audit_stats,
        "finite_lfc_sensitivity": {
            "method": "On each condition's true significant genes, exclude any nonfinite real or model logFC; use the same remaining genes and conditions for all models.",
            "affected_conditions": affected_conditions,
            "removed_condition_gene_occurrences": removed_gene_occurrences,
            "common_conditions": len(finite_common[MODELS[0]]),
            "mean_spearman": {
                model: float(np.mean(values)) if values else None
                for model, values in finite_common.items()
            },
        },
    }
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit(args.scores, args.predictions, args.output)


if __name__ == "__main__":
    main()
