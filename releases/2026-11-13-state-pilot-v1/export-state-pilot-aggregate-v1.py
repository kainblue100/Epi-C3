"""Export an audited STATE pilot's public aggregate statistics only."""

import argparse
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import sys

import torch
import yaml


MODELS = ("state", "perturb_mean", "context_mean")
METRICS = (
    "discrimination_score_l1",
    "pearson_delta",
    "pr_auc",
    "de_spearman_lfc_sig",
    "overlap_at_N",
    "de_count_spearman_all_conditions",
)
SOURCE_HASHES = {
    "K562": (
        "ReplogleWeissman2022_K562_gwps.h5ad",
        "d3269d9c863d96555eba768b42ab330cc1b57493cd33b4ac8fe7fbab8278f104",
    ),
    "RPE1": (
        "ReplogleWeissman2022_rpe1.h5ad",
        "12d6a0cf9378c4f09411e27f0ce07b59f1d97731bd576d6fadf112f6f088d1a4",
    ),
}
EXPECTED_STEPS = {"state": 512, "perturb_mean": 1, "context_mean": 1}
EXPECTED_SUPPORT = (28, 28, 24, 22, 24, 28)


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def checkpoint_metadata(runs, predictions):
    result = {}
    for model in MODELS:
        directory = runs / model
        checkpoint = directory / "checkpoints" / "final.ckpt"
        manifest = json.loads((predictions / model / "manifest.json").read_text(encoding="utf-8"))
        config = yaml.safe_load((directory / "config.yaml").read_text(encoding="utf-8"))
        if manifest["model"] != model or sha256(checkpoint) != manifest["checkpoint_sha256"]:
            raise ValueError(f"{model}: prediction/checkpoint mismatch")
        if sha256(directory / "config.yaml") != manifest["config_sha256"]:
            raise ValueError(f"{model}: prediction/configuration mismatch")
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if saved["global_step"] != EXPECTED_STEPS[model]:
            raise ValueError(f"{model}: unexpected actual training steps")
        if (
            config["model"]["name"] != model
            or config["model"]["kwargs"]["hidden_dim"] != 64
            or config["model"]["kwargs"]["cell_set_len"] != 8
            or config["training"]["train_seed"] != 42
            or config["training"]["max_steps"] != 512
        ):
            raise ValueError(f"{model}: unexpected training configuration")
        result[model] = {
            "actual_global_step": int(saved["global_step"]),
            "configured_max_steps": int(config["training"]["max_steps"]),
            "checkpoint_sha256": sha256(checkpoint),
            "config_sha256": sha256(directory / "config.yaml"),
        }
    return result


def build_public_summary(summary, audit, plan, checkpoints, summary_sha256, audit_sha256):
    if summary.get("format") != "epi-c3-state-replogle-pilot-scores-v1":
        raise ValueError("Unexpected scores format")
    if (
        audit.get("format") != "epi-c3-state-pilot-independent-metric-audit-v1"
        or audit.get("status") != "complete"
        or audit.get("source_summary_sha256") != summary_sha256
        or audit.get("conditions") != 28
        or audit.get("features") != 2000
    ):
        raise ValueError("Independent audit does not match the scores")
    if summary["cell_eval_version"] != "0.8.2" or summary["pdex_version"] != "0.3.0":
        raise ValueError("Unexpected scoring package versions")
    if summary["conditions"] != 28 or summary["cells_per_condition"] != 100 or summary["hvg_count"] != 2000:
        raise ValueError("Unexpected pilot dimensions")
    if (
        plan["seed"] != 20261113
        or plan["targets"]["eligible_count"] != 565
        or [len(plan["targets"][name]) for name in ("train", "val", "test")] != [10, 2, 28]
        or plan["cells_per_target_context"] != 100
    ):
        raise ValueError("Unexpected frozen split")
    for context, (filename, expected_hash) in SOURCE_HASHES.items():
        source = plan["sources"][context]
        if source["filename"] != filename or source["sha256"] != expected_hash:
            raise ValueError(f"Unexpected {context} source")
    if {m: checkpoints[m]["actual_global_step"] for m in MODELS} != EXPECTED_STEPS:
        raise ValueError("Unexpected checkpoint step counts")
    if summary["control_counts"] != {
        "perturbation_parent_pool": 280,
        "predicted_control": 280,
        "independent_real_control": 280,
    } or summary["cross_batch_parent_assignments"] != 33:
        raise ValueError("Unexpected control or batch counts")

    metrics = []
    for metric, expected_n in zip(METRICS, EXPECTED_SUPPORT, strict=True):
        scores = {}
        for model in MODELS:
            item = summary["aggregate"][model][metric]
            if item["n_conditions"] != expected_n:
                raise ValueError(f"{model}/{metric}: wrong common-condition count")
            value = item["value"]
            if value is not None and (not math.isfinite(value) or not -1 <= value <= 1):
                raise ValueError(f"{model}/{metric}: invalid public value")
            scores[model] = value
        if metric == "de_count_spearman_all_conditions":
            if scores["context_mean"] is not None:
                raise ValueError("Expected undefined constant-count correlation")
        elif any(value is None for value in scores.values()):
            raise ValueError(f"{metric}: unexpected undefined score")
        metrics.append({"key": metric, "common_conditions": expected_n, "scores": scores})

    sensitivity = audit["finite_lfc_sensitivity"]
    if (
        sensitivity["common_conditions"] != 22
        or sensitivity["affected_conditions"] != 7
        or sensitivity["removed_condition_gene_occurrences"] != 16
        or any(not math.isfinite(sensitivity["mean_spearman"][model]) for model in MODELS)
    ):
        raise ValueError("Unexpected shared-finite logFC sensitivity")
    if (
        summary["audit"]["perturb_mean"]["clipped_below_zero"] != 1666400
        or summary["audit"]["context_mean"]["conditions_with_one_unique_prediction_vector"] != 28
        or audit["model_audits"]["state"]["infinite_predicted_lfc"] != 704
        or audit["infinite_real_lfc"] != 3
    ):
        raise ValueError("Unexpected prediction diagnostics")

    return {
        "format": "epi-c3-state-replogle-pilot-public-v1",
        "scope": (
            "One two-cell-line, 40-target exploratory split; neither the published "
            "paper benchmark nor a comparison of the new Epi-C3 models."
        ),
        "source": {
            "collection_doi": "10.5281/zenodo.7041849",
            "study_doi": "10.1016/j.cell.2022.05.013",
            "collection_license": "CC-BY-4.0; original study terms also apply",
            "files": {
                context: {"filename": filename, "sha256": sha}
                for context, (filename, sha) in SOURCE_HASHES.items()
            },
        },
        "experiment": {
            "selection_seed": plan["seed"],
            "eligible_shared_targets": plan["targets"]["eligible_count"],
            "selected_targets": 40,
            "k562_training_targets": 40,
            "rpe1_targets": {"train": 10, "validation": 2, "test": 28},
            "cells_per_target_context": 100,
            "input_cells": 11230,
            "train_noncontrol_cells": 5000,
            "model_input_control_cells": 3230,
            "validation_cells": 200,
            "test_cells": 2800,
            "independent_reference_controls_available": 560,
            "controls_used": summary["control_counts"],
            "cross_batch_parent_assignments": summary["cross_batch_parent_assignments"],
            "features": 2000,
            "normalization": "log1p(counts / original measured-library total * 10000)",
            "feature_selection": "scanpy Seurat HVGs on training cells and training controls only",
        },
        "software": {
            "state_commit": "9bbfe78a434a55205e4de834e1ea99f85f7a3add",
            "arc_state": "0.11.3",
            "cell_load": "0.10.4",
            "cell_eval": summary["cell_eval_version"],
            "pdex": summary["pdex_version"],
            "python": sys.version.split()[0],
            "runtime_versions": {
                name: importlib.metadata.version(name)
                for name in (
                    "numpy", "scipy", "pandas", "polars", "scanpy", "anndata",
                    "h5py", "torch", "lightning", "scikit-learn", "matplotlib", "PyYAML",
                )
            },
        },
        "models": {
            model: {
                key: checkpoints[model][key]
                for key in ("actual_global_step", "configured_max_steps", "checkpoint_sha256", "config_sha256")
            }
            for model in MODELS
        },
        "metrics": metrics,
        "prediction_audit": {
            model: {
                "clipped_below_zero": summary["audit"][model]["clipped_below_zero"],
                "clipped_above_fourteen": summary["audit"][model]["clipped_above_fourteen"],
                "min_unique_vectors_per_condition": summary["audit"][model]["minimum_unique_prediction_vectors_per_condition"],
                "max_unique_vectors_per_condition": summary["audit"][model]["maximum_unique_prediction_vectors_per_condition"],
                "conditions_with_one_unique_vector": summary["audit"][model]["conditions_with_one_unique_prediction_vector"],
                "infinite_predicted_lfc": audit["model_audits"][model]["infinite_predicted_lfc"],
            }
            for model in MODELS
        },
        "independent_metric_audit": {
            "infinite_real_lfc": audit["infinite_real_lfc"],
            "bh_corrected_groups_per_model": audit["bh_corrected_groups_per_model"],
            "mann_whitney_spot_checks_per_model": audit["mann_whitney_spot_checks_per_model"],
            "finite_lfc_sensitivity": {
                "method": sensitivity["method"],
                "affected_conditions": sensitivity["affected_conditions"],
                "removed_condition_gene_occurrences": sensitivity["removed_condition_gene_occurrences"],
                "common_conditions": sensitivity["common_conditions"],
                "mean_spearman": {model: sensitivity["mean_spearman"][model] for model in MODELS},
            },
            "maximum_absolute_metric_differences": {
                model: audit["model_audits"][model]["maximum_absolute_metric_differences"]
                for model in MODELS
            },
            "overlap_cutoff_ties": {
                model: audit["model_audits"][model]["overlap_cutoff_ties"]
                for model in MODELS
            },
        },
        "provenance": {
            "private_score_summary_sha256": summary_sha256,
            "private_independent_audit_sha256": audit_sha256,
            "note": "The underlying per-cell predictions, plans and identities are not distributed.",
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    summary = json.loads(args.scores.read_text(encoding="utf-8"))
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    expected_versions = {"arc-state": "0.11.3", "cell-load": "0.10.4", "cell-eval": "0.8.2", "pdex": "0.3.0"}
    for package, version in expected_versions.items():
        if importlib.metadata.version(package) != version:
            raise ValueError(f"Unexpected installed version of {package}")
    public = build_public_summary(
        summary, audit, plan, checkpoint_metadata(args.runs, args.predictions),
        sha256(args.scores), sha256(args.audit),
    )
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(public, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
