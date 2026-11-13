"""Independently recompute saved K562 GRN population prediction errors."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def recompute(prediction: np.ndarray, truth: np.ndarray, genes: list[str], targets: list[str]) -> np.ndarray:
    values = np.asarray(prediction, dtype=np.float64)
    expected = np.asarray(truth, dtype=np.float64)
    if (
        values.ndim != 3 or expected.shape != (len(targets), len(genes))
        or values.shape[1:] != expected.shape or len(genes) < 2
        or len(set(genes)) != len(genes) or len(set(targets)) != len(targets)
        or not np.isfinite(values).all() or not np.isfinite(expected).all()
    ):
        raise ValueError("Invalid prediction, reference or identity axes")
    index = {gene: position for position, gene in enumerate(genes)}
    if any(target not in index for target in targets):
        raise ValueError("Every held-out target must occur on the gene axis")
    result = np.empty(values.shape[:2], dtype=np.float64)
    for column, target in enumerate(targets):
        difference = values[:, column] - expected[column]
        errors = np.delete(difference, index[target], axis=1) ** 2
        result[:, column] = errors.mean(axis=1, dtype=np.float64)
    return result


def run(root: Path, output: Path) -> None:
    prediction_dir = root / "epi-c3-population-grn-predictions-v2-test"
    reference_dir = root / "replogle-gwps-graph-test-reference-v1"
    score_dir = root / "epi-c3-population-grn-test-score-v2"
    prediction_state = json.loads((prediction_dir / "state.json").read_text())
    reference_state = json.loads((reference_dir / "state.json").read_text())
    score_state = json.loads((score_dir / "state.json").read_text())
    files = {
        prediction_dir / prediction_state["means_file"]: prediction_state["means_sha256"],
        reference_dir / reference_state["reference_means_file"]: reference_state["reference_means_sha256"],
        score_dir / "target-scores.csv": score_state["target_scores_sha256"],
        score_dir / "summary.json": score_state["summary_sha256"],
    }
    if any(sha256(path) != expected for path, expected in files.items()):
        raise ValueError("A frozen prediction, reference or score artifact changed")
    if (
        prediction_state.get("state") != "complete" or prediction_state.get("partition") != "test"
        or reference_state.get("state") != "complete" or reference_state.get("partition") != "test"
        or score_state.get("state") != "complete"
        or score_state["test_prediction_state_sha256"] != sha256(prediction_dir / "state.json")
        or score_state["test_reference_state_sha256"] != sha256(reference_dir / "state.json")
    ):
        raise ValueError("Prediction, reference and score provenance disagree")
    with np.load(prediction_dir / prediction_state["means_file"], allow_pickle=False) as predicted, \
            np.load(reference_dir / reference_state["reference_means_file"], allow_pickle=False) as reference:
        genes, targets = predicted["genes"].tolist(), predicted["targets"].tolist()
        arms, seeds = predicted["arms"].tolist(), predicted["seeds"].tolist()
        predictions = predicted["predictions"]
        truth = reference["treatment_means"]
        if (
            reference["genes"].tolist() != genes or reference["targets"].tolist() != targets
            or predictions.shape != (3, 4, 3, 347, 1861)
            or len(genes) != 1861 or len(targets) != 347 or len(arms) != 4 or len(seeds) != 3
            or prediction_state["test_prediction_axis"] != ["training-mean-zero", "validation-selected", "full"]
        ):
            raise ValueError("The frozen population prediction axes differ")
        results = {
            "no-effect": recompute(predicted["no_effect"][None], truth, genes, targets),
            "training-mean": recompute(predicted["training_mean"][None], truth, genes, targets),
        }
        for index, arm in enumerate(arms):
            for level, label in ((1, "selected"), (2, "full")):
                results[f"population-v2-{arm}-{label}"] = recompute(
                    predictions[:, index, level], truth, genes, targets,
                )
    saved = pd.read_csv(score_dir / "target-scores.csv")
    summary = json.loads((score_dir / "summary.json").read_text())
    max_difference = 0.0
    checked = 0
    verified_summary = {}
    for method, matrix in results.items():
        rows = saved.loc[saved["method"] == method, ["seed_index", "target", "target_excluded_mse"]]
        if len(rows) != matrix.size or rows.duplicated(["seed_index", "target"]).any():
            raise ValueError(f"Missing or duplicated per-target scores for {method}")
        table = rows.set_index(["seed_index", "target"])["target_excluded_mse"]
        for seed_index in range(matrix.shape[0]):
            for target_index, target in enumerate(targets):
                difference = abs(float(table.loc[(seed_index, target)]) - matrix[seed_index, target_index])
                max_difference = max(max_difference, difference)
                checked += 1
        by_seed = matrix.mean(axis=1, dtype=np.float64)
        reported = summary["all_target_macro_mse"][method]
        if (
            not np.allclose(reported["by_seed"], by_seed, rtol=0, atol=1e-14)
            or not np.isclose(reported["mean"], by_seed.mean(), rtol=0, atol=1e-14)
        ):
            raise ValueError(f"Stored macro MSE differs for {method}")
        verified_summary[method] = {"mean": float(by_seed.mean()), "by_seed": by_seed.tolist()}
    if max_difference > 1e-14:
        raise ValueError(f"Stored per-target scores differ by up to {max_difference}")
    real = results["population-v2-epi-real-graph-selected"]
    shuffled = results["population-v2-epi-shuffled-graph-selected"]
    target_differences = (real - shuffled).mean(axis=0)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir()
    with (output / "target-real-minus-shuffled.csv").open("x", encoding="utf-8", newline="") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(["target", "real_minus_shuffled_mse_seed_mean"])
        writer.writerows(zip(targets, target_differences, strict=True))
    result = {
        "state": "complete", "partition": "test", "targets": len(targets), "genes": len(genes),
        "seeds": seeds, "checked_target_scores": checked, "maximum_absolute_difference": max_difference,
        "macro_mse": verified_summary,
        "selected_real_minus_shuffled_macro_mse": float(target_differences.mean()),
        "targets_with_lower_real_mse": int((target_differences < 0).sum()),
        "targets_with_equal_mse": int((target_differences == 0).sum()),
        "targets_with_higher_real_mse": int((target_differences > 0).sum()),
        "input_sha256": {str(path.relative_to(root)): digest for path, digest in files.items()},
        "target_difference_sha256": sha256(output / "target-real-minus-shuffled.csv"),
        "limitations": [
            "This is an arithmetic audit of an exploratory post-hoc evaluation, not independent validation.",
            "Three model seeds share a single pretrained encoder; target errors are not biological replicates.",
            "Predictions are population means, so this audit does not establish single-cell distributional performance.",
        ],
    }
    with (output / "audit.json").open("x", encoding="utf-8") as file:
        json.dump(result, file, indent=2, allow_nan=False)
        file.write("\n")
    print(json.dumps({key: result[key] for key in (
        "checked_target_scores", "maximum_absolute_difference", "selected_real_minus_shuffled_macro_mse",
        "targets_with_lower_real_mse", "targets_with_higher_real_mse",
    )}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.root, args.output)
