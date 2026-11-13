"""Recompute published aggregate metrics using independent per-target loops."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


FILES = {
    "gears": (
        "replogle-gwps-gears-test-predictions-v1/test-means.npz",
        "4facf0b2ddb4d581828d47385fd455577fa390a23fe374fc1f4eaf4a7dee0888",
    ),
    "legacy_epi": (
        "epi-c3-population-grn-predictions-v2-test/means.npz",
        "2bb86f50aa5cefe82c2858d581f083c1303031bf4b3a44769a3e22d4d635f367",
    ),
    "reference": (
        "replogle-gwps-graph-test-reference-v1/reference-means.npz",
        "87bd60f9a6f62be13b7c69635cdc76fe6285aec6f686eccd5cedadc15ba16d82",
    ),
}


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def recompute(prediction, observed, input_control, reference_control, excluded):
    observed_effect = observed - reference_control
    predicted_effect = prediction - input_control
    mse, correlation, discrimination = [], [], []
    for index, gene_index in enumerate(excluded):
        prediction_row = np.delete(prediction[index], gene_index)
        observation_row = np.delete(observed[index], gene_index)
        mse.append(float(np.mean((prediction_row - observation_row) ** 2)))
        effect_row = np.delete(predicted_effect[index], gene_index)
        truth_row = np.delete(observed_effect[index], gene_index)
        if np.ptp(effect_row) == 0 or np.ptp(truth_row) == 0:
            correlation.append(np.nan)
        else:
            correlation.append(float(np.corrcoef(effect_row, truth_row)[0, 1]))
        differences = observed_effect - predicted_effect[index]
        differences[:, gene_index] = 0
        distances = np.sum(differences * differences, axis=1)
        correct = distances[index]
        ties = np.isclose(distances, correct, rtol=1e-12, atol=1e-12)
        less = np.count_nonzero((distances < correct) & ~ties)
        rank = less + (np.count_nonzero(ties) - 1) / 2
        discrimination.append(float(1 - rank / len(excluded)))
    return {
        "mse": np.asarray(mse),
        "delta_pearson": np.asarray(correlation),
        "discrimination": np.asarray(discrimination),
    }


def audit(root: Path, summary_path: Path) -> dict:
    for relative, expected in FILES.values():
        if digest(root / relative) != expected:
            raise ValueError("Frozen input hash mismatch")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary["format"] != "epi-c3-posthoc-legacy-k562-comparison-v1":
        raise ValueError("Unexpected summary format")
    expected_methods = {item["id"]: item for item in summary["methods"]}
    with (
        np.load(root / FILES["gears"][0], allow_pickle=False) as gears,
        np.load(root / FILES["legacy_epi"][0], allow_pickle=False) as epi,
        np.load(root / FILES["reference"][0], allow_pickle=False) as ref,
    ):
        genes, targets = ref["genes"].tolist(), ref["targets"].tolist()
        if (
            len(genes) != 1861 or len(targets) != 347
            or len(set(genes)) != len(genes) or len(set(targets)) != len(targets)
            or any(item["genes"].tolist() != genes or item["targets"].tolist() != targets
                   for item in (gears, epi))
            or not np.array_equal(gears["no_effect"], epi["no_effect"])
        ):
            raise ValueError("Frozen axes or controls differ")
        lookup = {name: index for index, name in enumerate(genes)}
        excluded = [lookup[target] for target in targets]
        arrays = {
            "legacy-epi-graph": epi["predictions"][:, 2, 1],
            "legacy-epi-shuffled": epi["predictions"][:, 3, 1],
            "legacy-epi-nodes": epi["predictions"][:, 1, 1],
            "legacy-rna": epi["predictions"][:, 0, 1],
            "gears": gears["means"][None],
            "training-mean": epi["training_mean"][None],
            "no-effect": epi["no_effect"][None],
        }
        if set(arrays) != set(expected_methods):
            raise ValueError("Summary methods differ from frozen inputs")
        maximum_difference = 0.0
        checked_finite = 0
        checked_undefined = 0
        for key, predictions in arrays.items():
            saved = expected_methods[key]
            if saved["prediction_runs"] != len(predictions):
                raise ValueError(f"{key} prediction-run count mismatch")
            seed_results = [
                recompute(prediction, ref["treatment_means"], epi["no_effect"],
                          ref["control_means"], excluded)
                for prediction in predictions
            ]
            for name in ("mse", "delta_pearson", "discrimination"):
                expected = saved["metrics"][name]
                recomputed_macros = []
                for seed_index, result in enumerate(seed_results):
                    finite = result[name][np.isfinite(result[name])]
                    if len(finite) != expected["finite_targets_by_run"][seed_index]:
                        raise ValueError(f"{key} {name} support mismatch")
                    checked_finite += len(finite)
                    checked_undefined += len(targets) - len(finite)
                    actual = float(finite.mean()) if len(finite) else None
                    prior = expected["by_run"][seed_index]
                    if actual is None or prior is None:
                        if actual is not None or prior is not None:
                            raise ValueError(f"{key} {name} undefined-value mismatch")
                    else:
                        maximum_difference = max(maximum_difference, abs(actual - prior))
                        recomputed_macros.append(actual)
                aggregate = float(np.mean(recomputed_macros)) if recomputed_macros else None
                prior = expected["mean_of_run_macros"]
                if aggregate is None or prior is None:
                    if aggregate is not None or prior is not None:
                        raise ValueError(f"{key} {name} aggregate undefined-value mismatch")
                else:
                    maximum_difference = max(maximum_difference, abs(aggregate - prior))
        if maximum_difference > 1e-12:
            raise ValueError(f"Aggregate metric disagreement: {maximum_difference}")
    return {
        "format": "epi-c3-posthoc-legacy-k562-independent-audit-v1",
        "status": "complete",
        "summary_sha256": digest(summary_path),
        "audit_script_sha256": digest(Path(__file__)),
        "targets": len(targets),
        "genes": len(genes),
        "checked_finite_target_metrics": checked_finite,
        "checked_undefined_target_metrics": checked_undefined,
        "maximum_aggregate_absolute_difference": maximum_difference,
        "limitations": [
            "This is an independent arithmetic replay of the same saved population means, not biological replication.",
            "It does not establish fair prior exposure, validate the original experimental design, or reproduce STATE.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = audit(args.root, args.summary)
    with args.output.open("x", encoding="utf-8", newline="\n") as file:
        json.dump(result, file, indent=2, allow_nan=False)
        file.write("\n")
    print(json.dumps({key: value for key, value in result.items() if key.startswith("checked_") or key.startswith("maximum_")}))


if __name__ == "__main__":
    main()
