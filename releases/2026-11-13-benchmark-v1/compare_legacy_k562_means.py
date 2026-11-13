"""Recompute an explicitly post-hoc comparison from frozen population means."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


PLAN_SHA = "e22d57934f3f661361f266b350906eecaef875521f1686d1cd9ee1d9ea5d40da"
ARTIFACTS = {
    "gears": {
        "directory": "replogle-gwps-gears-test-predictions-v1",
        "state_sha256": "c0c4d6b8313f7c84dc07275954f9a1b8a4f6532a493084d375f823f6fe3401fb",
        "data_file": "test-means.npz",
        "data_key": "means_sha256",
        "data_sha256": "4facf0b2ddb4d581828d47385fd455577fa390a23fe374fc1f4eaf4a7dee0888",
    },
    "legacy_epi": {
        "directory": "epi-c3-population-grn-predictions-v2-test",
        "state_sha256": "eba029f531c7b111026819e7723bb1c607ac730fe028ef152425dd954ac82a3b",
        "data_file": "means.npz",
        "data_key": "means_sha256",
        "data_sha256": "2bb86f50aa5cefe82c2858d581f083c1303031bf4b3a44769a3e22d4d635f367",
    },
    "reference": {
        "directory": "replogle-gwps-graph-test-reference-v1",
        "state_sha256": "93d2e0c268e34ba0ff21a81dcbc54cae9fdc2831c9cd03609059346eb4cd51ff",
        "data_file": "reference-means.npz",
        "data_key": "reference_means_sha256",
        "data_sha256": "87bd60f9a6f62be13b7c69635cdc76fe6285aec6f686eccd5cedadc15ba16d82",
    },
}
ARMS = ("rna-only", "epi-nodes-only", "epi-real-graph", "epi-shuffled-graph")
SEEDS = (314159, 271828, 161803)
AUDITED_MSE = {
    "legacy-rna": 0.015219834506276668,
    "legacy-epi-nodes": 0.015195956572086911,
    "legacy-epi-graph": 0.015187104665245118,
    "legacy-epi-shuffled": 0.015193984647963033,
    "gears": 0.01583080899741956,
    "training-mean": 0.015711020985759664,
    "no-effect": 0.01604481739394287,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact(root: Path, name: str) -> Path:
    item = ARTIFACTS[name]
    directory = root / item["directory"]
    state_path = directory / "state.json"
    if sha256(state_path) != item["state_sha256"]:
        raise ValueError(f"{name} state hash changed")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if (
        state.get("state") != "complete"
        or state.get("partition") != "test"
        or state.get("plan_sha256") != PLAN_SHA
        or state.get(item["data_key"]) != item["data_sha256"]
    ):
        raise ValueError(f"{name} state does not match the frozen test plan")
    path = directory / item["data_file"]
    if sha256(path) != item["data_sha256"]:
        raise ValueError(f"{name} data hash changed")
    if name == "legacy_epi" and state.get("test_prediction_axis") != [
        "training-mean-zero", "validation-selected", "full"
    ]:
        raise ValueError("Unexpected legacy Epi-C3 prediction axis")
    return path


def target_indices(genes: list[str], targets: list[str]) -> np.ndarray:
    if len(genes) < 3 or len(set(genes)) != len(genes) or not targets or len(set(targets)) != len(targets):
        raise ValueError("Gene or target axis is missing or duplicated")
    lookup = {name: index for index, name in enumerate(genes)}
    if any(target not in lookup for target in targets):
        raise ValueError("A target is absent from the gene axis")
    return np.asarray([lookup[target] for target in targets], dtype=np.int64)


def checked_arrays(
    predicted: np.ndarray, observed: np.ndarray, excluded: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    predicted = np.asarray(predicted, dtype=np.float64)
    observed = np.asarray(observed, dtype=np.float64)
    excluded = np.asarray(excluded)
    if (
        observed.ndim != 2 or observed.shape[0] < 1 or observed.shape[1] < 3
        or predicted.shape != observed.shape
        or excluded.shape != (observed.shape[0],)
        or not np.issubdtype(excluded.dtype, np.integer)
        or np.any(excluded < 0) or np.any(excluded >= observed.shape[1])
        or not np.isfinite(predicted).all() or not np.isfinite(observed).all()
    ):
        raise ValueError("Invalid prediction, reference, or target indices")
    return predicted, observed, excluded


def target_excluded_mse(
    predicted: np.ndarray, observed: np.ndarray, excluded: np.ndarray
) -> np.ndarray:
    predicted, observed, excluded = checked_arrays(predicted, observed, excluded)
    squared = np.square(predicted - observed)
    return (
        squared.sum(axis=1, dtype=np.float64) - squared[np.arange(len(excluded)), excluded]
    ) / (observed.shape[1] - 1)


def target_excluded_pearson(
    predicted_effect: np.ndarray, observed_effect: np.ndarray, excluded: np.ndarray
) -> np.ndarray:
    predicted_effect, observed_effect, excluded = checked_arrays(
        predicted_effect, observed_effect, excluded
    )
    keep = np.ones(predicted_effect.shape, dtype=bool)
    keep[np.arange(len(excluded)), excluded] = False
    shape = (len(excluded), predicted_effect.shape[1] - 1)
    x = predicted_effect[keep].reshape(shape)
    y = observed_effect[keep].reshape(shape)
    x = x - x.mean(axis=1, keepdims=True)
    y = y - y.mean(axis=1, keepdims=True)
    denominator = np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1)
    result = np.full(len(excluded), np.nan, dtype=np.float64)
    valid = denominator > 0
    result[valid] = np.einsum("ij,ij->i", x[valid], y[valid]) / denominator[valid]
    return np.clip(result, -1.0, 1.0)


def target_excluded_discrimination(
    predicted_effect: np.ndarray, observed_effect: np.ndarray, excluded: np.ndarray
) -> np.ndarray:
    predicted_effect, observed_effect, excluded = checked_arrays(
        predicted_effect, observed_effect, excluded
    )
    count = len(excluded)
    if count < 2:
        raise ValueError("Discrimination needs at least two perturbations")
    pred_norm = np.einsum("ij,ij->i", predicted_effect, predicted_effect)
    obs_norm = np.einsum("ij,ij->i", observed_effect, observed_effect)
    distances = pred_norm[:, None] + obs_norm[None, :] - 2 * predicted_effect @ observed_effect.T
    pred_excluded = predicted_effect[np.arange(count), excluded]
    obs_excluded = observed_effect[:, excluded].T
    distances -= np.square(pred_excluded[:, None] - obs_excluded)
    scale = max(1.0, float(pred_norm.max() + obs_norm.max()))
    if float(distances.min()) < -1e-10 * scale:
        raise ValueError("Invalid squared distances")
    distances = np.maximum(distances, 0)
    correct = distances[np.arange(count), np.arange(count)][:, None]
    ties = np.isclose(distances, correct, rtol=1e-12, atol=1e-12)
    less = (distances < correct) & ~ties
    rank = less.sum(axis=1) + (ties.sum(axis=1) - 1) / 2
    return 1.0 - rank / count


def summarize(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("Expected seed-by-target values")
    by_run = []
    finite_counts = []
    for row in values:
        finite = row[np.isfinite(row)]
        finite_counts.append(int(finite.size))
        by_run.append(float(finite.mean()) if finite.size else None)
    valid_runs = [value for value in by_run if value is not None]
    return {
        "mean_of_run_macros": float(np.mean(valid_runs)) if valid_runs else None,
        "by_run": by_run,
        "finite_targets_by_run": finite_counts,
        "total_targets": values.shape[1],
    }


def evaluate(
    predictions: np.ndarray,
    input_control: np.ndarray,
    observed: np.ndarray,
    reference_control: np.ndarray,
    excluded: np.ndarray,
) -> dict:
    predictions = np.asarray(predictions, dtype=np.float64)
    if predictions.ndim != 3 or predictions.shape[1:] != observed.shape:
        raise ValueError("Expected seed-by-target-by-gene predictions")
    observed_effect = observed - reference_control
    results = {"mse": [], "delta_pearson": [], "discrimination": []}
    for predicted in predictions:
        effect = predicted - input_control
        results["mse"].append(target_excluded_mse(predicted, observed, excluded))
        results["delta_pearson"].append(target_excluded_pearson(effect, observed_effect, excluded))
        results["discrimination"].append(target_excluded_discrimination(effect, observed_effect, excluded))
    return {name: summarize(np.stack(scores)) for name, scores in results.items()}


def compare(root: Path) -> dict:
    paths = {name: artifact(root, name) for name in ARTIFACTS}
    with (
        np.load(paths["gears"], allow_pickle=False) as gears,
        np.load(paths["legacy_epi"], allow_pickle=False) as epi,
        np.load(paths["reference"], allow_pickle=False) as reference,
    ):
        genes, targets = reference["genes"].tolist(), reference["targets"].tolist()
        excluded = target_indices(genes, targets)
        if (
            len(genes) != 1861 or len(targets) != 347
            or any(data["genes"].tolist() != genes or data["targets"].tolist() != targets
                   for data in (gears, epi))
            or epi["arms"].tolist() != list(ARMS) or epi["seeds"].tolist() != list(SEEDS)
            or epi["predictions"].shape != (3, 4, 3, 347, 1861)
            or not np.array_equal(gears["no_effect"], epi["no_effect"])
        ):
            raise ValueError("Frozen predictions do not share the expected axes or input controls")
        observed = reference["treatment_means"]
        reference_control = reference["control_means"]
        input_control = epi["no_effect"]
        for value in (observed, reference_control, input_control):
            if value.shape != (347, 1861) or not np.isfinite(value).all():
                raise ValueError("Invalid frozen reference or control matrix")
        definitions = [
            ("legacy-epi-graph", "Legacy Epi-C3: candidate graph", epi["predictions"][:, 2, 1]),
            ("legacy-epi-shuffled", "Legacy Epi-C3: shuffled graph", epi["predictions"][:, 3, 1]),
            ("legacy-epi-nodes", "Legacy Epi-C3: epigenomic nodes", epi["predictions"][:, 1, 1]),
            ("legacy-rna", "Legacy Epi-C3: RNA only", epi["predictions"][:, 0, 1]),
            ("gears", "GEARS, adapted", gears["means"][None]),
            ("training-mean", "Training mean", epi["training_mean"][None]),
            ("no-effect", "No effect", input_control[None]),
        ]
        methods = []
        for key, label, predictions in definitions:
            metrics = evaluate(predictions, input_control, observed, reference_control, excluded)
            actual = metrics["mse"]["mean_of_run_macros"]
            if actual is None or abs(actual - AUDITED_MSE[key]) > 1e-14:
                raise ValueError(f"{key} MSE does not match its previously audited score")
            seed_count = None if key in {"training-mean", "no-effect"} else len(predictions)
            methods.append({
                "id": key, "label": label, "prediction_runs": len(predictions),
                "seed_count": seed_count, "metrics": metrics,
            })
    return {
        "format": "epi-c3-posthoc-legacy-k562-comparison-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "Replogle K562 genetic perturbations",
        "partition": "Previously examined held-out gene targets within one cell line",
        "target_count": len(targets),
        "gene_count": len(genes),
        "scored_genes_per_target": len(genes) - 1,
        "official_state_benchmark": False,
        "official_cell_eval_result": False,
        "metrics": {
            "mse": "Macro mean per-target squared error of population mean log expression, excluding the targeted gene; lower is better.",
            "delta_pearson": "Macro mean per-target Pearson correlation across non-target genes between predicted minus input-control and observed minus reference-control population means; constant vectors are undefined, not zero.",
            "discrimination": "Per target, rank its predicted effect among all observed effects by Euclidean distance after excluding that target gene; score is 1 minus the zero-based mean tie rank divided by the number of targets. This custom tie-aware population-mean metric is not an official cell-eval result.",
        },
        "inputs_sha256": {f"{name}_means": item["data_sha256"] for name, item in ARTIFACTS.items()},
        "frozen_plan_sha256": PLAN_SHA,
        "script_sha256": sha256(Path(__file__)),
        "methods": methods,
        "limitations": [
            "The original validation and test outcomes were examined before the GEARS extension and this analysis.",
            "This is target holdout within K562, not the STATE underrepresented-context benchmark.",
            "GEARS uses a transductive public GO graph with held-out target identities and has one fitted seed.",
            "Legacy Epi-C3 has three response seeds sharing one pretrained encoder; these are not biological replicates.",
            "The candidate graph directly covers only two test targets; the shuffled graph is an important control.",
            "Input controls and batches are reused across targets; prediction and reference controls differ.",
            "Saved population means cannot establish single-cell distributions or differential-expression p values.",
            "These are legacy models, not the incomplete shared-foundation GRN or same-cell trimodal branches.",
            "STATE and the other image models were not evaluated in this comparison.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path, help="Private root containing the frozen artifacts")
    parser.add_argument("--output", required=True, type=Path, help="New aggregated JSON output")
    args = parser.parse_args()
    result = compare(args.root)
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"Saved aggregated results for {result['target_count']} targets and {len(result['methods'])} methods")


if __name__ == "__main__":
    main()
