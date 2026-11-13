"""Independently recompute the saved, post-hoc K562 population-GRN scores."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np


PLAN_SHA = "e22d57934f3f661361f266b350906eecaef875521f1686d1cd9ee1d9ea5d40da"
NETWORK_SHA = "edbf309cf0c91b2c1a6f813416cc52c3324b618bd17c1aadb25134390bdc498b"
PREDICTION_STATES = {
    "validation": "496d81a63647cf2de006df5dd08dbd36097adb412b5fbe20e5a7cacb0e1c49eb",
    "test": "eba029f531c7b111026819e7723bb1c607ac730fe028ef152425dd954ac82a3b",
}
REFERENCE_STATES = {
    "validation": "4791d8e58a7af2a27410be330c18de47bc5bff495f015e207ae2e8df7cc61a97",
    "test": "93d2e0c268e34ba0ff21a81dcbc54cae9fdc2831c9cd03609059346eb4cd51ff",
}
ARMS = ("rna-only", "epi-nodes-only", "epi-real-graph", "epi-shuffled-graph")
SEEDS = (314159, 271828, 161803)
STRENGTHS = (0.0, 0.25, 0.5, 0.75, 1.0)
TOLERANCE = 1e-12


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path, expected_sha: str | None = None) -> dict:
    actual = sha256(path)
    if expected_sha is not None and actual != expected_sha:
        raise ValueError(f"Unexpected digest for {path.name}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected an object in {path.name}")
    return value


def artifact(directory: Path, state: dict, file_key: str, sha_key: str) -> Path:
    name = state[file_key]
    if not isinstance(name, str) or Path(name).name != name:
        raise ValueError("Artifact filename must be a basename")
    path = directory / name
    if sha256(path) != state[sha_key]:
        raise ValueError(f"Artifact digest changed: {name}")
    return path


def axes(genes: list[str], targets: list[str]) -> np.ndarray:
    if len(genes) < 2 or len(set(genes)) != len(genes) or not targets or len(set(targets)) != len(targets):
        raise ValueError("Gene or target axis is empty or duplicated")
    lookup = {name: index for index, name in enumerate(genes)}
    if any(target not in lookup for target in targets):
        raise ValueError("Target missing from gene axis")
    return np.asarray([lookup[target] for target in targets], dtype=np.int64)


def target_mse(predicted: np.ndarray, observed: np.ndarray, excluded: np.ndarray) -> np.ndarray:
    predicted = np.asarray(predicted, dtype=np.float64)
    observed = np.asarray(observed, dtype=np.float64)
    excluded = np.asarray(excluded)
    if (
        observed.ndim != 2 or observed.shape[0] == 0 or observed.shape[1] < 2
        or predicted.shape != observed.shape or excluded.shape != (observed.shape[0],)
        or not np.issubdtype(excluded.dtype, np.integer)
        or np.any(excluded < 0) or np.any(excluded >= observed.shape[1])
        or not np.isfinite(predicted).all() or not np.isfinite(observed).all()
    ):
        raise ValueError("Invalid prediction, reference, or target indices")
    squared = np.square(predicted - observed)
    total = np.sum(squared, axis=1, dtype=np.float64)
    return (total - squared[np.arange(len(excluded)), excluded]) / (observed.shape[1] - 1)


def graph_metrics(predicted: np.ndarray, observed: np.ndarray, reference: np.ndarray,
                  input_control: np.ndarray, neighbors: np.ndarray) -> dict:
    arrays = [np.asarray(value, dtype=np.float64) for value in (predicted, observed, reference, input_control)]
    neighbors = np.asarray(neighbors, dtype=np.int64)
    if (
        any(value.ndim != 1 or value.shape != arrays[0].shape or not np.isfinite(value).all() for value in arrays)
        or neighbors.ndim != 1 or len(neighbors) == 0 or len(np.unique(neighbors)) != len(neighbors)
        or np.any(neighbors < 0) or np.any(neighbors >= len(arrays[0]))
    ):
        raise ValueError("Invalid graph-neighbor inputs")
    predicted, observed, reference, input_control = arrays
    pred_delta = predicted[neighbors] - input_control[neighbors]
    obs_delta = observed[neighbors] - reference[neighbors]
    eligible = np.abs(obs_delta) >= 0.1
    nonzero = eligible & (np.abs(pred_delta) >= 1e-6)
    correct = nonzero & (np.sign(pred_delta) == np.sign(obs_delta))
    eligible_count, nonzero_count, correct_count = (int(mask.sum()) for mask in (eligible, nonzero, correct))
    return {
        "neighbor_count": len(neighbors),
        "neighbor_mse": float(np.mean(np.square(predicted[neighbors] - observed[neighbors]), dtype=np.float64)),
        "eligible_effect_count": eligible_count,
        "nonzero_predicted_effect_count": nonzero_count,
        "correct_direction_count": correct_count,
        "direction_coverage": nonzero_count / eligible_count if eligible_count else None,
        "direction_accuracy_all_eligible": correct_count / eligible_count if eligible_count else None,
        "direction_accuracy_given_nonzero": correct_count / nonzero_count if nonzero_count else None,
    }


def check_close(actual: float, expected: float, what: str) -> float:
    actual, expected = float(actual), float(expected)
    if not math.isfinite(actual) or not math.isfinite(expected):
        raise ValueError(f"Nonfinite {what}")
    difference = abs(actual - expected)
    if difference > TOLERANCE:
        raise ValueError(f"{what} differs by {difference}")
    return difference


def load_partition(root: Path, partition: str, genes: list[str], targets: list[str]) -> tuple:
    pred_dir = root / f"epi-c3-population-grn-predictions-v2-{partition}"
    ref_dir = root / f"replogle-gwps-graph-{partition}-reference-v1"
    pred = read_json(pred_dir / "state.json", PREDICTION_STATES[partition])
    ref = read_json(ref_dir / "state.json", REFERENCE_STATES[partition])
    if any(item.get("state") != "complete" or item.get("partition") != partition
           or item.get("plan_sha256") != PLAN_SHA for item in (pred, ref)):
        raise ValueError("Prediction or reference provenance does not match the frozen partition")
    pred_file = artifact(pred_dir, pred, "means_file", "means_sha256")
    ref_file = artifact(ref_dir, ref, "reference_means_file", "reference_means_sha256")
    with np.load(pred_file, allow_pickle=False) as p, np.load(ref_file, allow_pickle=False) as r:
        width = 5 if partition == "validation" else 3
        if (
            p["genes"].tolist() != genes or r["genes"].tolist() != genes
            or p["targets"].tolist() != targets or r["targets"].tolist() != targets
            or p["arms"].tolist() != list(ARMS) or p["seeds"].tolist() != list(SEEDS)
            or p["predictions"].shape != (3, 4, width, len(targets), len(genes))
        ):
            raise ValueError("Saved prediction and reference axes differ")
        result = (p["predictions"], p["no_effect"], p["training_mean"],
                  r["treatment_means"], r["control_means"])
    if any(not np.isfinite(value).all() for value in result):
        raise ValueError("Saved prediction or reference is nonfinite")
    return pred, ref, result


def read_other_prediction(root: Path, name: str, expected_state_sha: str,
                          genes: list[str], targets: list[str]) -> dict:
    directory = root / name
    state = read_json(directory / "state.json", expected_state_sha)
    if state.get("state") != "complete" or state.get("partition") != "test" or state.get("plan_sha256") != PLAN_SHA:
        raise ValueError("Prior prediction provenance differs")
    path = artifact(directory, state, "means_file", "means_sha256")
    with np.load(path, allow_pickle=False) as data:
        if data["genes"].tolist() != genes or data["targets"].tolist() != targets:
            raise ValueError("Prior prediction axes differ")
        return {key: data[key] for key in data.files}


def neighbor_indices(path: Path, genes: list[str], targets: list[str]) -> dict[str, np.ndarray]:
    lookup = {gene: index for index, gene in enumerate(genes)}
    found = {target: set() for target in targets}
    seen = set()
    with path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            source, target = row["source"], row["target"]
            if source not in lookup or target not in lookup or source == target or (source, target) in seen:
                raise ValueError("Invalid or duplicate candidate edge")
            seen.add((source, target))
            if row["evidence_origin"] != "unpaired_bulk_candidate" or row["regulatory_sign"] != "unknown":
                raise ValueError("Unexpected graph evidence semantics")
            if source in found:
                found[source].add(lookup[target])
    if not seen or any(not values for values in found.values()):
        raise ValueError("A graph-source target has no candidate neighbors")
    return {target: np.asarray(sorted(values), dtype=np.int64) for target, values in found.items()}


def audit(root: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError(output)
    plan = read_json(root / "replogle-gwps-graph-plan-v1/plan.json", PLAN_SHA)
    genes = plan["gene_axis"]["symbols"]
    score_dir = root / "epi-c3-population-grn-test-score-v2"
    score_state = read_json(score_dir / "state.json")
    if (
        score_state.get("state") != "complete" or score_state.get("plan_sha256") != PLAN_SHA
        or score_state.get("test_prediction_state_sha256") != PREDICTION_STATES["test"]
        or score_state.get("test_reference_state_sha256") != REFERENCE_STATES["test"]
    ):
        raise ValueError("Score provenance differs")
    summary_path = artifact(score_dir, {**score_state, "summary_file": "summary.json"},
                            "summary_file", "summary_sha256")
    csv_path = artifact(score_dir, {**score_state, "target_scores_file": "target-scores.csv"},
                        "target_scores_file", "target_scores_sha256")
    summary = read_json(summary_path)
    selection_dir = root / "epi-c3-population-grn-validation-selection-v2"
    selection_state = read_json(selection_dir / "state.json")
    selection_path = artifact(selection_dir, {**selection_state, "selection_file": "selection.json"},
                              "selection_file", "selection_sha256")
    scores_path = artifact(selection_dir, {**selection_state, "scores_file": "validation-scores.npz"},
                           "scores_file", "scores_sha256")
    selection = read_json(selection_path, score_state["selection_sha256"])
    if (
        selection.get("validation_prediction_state_sha256") != PREDICTION_STATES["validation"]
        or selection.get("validation_reference_state_sha256") != REFERENCE_STATES["validation"]
        or selection.get("test_treatment_expression_used_for_selection") is not False
    ):
        raise ValueError("Selection provenance differs")

    validation_targets = sorted(plan["conditions"]["validation"])
    excluded = axes(genes, validation_targets)
    _, validation_ref, (validation_pred, val_no_effect, val_training_mean, truth, _) = load_partition(
        root, "validation", genes, validation_targets,
    )
    if selection["validation_reference_means_sha256"] != validation_ref["reference_means_sha256"]:
        raise ValueError("Selection reference mismatch")
    with np.load(scores_path, allow_pickle=False) as saved:
        if (saved["arms"].tolist() != list(ARMS) or saved["seeds"].tolist() != list(SEEDS)
                or saved["targets"].tolist() != validation_targets or saved["strengths"].tolist() != list(STRENGTHS)):
            raise ValueError("Validation score axes differ")
        saved_model, saved_baselines = saved["model_mse"], saved["baseline_mse"]
    maximum_validation_difference = 0.0
    validation_objectives = {}
    for arm_index, arm in enumerate(ARMS):
        objectives = []
        for level_index, strength in enumerate(STRENGTHS):
            per_seed = []
            for seed_index in range(len(SEEDS)):
                independent = target_mse(validation_pred[seed_index, arm_index, level_index], truth, excluded)
                maximum_validation_difference = max(
                    maximum_validation_difference,
                    float(np.max(np.abs(independent - saved_model[arm_index, seed_index, level_index]))),
                )
                per_seed.append(float(independent.mean(dtype=np.float64)))
            objective = float(np.mean(per_seed, dtype=np.float64))
            original = selection["selection_candidates"][arm]["candidates"][level_index]
            if float(original["strength"]) != strength:
                raise ValueError("Validation strength order changed")
            check_close(objective, original["objective"], "validation objective")
            objectives.append((objective, strength))
        chosen = min(objectives)[1]
        if selection["selected_strengths"][arm] != chosen:
            raise ValueError("Validation-selected strength differs")
        validation_objectives[arm] = [value for value, _ in objectives]
    for index, values in enumerate((val_no_effect, val_training_mean)):
        independent = target_mse(values, truth, excluded)
        maximum_validation_difference = max(
            maximum_validation_difference, float(np.max(np.abs(independent - saved_baselines[index]))),
        )
    if maximum_validation_difference > TOLERANCE:
        raise ValueError("Validation per-target scores differ")
    del validation_pred, val_no_effect, val_training_mean, truth, saved_model, saved_baselines

    targets = sorted(plan["conditions"]["test"])
    excluded = axes(genes, targets)
    test_state, test_ref, (predictions, no_effect, training_mean, truth, controls) = load_partition(
        root, "test", genes, targets,
    )
    if (
        test_state["selection_sha256"] != sha256(selection_path)
        or test_state["selected_strengths"] != selection["selected_strengths"]
        or score_state["test_reference_means_sha256"] != test_ref["reference_means_sha256"]
        or summary["selected_strengths"] != selection["selected_strengths"]
    ):
        raise ValueError("Test predictions or reference mismatch")
    previous = read_other_prediction(
        root, "replogle-gwps-graph-test-predictions-v1",
        score_state["comparator_states"]["previous_c3_prediction_state_sha256"], genes, targets,
    )
    gears = read_other_prediction(
        root, "replogle-gwps-gears-test-predictions-v1",
        score_state["comparator_states"]["gears_prediction_state_sha256"], genes, targets,
    )
    old_gene = read_other_prediction(
        root, "epi-c3-gene-grn-predictions-v1-test",
        score_state["previous_gene_grn_prediction_state_sha256"], genes, targets,
    )
    if any(not np.allclose(value, no_effect, rtol=0, atol=TOLERANCE)
           for value in (previous["no_effect"], gears["no_effect"], old_gene["no_effect"])):
        raise ValueError("Comparator input-control means differ")
    if old_gene["arms"].tolist() != list(ARMS) or old_gene["seeds"].tolist() != list(SEEDS):
        raise ValueError("Prior gene-GRN axes differ")
    methods = {
        "no-effect": no_effect[None],
        "training-mean": training_mean[None],
        "previous-c3-flow-selected": previous["predictions"][:, 0, 0],
        "gears-posthoc-extra-go-selected": gears["means"][None],
    }
    for index, arm in enumerate(ARMS):
        methods[f"population-v2-{arm}-selected"] = predictions[:, index, 1]
        methods[f"population-v2-{arm}-full"] = predictions[:, index, 2]
        methods[f"gene-v1-{arm}-selected"] = old_gene["predictions"][:, index, 1]
    if set(methods) != set(summary["all_target_macro_mse"]):
        raise ValueError("Published and reconstructed comparator lists differ")

    network_dir = root / "k562-evidence-grn-v1"
    manifest = read_json(network_dir / "manifest.json", NETWORK_SHA)
    edges = network_dir / "edges-real.csv"
    if sha256(edges) != manifest["artifacts"]["edges-real.csv"]:
        raise ValueError("Candidate graph edges changed")
    graph_targets = plan["graph_source_targets"]["test"]
    neighbor_map = neighbor_indices(edges, genes, graph_targets)
    with csv_path.open(encoding="utf-8", newline="") as stream:
        saved_rows = {}
        for row in csv.DictReader(stream):
            key = (row["method"], int(row["seed_index"]), row["target"])
            if key in saved_rows:
                raise ValueError("Duplicate published per-target score")
            saved_rows[key] = row
    maximum_test_difference = 0.0
    maximum_neighbor_difference = 0.0
    comparisons = {}
    expected_rows = set()
    graph_detail = {}
    for name, means in methods.items():
        scores = []
        for seed_index in range(len(means)):
            actual = target_mse(means[seed_index], truth, excluded)
            scores.append(actual)
            for target_index, target in enumerate(targets):
                key = (name, seed_index, target)
                expected_rows.add(key)
                row = saved_rows[key]
                maximum_test_difference = max(
                    maximum_test_difference,
                    check_close(actual[target_index], row["target_excluded_mse"], "test per-target MSE"),
                )
                if target not in neighbor_map:
                    continue
                result = graph_metrics(
                    means[seed_index, target_index], truth[target_index], controls[target_index],
                    no_effect[target_index], neighbor_map[target],
                )
                for metric, value in result.items():
                    if value is None:
                        if row[metric] != "":
                            raise ValueError("Undefined neighbor metric was saved as a value")
                    else:
                        maximum_neighbor_difference = max(
                            maximum_neighbor_difference,
                            check_close(value, row[metric], f"neighbor {metric}"),
                        )
                if name in ("population-v2-epi-real-graph-selected", "population-v2-epi-shuffled-graph-selected"):
                    graph_detail.setdefault(target, {}).setdefault(name, []).append(result)
            published = summary["all_target_macro_mse"][name]["by_seed"][seed_index]
            check_close(actual.mean(dtype=np.float64), published, "test seed macro MSE")
        scores = np.stack(scores)
        check_close(scores.mean(dtype=np.float64), summary["all_target_macro_mse"][name]["mean"], "test macro MSE")
        comparisons[name] = scores
    if set(saved_rows) != expected_rows:
        raise ValueError("Published per-target score coverage differs")
    real = comparisons["population-v2-epi-real-graph-selected"]
    shuffled = comparisons["population-v2-epi-shuffled-graph-selected"]
    delta = real - shuffled
    mean_delta = delta.mean(axis=0, dtype=np.float64)
    report = {
        "format": "epi-c3-population-grn-independent-audit-v1",
        "status": "verified-saved-aggregates",
        "plan_sha256": PLAN_SHA,
        "network_manifest_sha256": NETWORK_SHA,
        "validation_prediction_state_sha256": PREDICTION_STATES["validation"],
        "test_prediction_state_sha256": PREDICTION_STATES["test"],
        "validation_reference_state_sha256": REFERENCE_STATES["validation"],
        "test_reference_state_sha256": REFERENCE_STATES["test"],
        "selection_sha256": sha256(selection_path),
        "published_summary_sha256": sha256(summary_path),
        "published_target_scores_sha256": sha256(csv_path),
        "script_sha256": sha256(Path(__file__)),
        "test_targets": len(targets),
        "genes": len(genes),
        "validated_test_rows": len(expected_rows),
        "max_validation_per_target_absolute_difference": maximum_validation_difference,
        "max_test_per_target_absolute_difference": maximum_test_difference,
        "max_neighbor_metric_absolute_difference": maximum_neighbor_difference,
        "validation_objectives_by_arm_and_strength_order": validation_objectives,
        "selected_strengths": selection["selected_strengths"],
        "real_minus_shuffled_mse": {
            "mean": float(delta.mean(dtype=np.float64)),
            "by_seed": delta.mean(axis=1, dtype=np.float64).tolist(),
            "targets_real_lower_after_seed_averaging": int((mean_delta < 0).sum()),
            "targets_shuffled_lower_after_seed_averaging": int((mean_delta > 0).sum()),
            "targets_tied_after_seed_averaging": int((mean_delta == 0).sum()),
        },
        "graph_source_targets": graph_detail,
        "limitations": [
            "Recomputed from saved group means and candidate edges, not independently reconstructed from raw cells.",
            "Post-hoc internal split and model design are not an independent blind evaluation.",
            "Candidate edges are unpaired bulk associations; only two test targets directly source edges.",
            "The shuffled graph changes only 31.25% of edges, and three response seeds share one encoder.",
            "Target-wise wins and seed variation are descriptive, not biological replicates or significance tests.",
            "The neighbor predicted direction includes the training-mean baseline shift and uses real-graph neighbors.",
        ],
        "official_challenge_score": False,
    }
    output.mkdir()
    with (output / "target-differences.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["target", "mean_real_minus_shuffled_mse", *[f"seed_{seed}" for seed in SEEDS]])
        for index, target in enumerate(targets):
            writer.writerow([target, float(mean_delta[index]), *delta[:, index].tolist()])
        stream.flush()
        os.fsync(stream.fileno())
    report["target_differences_sha256"] = sha256(output / "target-differences.csv")
    with (output / "audit.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({key: report[key] for key in (
        "status", "validated_test_rows", "max_validation_per_target_absolute_difference",
        "max_test_per_target_absolute_difference", "max_neighbor_metric_absolute_difference",
        "real_minus_shuffled_mse",
    )}, allow_nan=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output != args.root / "epi-c3-population-grn-independent-audit-v1":
        raise ValueError("Unexpected output path")
    audit(args.root, args.output)


if __name__ == "__main__":
    main()
