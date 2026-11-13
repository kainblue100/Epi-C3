"""Independently recompute graph-neighbor scores for saved population predictions."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np


FIELDS = (
    "neighbor_count", "neighbor_mse", "eligible_effect_count",
    "nonzero_predicted_effect_count", "correct_direction_count", "direction_coverage",
    "direction_accuracy_all_eligible", "direction_accuracy_given_nonzero",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def score_neighbors(prediction, treatment, reference_control, input_control, indices) -> dict:
    if not indices or len(set(indices)) != len(indices):
        raise ValueError("A nonempty set of distinct graph neighbors is required")
    if not (len(prediction) == len(treatment) == len(reference_control) == len(input_control)):
        raise ValueError("Expression vectors have different gene axes")
    errors = []
    eligible = nonzero = correct = 0
    for index in indices:
        if not 0 <= index < len(prediction):
            raise ValueError("Graph neighbor is outside the gene axis")
        predicted, observed, ref, control = (
            float(prediction[index]), float(treatment[index]),
            float(reference_control[index]), float(input_control[index]),
        )
        if not all(math.isfinite(value) for value in (predicted, observed, ref, control)):
            raise ValueError("Graph-neighbor expression is nonfinite")
        errors.append((predicted - observed) ** 2)
        actual_change = observed - ref
        predicted_change = predicted - control
        if abs(actual_change) >= 0.1:
            eligible += 1
            if abs(predicted_change) >= 1e-6:
                nonzero += 1
                if (actual_change > 0) == (predicted_change > 0):
                    correct += 1
    return {
        "neighbor_count": len(indices),
        "neighbor_mse": math.fsum(errors) / len(errors),
        "eligible_effect_count": eligible,
        "nonzero_predicted_effect_count": nonzero,
        "correct_direction_count": correct,
        "direction_coverage": nonzero / eligible if eligible else None,
        "direction_accuracy_all_eligible": correct / eligible if eligible else None,
        "direction_accuracy_given_nonzero": correct / nonzero if nonzero else None,
    }


def neighbor_indices(path: Path, genes: list[str], sources: list[str]) -> dict[str, list[int]]:
    if len(set(genes)) != len(genes) or len(set(sources)) != len(sources):
        raise ValueError("Gene and graph-source axes must be unique")
    positions = {gene: index for index, gene in enumerate(genes)}
    if any(source not in positions for source in sources):
        raise ValueError("A graph source is missing from the gene axis")
    neighbors = {source: set() for source in sources}
    seen = set()
    with path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            source, target = row["source"], row["target"]
            pair = (source, target)
            if source not in positions or target not in positions or source == target or pair in seen:
                raise ValueError("Invalid or duplicated candidate graph edge")
            seen.add(pair)
            if source in neighbors:
                neighbors[source].add(positions[target])
    if not seen or any(not values for values in neighbors.values()):
        raise ValueError("A requested graph source has no candidate neighbors")
    return {source: sorted(values) for source, values in neighbors.items()}


def verify_value(actual, expected, label: str) -> float:
    if expected in ("", None):
        if actual is not None:
            raise ValueError(f"{label}: expected an undefined value")
        return 0.0
    if actual is None:
        raise ValueError(f"{label}: an undefined value was saved as a number")
    observed = float(expected)
    difference = abs(float(actual) - observed)
    if not math.isfinite(observed) or difference > 1e-12:
        raise ValueError(f"{label}: reconstructed value differs by {difference}")
    return difference


def run(root: Path, output: Path) -> None:
    prediction_dir = root / "epi-c3-population-grn-predictions-v2-test"
    reference_dir = root / "replogle-gwps-graph-test-reference-v1"
    score_dir = root / "epi-c3-population-grn-test-score-v2"
    network_dir = root / "k562-evidence-grn-v1"
    prediction_state_path = prediction_dir / "state.json"
    reference_state_path = reference_dir / "state.json"
    score_state_path = score_dir / "state.json"
    manifest_path = network_dir / "manifest.json"
    prediction_state = json.loads(prediction_state_path.read_text(encoding="utf-8"))
    reference_state = json.loads(reference_state_path.read_text(encoding="utf-8"))
    score_state = json.loads(score_state_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    inputs = {
        prediction_dir / prediction_state["means_file"]: prediction_state["means_sha256"],
        reference_dir / reference_state["reference_means_file"]: reference_state["reference_means_sha256"],
        score_dir / "target-scores.csv": score_state["target_scores_sha256"],
        score_dir / "summary.json": score_state["summary_sha256"],
        network_dir / "edges-real.csv": manifest["artifacts"]["edges-real.csv"],
        prediction_state_path: score_state["test_prediction_state_sha256"],
        reference_state_path: score_state["test_reference_state_sha256"],
        manifest_path: prediction_state["network_manifest_sha256"],
    }
    if any(sha256(path) != expected for path, expected in inputs.items()):
        raise ValueError("A frozen graph, prediction, reference, or score artifact changed")
    if any(state.get("state") != "complete" for state in (prediction_state, reference_state, score_state, manifest)):
        raise ValueError("The audit requires complete frozen inputs")
    if prediction_state.get("partition") != "test" or reference_state.get("partition") != "test":
        raise ValueError("The audit requires the frozen test partition")
    summary = json.loads((score_dir / "summary.json").read_text(encoding="utf-8"))
    sources = sorted(summary["graph_source_targets"])
    if len(sources) != 2 or set(sources) != set(manifest["graph_source_targets_by_split"]["test"]):
        raise ValueError("Graph-source targets differ from the frozen network")
    with (score_dir / "target-scores.csv").open(encoding="utf-8", newline="") as stream:
        saved = {}
        for row in csv.DictReader(stream):
            if row["target"] in sources:
                key = (row["method"], int(row["seed_index"]), row["target"])
                if key in saved:
                    raise ValueError("Duplicate graph-neighbor score")
                saved[key] = row

    results = []
    max_difference = 0.0
    comparisons = 0
    with np.load(prediction_dir / prediction_state["means_file"], allow_pickle=False) as predicted, \
            np.load(reference_dir / reference_state["reference_means_file"], allow_pickle=False) as reference:
        genes, targets = predicted["genes"].tolist(), predicted["targets"].tolist()
        if reference["genes"].tolist() != genes or reference["targets"].tolist() != targets:
            raise ValueError("Prediction and reference axes differ")
        if len(set(targets)) != len(targets) or any(source not in targets for source in sources):
            raise ValueError("Graph sources do not match the test target axis")
        neighbors = neighbor_indices(network_dir / "edges-real.csv", genes, sources)
        predictions, no_effect = predicted["predictions"], predicted["no_effect"]
        truth, reference_control = reference["treatment_means"], reference["control_means"]
        arms, seeds = predicted["arms"].tolist(), predicted["seeds"].tolist()
        if predictions.shape != (len(seeds), len(arms), 3, len(targets), len(genes)):
            raise ValueError("Unexpected saved prediction axes")
        methods = {
            "no-effect": no_effect[None],
            "training-mean": predicted["training_mean"][None],
        }
        for arm_index, arm in enumerate(arms):
            for level, label in ((1, "selected"), (2, "full")):
                methods[f"population-v2-{arm}-{label}"] = predictions[:, arm_index, level]
        for target in sources:
            case = targets.index(target)
            for method, matrix in methods.items():
                for seed_index, values in enumerate(matrix):
                    key = (method, seed_index, target)
                    if key not in saved:
                        raise ValueError(f"Missing graph-neighbor score for {key}")
                    actual = score_neighbors(
                        values[case], truth[case], reference_control[case], no_effect[case], neighbors[target],
                    )
                    summary_rows = summary["graph_source_targets"][target][method]
                    matching = [row for row in summary_rows if row["seed_index"] == seed_index]
                    if len(matching) != 1:
                        raise ValueError(f"Missing or duplicated graph summary for {key}")
                    for field in FIELDS:
                        for expected in (saved[key][field], matching[0][field]):
                            max_difference = max(
                                max_difference, verify_value(actual[field], expected, f"{key}/{field}"),
                            )
                            comparisons += 1
                    results.append({"target": target, "method": method, "seed_index": seed_index, **actual})
    if output.exists():
        raise FileExistsError(output)
    output.mkdir()
    report = {
        "format": "epi-c3-grn-neighbor-audit-v1",
        "state": "complete",
        "partition": "test",
        "target_count": len(sources),
        "audited_rows": len(results),
        "value_comparisons_against_csv_and_summary": comparisons,
        "maximum_absolute_difference": max_difference,
        "observed_effect_threshold_log_cpm": 0.1,
        "predicted_effect_threshold_log_cpm": 1e-6,
        "inputs_sha256": {str(path.relative_to(root)): digest for path, digest in inputs.items()},
        "results": results,
        "limitations": [
            "This is an arithmetic audit of an observed post-hoc experiment, not independent validation.",
            "Only two graph-source test targets are covered; candidates are unpaired bulk associations.",
            "Three response seeds share one pretrained RNA encoder and are not biological replicates.",
            "Predicted direction includes the training-mean shift relative to input controls.",
            "Only the new population arms and the no-effect and training-mean baselines were recomputed.",
        ],
    }
    with (output / "audit.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({key: report[key] for key in (
        "target_count", "audited_rows", "value_comparisons_against_csv_and_summary", "maximum_absolute_difference",
    )}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.root, args.output)
