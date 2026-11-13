"""Audit frozen STATE-pilot inputs against the existing K562 graph and RNA axis."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil

import numpy as np


PILOT_PLAN_SHA = "eb12e237bc459d24630c7399099393a17fe9bde2616b59508c78765790b6dbe3"
PILOT_MANIFEST_SHA = "6bb5a8d09b310dac58fccba46354370f0fd8a81b867a5491e79e0eecd20d0aac"
FOUNDATION_PLAN_SHA = "77192b5b29c5b096b1b64dd17c3ec9d989acfba41ca97f4c5b7aebbcaf8550e9"
GRAPH_PLAN_SHA = "e22d57934f3f661361f266b350906eecaef875521f1686d1cd9ee1d9ea5d40da"
GRAPH_MANIFEST_SHA = "edbf309cf0c91b2c1a6f813416cc52c3324b618bd17c1aadb25134390bdc498b"
CONTROL = "non-targeting"
RESERVE_WITH_HEADROOM = 201 * 1024**3


def sha256(path: Path) -> str:
    before = path.stat()
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError(f"{path.name} changed while hashing")
    return value.hexdigest()


def read_json(path: Path, expected_sha: str) -> dict:
    if sha256(path) != expected_sha:
        raise ValueError(f"Frozen input hash mismatch: {path.name}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path.name}")
    return value


def unique_strings(values, label: str) -> list[str]:
    if (
        not isinstance(values, list) or not values
        or any(not isinstance(value, str) or not value.strip() for value in values)
        or len(set(values)) != len(values)
    ):
        raise ValueError(f"{label} must contain unique nonempty strings")
    return values


def axis_sha256(genes: list[str]) -> str:
    encoded = json.dumps(genes, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def pilot_features(pilot: dict) -> dict[str, str]:
    pairs = pilot["features"]
    if (
        not isinstance(pairs, list) or not pairs
        or any(not isinstance(pair, list) or len(pair) != 2 for pair in pairs)
    ):
        raise ValueError("Pilot features must be symbol and Ensembl-ID pairs")
    symbols = unique_strings([pair[0] for pair in pairs], "Pilot symbols")
    identifiers = unique_strings([pair[1] for pair in pairs], "Pilot identifiers")
    return dict(zip(symbols, identifiers, strict=True))


def target_partitions(pilot: dict, graph_plan: dict, features: dict[str, str]) -> dict[str, set[str]]:
    partitions = {}
    seen = set()
    for role in ("train", "val", "test"):
        targets = set(unique_strings(pilot["targets"][role], f"Pilot {role} targets"))
        if seen & targets or not targets.issubset(features):
            raise ValueError("Pilot target partitions overlap or lack frozen feature identifiers")
        partitions[role] = targets
        seen.update(targets)
    partitions["all_selected"] = seen
    graph_splits = graph_plan["target_split"]
    if set(graph_splits) != {"train", "validation", "test"}:
        raise ValueError("Unexpected legacy graph target partitions")
    seen_graph = set()
    for split in ("train", "validation", "test"):
        targets = set(unique_strings(graph_splits[split], f"Graph {split} targets"))
        if seen_graph & targets or set(graph_plan["conditions"][split]) != targets:
            raise ValueError("Legacy graph target partitions or condition records disagree")
        seen_graph.update(targets)
    return partitions


def read_graph(path: Path, manifest: dict, graph_genes: set[str], *, shuffled: bool) -> set[tuple[str, str]]:
    if sha256(path) != manifest["artifacts"][path.name]:
        raise ValueError("Candidate graph artifact hash mismatch")
    expected_origin = "synthetic_structural_control" if shuffled else "unpaired_bulk_candidate"
    edges = set()
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"source", "target", "evidence_origin", "regulatory_sign"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("Candidate graph lacks required edge fields")
        for row in reader:
            source, target = row["source"], row["target"]
            pair = (source, target)
            if (
                source not in graph_genes or target not in graph_genes or source == target
                or pair in edges or row["evidence_origin"] != expected_origin
                or row["regulatory_sign"] != "unknown"
            ):
                raise ValueError("Candidate graph has invalid, duplicated or mislabeled edges")
            edges.add(pair)
    if not edges or len(edges) != manifest["edge_count"]:
        raise ValueError("Candidate graph edge count differs from its manifest")
    return edges


def degree_differences(real: set[tuple[str, str]], shuffled: set[tuple[str, str]]) -> dict:
    def counts(edges, position):
        return Counter(pair[position] for pair in edges)

    def differences(left, right):
        return sum(left[node] != right[node] for node in left.keys() | right.keys())

    return {
        "real_edges": len(real),
        "shuffled_edges": len(shuffled),
        "real_edges_absent_from_shuffled": len(real - shuffled),
        "shuffled_edges_absent_from_real": len(shuffled - real),
        "nodes_with_different_out_degree": differences(counts(real, 0), counts(shuffled, 0)),
        "nodes_with_different_in_degree": differences(counts(real, 1), counts(shuffled, 1)),
    }


def summarize_axes(
    pilot: dict,
    foundation: dict,
    graph_plan: dict,
    hvg: list[str],
    real: set[tuple[str, str]],
    shuffled: set[tuple[str, str]],
) -> dict:
    features = pilot_features(pilot)
    foundation_genes = unique_strings(foundation["genes"], "Foundation gene axis")
    if axis_sha256(foundation_genes) != foundation["gene_axis_sha256"]:
        raise ValueError("Foundation gene-axis hash mismatch")
    graph_genes = set(unique_strings(graph_plan["gene_axis"]["symbols"], "Graph gene axis"))
    hvg = unique_strings(hvg, "Pilot HVG axis")
    if not set(hvg).issubset(features):
        raise ValueError("Pilot HVGs are missing from the frozen symbol/Ensembl mapping")
    partitions = target_partitions(pilot, graph_plan, features)
    foundation_set, hvg_set = set(foundation_genes), set(hvg)
    real_panel = {(source, target) for source, target in real if source in hvg_set and target in hvg_set}
    shuffled_panel = {
        (source, target) for source, target in shuffled if source in hvg_set and target in hvg_set
    }
    real_sources = {source for source, _ in real}
    shuffled_sources = {source for source, _ in shuffled}
    real_panel_sources = {source for source, _ in real_panel}
    split = graph_plan["target_split"]
    partition_summary = {}
    for role, targets in partitions.items():
        partition_summary[role] = {
            "targets": len(targets),
            "targets_on_hvg_axis": len(targets & hvg_set),
            "targets_with_exact_ensembl_id_on_foundation_axis": sum(
                features[target] in foundation_set for target in targets
            ),
            "targets_on_legacy_graph_axis": len(targets & graph_genes),
            "targets_directly_sourcing_real_edges": len(targets & real_sources),
            "targets_directly_sourcing_shuffled_edges": len(targets & shuffled_sources),
            "targets_directly_sourcing_real_hvg_edges": len(targets & real_panel_sources),
            "legacy_graph_target_split_overlaps": {
                name: len(targets & set(split[name])) for name in ("train", "validation", "test")
            },
        }
    return {
        "shared_feature_count": len(features),
        "shared_features_with_exact_ensembl_id_on_foundation_axis": sum(
            identifier in foundation_set for identifier in features.values()
        ),
        "hvg_count": len(hvg),
        "hvgs_with_exact_ensembl_id_on_foundation_axis": sum(
            features[gene] in foundation_set for gene in hvg
        ),
        "missing_hvgs_with_foundation_symbol_fallback": sum(
            features[gene] not in foundation_set and "HUMAN_SYMBOL:" + gene in foundation_set
            for gene in hvg
        ),
        "hvgs_on_legacy_graph_axis": len(hvg_set & graph_genes),
        "pilot_target_partitions": partition_summary,
        "graph_full_axis": degree_differences(real, shuffled),
        "graph_projected_to_pilot_hvgs": degree_differences(real_panel, shuffled_panel),
    }


def training_row_overlap(pilot: dict, graph_plan: dict) -> dict:
    source = pilot["sources"]["K562"]
    if source["sha256"] != graph_plan["source_sha256"]:
        raise ValueError("Pilot K562 and legacy graph do not identify the same source file")
    total_rows = source["rows"]
    if type(total_rows) is not int or total_rows < 1:
        raise ValueError("Pilot source-row count is invalid")
    selected = {}
    all_pilot_rows = set()
    for item in pilot["records"]["K562"]:
        row, target, role = item["row"], item["target"], item["role"]
        if type(row) is not int or not 0 <= row < total_rows or row in all_pilot_rows:
            raise ValueError("Pilot K562 source-row records are invalid or duplicated")
        all_pilot_rows.add(row)
        if item["context"] != "K562":
            raise ValueError("Pilot K562 record has an unexpected context")
        if role == "train":
            if target == CONTROL:
                raise ValueError("Pilot K562 treatment record is labeled as a control")
            selected[row] = target
        elif role != "train_control" or target != CONTROL:
            raise ValueError("Pilot K562 has an unexpected training role")
    if not selected:
        raise ValueError("Pilot K562 has no selected training treatment rows")
    seen = set()
    result = {}
    for split in ("train", "validation", "test"):
        row_key = "rows" if split == "train" else "treatment_rows"
        overlap_rows = overlap_conditions = 0
        for target, item in graph_plan["conditions"][split].items():
            rows = item[row_key]
            if not isinstance(rows, list) or any(
                type(row) is not int or not 0 <= row < total_rows for row in rows
            ):
                raise ValueError("Legacy graph contains invalid source-row records")
            row_set = set(rows)
            if len(row_set) != len(rows) or seen & row_set:
                raise ValueError("Legacy graph treatment rows overlap or repeat")
            seen.update(row_set)
            overlap = row_set & selected.keys()
            if any(selected[row] != target for row in overlap):
                raise ValueError("The same source row has different perturbation labels")
            overlap_rows += len(overlap)
            overlap_conditions += bool(overlap)
        result[split] = {
            "overlapping_source_rows": overlap_rows,
            "overlapping_perturbation_conditions": overlap_conditions,
        }
    return {
        "pilot_k562_training_treatment_rows": len(selected),
        "legacy_graph_partitions": result,
    }


def normalization_diagnostic(x: np.ndarray) -> dict:
    values = np.asarray(x, dtype=np.float64)
    if values.ndim != 2 or not values.shape[0] or not values.shape[1]:
        raise ValueError("Control expression must be a nonempty two-dimensional array")
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("Control expression must be finite and nonnegative")
    linear = np.expm1(values)
    panel_totals = linear.sum(axis=1, keepdims=True)
    input_totals = values.sum(axis=1, keepdims=True)
    if (
        not np.isfinite(linear).all() or np.any(panel_totals <= 0)
        or np.any(input_totals <= 0) or np.any(panel_totals > 10000.1)
    ):
        raise ValueError("Control data are inconsistent with frozen original-library normalization")
    local = np.log1p(linear / panel_totals * 10000.0)
    naive = np.log1p(values / input_totals * 10000.0)
    panel_fraction = panel_totals[:, 0] / 10000.0
    return {
        "training_control_cells": len(values),
        "features": values.shape[1],
        "hvg_fraction_of_original_library": {
            "minimum": float(panel_fraction.min()),
            "mean": float(panel_fraction.mean()),
            "maximum": float(panel_fraction.max()),
        },
        "local_panel_vs_original_library_mean_absolute_difference": float(np.abs(local - values).mean()),
        "naive_log_input_vs_local_panel_mean_absolute_difference": float(np.abs(naive - local).mean()),
        "naive_log_input_vs_local_panel_max_absolute_difference": float(np.abs(naive - local).max()),
    }


def read_h5_strings(node, h5py) -> list[str]:
    if isinstance(node, h5py.Dataset):
        return node.asstr()[:].tolist()
    if not isinstance(node, h5py.Group) or set(node.keys()) != {"categories", "codes"}:
        raise ValueError("Unsupported HDF5 text encoding")
    categories = node["categories"].asstr()[:]
    codes = node["codes"][:]
    if np.any(codes < 0) or np.any(codes >= len(categories)):
        raise ValueError("HDF5 categorical metadata contains invalid codes")
    return categories[codes].tolist()


def read_pilot_input(path: Path, manifest: dict, *, sample_size: int = 64) -> tuple[list[str], dict]:
    import h5py

    before = path.stat()
    if sha256(path) != manifest["dataset_sha256"]:
        raise ValueError("Pilot dataset hash mismatch")
    with h5py.File(path, "r") as file:
        hvg = read_h5_strings(file["var/_index"], h5py)
        roles = np.asarray(read_h5_strings(file["obs/role"], h5py), dtype=str)
        contexts = np.asarray(read_h5_strings(file["obs/context"], h5py), dtype=str)
        labels = np.asarray(read_h5_strings(file["obs/target_gene"], h5py), dtype=str)
        matrix = file["X"]
        if (
            matrix.shape != (manifest["cells"], manifest["feature_count"])
            or len(hvg) != matrix.shape[1]
            or any(len(values) != matrix.shape[0] for values in (roles, contexts, labels))
        ):
            raise ValueError("Pilot dataset dimensions differ from its frozen manifest")
        expected_roles = {name: count for name, count in manifest["role_counts"].items()
                          if name != "reference_control"}
        if dict(Counter(roles.tolist())) != expected_roles:
            raise ValueError("Pilot dataset roles differ from the frozen manifest")
        selected = np.flatnonzero((roles == "train_control") & (contexts == "RPE1"))
        if len(selected) < sample_size or not np.all(labels[selected[:sample_size]] == CONTROL):
            raise ValueError("Pilot lacks the required RPE1 training controls")
        sample = np.asarray(matrix[selected[:sample_size], :], dtype=np.float64)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("Pilot dataset changed during the audit")
    return hvg, normalization_diagnostic(sample)


def audit(root: Path) -> dict:
    pilot_root = root / "state-replogle-pilot-v1"
    foundation_path = root / "rna-foundation-stage1-v1" / "plan.json"
    pilot_path = pilot_root / "frozen-plan.json"
    manifest_path = pilot_root / "inputs-cellload-v2" / "manifest.json"
    data_path = pilot_root / "inputs-cellload-v2" / "data" / "replogle.h5ad"
    graph_plan_path = root / "replogle-gwps-graph-plan-v1" / "plan.json"
    graph_root = root / "k562-evidence-grn-v1"
    graph_manifest_path = graph_root / "manifest.json"

    pilot = read_json(pilot_path, PILOT_PLAN_SHA)
    manifest = read_json(manifest_path, PILOT_MANIFEST_SHA)
    foundation = read_json(foundation_path, FOUNDATION_PLAN_SHA)
    graph_plan = read_json(graph_plan_path, GRAPH_PLAN_SHA)
    graph_manifest = read_json(graph_manifest_path, GRAPH_MANIFEST_SHA)
    if (
        pilot.get("format") != "epi-c3-state-replogle-pilot-plan-v1"
        or manifest.get("format") != "epi-c3-state-replogle-pilot-inputs-v1"
        or manifest.get("status") != "complete"
        or manifest.get("plan_sha256") != PILOT_PLAN_SHA
        or foundation.get("format") != "epi-c3-shared-rna-stage1-plan-v1"
        or foundation.get("scope") != "incomplete-stage1-not-a-full-foundation"
        or graph_plan.get("format") != "epi-c3-replogle-gwps-graph-plan-v1"
        or graph_manifest.get("format") != "epi-c3-evidence-grn-v1"
        or graph_manifest.get("state") != "complete"
        or graph_manifest.get("context") != "K562"
        or graph_manifest.get("assembly") != "GRCh38"
        or graph_manifest.get("source_plan_sha256") != GRAPH_PLAN_SHA
    ):
        raise ValueError("Input formats or provenance do not match the frozen experiment")
    if manifest["normalization"] != "log1p(raw counts / original measured-library total * 10000)":
        raise ValueError("Pilot normalization differs from the frozen protocol")

    graph_genes = set(unique_strings(graph_plan["gene_axis"]["symbols"], "Graph gene axis"))
    real = read_graph(graph_root / "edges-real.csv", graph_manifest, graph_genes, shuffled=False)
    shuffled = read_graph(graph_root / "edges-shuffled.csv", graph_manifest, graph_genes, shuffled=True)
    hvg, normalization = read_pilot_input(data_path, manifest)
    axes = summarize_axes(pilot, foundation, graph_plan, hvg, real, shuffled)
    overlap = training_row_overlap(pilot, graph_plan)
    import h5py

    return {
        "format": "epi-c3-grn-state-pilot-preflight-v1",
        "status": "preflight-complete-not-a-model",
        "scope": "Read-only compatibility analysis of frozen metadata and 64 pilot training controls",
        "input_sha256": {
            "pilot_plan": PILOT_PLAN_SHA,
            "pilot_manifest": PILOT_MANIFEST_SHA,
            "pilot_dataset": manifest["dataset_sha256"],
            "foundation_stage1_plan": FOUNDATION_PLAN_SHA,
            "legacy_graph_plan": GRAPH_PLAN_SHA,
            "graph_manifest": GRAPH_MANIFEST_SHA,
            "real_graph_edges": graph_manifest["artifacts"]["edges-real.csv"],
            "shuffled_graph_edges": graph_manifest["artifacts"]["edges-shuffled.csv"],
            "audit_script": sha256(Path(__file__)),
        },
        "software_versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "h5py": h5py.__version__,
        },
        "axis_and_graph": axes,
        "same_source_training_row_overlap": overlap,
        "training_control_normalization": normalization,
        "limitations": [
            "This is a post-hoc preflight, not a trained model, independent benchmark or official cell-eval result.",
            "The original source-file hashes in the frozen plans agree; the original source files were not rehashed by this audit.",
            "Row overlap establishes membership in saved training and legacy evaluation inputs, not whether an optimizer consumed each row.",
            "The candidate graph is K562 unpaired-bulk evidence, not RPE1-specific or proven causal regulation.",
            "Foundation stage one is incomplete in the audited plan; no checkpoint or new perturbation model was loaded or trained.",
            "Cross-source biological identity and condition overlap, including stage-one and replay sources, remain unverified.",
            "The normalization diagnostic uses the first 64 RPE1 training controls in stored input order, not held-out treatment expression.",
            "The planned Tahoe continuation is not connected to a candidate five-context isolation procedure; its full stream would expose at least 7562326 paper-test perturbation row occurrences.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Private directory containing the frozen runs")
    parser.add_argument("--output", type=Path, required=True, help="New aggregate JSON output")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if shutil.disk_usage(args.output.parent).free < RESERVE_WITH_HEADROOM:
        raise OSError("Insufficient disk space above the 200 GiB reserve and 1 GiB headroom")
    result = audit(args.root)
    descriptor = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({"status": result["status"], "output_sha256": sha256(args.output)}), flush=True)


if __name__ == "__main__":
    main()
