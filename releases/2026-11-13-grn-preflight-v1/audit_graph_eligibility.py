"""Audit graph-source coverage in the frozen pilot's eligible target universe."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import platform
import shutil

import numpy as np

import audit_grn_pilot as common


PREPARATION_SHA = "0e799ad6b52cfcf4312c467546abd40665ef76d55548a1ccea7592a16ae3a18d"


def load_preparation(path: Path):
    if common.sha256(path) != PREPARATION_SHA:
        raise ValueError("Frozen STATE pilot preparation code hash mismatch")
    spec = importlib.util.spec_from_file_location("frozen_pilot_preparation", path)
    if spec is None or spec.loader is None:
        raise ImportError("Cannot load frozen STATE pilot preparation code")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def eligible_candidates(left: dict, right: dict, pilot: dict) -> set[str]:
    features = common.pilot_features(pilot)
    minimum = pilot["cells_per_target_context"]
    if type(minimum) is not int or minimum < 1:
        raise ValueError("The frozen minimum number of cells is invalid")
    candidates = {
        target for target in left["counts"].keys() & right["counts"].keys() & features.keys()
        if target != common.CONTROL and left["counts"][target] >= minimum and right["counts"][target] >= minimum
    }
    if len(candidates) != pilot["targets"]["eligible_count"]:
        raise ValueError("Eligible-target count differs from the frozen pilot plan")
    return candidates


def summarize_candidates(
    left: dict,
    right: dict,
    pilot: dict,
    foundation: dict,
    graph_plan: dict,
    real_edges: set[tuple[str, str]],
) -> dict:
    features = common.pilot_features(pilot)
    foundation_genes = set(common.unique_strings(foundation["genes"], "Foundation gene axis"))
    if common.axis_sha256(foundation["genes"]) != foundation["gene_axis_sha256"]:
        raise ValueError("Foundation gene-axis hash mismatch")
    minimum = pilot["cells_per_target_context"]
    candidates = eligible_candidates(left, right, pilot)
    partitions = common.target_partitions(pilot, graph_plan, features)
    if not partitions["all_selected"].issubset(candidates):
        raise ValueError("A frozen pilot target is no longer eligible")
    graph_sources = {source for source, _ in real_edges}
    covered = candidates & graph_sources
    split = graph_plan["target_split"]
    covered_by_split = {name: len(covered & set(split[name])) for name in ("train", "validation", "test")}
    covered_on_split = set().union(*(set(values) for values in split.values()))
    return {
        "minimum_single_perturbation_cells_per_context": minimum,
        "eligible_shared_targets": len(candidates),
        "eligible_targets_sourcing_real_graph_edges": len(covered),
        "covered_targets_with_exact_ensembl_id_on_foundation_axis": sum(
            features[target] in foundation_genes for target in covered
        ),
        "candidate_edges_from_eligible_graph_sources": sum(source in covered for source, _ in real_edges),
        "eligible_graph_sources_in_legacy_target_splits": covered_by_split,
        "eligible_graph_sources_outside_legacy_target_splits": len(covered - covered_on_split),
        "already_selected_pilot_targets_sourcing_real_graph_edges": len(
            partitions["all_selected"] & graph_sources
        ),
    }


def verify_source(path: Path, expected: dict) -> tuple[int, int]:
    before = path.stat()
    if before.st_size != expected["size"] or common.sha256(path) != expected["sha256"]:
        raise ValueError("A Replogle source differs from the frozen pilot plan")
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("A Replogle source changed during verification")
    return after.st_size, after.st_mtime_ns


def audit(root: Path, k562: Path, rpe1: Path, preparation: Path) -> dict:
    pilot_path = root / "state-replogle-pilot-v1" / "frozen-plan.json"
    foundation_path = root / "rna-foundation-stage1-v1" / "plan.json"
    graph_plan_path = root / "replogle-gwps-graph-plan-v1" / "plan.json"
    graph_root = root / "k562-evidence-grn-v1"
    pilot = common.read_json(pilot_path, common.PILOT_PLAN_SHA)
    foundation = common.read_json(foundation_path, common.FOUNDATION_PLAN_SHA)
    graph_plan = common.read_json(graph_plan_path, common.GRAPH_PLAN_SHA)
    graph_manifest = common.read_json(graph_root / "manifest.json", common.GRAPH_MANIFEST_SHA)
    if (
        pilot.get("format") != "epi-c3-state-replogle-pilot-plan-v1"
        or foundation.get("format") != "epi-c3-shared-rna-stage1-plan-v1"
        or graph_plan.get("format") != "epi-c3-replogle-gwps-graph-plan-v1"
        or graph_manifest.get("format") != "epi-c3-evidence-grn-v1"
        or graph_manifest.get("state") != "complete"
        or graph_manifest.get("context") != "K562"
        or graph_manifest.get("source_plan_sha256") != common.GRAPH_PLAN_SHA
        or graph_plan.get("source_sha256") != pilot["sources"]["K562"]["sha256"]
    ):
        raise ValueError("Frozen inputs have incompatible provenance")
    source_stats = {
        "K562": verify_source(k562, pilot["sources"]["K562"]),
        "RPE1": verify_source(rpe1, pilot["sources"]["RPE1"]),
    }
    helper = load_preparation(preparation)
    if pilot["cells_per_target_context"] != helper.CELLS_PER_TARGET:
        raise ValueError("The frozen target threshold differs from its preparation code")
    metadata = {
        "K562": helper.load_metadata(k562, "K562"),
        "RPE1": helper.load_metadata(rpe1, "RPE1"),
    }
    axis = helper.shared_axis(metadata["K562"], metadata["RPE1"])
    if [list(pair) for pair in axis] != pilot["features"]:
        raise ValueError("Current source annotations differ from the frozen shared gene axis")
    selected = helper.select_targets(metadata["K562"], metadata["RPE1"], {symbol for symbol, _ in axis})
    if selected != pilot["targets"]:
        raise ValueError("Current source metadata do not reconstruct the frozen target selection")
    for name, path in (("K562", k562), ("RPE1", rpe1)):
        stat = path.stat()
        if (stat.st_size, stat.st_mtime_ns) != source_stats[name]:
            raise ValueError("A Replogle source changed during metadata inspection")
    graph_genes = set(common.unique_strings(graph_plan["gene_axis"]["symbols"], "Graph gene axis"))
    real = common.read_graph(graph_root / "edges-real.csv", graph_manifest, graph_genes, shuffled=False)
    summary = summarize_candidates(metadata["K562"], metadata["RPE1"], pilot, foundation, graph_plan, real)
    import h5py

    return {
        "format": "epi-c3-grn-pilot-eligible-targets-v1",
        "status": "metadata-audit-complete-not-a-new-split",
        "scope": "Exact-symbol graph coverage among metadata-eligible shared K562 and RPE1 perturbations",
        "input_sha256": {
            "k562_source": pilot["sources"]["K562"]["sha256"],
            "rpe1_source": pilot["sources"]["RPE1"]["sha256"],
            "pilot_plan": common.PILOT_PLAN_SHA,
            "foundation_stage1_plan": common.FOUNDATION_PLAN_SHA,
            "legacy_graph_plan": common.GRAPH_PLAN_SHA,
            "graph_manifest": common.GRAPH_MANIFEST_SHA,
            "real_graph_edges": graph_manifest["artifacts"]["edges-real.csv"],
            "pilot_preparation_script": PREPARATION_SHA,
            "common_preflight_script": common.sha256(Path(common.__file__)),
            "audit_script": common.sha256(Path(__file__)),
        },
        "software_versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "h5py": h5py.__version__,
        },
        "eligibility": summary,
        "limitations": [
            "This is a post-hoc metadata feasibility audit, not a new held-out split or model result.",
            "Target and graph-source matches are exact case-sensitive symbols without alias harmonization.",
            "Only single-gene conditions with at least the frozen minimum number of cells in both source files are eligible.",
            "The candidate graph is K562 unpaired-bulk evidence, not RPE1-specific or proven causal.",
            "Some legacy validation and test outcomes have already been observed; this audit cannot establish a blind selection.",
            "Foundation training and cross-source exposure isolation are incomplete; no checkpoint or held-out expression was read.",
            "If the scheduled Tahoe stream is fully consumed, the separate tight audit found at least 7562326 exposed paper-test perturbation row occurrences.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--k562", type=Path, required=True)
    parser.add_argument("--rpe1", type=Path, required=True)
    parser.add_argument("--preparation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if shutil.disk_usage(args.output.parent).free < common.RESERVE_WITH_HEADROOM:
        raise OSError("Insufficient disk space above the 200 GiB reserve and 1 GiB headroom")
    result = audit(args.root, args.k562, args.rpe1, args.preparation)
    descriptor = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({"status": result["status"], "output_sha256": common.sha256(args.output)}), flush=True)


if __name__ == "__main__":
    main()
