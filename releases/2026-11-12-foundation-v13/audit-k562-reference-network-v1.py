"""Audit a candidate K562 reference network against the frozen graph and target split."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import os
from pathlib import Path


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def verified_json(path: Path, expected: str) -> dict:
    if digest(path) != expected:
        raise ValueError(f"{path.name} does not match its frozen SHA-256")
    with path.open(encoding="utf-8") as stream:
        data = json.load(stream)
    if not isinstance(data, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return data


def read_reference(path: Path, expected: str) -> tuple[dict[tuple[str, str], set[str]], Counter]:
    if digest(path) != expected:
        raise ValueError("Reference network differs from its frozen SHA-256")
    pairs: dict[tuple[str, str], set[str]] = defaultdict(set)
    triples: set[tuple[str, str, str]] = set()
    counts = Counter()
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        if reader.fieldnames != ["Source", "Target", "Relationship"]:
            raise ValueError("Unexpected reference network columns")
        for row in reader:
            if None in row or any(not isinstance(value, str) or not value or value != value.strip()
                                  for value in row.values()):
                raise ValueError("Malformed reference network row")
            source, target, relation = row["Source"], row["Target"], row["Relationship"]
            if relation not in {"Activation", "Repression"}:
                raise ValueError("Unexpected reference network relationship")
            triple = source, target, relation
            counts["rows"] += 1
            counts[relation] += 1
            counts["duplicate_identical_rows"] += triple in triples
            triples.add(triple)
            pairs[(source, target)].add(relation)
    if not pairs:
        raise ValueError("Reference network is empty")
    return pairs, counts


def read_graph(path: Path, manifest: dict, genes: set[str]) -> set[tuple[str, str]]:
    if (
        manifest.get("format") != "epi-c3-evidence-grn-v1"
        or manifest.get("state") != "complete"
        or manifest.get("context") != "K562"
        or manifest.get("assembly") != "GRCh38"
        or digest(path) != manifest.get("artifacts", {}).get(path.name)
    ):
        raise ValueError("Candidate graph metadata or artifact digest differs")
    edges = set()
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"source", "target", "evidence_origin", "regulatory_sign"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("Candidate graph lacks required columns")
        for row in reader:
            source, target = row["source"], row["target"]
            pair = source, target
            if (
                source not in genes or target not in genes or source == target or pair in edges
                or row["evidence_origin"] != "unpaired_bulk_candidate"
                or row["regulatory_sign"] != "unknown"
            ):
                raise ValueError("Invalid candidate graph edge or evidence semantics")
            edges.add(pair)
    if not edges or len(edges) != manifest.get("edge_count"):
        raise ValueError("Candidate graph edge count differs from its manifest")
    return edges


def audit(reference: Path, reference_sha: str, plan_path: Path, plan_sha: str,
          manifest_path: Path, manifest_sha: str, graph_path: Path, catalog_rows: int) -> dict:
    if reference.name != "RN119.tsv":
        raise ValueError("This audit expects the RN119 reference network")
    plan = verified_json(plan_path, plan_sha)
    manifest = verified_json(manifest_path, manifest_sha)
    if plan.get("format") != "epi-c3-replogle-gwps-graph-plan-v1":
        raise ValueError("Unexpected frozen evaluation plan")
    if manifest.get("source_plan_sha256") != plan_sha:
        raise ValueError("Candidate graph was not derived from this frozen evaluation plan")
    genes_list = plan["gene_axis"]["symbols"]
    if not isinstance(genes_list, list) or not genes_list or len(set(genes_list)) != len(genes_list):
        raise ValueError("Invalid frozen gene axis")
    genes = set(genes_list)
    split_values = plan["target_split"]
    if set(split_values) != {"train", "validation", "test"}:
        raise ValueError("Unexpected frozen target partitions")
    splits = {name: set(values) for name, values in split_values.items()}
    if any(len(splits[name]) != len(split_values[name]) for name in splits):
        raise ValueError("Duplicate target within a frozen partition")
    if any(splits[a] & splits[b] for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))):
        raise ValueError("Frozen target partitions overlap")
    graph_targets = plan["graph_source_targets"]
    if set(graph_targets) != set(splits) or any(
        not set(graph_targets[name]).issubset(splits[name]) for name in splits
    ):
        raise ValueError("Graph-source target partitions are inconsistent")
    pairs, counts = read_reference(reference, reference_sha)
    graph = read_graph(graph_path, manifest, genes)
    reference_edges = set(pairs)
    reference_sources = {source for source, _ in reference_edges}
    reference_targets = {target for _, target in reference_edges}
    graph_sources = {source for source, _ in graph}
    axis_edges = {pair for pair in reference_edges if pair[0] in genes and pair[1] in genes}
    common_edges = reference_edges & graph
    conflict_edges = {pair for pair, signs in pairs.items() if len(signs) > 1}
    split_report = {}
    for name in ("train", "validation", "test"):
        targets = splits[name]
        graph_target_set = set(graph_targets[name])
        split_report[name] = {
            "frozen_targets": len(targets),
            "reference_regulators_in_partition": len(reference_sources & targets),
            "reference_unique_edges_from_partition": sum(source in targets for source, _ in reference_edges),
            "graph_source_targets": len(graph_target_set),
            "graph_source_targets_with_reference_edges": len(reference_sources & graph_target_set),
            "reference_edges_from_graph_source_targets": sum(
                source in graph_target_set for source, _ in reference_edges
            ),
        }
    return {
        "format": "epi-c3-k562-reference-network-audit-v1",
        "status": "complete",
        "reference_network_id": "RN119",
        "reference_network_url": "https://scmogrndb.psu.edu/RN_TSV/RN119.tsv",
        "reference_network_sha256": reference_sha,
        "frozen_plan_sha256": plan_sha,
        "candidate_graph_manifest_sha256": manifest_sha,
        "candidate_graph_edges_sha256": digest(graph_path),
        "reference_catalog_rows": catalog_rows,
        "reference": {
            "rows": counts["rows"],
            "catalog_row_count_matches": counts["rows"] == catalog_rows,
            "activation_rows": counts["Activation"],
            "repression_rows": counts["Repression"],
            "duplicate_identical_rows": counts["duplicate_identical_rows"],
            "unique_directed_gene_pairs": len(reference_edges),
            "pairs_with_both_signs": len(conflict_edges),
            "unique_regulators": len(reference_sources),
            "unique_targets": len(reference_targets),
        },
        "frozen_axis_comparison": {
            "genes": len(genes),
            "reference_regulators_on_axis": len(reference_sources & genes),
            "reference_unique_edges_both_ends_on_axis": len(axis_edges),
            "conflicting_edges_both_ends_on_axis": len(axis_edges & conflict_edges),
            "candidate_graph_edges": len(graph),
            "shared_regulators": len(reference_sources & graph_sources),
            "exact_directed_gene_pair_overlap": len(common_edges),
            "overlapping_pairs_with_both_signs": len(common_edges & conflict_edges),
        },
        "frozen_target_partitions": split_report,
        "used_for_training_or_evaluation": False,
        "limitations": [
            "This is a post-hoc exact, case-sensitive gene-symbol comparison; aliases and reference-gene versions were not harmonized.",
            "Opposite labels for the same pair may reflect heterogeneous source experiments; they cannot be silently assigned a single regulatory sign.",
            "Reference target overlap does not by itself prove leakage, but underlying experiments and benchmark overlap must be audited before use.",
            "Candidate edges and reference labels do not establish direct causal regulation.",
            "The local reference checksum is not an independently supplied source checksum; usage and redistribution rights remain to be verified.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--reference-sha256", required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--catalog-rows", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.catalog_rows <= 0:
        raise ValueError("The published catalog row count must be positive")
    result = audit(
        args.reference, args.reference_sha256, args.plan, args.plan_sha256,
        args.manifest, args.manifest_sha256, args.graph, args.catalog_rows,
    )
    result["script_sha256"] = digest(Path(__file__))
    args.output.mkdir(mode=0o700, parents=True, exist_ok=False)
    descriptor = os.open(args.output / "audit.json", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
