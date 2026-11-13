"""Generate two additional fixed-seed, degree-matched graph controls."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import shutil

import build_encode_promoter_graph as builder
import select_encode_prior as selection


EXTRA_SEEDS = (20261114, 20261115)


def graph_rows(edges: set[tuple[str, str]], ids: dict[str, str]) -> list[dict]:
    return [{
        "source": source, "target": target,
        "source_ensembl_id": ids[source], "target_ensembl_id": ids[target],
        "evidence_origin": "synthetic_structural_control", "regulatory_sign": "unknown",
        "chip_experiment_accessions": "", "chip_file_accessions": "", "supporting_peak_gene_overlaps": "",
    } for source, target in sorted(edges)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph-dir", type=Path, required=True)
    parser.add_argument("--summary-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    summary = selection.common.read_json(args.graph_dir / "graph-summary.json", args.summary_sha256)
    if (
        summary.get("format") != "epi-c3-encode-k562-promoter-graph-summary-v1"
        or summary["input_sha256"]["builder_script"] != selection.common.sha256(Path(builder.__file__))
    ):
        raise ValueError("The frozen graph summary does not match the builder")
    real_path = args.graph_dir / "edges-real.csv"
    if selection.common.sha256(real_path) != summary["input_sha256"]["real_edges"]:
        raise ValueError("The frozen real graph changed")
    with real_path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    edges = {(row["source"], row["target"]) for row in rows}
    if len(edges) != len(rows) or len(edges) != summary["graph"]["candidate_edges"]:
        raise ValueError("Real graph edges are duplicated or do not match the summary")
    ids = {}
    for row in rows:
        for label, identifier in ((row["source"], row["source_ensembl_id"]),
                                  (row["target"], row["target_ensembl_id"])):
            if label in ids and ids[label] != identifier:
                raise ValueError("A graph symbol has inconsistent Ensembl identifiers")
            ids[label] = identifier
    controls, statistics = {}, []
    for seed in EXTRA_SEEDS:
        controls[seed], details = builder.shuffled_edges(edges, seed=seed)
        statistics.append(details)
    if shutil.disk_usage(args.output_dir.parent).free < selection.common.RESERVE_WITH_HEADROOM:
        raise OSError("Insufficient disk space above the 200 GiB reserve and 1 GiB headroom")
    args.output_dir.mkdir(mode=0o700)
    for details in statistics:
        seed = details["seed"]
        details["filename"] = f"edges-shuffled-{seed}.csv"
        details["sha256"] = builder.write_graph(args.output_dir / details["filename"], graph_rows(controls[seed], ids))
    result = {
        "format": "epi-c3-encode-k562-additional-degree-controls-v1",
        "status": "structural-controls-only-not-model-results",
        "input_sha256": {
            "graph_summary": args.summary_sha256, "real_edges": summary["input_sha256"]["real_edges"],
            "builder_script": selection.common.sha256(Path(builder.__file__)),
            "control_script": selection.common.sha256(Path(__file__)),
        },
        "additional_controls": statistics,
        "additional_controls_pairwise_different_edges": len(controls[EXTRA_SEEDS[0]] - controls[EXTRA_SEEDS[1]]),
        "limitations": [
            "Each control is one deterministic finite edge-swap chain; uniform sampling and convergence are not established.",
            "Degree matching does not preserve biological edge evidence, and no model was trained or evaluated.",
        ],
    }
    digest = selection.write_json(args.output_dir / "controls-summary.json", result)
    print(json.dumps({"state": "complete", "sha256": digest, "additional_controls": statistics,
                      "pairwise_different_edges": result["additional_controls_pairwise_different_edges"]}), flush=True)


if __name__ == "__main__":
    main()
