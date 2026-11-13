"""Independently audit graph structure and sampled ChIP/ATAC/promoter overlaps."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path
import random
import re
import shutil


RESERVE_WITH_HEADROOM = 201 * 1024**3
PILOT_SHA = "eb12e237bc459d24630c7399099393a17fe9bde2616b59508c78765790b6dbe3"
FOUNDATION_SHA = "77192b5b29c5b096b1b64dd17c3ec9d989acfba41ca97f4c5b7aebbcaf8550e9"
GENCODE_SHA = "01f817afed65feee863361b4baf30a95e722ee5c5d508ff77b04106ef7ba20d3"
ATAC_SHA = "47a71eaac0bfbad586a14d89971398c8922bf8864b5eeab4cb58d37ae925e0e9"
EXPECTED_FIELDS = [
    "source", "target", "source_ensembl_id", "target_ensembl_id", "evidence_origin", "regulatory_sign",
    "chip_experiment_accessions", "chip_file_accessions", "supporting_peak_gene_overlaps",
]


def digest(path: Path) -> str:
    before = path.stat()
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError(f"{path.name} changed while hashing")
    return value.hexdigest()


def load_json(path: Path, expected: str) -> dict:
    if digest(path) != expected:
        raise ValueError(f"Hash mismatch: {path.name}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected an object in {path.name}")
    return value


def load_edges(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != EXPECTED_FIELDS:
            raise ValueError("Unexpected graph CSV fields")
        return list(reader)


def audit_structure(real: list[dict], shuffled: list[dict], features: dict, foundation: set, sources: dict) -> dict:
    pairs = []
    for rows, evidence in ((real, "unpaired_encode_k562_chip_atac_promoter"),
                           (shuffled, "synthetic_structural_control")):
        edges = set()
        for item in rows:
            source, target = item["source"], item["target"]
            if (
                source == target or (source, target) in edges or item["evidence_origin"] != evidence
                or item["regulatory_sign"] != "unknown"
                or features.get(source) != item["source_ensembl_id"]
                or features.get(target) != item["target_ensembl_id"]
                or item["source_ensembl_id"] not in foundation or item["target_ensembl_id"] not in foundation
            ):
                raise ValueError("Graph has a duplicate, self-edge, unmapped gene, or invalid evidence label")
            edges.add((source, target))
            if evidence == "synthetic_structural_control":
                if any(item[key] for key in ("chip_experiment_accessions", "chip_file_accessions",
                                             "supporting_peak_gene_overlaps")):
                    raise ValueError("Shuffled edges must not retain biological evidence")
            else:
                file_ids = item["chip_file_accessions"].split(";")
                experiment_ids = set(item["chip_experiment_accessions"].split(";"))
                if (
                    not file_ids or len(file_ids) != len(set(file_ids))
                    or any(file_id not in sources or sources[file_id]["source_symbol"] != source
                           for file_id in file_ids)
                    or {sources[file_id]["experiment_accession"] for file_id in file_ids} != experiment_ids
                    or not item["supporting_peak_gene_overlaps"].isdigit()
                    or int(item["supporting_peak_gene_overlaps"]) < len(file_ids)
                ):
                    raise ValueError("A real edge has invalid source-file attribution")
        if not edges:
            raise ValueError("Graph must not be empty")
        pairs.append(edges)
    original, control = pairs
    for axis in (0, 1):
        if Counter(item[axis] for item in original) != Counter(item[axis] for item in control):
            raise ValueError("Shuffled graph does not preserve directed degrees")
    source_genes = {source for source, _ in original}
    indegree = Counter(target for _, target in original)
    outdegree = Counter(source for source, _ in original)
    changed_upper_bound = sum(
        min(degree, len(source_genes) - int(target in source_genes) - degree)
        for target, degree in indegree.items()
    )
    source_upper_bound = sum(
        min(degree, len(indegree) - int(source in indegree) - degree)
        for source, degree in outdegree.items()
    )
    return {
        "real_edges": len(original),
        "shuffled_edges": len(control),
        "sources": len(source_genes),
        "targets": len(indegree),
        "degree_difference_nodes": 0,
        "real_edges_absent_from_shuffled": len(original - control),
        "changed_fraction": len(original - control) / len(original),
        "changed_edge_fraction_upper_bound_from_target_indegrees": changed_upper_bound / len(original),
        "changed_edge_fraction_upper_bound_from_source_outdegrees": source_upper_bound / len(original),
        "multiple_experiment_edges": sum(";" in item["chip_experiment_accessions"] for item in real),
        "sources_with_multiple_experiment_edges": len({
            item["source"] for item in real if ";" in item["chip_experiment_accessions"]
        }),
    }


def overlap(left: tuple[int, int], right: tuple[int, int]) -> bool:
    return max(left[0], right[0]) < min(left[1], right[1])


def iter_intervals(path: Path):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip() or line.startswith(("#", "track ", "browser ")):
                continue
            fields = line.rstrip("\r\n").split("\t")
            start, end, summit = int(fields[1]), int(fields[2]), int(fields[9])
            if summit != -1 and summit >= end - start:
                continue
            yield fields[0], start, end


def spot_check(real: list[dict], peak_dir: Path, atac: Path, gencode: Path) -> dict:
    grouped = defaultdict(list)
    for item in real:
        grouped[item["source"]].append(item)
    rng = random.Random(20261113)
    samples = []
    for source in sorted(grouped):
        options = sorted(grouped[source], key=lambda item: item["target"])
        samples.extend(rng.sample(options, min(2, len(options))))
    targets = {item["target_ensembl_id"]: item["target"] for item in samples}
    promoters = defaultdict(set)
    with gzip.open(gencode, "rt", encoding="utf-8") as stream:
        for line in stream:
            if line.startswith("#") or not line.strip():
                continue
            fields = line.rstrip("\r\n").split("\t")
            if fields[2] != "transcript":
                continue
            attributes = {}
            for entry in fields[8].split(";"):
                pieces = entry.strip().split(" ", 1)
                if len(pieces) == 2:
                    attributes[pieces[0]] = pieces[1].strip('"')
            identifier = attributes.get("gene_id", "").split(".", 1)[0]
            if identifier not in targets or attributes.get("gene_name") != targets[identifier]:
                continue
            tss = int(fields[3]) - 1 if fields[6] == "+" else int(fields[4]) - 1
            promoters[identifier].add((fields[0], max(0, tss - 1000), tss + 1001))
    if any(not promoters[identifier] for identifier in targets):
        raise ValueError("A sampled target has no independently reconstructed promoter")
    by_chrom = defaultdict(list)
    for identifier, windows in promoters.items():
        for chrom, start, end in windows:
            by_chrom[chrom].append((start, end, identifier))
    accessible = defaultdict(set)
    for chrom, start, end in iter_intervals(atac):
        for window_start, window_end, identifier in by_chrom[chrom]:
            if overlap((start, end), (window_start, window_end)):
                accessible[identifier].add((chrom, max(start, window_start), min(end, window_end)))
    if any(not accessible[identifier] for identifier in targets):
        raise ValueError("A sampled target has no accessible promoter")
    file_ids = {file_id for item in samples for file_id in item["chip_file_accessions"].split(";")}
    chip = {file_id: list(iter_intervals(peak_dir / f"{file_id}.bed.gz")) for file_id in file_ids}
    for index, item in enumerate(samples):
        segments = accessible[item["target_ensembl_id"]]
        count = 0
        for file_id in item["chip_file_accessions"].split(";"):
            matches = sum(
                any(chrom == segment[0] and overlap((start, end), (segment[1], segment[2]))
                    for segment in segments)
                for chrom, start, end in chip[file_id]
            )
            if not matches:
                raise ValueError(f"Sampled edge {index} lacks a triple overlap in one attributed ChIP file")
            count += matches
        if count != int(item["supporting_peak_gene_overlaps"]):
            raise ValueError(f"Sampled edge {index} has a different supporting peak count")
    return {
        "seed": 20261113,
        "sampled_edges": len(samples),
        "sources_represented": len(grouped),
        "independent_three_way_overlap_checks_passed": len(samples),
        "sampled_source_files": len(file_ids),
        "sampling": "Up to two fixed-seed randomly selected edges per source gene.",
    }


def audit_extra_controls(
    directory: Path, controls: dict, graph_summary_sha: str, real: list[dict],
    features: dict, foundation: set, sources: dict,
) -> dict:
    if (
        controls.get("format") != "epi-c3-encode-k562-additional-degree-controls-v1"
        or controls["input_sha256"]["graph_summary"] != graph_summary_sha
    ):
        raise ValueError("Additional controls do not match the frozen graph")
    items = controls.get("additional_controls")
    if not isinstance(items, list) or len(items) != 2 or len({item["seed"] for item in items}) != 2:
        raise ValueError("Expected two additional fixed-seed control graphs")
    checked, sets = [], []
    for item in items:
        seed, filename = item["seed"], item["filename"]
        if type(seed) is not int or filename != f"edges-shuffled-{seed}.csv":
            raise ValueError("Invalid additional control filename")
        path = directory / filename
        if digest(path) != item["sha256"]:
            raise ValueError("An additional control graph changed")
        control = load_edges(path)
        structure = audit_structure(real, control, features, foundation, sources)
        if (
            structure["real_edges_absent_from_shuffled"] != item["real_edges_absent_from_shuffled"]
            or structure["changed_fraction"] != item["changed_fraction"]
        ):
            raise ValueError("Additional control statistics differ from their files")
        sets.append({(row["source"], row["target"]) for row in control})
        checked.append({
            "seed": seed, "edges": structure["shuffled_edges"],
            "real_edges_absent_from_shuffled": structure["real_edges_absent_from_shuffled"],
            "changed_fraction": structure["changed_fraction"], "degree_difference_nodes": 0,
        })
    pairwise = len(sets[0] - sets[1])
    if pairwise != controls["additional_controls_pairwise_different_edges"]:
        raise ValueError("Additional control pairwise difference is inconsistent")
    return {"controls": checked, "pairwise_different_edges": pairwise}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--graph-dir", type=Path, required=True)
    parser.add_argument("--summary-sha256", required=True)
    parser.add_argument("--peaks-dir", type=Path, required=True)
    parser.add_argument("--atac", type=Path, required=True)
    parser.add_argument("--gencode", type=Path, required=True)
    parser.add_argument("--controls-dir", type=Path)
    parser.add_argument("--controls-summary-sha256")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (args.controls_dir is None) != (args.controls_summary_sha256 is None):
        raise ValueError("Additional controls require both their directory and summary hash")
    summary = load_json(args.graph_dir / "graph-summary.json", args.summary_sha256)
    inputs = summary["input_sha256"]
    pilot = load_json(args.root / "state-replogle-pilot-v1" / "frozen-plan.json", PILOT_SHA)
    foundation = load_json(args.root / "rna-foundation-stage1-v1" / "plan.json", FOUNDATION_SHA)
    verified = load_json(args.peaks_dir / "verified-files.json", inputs["verified_peak_manifest"])
    if (
        summary.get("format") != "epi-c3-encode-k562-promoter-graph-summary-v1"
        or inputs["pilot_plan"] != PILOT_SHA or inputs["foundation_stage1_plan"] != FOUNDATION_SHA
        or inputs["gencode_v44_gtf"] != GENCODE_SHA or digest(args.gencode) != GENCODE_SHA
        or digest(args.atac) != ATAC_SHA
    ):
        raise ValueError("Graph input provenance does not match the frozen audit")
    sources = {}
    for item in verified["files"]:
        accession = item["file_accession"]
        if not re.fullmatch(r"ENCFF[0-9]{3}[A-Z]{3}", accession) or accession in sources:
            raise ValueError("Invalid or duplicate verified peak file")
        if digest(args.peaks_dir / f"{accession}.bed.gz") != item["sha256"]:
            raise ValueError("A ChIP peak file changed")
        sources[accession] = item
    real_path, control_path = args.graph_dir / "edges-real.csv", args.graph_dir / "edges-shuffled.csv"
    if digest(real_path) != inputs["real_edges"] or digest(control_path) != inputs["shuffled_edges"]:
        raise ValueError("Graph file hash mismatch")
    real, shuffled = load_edges(real_path), load_edges(control_path)
    features = dict(pilot["features"])
    foundation_genes = set(foundation["genes"])
    structure = audit_structure(real, shuffled, features, foundation_genes, sources)
    spot = spot_check(real, args.peaks_dir, args.atac, args.gencode)
    extras = None
    if args.controls_dir is not None:
        controls = load_json(args.controls_dir / "controls-summary.json", args.controls_summary_sha256)
        if controls["input_sha256"]["real_edges"] != inputs["real_edges"]:
            raise ValueError("Additional controls use a different real graph")
        extras = audit_extra_controls(
            args.controls_dir, controls, args.summary_sha256, real, features, foundation_genes, sources
        )
    if (
        structure["real_edges"] != summary["graph"]["candidate_edges"]
        or structure["sources"] != summary["graph"]["source_genes"]
        or structure["targets"] != summary["graph"]["target_genes"]
        or structure["multiple_experiment_edges"] != summary["graph"]["edges_with_multiple_chip_experiments"]
        or structure["real_edges_absent_from_shuffled"] != summary["degree_preserving_shuffle"]["real_edges_absent_from_shuffled"]
    ):
        raise ValueError("Independently calculated graph counts differ from the construction summary")
    report = {
        "format": "epi-c3-encode-k562-promoter-graph-independent-audit-v1",
        "status": "structure-and-sampled-overlaps-verified-not-a-model",
        "input_sha256": {
            "graph_summary": args.summary_sha256, "real_edges": inputs["real_edges"],
            "shuffled_edges": inputs["shuffled_edges"], "verified_peak_manifest": inputs["verified_peak_manifest"],
            "additional_controls_summary": args.controls_summary_sha256,
            "gencode_v44_gtf": GENCODE_SHA, "atac_peak": ATAC_SHA, "audit_script": digest(Path(__file__)),
        },
        "structure": structure,
        "spot_check": spot,
        "additional_control_checks": extras,
        "limitations": [
            "Three-way overlap was independently recomputed for the sampled edges only, not for every graph edge.",
            "The source-only and target-only degree bounds do not incorporate both sets of constraints and may be unattainable.",
            "Multiple distinct ENCODE experiments do not establish independent biological cultures.",
            "This audit does not establish biological causality, clinical validity, or held-out prediction performance.",
        ],
    }
    if args.output.exists():
        raise FileExistsError(args.output)
    if shutil.disk_usage(args.output.parent).free < RESERVE_WITH_HEADROOM:
        raise OSError("Insufficient disk space above the 200 GiB reserve and 1 GiB headroom")
    data = (json.dumps(report, indent=2, allow_nan=False) + "\n").encode("utf-8")
    descriptor = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({"state": "complete", "sha256": hashlib.sha256(data).hexdigest(),
                      "structure": structure, "spot_check": spot}), flush=True)


if __name__ == "__main__":
    main()
