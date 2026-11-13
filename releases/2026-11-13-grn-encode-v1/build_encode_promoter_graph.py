"""Build a frozen unpaired K562 ChIP/ATAC promoter-binding candidate graph."""

from __future__ import annotations

import argparse
from bisect import bisect_left
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

import download_encode_peaks as peaks
import select_encode_prior as selection
import verify_atac_prior as atac_verifier


GENCODE_SHA = "01f817afed65feee863361b4baf30a95e722ee5c5d508ff77b04106ef7ba20d3"
GENCODE_MD5 = "ee330cfe6d0654ba9b9cf434d5c1bfb1"
GENCODE_BYTES = 49721965
PRIMARY_CHROMOSOMES = {f"chr{i}" for i in range(1, 23)} | {"chrX", "chrY", "chrM"}
PROMOTER_RADIUS = 1000
SHUFFLE_SEED = 20261113
SWAPS_PER_EDGE = 5
MAX_ATTEMPTS_PER_EDGE = 200
GENE_ID = re.compile(r'(?:^|;\s*)gene_id "([^"]+)"')
GENE_NAME = re.compile(r'(?:^|;\s*)gene_name "([^"]+)"')


def verify_gencode(path: Path) -> None:
    before = path.stat()
    md5, sha = hashlib.md5(), hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            md5.update(block)
            sha.update(block)
    after = path.stat()
    if (
        before.st_size != GENCODE_BYTES or (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns)
        or md5.hexdigest() != GENCODE_MD5 or sha.hexdigest() != GENCODE_SHA
    ):
        raise ValueError("GENCODE v44 annotation size or checksum mismatch")


def gene_tss(path: Path, features: dict[str, str], foundation: set[str]) -> tuple[dict, dict]:
    selected = {identifier: symbol for symbol, identifier in features.items() if identifier in foundation}
    if len(selected) != sum(identifier in foundation for identifier in features.values()):
        raise ValueError("The common gene axis contains duplicate identifiers")
    loci, names, transcripts = defaultdict(set), defaultdict(set), defaultdict(set)
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if line.startswith("#") or not line.strip():
                continue
            fields = line.rstrip("\r\n").split("\t")
            if len(fields) != 9:
                raise ValueError(f"GENCODE row {number} does not have nine columns")
            if fields[2] not in ("gene", "transcript"):
                continue
            id_match, name_match = GENE_ID.search(fields[8]), GENE_NAME.search(fields[8])
            if id_match is None or name_match is None:
                raise ValueError(f"GENCODE row {number} lacks a gene identifier or name")
            identifier = id_match.group(1).split(".", 1)[0]
            if identifier not in selected:
                continue
            try:
                start, end = int(fields[3]), int(fields[4])
            except ValueError as exc:
                raise ValueError(f"GENCODE row {number} has invalid coordinates") from exc
            if not fields[0] or start < 1 or end < start or fields[6] not in ("+", "-"):
                raise ValueError(f"GENCODE row {number} has invalid coordinates or strand")
            item = (fields[0], start, end, fields[6])
            if fields[2] == "gene":
                loci[identifier].add(item)
                names[identifier].add(name_match.group(1))
            else:
                transcripts[identifier].add((name_match.group(1), *item))

    mapped, rejected = {}, Counter()
    for identifier, symbol in sorted(selected.items(), key=lambda item: item[1]):
        if not loci[identifier]:
            rejected["no_gencode_gene_record"] += 1
            continue
        if len(loci[identifier]) != 1 or names[identifier] != {symbol}:
            rejected["ambiguous_or_mismatched_gene"] += 1
            continue
        chrom, start, end, strand = next(iter(loci[identifier]))
        if chrom not in PRIMARY_CHROMOSOMES:
            rejected["nonprimary_gene_locus"] += 1
            continue
        tx = transcripts[identifier]
        if not tx:
            rejected["no_gencode_transcript"] += 1
            continue
        if any(
            name != symbol or tx_chrom != chrom or tx_strand != strand
            or tx_start < start or tx_end > end
            for name, tx_chrom, tx_start, tx_end, tx_strand in tx
        ):
            rejected["ambiguous_or_out_of_gene_transcript"] += 1
            continue
        tss = {tx_start - 1 if strand == "+" else tx_end - 1 for _, _, tx_start, tx_end, _ in tx}
        mapped[symbol] = {"ensembl_id": identifier, "chromosome": chrom, "tss_zero_based": sorted(tss)}
    return mapped, {
        "common_foundation_features": len(selected),
        "features_with_unambiguous_primary_gencode_transcripts": len(mapped),
        "excluded_features_by_reason": dict(sorted(rejected.items())),
        "distinct_transcript_tss": sum(len(item["tss_zero_based"]) for item in mapped.values()),
    }


class IntervalIndex:
    def __init__(self, intervals):
        grouped = defaultdict(list)
        for chrom, start, end, value in intervals:
            if start < 0 or end <= start:
                raise ValueError("Cannot index an invalid half-open interval")
            grouped[chrom].append((start, end, value))
        self.data = {}
        for chrom, values in grouped.items():
            values.sort(key=lambda item: (item[0], item[1], str(item[2])))
            starts, prefix_max, maximum = [], [], -1
            for start, end, _ in values:
                starts.append(start)
                maximum = max(maximum, end)
                prefix_max.append(maximum)
            self.data[chrom] = starts, prefix_max, values

    def overlaps(self, chrom: str, start: int, end: int):
        if start < 0 or end <= start:
            raise ValueError("Cannot query an invalid half-open interval")
        if chrom not in self.data:
            return
        starts, prefix_max, values = self.data[chrom]
        index = bisect_left(starts, end) - 1
        while index >= 0 and prefix_max[index] > start:
            item = values[index]
            if item[1] > start:
                yield item
            index -= 1


def accessible_promoters(mapped: dict, atac_intervals) -> tuple[IntervalIndex, dict]:
    atac = IntervalIndex((chrom, start, end, None) for chrom, start, end in atac_intervals
                         if chrom in PRIMARY_CHROMOSOMES)
    segments = set()
    genes = set()
    for symbol, item in mapped.items():
        chrom = item["chromosome"]
        for tss in item["tss_zero_based"]:
            start, end = max(0, tss - PROMOTER_RADIUS), tss + PROMOTER_RADIUS + 1
            for atac_start, atac_end, _ in atac.overlaps(chrom, start, end):
                segments.add((chrom, max(start, atac_start), min(end, atac_end), symbol))
                genes.add(symbol)
    return IntervalIndex(segments), {
        "promoter_radius_bp": PROMOTER_RADIUS,
        "promoter_atac_intersection_segments": len(segments),
        "genes_with_accessible_annotated_promoter": len(genes),
    }


def construct_edges(sources: list[tuple[dict, Path]], mapped: dict, promoter_index: IntervalIndex) -> tuple[dict, dict]:
    support = defaultdict(lambda: {"experiments": set(), "files": set(), "peak_gene_overlaps": 0})
    self_edges = set()
    source_genes_missing_annotation = set()
    nonprimary_rows = 0
    for source, path in sources:
        symbol = source["source_symbol"]
        if symbol not in mapped or mapped[symbol]["ensembl_id"] != source["source_ensembl_id"]:
            source_genes_missing_annotation.add(symbol)
            continue
        for chrom, start, end in peaks.usable_intervals(path):
            if chrom not in PRIMARY_CHROMOSOMES:
                nonprimary_rows += 1
                continue
            targets = {item[2] for item in promoter_index.overlaps(chrom, start, end)}
            for target in targets:
                if symbol == target:
                    self_edges.add((symbol, target))
                    continue
                edge = support[(symbol, target)]
                edge["experiments"].add(source["experiment_accession"])
                edge["files"].add(source["file_accession"])
                edge["peak_gene_overlaps"] += 1
    if not support:
        raise ValueError("No ChIP/ATAC/promoter candidate edges passed the frozen filters")
    return dict(support), {
        "source_genes_missing_unambiguous_annotation": len(source_genes_missing_annotation),
        "unique_self_edges_excluded": len(self_edges),
        "usable_nonprimary_chip_rows_excluded": nonprimary_rows,
    }


def shuffled_edges(real: set[tuple[str, str]], seed: int = SHUFFLE_SEED) -> tuple[set[tuple[str, str]], dict]:
    if len(real) < 2:
        raise ValueError("At least two candidate edges are required for a shuffled control")
    rng = random.Random(seed)
    edges = sorted(real)
    current = set(edges)
    common = len(real)
    successful = attempts = 0
    desired_swaps = len(edges) * SWAPS_PER_EDGE
    max_attempts = max(1000, len(edges) * MAX_ATTEMPTS_PER_EDGE)
    while successful < desired_swaps and attempts < max_attempts:
        attempts += 1
        first, second = rng.sample(range(len(edges)), 2)
        a, b = edges[first]
        c, d = edges[second]
        swap_one, swap_two = (a, d), (c, b)
        if a == c or b == d or a == d or c == b or swap_one in current or swap_two in current:
            continue
        old_one, old_two = edges[first], edges[second]
        common -= int(old_one in real) + int(old_two in real)
        common += int(swap_one in real) + int(swap_two in real)
        current.remove(old_one)
        current.remove(old_two)
        current.add(swap_one)
        current.add(swap_two)
        edges[first], edges[second] = swap_one, swap_two
        successful += 1
    changed = len(edges) - common
    if successful < desired_swaps or changed == 0:
        raise ValueError(
            f"Degree-preserving swaps could not complete: {successful}/{desired_swaps} swaps, "
            f"{changed}/{len(edges)} edges changed"
        )
    for position in (0, 1):
        if Counter(edge[position] for edge in real) != Counter(edge[position] for edge in current):
            raise ValueError("Shuffled graph did not preserve the directed degree sequence")
    return current, {
        "seed": seed, "attempted_swaps": attempts, "successful_swaps": successful,
        "requested_successful_swaps": desired_swaps,
        "real_edges_absent_from_shuffled": changed, "changed_fraction": changed / len(edges),
        "nodes_with_different_in_degree": 0, "nodes_with_different_out_degree": 0,
    }


def write_graph(path: Path, rows: list[dict]) -> str:
    if shutil.disk_usage(path.parent).free < selection.common.RESERVE_WITH_HEADROOM:
        raise OSError("Insufficient disk space above the 200 GiB reserve and 1 GiB headroom")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=[
            "source", "target", "source_ensembl_id", "target_ensembl_id",
            "evidence_origin", "regulatory_sign", "chip_experiment_accessions",
            "chip_file_accessions", "supporting_peak_gene_overlaps",
        ], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        stream.flush()
        os.fsync(stream.fileno())
    return selection.common.sha256(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--selection-sha256", required=True)
    parser.add_argument("--peaks-dir", type=Path, required=True)
    parser.add_argument("--verified-peaks-sha256", required=True)
    parser.add_argument("--atac", type=Path, required=True)
    parser.add_argument("--atac-source", type=Path, required=True)
    parser.add_argument("--atac-source-sha256", required=True)
    parser.add_argument("--gencode", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = selection.common.read_json(args.selection, args.selection_sha256)
    verified = selection.common.read_json(args.peaks_dir / "verified-files.json", args.verified_peaks_sha256)
    atac = selection.common.read_json(args.atac_source, args.atac_source_sha256)
    if (
        manifest.get("format") != "epi-c3-encode-k562-chip-selection-v1"
        or verified.get("format") != "epi-c3-encode-k562-chip-verified-files-v1"
        or atac.get("format") != "epi-c3-encode-k562-atac-verified-v1"
        or verified["input_sha256"]["selection"] != args.selection_sha256
        or verified["input_sha256"]["downloader_script"] != selection.common.sha256(Path(peaks.__file__))
        or atac["input_sha256"]["peak_validator_script"] != selection.common.sha256(Path(peaks.__file__))
        or atac["input_sha256"]["verifier_script"] != selection.common.sha256(Path(atac_verifier.__file__))
        or manifest["input_sha256"]["selector_script"] != selection.common.sha256(Path(selection.__file__))
        or atac["input_sha256"]["selector_script"] != selection.common.sha256(Path(selection.__file__))
        or atac["source"]["file_accession"] != atac_verifier.ATAC_ACCESSION
        or atac["source"]["experiment_accession"] != atac_verifier.ATAC_EXPERIMENT
        or atac["source"]["sha256"] != atac_verifier.ATAC_SHA
        or atac["source"]["official_md5"] != atac_verifier.ATAC_MD5
    ):
        raise ValueError("Verified source manifests do not share the frozen provenance")
    expected = manifest["files"]
    actual = verified["files"]
    if len(expected) != len(actual) or any(
        any(record.get(key) != original.get(key) for key in original)
        for original, record in zip(expected, actual, strict=True)
    ):
        raise ValueError("Downloaded peaks differ from the selected source list")
    sources = []
    for item in actual:
        peaks.validate_selected_record(item)
        path = args.peaks_dir / f'{item["file_accession"]}.bed.gz'
        checks = peaks.verify_peak_file(path, item["bytes"], item["official_md5"], item["sha256"])
        if any(checks[key] != item[key] for key in checks):
            raise ValueError("A verified ChIP peak file changed")
        sources.append((item, path))
    atac_item = atac["source"]
    atac_checks = peaks.verify_peak_file(args.atac, atac_item["bytes"], atac_item["official_md5"], atac_item["sha256"])
    if any(atac_checks[key] != atac_item[key] for key in atac_checks):
        raise ValueError("The verified ATAC peak file changed")
    verify_gencode(args.gencode)

    pilot = selection.common.read_json(
        args.root / "state-replogle-pilot-v1" / "frozen-plan.json", selection.common.PILOT_PLAN_SHA
    )
    foundation = selection.common.read_json(
        args.root / "rna-foundation-stage1-v1" / "plan.json", selection.common.FOUNDATION_PLAN_SHA
    )
    if (
        manifest["input_sha256"]["pilot_plan"] != selection.common.PILOT_PLAN_SHA
        or manifest["input_sha256"]["foundation_stage1_plan"] != selection.common.FOUNDATION_PLAN_SHA
        or pilot.get("format") != "epi-c3-state-replogle-pilot-plan-v1"
        or foundation.get("format") != "epi-c3-shared-rna-stage1-plan-v1"
        or selection.common.axis_sha256(foundation["genes"]) != foundation["gene_axis_sha256"]
    ):
        raise ValueError("Model gene-axis or frozen pilot provenance changed")
    features = selection.common.pilot_features(pilot)
    mapped, mapping = gene_tss(args.gencode, features, set(foundation["genes"]))
    promoter_index, promoter_summary = accessible_promoters(mapped, peaks.usable_intervals(args.atac))
    support, construction = construct_edges(sources, mapped, promoter_index)
    real = set(support)
    shuffled, shuffle = shuffled_edges(real)
    source_labels = {source for source, _ in real}
    real_rows = [
        {
            "source": source, "target": target,
            "source_ensembl_id": mapped[source]["ensembl_id"], "target_ensembl_id": mapped[target]["ensembl_id"],
            "evidence_origin": "unpaired_encode_k562_chip_atac_promoter",
            "regulatory_sign": "unknown",
            "chip_experiment_accessions": ";".join(sorted(support[(source, target)]["experiments"])),
            "chip_file_accessions": ";".join(sorted(support[(source, target)]["files"])),
            "supporting_peak_gene_overlaps": support[(source, target)]["peak_gene_overlaps"],
        } for source, target in sorted(real)
    ]
    shuffled_rows = [
        {
            "source": source, "target": target,
            "source_ensembl_id": mapped[source]["ensembl_id"], "target_ensembl_id": mapped[target]["ensembl_id"],
            "evidence_origin": "synthetic_structural_control", "regulatory_sign": "unknown",
            "chip_experiment_accessions": "", "chip_file_accessions": "", "supporting_peak_gene_overlaps": "",
        } for source, target in sorted(shuffled)
    ]
    if shutil.disk_usage(args.output_dir.parent).free < selection.common.RESERVE_WITH_HEADROOM:
        raise OSError("Insufficient disk space above the 200 GiB reserve and 1 GiB headroom")
    args.output_dir.mkdir(mode=0o700)
    real_sha = write_graph(args.output_dir / "edges-real.csv", real_rows)
    shuffled_sha = write_graph(args.output_dir / "edges-shuffled.csv", shuffled_rows)
    graph_summary = {
        "candidate_edges": len(real),
        "source_genes": len(source_labels),
        "target_genes": len({target for _, target in real}),
        "edges_with_multiple_chip_experiments": sum(len(item["experiments"]) > 1 for item in support.values()),
        "pilot_target_sources_by_split": {
            role: len(source_labels & set(pilot["targets"][role])) for role in ("train", "val", "test")
        },
        "excluded_chip_summit_rows": sum(item["invalid_summit_rows_excluded"] for item in actual),
        "excluded_atac_summit_rows": atac_item["invalid_summit_rows_excluded"],
        **construction,
    }
    inputs = {
        "pilot_plan": selection.common.PILOT_PLAN_SHA,
        "foundation_stage1_plan": selection.common.FOUNDATION_PLAN_SHA,
        "selected_peak_manifest": args.selection_sha256,
        "verified_peak_manifest": args.verified_peaks_sha256,
        "verified_atac_source": args.atac_source_sha256,
        "gencode_v44_gtf": GENCODE_SHA,
        "builder_script": selection.common.sha256(Path(__file__)),
        "real_edges": real_sha,
        "shuffled_edges": shuffled_sha,
    }
    report = {
        "format": "epi-c3-encode-k562-promoter-graph-summary-v1",
        "status": "candidate-prior-not-a-trained-model",
        "input_sha256": inputs,
        "gene_mapping": mapping,
        "promoters": promoter_summary,
        "graph": graph_summary,
        "degree_preserving_shuffle": shuffle,
        "limitations": [
            "The graph uses unpaired bulk K562 ChIP-seq and ATAC-seq, not matched single-cell or benchmark cultures.",
            "Released public metadata exclude reported treatments and genetic modifications, but biological sample independence is not established.",
            "Exact symbol and stable Ensembl matching with GENCODE v44 excludes unresolved aliases and non-primary or ambiguous loci.",
            "A ChIP peak overlapping an accessible annotated promoter indicates binding evidence, not causal regulation or regulatory sign.",
            "The K562 prior is a cross-cell-line transfer for RPE1 and does not establish RPE1-specific regulation.",
            "This is a post-hoc graph candidate, not a held-out split, trained model, measured graph effect, or official cell-eval result.",
            "One fixed-seed degree-preserving shuffled graph is reported; uniform sampling or convergence is not established.",
            "All 40 pilot targets appear in K562 training and the shared foundation's cross-source exposure remains unaudited.",
        ],
    }
    digest = selection.write_json(args.output_dir / "graph-summary.json", report)
    print(json.dumps({"state": "complete", "summary_sha256": digest, "graph": graph_summary,
                      "degree_preserving_shuffle": shuffle}), flush=True)


if __name__ == "__main__":
    main()
