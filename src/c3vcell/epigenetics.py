"""Build and validate context-specific, bulk epigenomic priors."""

from __future__ import annotations

import bisect
import csv
import gzip
import hashlib
import json
import math
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import urlsplit

import anndata as ad
import numpy as np
import pandas as pd

from .integrity import file_sha256

FEATURES = ("atac", "ctcf", "sp1", "usf1", "hic_loop")
NETWORK_FEATURES = (
    "atac", "ctcf", "sp1", "usf1", "chromatin_contact",
    "promoter_methylation", "linked_element_methylation",
)
FEATURES_BY_VERSION = {1: FEATURES, 2: NETWORK_FEATURES}
TRACKS = ("atac", "ctcf", "sp1", "usf1")
FORMAT_VERSION = 1
COORDINATES = "zero-based-half-open"
CONDITION = "no-reported-treatment"


def _axis(data: ad.AnnData) -> list[tuple[str, str]]:
    if not data.var_names.is_unique or "ensembl_ids" not in data.var:
        raise ValueError("Epigenomic priors require unique genes and var['ensembl_ids']")
    if data.var["ensembl_ids"].isna().any():
        raise ValueError("Gene-axis Ensembl identifiers cannot be missing")
    return list(zip(data.var_names.astype(str), data.var["ensembl_ids"].astype(str)))


def _axis_digest(axis: list[tuple[str, str]]) -> str:
    encoded = json.dumps(axis, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_context(data: ad.AnnData, metadata: dict) -> None:
    if "context" not in data.obs or data.obs["context"].isna().any():
        raise ValueError("Epigenomic priors require explicit, nonmissing cell contexts")
    if set(data.obs["context"].astype(str)) != {metadata["context"]}:
        raise ValueError("Cell context does not match the epigenomic reference")
    if data.uns.get("c3vcell_genome_assembly") != metadata["assembly"]:
        raise ValueError("Cell data genome assembly does not match the epigenomic reference")


def validate_prior_data(data: ad.AnnData, metadata: dict) -> None:
    validate_context(data, metadata)
    if metadata.get("gene_axis_sha256") != _axis_digest(_axis(data)):
        raise ValueError("Epigenomic bundle gene axis or Ensembl identities differ")


def validate_checkpoint_context(checkpoint: dict, data: ad.AnnData) -> None:
    metadata = checkpoint.get("epigenetics")
    if metadata is not None:
        validate_prior_data(data, metadata)


def _stable_id(value: str) -> str | None:
    if "|" in value or not value:
        return None
    result = value.split(".", 1)[0]
    return result if re.fullmatch(r"ENSG[0-9]+", result) else None


class _IntervalIndex:
    def __init__(self, intervals: list[tuple[str, int, int, str]]) -> None:
        by_chrom: dict[str, list[tuple[int, int, str]]] = defaultdict(list)
        for chrom, start, end, value in intervals:
            by_chrom[chrom].append((start, end, value))
        self.data = {}
        for chrom, rows in by_chrom.items():
            rows.sort()
            starts = [row[0] for row in rows]
            maxima = []
            maximum = -1
            for _, end, _ in rows:
                maximum = max(maximum, end)
                maxima.append(maximum)
            self.data[chrom] = starts, maxima, rows

    def overlaps(self, chrom: str, start: int, end: int) -> list[tuple[int, int, str]]:
        if chrom not in self.data:
            return []
        starts, maxima, rows = self.data[chrom]
        index = bisect.bisect_left(starts, end)
        found = []
        while index and maxima[index - 1] > start:
            index -= 1
            if rows[index][1] > start:
                found.append(rows[index])
        return found


def _open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def _bed(path: Path) -> _IntervalIndex:
    intervals = []
    with _open_text(path) as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip() or line.startswith(("#", "track ", "browser ")):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 3:
                raise ValueError(f"{path}:{number}: BED requires three tab-separated columns")
            try:
                start, end = int(fields[1]), int(fields[2])
            except ValueError as exc:
                raise ValueError(f"{path}:{number}: invalid BED coordinates") from exc
            if not fields[0] or start < 0 or end <= start:
                raise ValueError(f"{path}:{number}: invalid BED interval")
            intervals.append((fields[0], start, end, ""))
    if not intervals:
        raise ValueError(f"No intervals in {path}")
    return _IntervalIndex(intervals)


def _promoters(
    path: Path, axis: list[tuple[str, str]], flank: int,
) -> tuple[dict[str, tuple[str, int, int]], dict]:
    records: dict[str, set[tuple[str, int, int]]] = defaultdict(set)
    pattern = re.compile(r'(?:^|;\s*)gene_id "([^"]+)"')
    with _open_text(path) as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip() or line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 9:
                raise ValueError(f"{path}:{number}: GTF requires nine columns")
            if fields[2] != "gene":
                continue
            match = pattern.search(fields[8])
            if match is None:
                raise ValueError(f"{path}:{number}: GTF gene lacks gene_id")
            identifier = _stable_id(match.group(1))
            if identifier is None:
                continue
            try:
                start, end = int(fields[3]), int(fields[4])
            except ValueError as exc:
                raise ValueError(f"{path}:{number}: invalid GTF coordinates") from exc
            if start < 1 or end < start or fields[6] not in ("+", "-"):
                raise ValueError(f"{path}:{number}: invalid GTF gene interval or strand")
            tss = start - 1 if fields[6] == "+" else end - 1
            records[identifier].add((fields[0], max(0, tss - flank), tss + flank + 1))
    ids = [_stable_id(ensembl) for _, ensembl in axis]
    counts = Counter(identifier for identifier in ids if identifier is not None)
    promoters = {}
    for (gene, _), identifier in zip(axis, ids):
        if identifier is not None and counts[identifier] == 1 and len(records[identifier]) == 1:
            promoters[gene] = next(iter(records[identifier]))
    if not promoters:
        raise ValueError("No unambiguous Ensembl gene identifiers matched the annotation")
    return promoters, {
        "genes": len(axis),
        "mapped_unambiguous_genes": len(promoters),
        "unmapped_or_ambiguous_genes": len(axis) - len(promoters),
    }


def _active_tf(
    chrom: str, start: int, end: int, tf: _IntervalIndex, atac: _IntervalIndex,
) -> bool:
    return bool(_active_tf_regions(chrom, start, end, tf, atac))


def _active_tf_regions(
    chrom: str, start: int, end: int, tf: _IntervalIndex, atac: _IntervalIndex,
) -> set[tuple[str, int, int]]:
    result = set()
    for left, right, _ in tf.overlaps(chrom, start, end):
        left, right = max(start, left), min(end, right)
        for peak_start, peak_end, _ in atac.overlaps(chrom, left, right):
            result.add((chrom, max(left, peak_start), min(right, peak_end)))
    return result


class _CpGMethylationIndex:
    def __init__(self, path: Path, min_coverage: int) -> None:
        rows: dict[str, list[tuple[int, float, int]]] = defaultdict(list)
        observed = set()
        with _open_text(path) as stream:
            for number, line in enumerate(stream, 1):
                if not line.strip() or line.startswith("#"):
                    continue
                fields = line.rstrip("\n").split("\t")
                if len(fields) != 5:
                    raise ValueError(f"{path}:{number}: expected chromosome, start, end, fraction, coverage")
                try:
                    start, end = int(fields[1]), int(fields[2])
                    fraction, coverage = float(fields[3]), int(fields[4])
                except ValueError as exc:
                    raise ValueError(f"{path}:{number}: invalid CpG methylation value") from exc
                key = (fields[0], start)
                if (
                    not fields[0] or start < 0 or end != start + 1
                    or not np.isfinite(fraction) or not 0 <= fraction <= 1
                    or coverage < min_coverage or key in observed
                ):
                    raise ValueError(f"{path}:{number}: invalid or duplicated CpG measurement")
                observed.add(key)
                rows[fields[0]].append((start, fraction, coverage))
        if not observed:
            raise ValueError(f"No valid CpG measurements in {path}")
        self._index(rows)

    def _index(self, rows: dict[str, list[tuple[int, float, int]]]) -> None:
        self.data = {}
        for chrom, sites in rows.items():
            sites.sort()
            starts = [row[0] for row in sites]
            numerator = np.r_[0.0, np.cumsum([fraction * coverage for _, fraction, coverage in sites])]
            denominator = np.r_[0, np.cumsum([coverage for _, _, coverage in sites], dtype=np.int64)]
            self.data[chrom] = starts, numerator, denominator

    @classmethod
    def from_uncovered_fractions(
        cls, path: Path, regions: set[tuple[str, int, int]], covered_chromosomes: set[str],
    ) -> _CpGMethylationIndex:
        by_chrom: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for chrom, start, end in regions:
            by_chrom[chrom].append((start, end))
        queries = {}
        for chrom, intervals in by_chrom.items():
            merged: list[tuple[int, int]] = []
            for start, end in sorted(intervals):
                if merged and start <= merged[-1][1]:
                    merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
                else:
                    merged.append((start, end))
            queries[chrom] = [row[0] for row in merged], [row[1] for row in merged]
        rows: dict[str, list[tuple[int, float, int]]] = defaultdict(list)
        observed = 0
        seen_chromosomes = set()
        previous = None
        with _open_text(path) as stream:
            for number, line in enumerate(stream, 1):
                if not line.strip() or line.startswith(("#", "track ", "browser ")):
                    continue
                fields = line.rstrip("\n").split("\t")
                if len(fields) != 4:
                    raise ValueError(f"{path}:{number}: expected chromosome, start, end, CpG fraction")
                try:
                    start, end = int(fields[1]), int(fields[2])
                    fraction = float(fields[3])
                except ValueError as exc:
                    raise ValueError(f"{path}:{number}: invalid CpG fraction") from exc
                key = (fields[0], start)
                if (
                    not fields[0] or start < 0 or end != start + 1
                    or not math.isfinite(fraction) or not 0 <= fraction <= 1
                    or fields[0] not in covered_chromosomes
                ):
                    raise ValueError(f"{path}:{number}: invalid CpG coordinate, fraction, or coverage")
                if previous is not None and key <= previous:
                    raise ValueError(f"{path}:{number}: duplicated or unsorted CpG dyad")
                previous = key
                seen_chromosomes.add(fields[0])
                intervals = queries.get(fields[0])
                if intervals is None:
                    continue
                starts, ends = intervals
                index = bisect.bisect_right(starts, start) - 1
                if index < 0 or start >= ends[index]:
                    continue
                observed += 1
                rows[fields[0]].append((start, fraction, 1))
        if seen_chromosomes != covered_chromosomes:
            raise ValueError("Declared methylation chromosome coverage differs from the CpG file")
        if not observed:
            raise ValueError(f"No observed CpGs overlap the requested regions in {path}")
        result = cls.__new__(cls)
        result._index(rows)
        return result

    def totals(self, chrom: str, start: int, end: int) -> tuple[float, int]:
        if chrom not in self.data:
            return 0.0, 0
        starts, numerator, denominator = self.data[chrom]
        left, right = bisect.bisect_left(starts, start), bisect.bisect_left(starts, end)
        return float(numerator[right] - numerator[left]), int(denominator[right] - denominator[left])

    def mean(self, regions: set[tuple[str, int, int]]) -> tuple[float, bool]:
        merged: list[tuple[str, int, int]] = []
        for chrom, start, end in sorted(regions):
            if merged and merged[-1][0] == chrom and start <= merged[-1][2]:
                previous = merged[-1]
                merged[-1] = (chrom, previous[1], max(previous[2], end))
            else:
                merged.append((chrom, start, end))
        totals = [self.totals(chrom, start, end) for chrom, start, end in merged]
        numerator = sum(total[0] for total in totals)
        denominator = sum(total[1] for total in totals)
        return (numerator / denominator, True) if denominator else (0.0, False)


def feature_names(metadata: dict) -> tuple[str, ...]:
    names = FEATURES_BY_VERSION.get(metadata.get("format_version"))
    if names is None or metadata.get("features") != list(names):
        raise ValueError("Unsupported epigenomic bundle feature schema")
    return names


def _source_records(spec_path: Path, spec: dict) -> tuple[dict[str, Path], list[dict]]:
    context, assembly = spec.get("context"), spec.get("assembly")
    if not isinstance(context, str) or not context or assembly != "GRCh38":
        raise ValueError("A nonempty context and GRCh38 assembly are required")
    version = spec.get("format_version")
    if spec.get("condition") != CONDITION or version not in FEATURES_BY_VERSION:
        raise ValueError("Unsupported epigenomic specification or condition")
    sources = spec.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("The epigenomic specification needs source records")
    paths = {}
    provenance = []
    for entry in sources:
        if not isinstance(entry, dict):
            raise ValueError("Each epigenomic source must be an object")
        role = entry.get("role")
        allowed = {*TRACKS, "annotation", "hic_loops"}
        if version == 2:
            allowed.update(("chromatin_contacts", "methylation"))
        if role not in allowed or role in paths:
            raise ValueError(f"Unknown or duplicate epigenomic source role: {role!r}")
        if entry.get("assembly") != assembly:
            raise ValueError(f"Assembly mismatch for epigenomic source {role}")
        url = entry.get("source_url")
        if not isinstance(url, str) or urlsplit(url).scheme != "https" or not urlsplit(url).netloc:
            raise ValueError(f"Source {role} needs an HTTPS provenance URL")
        digest = entry.get("sha256")
        if not isinstance(digest, str) or re.fullmatch(r"[a-f0-9]{64}", digest) is None:
            raise ValueError(f"Source {role} needs a lowercase SHA-256")
        if role != "annotation" and (
            entry.get("context") != context or entry.get("condition") != CONDITION
        ):
            raise ValueError(f"Context or treatment mismatch for epigenomic source {role}")
        if role in TRACKS:
            expected_assay = "ATAC-seq" if role == "atac" else "TF ChIP-seq"
            expected_target = None if role == "atac" else role.upper()
            if (
                entry.get("assay") != expected_assay
                or entry.get("target") != expected_target
                or entry.get("coordinates") != COORDINATES
                or not isinstance(entry.get("accession"), str)
                or not entry["accession"]
            ):
                raise ValueError(f"Peak track metadata does not match its role: {role}")
        if role in ("hic_loops", "chromatin_contacts"):
            chromosomes = entry.get("covered_chromosomes")
            assay = entry.get("assay")
            resolution = entry.get("resolution_bp")
            anchor_widths = entry.get("anchor_widths_bp")
            multi_resolution = (
                version == 2 and assay in ("Hi-C", "Micro-C")
                and entry.get("evidence_type") == "called-loop"
                and resolution is None and entry.get("resolution_kind") == "multi-resolution-anchor-interval"
                and isinstance(anchor_widths, list) and bool(anchor_widths)
                and all(
                    isinstance(width, int) and not isinstance(width, bool) and 0 < width <= 1_000_000
                    for width in anchor_widths
                )
                and anchor_widths == sorted(set(anchor_widths))
            )
            resolution_valid = (
                assay == "ChIA-PET" and entry.get("evidence_type") == "called-loop"
                and resolution is None and entry.get("resolution_kind") == "anchor-interval"
                and isinstance(entry.get("target"), str) and bool(entry["target"])
            ) or (
                assay in ("Hi-C", "Micro-C")
                and isinstance(resolution, int) and not isinstance(resolution, bool) and resolution > 0
                and entry.get("resolution_kind") != "multi-resolution-anchor-interval"
            ) or multi_resolution
            if (
                not resolution_valid or entry.get("coordinates") != COORDINATES
                or entry.get("evidence_type") not in ("called-loop", "distance-normalized-contact")
                or not isinstance(chromosomes, list)
                or not chromosomes
                or not all(isinstance(chrom, str) and chrom for chrom in chromosomes)
                or len(set(chromosomes)) != len(chromosomes)
            ):
                raise ValueError(
                    "Chromatin contacts need assay, evidence type, coordinate, resolution and coverage metadata"
                )
        if role == "methylation":
            covered_cpg = (
                entry.get("assay") in ("RRBS", "WGBS", "EM-seq")
                and entry.get("file_format") == "cpg-fraction-bed"
                and isinstance(entry.get("min_coverage"), int)
                and not isinstance(entry.get("min_coverage"), bool)
                and entry["min_coverage"] >= 1
            )
            fraction_only = (
                version == 2 and entry.get("assay") == "WGBS"
                and entry.get("file_format") == "cpg-fraction-bed-no-coverage"
                and "min_coverage" in entry and entry["min_coverage"] is None
                and entry.get("site_coverage") == "not-available"
                and entry.get("aggregation") == "equal-strand-then-cpg-mean"
                and entry.get("reference_sequence_validated") is True
                and isinstance(entry.get("covered_chromosomes"), list)
                and bool(entry["covered_chromosomes"])
                and all(
                    isinstance(chrom, str) and chrom for chrom in entry["covered_chromosomes"]
                )
                and len(set(entry["covered_chromosomes"])) == len(entry["covered_chromosomes"])
            )
            if entry.get("coordinates") != COORDINATES or not (covered_cpg or fraction_only):
                raise ValueError("Methylation needs explicit CpG fraction, coverage and assay metadata")
        raw_path = entry.get("path")
        if not isinstance(raw_path, str) or not raw_path:
            raise ValueError(f"Source {role} needs a local file path")
        path = Path(raw_path)
        if not path.is_absolute():
            path = spec_path.parent / path
        if not path.is_file() or file_sha256(path) != digest:
            raise ValueError(f"Missing or changed epigenomic source {role}: {path}")
        paths[role] = path
        provenance.append(dict(entry))
    if "annotation" not in paths or not (set(paths) - {"annotation"}):
        raise ValueError("A gene annotation and at least one epigenomic source are required")
    if {"hic_loops", "chromatin_contacts"} <= set(paths):
        raise ValueError("Specify only one combined chromatin-contact source")
    return paths, provenance


def _loops(path: Path):
    with _open_text(path) as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip() or line.startswith(("#", "track ", "browser ")):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 6:
                raise ValueError(f"{path}:{number}: BEDPE requires six columns")
            try:
                start1, end1, start2, end2 = map(int, (fields[1], fields[2], fields[4], fields[5]))
            except ValueError as exc:
                raise ValueError(f"{path}:{number}: invalid BEDPE coordinates") from exc
            if (
                not fields[0] or not fields[3] or min(start1, start2) < 0
                or end1 <= start1 or end2 <= start2
                or max(end1 - start1, end2 - start2) > 1_000_000
            ):
                raise ValueError(f"{path}:{number}: invalid or excessively wide BEDPE anchor")
            yield (fields[0], start1, end1), (fields[3], start2, end2)


def build_epigenetic_bundle(
    gene_data_path: Path, spec_path: Path, outdir: Path, *, promoter_flank: int = 2000,
) -> Path:
    """Create static reference features and candidate messages, without reading expression."""
    if promoter_flank < 0 or promoter_flank > 100_000:
        raise ValueError("Promoter flank must be between 0 and 100000 bases")
    if outdir.exists() and any(outdir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty directory: {outdir}")
    spec_digest = file_sha256(spec_path)
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    if not isinstance(spec, dict):
        raise ValueError("Epigenomic specification must be a JSON object")
    paths, sources = _source_records(spec_path, spec)
    version = spec["format_version"]
    features = FEATURES_BY_VERSION[version]
    data = ad.read_h5ad(gene_data_path, backed="r")
    try:
        validate_context(data, spec)
        axis = _axis(data)
    finally:
        data.file.close()
    promoters, mapping = _promoters(paths["annotation"], axis, promoter_flank)
    indexes = {role: _bed(paths[role]) for role in TRACKS if role in paths}
    methyl_source = next((entry for entry in sources if entry["role"] == "methylation"), None)
    graph: dict[tuple[str, str], set[str]] = defaultdict(set)
    linked_regions: dict[str, set[tuple[str, int, int]]] = defaultdict(set)
    values = np.zeros((len(axis), len(features)), dtype=np.float32)
    masks = np.zeros_like(values)
    gene_index = {gene: index for index, (gene, _) in enumerate(axis)}
    for gene, (chrom, start, end) in promoters.items():
        row = gene_index[gene]
        for column, role in enumerate(TRACKS):
            if role in indexes:
                masks[row, column] = 1
                values[row, column] = bool(indexes[role].overlaps(chrom, start, end))
        if "atac" in indexes:
            for role, tf_gene in (("sp1", "SP1"), ("usf1", "USF1")):
                if (
                    role in indexes and tf_gene in gene_index and tf_gene != gene
                    and _active_tf(chrom, start, end, indexes[role], indexes["atac"])
                ):
                    graph[(tf_gene, gene)].add("promoter_tf_atac")
    loop_count = 0
    contact_role = "chromatin_contacts" if "chromatin_contacts" in paths else "hic_loops"
    if contact_role in paths:
        loop_source = next(entry for entry in sources if entry["role"] == contact_role)
        covered = set(loop_source["covered_chromosomes"])
        multi_resolution = loop_source.get("resolution_kind") == "multi-resolution-anchor-interval"
        anchor_widths = set(loop_source.get("anchor_widths_bp", [])) if multi_resolution else set()
        seen_widths = set()
        promoter_index = _IntervalIndex([
            (chrom, start, end, gene) for gene, (chrom, start, end) in promoters.items()
        ])
        for gene, (chrom, _, _) in promoters.items():
            if chrom in covered:
                masks[gene_index[gene], 4] = 1
        for first, second in _loops(paths[contact_role]):
            if first[0] not in covered or second[0] not in covered:
                raise ValueError("A loop anchor is outside the declared chromosome coverage")
            if multi_resolution:
                widths = {first[2] - first[1], second[2] - second[1]}
                if not widths <= anchor_widths:
                    raise ValueError("A loop anchor has an undeclared interval width")
                seen_widths.update(widths)
            loop_count += 1
            left = {row[2] for row in promoter_index.overlaps(*first)}
            right = {row[2] for row in promoter_index.overlaps(*second)}
            if len(left) * len(right) > 10_000:
                raise ValueError("A Hi-C loop maps to too many promoter pairs")
            for gene in left | right:
                values[gene_index[gene], 4] = 1
            for a in left:
                for b in right:
                    if a != b:
                        evidence = "hic_promoter_contact" if version == 1 else "contact_promoter"
                        graph[(a, b)].add(evidence)
                        graph[(b, a)].add(evidence)
            if "atac" in indexes:
                for anchor, destinations in ((first, right), (second, left)):
                    if methyl_source is not None:
                        chrom, start, end = anchor
                        accessible_regions = {
                            (chrom, max(start, left), min(end, right))
                            for left, right, _ in indexes["atac"].overlaps(chrom, start, end)
                        }
                        for gene in destinations:
                            linked_regions[gene].update(accessible_regions)
                    for role, tf_gene in (("sp1", "SP1"), ("usf1", "USF1")):
                        if role in indexes and tf_gene in gene_index:
                            regions = _active_tf_regions(*anchor, indexes[role], indexes["atac"])
                        else:
                            regions = set()
                        if regions:
                            for gene in destinations - {tf_gene}:
                                evidence = "hic_tf_atac" if version == 1 else "contact_tf_atac"
                                graph[(tf_gene, gene)].add(evidence)
        if multi_resolution and seen_widths != anchor_widths:
            raise ValueError("Declared multi-resolution anchor widths do not match the loops")

    if methyl_source is not None:
        if methyl_source["file_format"] == "cpg-fraction-bed-no-coverage":
            regions = set(promoters.values())
            for linked in linked_regions.values():
                regions.update(linked)
            methylation = _CpGMethylationIndex.from_uncovered_fractions(
                paths["methylation"], regions, set(methyl_source["covered_chromosomes"]),
            )
        else:
            methylation = _CpGMethylationIndex(paths["methylation"], methyl_source["min_coverage"])
        for gene, region in promoters.items():
            fraction, measured = methylation.mean({region})
            row = gene_index[gene]
            values[row, 5], masks[row, 5] = fraction, measured
        for gene, regions in linked_regions.items():
            fraction, measured = methylation.mean(regions)
            row = gene_index[gene]
            values[row, 6], masks[row, 6] = fraction, measured

    frame = pd.DataFrame({"gene": [x[0] for x in axis], "ensembl_ids": [x[1] for x in axis]})
    for index, feature in enumerate(features):
        frame[feature] = values[:, index]
        frame[f"{feature}_mask"] = masks[:, index]
    outdir.mkdir(parents=True, exist_ok=True)
    features_path, graph_path = outdir / "features.csv", outdir / "graph.csv"
    with features_path.open("x", encoding="utf-8", newline="") as stream:
        frame.to_csv(stream, index=False)
    with graph_path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("source", "target", "weight", "evidence"))
        for (source, target), evidence in sorted(graph.items()):
            writer.writerow((source, target, len(evidence), "|".join(sorted(evidence))))
    for entry in sources:
        if file_sha256(paths[entry["role"]]) != entry["sha256"]:
            raise RuntimeError(f"Epigenomic source changed during processing: {entry['role']}")
    if file_sha256(spec_path) != spec_digest:
        raise RuntimeError("Epigenomic specification changed during processing")
    manifest = {
        "format_version": version,
        "context": spec["context"],
        "assembly": spec["assembly"],
        "condition": spec["condition"],
        "features": list(features),
        "gene_axis_sha256": _axis_digest(axis),
        "specification_sha256": spec_digest,
        "sources": sources,
        "promoter_flank_bp": promoter_flank,
        "mapping": mapping,
        "edge_count": len(graph),
        "loop_count": loop_count,
        "files": {
            "features.csv": file_sha256(features_path),
            "graph.csv": file_sha256(graph_path),
        },
        "semantics": {
            "mask": (
                "For peaks/contacts, 1 means an available source and unambiguous gene; "
                "for methylation, 1 requires at least one observed CpG in the specified region."
            ),
            "zero": (
                "A binary zero means no peak/contact was called, not biological absence. "
                "A methylation zero with mask 1 is an observed fraction; mask 0 means missing."
            ),
            "graph": "row=perturbed source gene, column=neighbor read by the model",
            "contacts": (
                "Externally called assay-specific loops or explicitly labeled distance-normalized "
                "candidates; the builder performs no normalization or loop calling. ChIA-PET "
                "is protein-enriched and is not Hi-C."
            ),
            "methylation": (
                "Equal mean across reference-sequence-validated CpG dyads over promoters or "
                "ATAC-overlapping distal anchors. Available strand fractions were averaged "
                "equally per dyad; read coverage is unavailable and was not used. "
                "Methylation sign and causality are not assumed."
                if methyl_source is not None
                and methyl_source["file_format"] == "cpg-fraction-bed-no-coverage"
                else "Coverage-weighted CpG fraction over promoters or ATAC-overlapping distal "
                "anchors linked to that gene; methylation sign and causality are not assumed."
            ),
            "causality": "bulk reference associations are not paired measurements or causal edges",
        },
    }
    manifest_path = outdir / "manifest.json"
    with manifest_path.open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return manifest_path


def load_epigenetic_bundle(bundle: Path, data: ad.AnnData) -> tuple[np.ndarray, np.ndarray, dict]:
    manifest_path = bundle / "manifest.json"
    digest = file_sha256(manifest_path)
    metadata = json.loads(manifest_path.read_text(encoding="utf-8"))
    features = feature_names(metadata)
    if metadata.get("condition") != CONDITION or metadata.get("assembly") != "GRCh38":
        raise ValueError("Unsupported epigenomic bundle")
    axis = _axis(data)
    validate_prior_data(data, metadata)
    for name in ("features.csv", "graph.csv"):
        if file_sha256(bundle / name) != metadata.get("files", {}).get(name):
            raise ValueError(f"Epigenomic bundle file changed: {name}")
    frame = pd.read_csv(bundle / "features.csv", dtype={"gene": str, "ensembl_ids": str})
    if list(zip(frame["gene"], frame["ensembl_ids"])) != axis:
        raise ValueError("Epigenomic feature rows do not match the gene axis")
    values = frame[list(features)].to_numpy(dtype=np.float32)
    masks = frame[[f"{name}_mask" for name in features]].to_numpy(dtype=np.float32)
    if (
        not np.isfinite(values).all() or np.any((values < 0) | (values > 1))
        or not np.isin(values[:, :5], (0, 1)).all()
        or not np.isin(masks, (0, 1)).all() or np.any(values[masks == 0] != 0)
    ):
        raise ValueError("Epigenomic feature values or availability masks are invalid")
    edges = pd.read_csv(bundle / "graph.csv", keep_default_na=False)
    if (
        list(edges.columns) != ["source", "target", "weight", "evidence"]
        or len(edges) != metadata.get("edge_count")
        or edges.duplicated(["source", "target"]).any()
    ):
        raise ValueError("Epigenomic graph structure does not match the manifest")
    genes = {gene for gene, _ in axis}
    evidence_types = (
        {"promoter_tf_atac", "hic_promoter_contact", "hic_tf_atac"}
        if metadata["format_version"] == 1
        else {"promoter_tf_atac", "contact_promoter", "contact_tf_atac"}
    )
    for edge in edges.itertuples(index=False):
        types = str(edge.evidence).split("|")
        if (
            edge.source not in genes or edge.target not in genes or edge.source == edge.target
            or len(types) != len(set(types)) or not set(types) <= evidence_types
            or not np.isfinite(edge.weight) or edge.weight != len(types)
        ):
            raise ValueError("Epigenomic graph contains an invalid evidence edge")
    if file_sha256(manifest_path) != digest:
        raise RuntimeError("Epigenomic bundle manifest changed during loading")
    metadata["manifest_sha256"] = digest
    return values, masks, metadata


def _shuffle_graph_targets(
    edges: pd.DataFrame, genes: np.ndarray, rng: np.random.Generator,
) -> tuple[pd.DataFrame, int]:
    indices = {gene: index for index, gene in enumerate(genes)}
    original = {(source, target) for source, target in zip(edges["source"], edges["target"])}
    forbidden = {(indices[target], indices[source]) for source, target in original}
    permutation = np.arange(len(genes))
    swaps = 0
    for _ in range(10):
        for left, right in rng.integers(0, len(genes), size=(max(100, 12 * len(genes)), 2)):
            if left == right or (left, permutation[right]) in forbidden or (
                right, permutation[left]
            ) in forbidden:
                continue
            permutation[left], permutation[right] = permutation[right], permutation[left]
            swaps += 1
        shuffled = edges.copy()
        shuffled["target"] = [genes[permutation[indices[target]]] for target in edges["target"]]
        changed = {(source, target) for source, target in zip(shuffled["source"], shuffled["target"])}
        if changed != original:
            return shuffled.sort_values(["source", "target"]), swaps
    raise ValueError("Cannot create a structurally randomized, self-edge-free graph")


def shuffle_epigenetic_bundle(
    bundle: Path, gene_data_path: Path, outdir: Path, *, seed: int,
    shuffle_features: bool = True,
) -> Path:
    """Create a labeled randomization control without using perturbation outcomes."""
    if seed < 0:
        raise ValueError("Shuffle seed must be nonnegative")
    if not isinstance(shuffle_features, bool):
        raise ValueError("shuffle_features must be a boolean")
    if outdir.exists() and any(outdir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty directory: {outdir}")
    data = ad.read_h5ad(gene_data_path, backed="r")
    try:
        values, masks, metadata = load_epigenetic_bundle(bundle, data)
        axis = _axis(data)
    finally:
        data.file.close()
    if metadata.get("control") is not None:
        raise ValueError("A shuffled bundle cannot be used as the real reference")
    features = feature_names(metadata)
    rng = np.random.default_rng(seed)
    shuffled = values.copy()
    if shuffle_features:
        for column in range(len(features)):
            measured = np.flatnonzero(masks[:, column] == 1)
            shuffled[measured, column] = rng.permutation(values[measured, column])
    edges = pd.read_csv(bundle / "graph.csv", keep_default_na=False)
    if file_sha256(bundle / "graph.csv") != metadata["files"]["graph.csv"]:
        raise RuntimeError("Epigenomic graph changed during randomization")
    genes = np.asarray([gene for gene, _ in axis], dtype=object)
    graph_swaps = 0
    if len(edges):
        edges, graph_swaps = _shuffle_graph_targets(edges, genes, rng)
    if not len(edges) and np.array_equal(values, shuffled):
        raise ValueError("The reference has no randomizable signal")
    outdir.mkdir(parents=True, exist_ok=True)
    features_path, graph_path = outdir / "features.csv", outdir / "graph.csv"
    if shuffle_features:
        frame = pd.DataFrame({"gene": genes, "ensembl_ids": [ensembl for _, ensembl in axis]})
        for index, feature in enumerate(features):
            frame[feature] = shuffled[:, index]
            frame[f"{feature}_mask"] = masks[:, index]
        with features_path.open("x", encoding="utf-8", newline="") as stream:
            frame.to_csv(stream, index=False)
    else:
        with (bundle / "features.csv").open("rb") as source, features_path.open("xb") as target:
            shutil.copyfileobj(source, target)
        if file_sha256(features_path) != metadata["files"]["features.csv"]:
            raise RuntimeError("Epigenomic features changed during randomization")
    with graph_path.open("x", encoding="utf-8", newline="") as stream:
        edges.to_csv(stream, index=False)
    reference_digest = metadata.pop("manifest_sha256")
    if (
        file_sha256(bundle / "manifest.json") != reference_digest
        or file_sha256(bundle / "features.csv") != metadata["files"]["features.csv"]
        or file_sha256(bundle / "graph.csv") != metadata["files"]["graph.csv"]
    ):
        raise RuntimeError("Epigenomic reference changed during randomization")
    metadata["files"] = {
        "features.csv": file_sha256(features_path), "graph.csv": file_sha256(graph_path),
    }
    metadata["control"] = {
        "kind": (
            "within-observed-feature-and-graph-target-shuffle" if shuffle_features
            else "graph-target-shuffle-only"
        ),
        "seed": seed,
        "features_shuffled": shuffle_features,
        "reference_manifest_sha256": reference_digest,
        "graph_permutation": "constrained random transpositions",
        "successful_graph_swaps": graph_swaps,
        "note": (
            "The original source evidence labels no longer describe these randomized edges. "
            "The constrained transpositions are not a uniform sample of all valid graphs."
        ),
    }
    path = outdir / "manifest.json"
    with path.open("x", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return path
