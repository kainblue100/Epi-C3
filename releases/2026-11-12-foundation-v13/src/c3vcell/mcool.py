"""Extract bounded distance-normalized contact candidates from a local .mcool."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from urllib.parse import urlsplit

import h5py
import numpy as np

from .epigenetics import COORDINATES
from .integrity import file_sha256


def _chrom_sizes(path: Path) -> dict[str, int]:
    result = {}
    with path.open("r", encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 2 or not fields[0] or fields[0] in result:
                raise ValueError(f"{path}:{number}: invalid or duplicate chromosome size")
            try:
                size = int(fields[1])
            except ValueError as exc:
                raise ValueError(f"{path}:{number}: invalid chromosome size") from exc
            if size < 1:
                raise ValueError(f"{path}:{number}: chromosome size must be positive")
            result[fields[0]] = size
    return result


def _integer_array(values: np.ndarray, label: str) -> np.ndarray:
    array = np.asarray(values)
    if array.dtype.kind not in "iu" or (
        array.dtype.kind == "u" and np.any(array > np.iinfo(np.int64).max)
    ):
        raise ValueError(f"{label} must contain signed 64-bit-compatible integers")
    return array.astype(np.int64, copy=False)


def extract_contact_candidates(
    mcool_path: Path,
    source_metadata_path: Path,
    chrom_sizes_path: Path,
    outdir: Path,
    *,
    resolution: int,
    chromosomes: list[str],
    max_distance: int = 2_000_000,
    min_count: int = 10,
    min_observed_expected: float = 2.0,
    allow_assembly_inference: bool = False,
    chunk_pixels: int = 1_000_000,
    max_candidate_pixels: int = 2_000_000,
) -> Path:
    """Return cis-bin contacts, not statistically called or causal loops."""
    if outdir.exists() and any(outdir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty directory: {outdir}")
    if (
        min(resolution, max_distance, min_count, chunk_pixels, max_candidate_pixels) < 1
        or max_distance < resolution
        or not math.isfinite(min_observed_expected) or min_observed_expected <= 0
        or not chromosomes or len(set(chromosomes)) != len(chromosomes)
    ):
        raise ValueError("Invalid contact extraction parameters")
    source_metadata_digest = file_sha256(source_metadata_path)
    source = json.loads(source_metadata_path.read_text(encoding="utf-8"))
    url = source.get("source_url", "")
    if (
        not isinstance(url, str) or urlsplit(url).scheme != "https" or not urlsplit(url).netloc
        or source.get("assay") not in ("Hi-C", "Micro-C")
        or not isinstance(source.get("context"), str) or not source["context"]
        or source.get("condition") not in ("no-reported-treatment", "unknown")
        or source.get("assembly") not in ("GRCh38", "unknown")
    ):
        raise ValueError("Incomplete or unsupported source metadata for the contact map")
    source_digest = file_sha256(mcool_path)
    if source_digest != source.get("sha256"):
        raise ValueError("Contact-map SHA-256 does not match the source metadata")
    sizes_digest = file_sha256(chrom_sizes_path)
    reference_sizes = _chrom_sizes(chrom_sizes_path)
    max_bins = max_distance // resolution
    candidate_parts = []
    candidate_count = 0
    with h5py.File(mcool_path, "r") as file:
        key = f"resolutions/{resolution}"
        if key not in file:
            raise ValueError(f"The .mcool does not contain resolution {resolution}")
        group = file[key]
        if group.attrs.get("bin-size") != resolution:
            raise ValueError("Contact-map bin size does not match the requested resolution")
        declared = group.attrs.get("genome-assembly", "unknown")
        if isinstance(declared, bytes):
            declared = declared.decode("utf-8")
        if declared not in ("GRCh38", "hg38", "unknown", ""):
            raise ValueError(f"Unsupported contact-map genome assembly: {declared}")
        inferred = declared in ("unknown", "") or source["assembly"] == "unknown"
        if inferred and not allow_assembly_inference:
            raise ValueError("Contact-map assembly is unknown; explicit size-based inference is required")
        names = [
            item.decode("utf-8") if isinstance(item, bytes) else str(item)
            for item in group["chroms/name"][:]
        ]
        lengths = _integer_array(group["chroms/length"][:], "Chromosome lengths")
        if len(set(names)) != len(names) or len(names) != len(lengths):
            raise ValueError("Contact-map chromosomes must be unique")
        for name, length in zip(names, lengths):
            if name not in reference_sizes or reference_sizes[name] != length:
                raise ValueError(f"Contact-map chromosome size does not match GRCh38: {name}")
        if set(chromosomes) - set(names):
            raise ValueError("Requested chromosome is absent from the contact map")
        chrom_ids = _integer_array(group["bins/chrom"][:], "Bin chromosome IDs")
        starts = _integer_array(group["bins/start"][:], "Bin starts")
        ends = _integer_array(group["bins/end"][:], "Bin ends")
        if "weight" not in group["bins"]:
            raise ValueError("Contact map has no multiplicative balanced bin weights")
        weights = np.asarray(group["bins/weight"][:], dtype=np.float64)
        if (
            not (len(chrom_ids) == len(starts) == len(ends) == len(weights))
            or np.any(chrom_ids < 0) or np.any(chrom_ids >= len(names))
            or np.any(np.diff(chrom_ids) < 0)
        ):
            raise ValueError("Contact-map bin arrays are inconsistent")
        valid = np.isfinite(weights) & (weights > 0)
        selected_ids = np.asarray([names.index(name) for name in chromosomes], dtype=np.int64)
        possible = np.zeros(max_bins + 1, dtype=np.int64)
        for chrom_id in selected_ids:
            indices = np.flatnonzero(chrom_ids == chrom_id)
            expected_starts = np.arange(len(indices), dtype=np.int64) * resolution
            if (
                not len(indices) or not np.array_equal(starts[indices], expected_starts)
                or not np.array_equal(ends[indices], np.minimum(expected_starts + resolution, lengths[chrom_id]))
            ):
                raise ValueError(f"Invalid fixed bins for {names[chrom_id]}")
            present = valid[indices]
            for distance in range(1, min(max_bins, len(indices) - 1) + 1):
                possible[distance] += np.count_nonzero(present[:-distance] & present[distance:])
        totals = np.zeros(max_bins + 1, dtype=np.float64)
        pixels = group["pixels"]
        size = len(pixels["count"])
        if len(pixels["bin1_id"]) != size or len(pixels["bin2_id"]) != size:
            raise ValueError("Contact-map pixel arrays have different lengths")
        last_first = -1
        seen_second: set[int] = set()
        for offset in range(0, size, chunk_pixels):
            stop = min(offset + chunk_pixels, size)
            first = _integer_array(pixels["bin1_id"][offset:stop], "First pixel bin IDs")
            second = _integer_array(pixels["bin2_id"][offset:stop], "Second pixel bin IDs")
            raw_counts = np.asarray(pixels["count"][offset:stop])
            if raw_counts.dtype.kind not in "iuf":
                raise ValueError("Contact-map raw counts must be nonnegative integers")
            counts = raw_counts.astype(np.float64, copy=False)
            if (
                np.any(first < 0) or np.any(second >= len(chrom_ids))
                or np.any(first > second) or not np.isfinite(counts).all()
                or np.any(counts < 0) or np.any(counts > 2**53)
                or np.any(counts != np.floor(counts))
            ):
                raise ValueError("Contact-map pixels require valid IDs and nonnegative raw integer counts")
            if len(first):
                if int(first[0]) < last_first or np.any(first[1:] < first[:-1]):
                    raise ValueError("Contact-map pixels must be sorted by bin1 and unique")
                boundaries = np.r_[0, np.flatnonzero(first[1:] != first[:-1]) + 1, len(first)]
                for start, end in zip(boundaries[:-1], boundaries[1:]):
                    row = int(first[start])
                    if row != last_first:
                        seen_second.clear()
                    values = second[start:end].tolist()
                    unique = set(values)
                    if len(unique) != len(values) or seen_second.intersection(unique):
                        raise ValueError("Contact-map pixels must be sorted by bin1 and unique")
                    seen_second.update(unique)
                    last_first = row
            distance = second - first
            keep = (
                (distance > 0) & (distance <= max_bins)
                & (chrom_ids[first] == chrom_ids[second])
                & np.isin(chrom_ids[first], selected_ids) & valid[first] & valid[second]
            )
            a, b, d, raw = first[keep], second[keep], distance[keep], counts[keep]
            balanced = raw * weights[a] * weights[b]
            totals += np.bincount(d, weights=balanced, minlength=max_bins + 1)
            selected = raw >= min_count
            if np.any(selected):
                candidate_count += int(selected.sum())
                if candidate_count > max_candidate_pixels:
                    raise ValueError("Too many candidate pixels; narrow chromosomes or raise min_count")
                candidate_parts.append((a[selected], b[selected], d[selected], raw[selected], balanced[selected]))
        expected = np.divide(totals, possible, out=np.zeros_like(totals), where=possible > 0)
        outdir.mkdir(parents=True, exist_ok=True)
        bedpe = outdir / "contacts.bedpe"
        contacts = 0
        with bedpe.open("x", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
            stream.write("#chrom1\tstart1\tend1\tchrom2\tstart2\tend2\traw_count\tobserved_expected\n")
            for first, second, distance, raw, balanced in candidate_parts:
                denominator = expected[distance]
                oe = np.divide(balanced, denominator, out=np.zeros_like(balanced), where=denominator > 0)
                for a, b, count, enrichment in zip(first[oe >= min_observed_expected],
                                                    second[oe >= min_observed_expected],
                                                    raw[oe >= min_observed_expected],
                                                    oe[oe >= min_observed_expected]):
                    writer.writerow((
                        names[chrom_ids[a]], starts[a], ends[a], names[chrom_ids[b]],
                        starts[b], ends[b], int(count), format(float(enrichment), ".8g"),
                    ))
                    contacts += 1
    if (
        file_sha256(mcool_path) != source_digest or file_sha256(chrom_sizes_path) != sizes_digest
        or file_sha256(source_metadata_path) != source_metadata_digest
    ):
        raise RuntimeError("Contact map, metadata, or chromosome sizes changed during processing")
    if not contacts:
        raise ValueError("No candidate contacts passed the selected thresholds")
    derived = {
        "role": "hic_loops",
        "path": str(bedpe.resolve()),
        "sha256": file_sha256(bedpe),
        "source_url": source["source_url"],
        "input_mcool_sha256": source_digest,
        "source_metadata_sha256": source_metadata_digest,
        "input_scope": source.get("input_scope", "complete source object"),
        "chrom_sizes_sha256": sizes_digest,
        "assembly": "GRCh38",
        "assembly_evidence": (
            "inferred from all available chromosome lengths; not definitive"
            if inferred else "source declaration and chromosome lengths"
        ),
        "context": source["context"],
        "condition": source["condition"],
        "assay": source["assay"],
        "coordinates": COORDINATES,
        "evidence_type": "distance-normalized-contact",
        "resolution_bp": resolution,
        "covered_chromosomes": chromosomes,
        "parameters": {
            "max_distance_bp": max_distance,
            "min_raw_count": min_count,
            "min_observed_expected": min_observed_expected,
            "expected": "mean balanced cis count including zero pixels at each bin separation",
            "expected_chromosomes": chromosomes,
            "output_contacts": contacts,
        },
        "note": (
            "Thresholded contact candidates, not FDR-controlled loop calls, paired "
            "single-cell measurements, or causal enhancer-promoter links."
        ),
    }
    metadata = outdir / "source.json"
    with metadata.open("x", encoding="utf-8") as stream:
        json.dump(derived, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return metadata
