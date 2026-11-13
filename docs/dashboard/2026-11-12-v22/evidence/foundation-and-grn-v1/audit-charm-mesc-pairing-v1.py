"""Audit source-backed CHARM mESC identifiers without manufacturing cell pairs."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import gzip
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import tarfile

import numpy as np


IDENTIFIER = re.compile(r"P[0-9]+\Z")
FRACTION_TOLERANCE = 1e-6


def describe_input(path: Path) -> dict:
    if path.name.endswith(".part"):
        raise ValueError("A partial download cannot be audited")
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError(f"Input changed while hashing: {path.name}")
    return {"name": path.name, "bytes": before.st_size, "sha256": digest.hexdigest(),
            "mtime_ns": before.st_mtime_ns}


def rna_summary(path: Path) -> tuple[dict, dict[str, float]]:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as stream:
        header = stream.readline().rstrip("\r\n").split("\t")
        cells = header[1:]
        if (
            header[0] != "gene" or not cells or len(set(cells)) != len(cells)
            or any(IDENTIFIER.fullmatch(cell) is None for cell in cells)
        ):
            raise ValueError("The mESC RNA matrix needs unique, well-formed cell columns")
        totals = np.zeros(len(cells), dtype=np.float64)
        detected = np.zeros(len(cells), dtype=np.int64)
        genes: set[str] = set()
        fractional = nonzero = 0
        for line in stream:
            gene, separator, values = line.rstrip("\r\n").partition("\t")
            if not separator or not gene or gene in genes:
                raise ValueError("The mESC RNA matrix has a missing or duplicated gene")
            genes.add(gene)
            fields = values.split("\t")
            if len(fields) != len(cells):
                raise ValueError("The mESC RNA matrix contains malformed counts")
            row = np.asarray(fields, dtype=np.float64)
            if not np.isfinite(row).all() or np.any(row < 0):
                raise ValueError("The mESC RNA matrix contains malformed counts")
            totals += row
            detected += row > 0
            nonzero += int(np.count_nonzero(row))
            fractional += int(np.count_nonzero(np.abs(row - np.rint(row)) > FRACTION_TOLERANCE))
    if not genes:
        raise ValueError("The mESC RNA matrix has no genes")
    return {
        "cells": len(cells), "genes": len(genes), "entries": len(cells) * len(genes),
        "nonzero_entries": nonzero, "fractional_entries": fractional,
        "fractional_tolerance": FRACTION_TOLERANCE,
        "cells_with_positive_expression": int(np.count_nonzero(detected)),
    }, dict(zip(cells, totals.tolist(), strict=True))


def qc_summary(path: Path) -> tuple[dict, dict[str, dict]]:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        required = {"cellname", "celltype", "nUMIs", "nGenes", "cleaned_contacts",
                    "nCount_accessibility", "histone_modification_type"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("The CHARM QC table is missing required columns")
        seen: set[str] = set()
        cells = {}
        for row in reader:
            name = row["cellname"]
            if not name or name in seen:
                raise ValueError("The CHARM QC table has a missing or duplicated cell name")
            seen.add(name)
            if row["celltype"] != "mESC":
                continue
            if IDENTIFIER.fullmatch(name) is None:
                raise ValueError("An mESC QC identifier has an unexpected format")
            metrics = {key: float(row[key]) for key in
                       ("nUMIs", "nGenes", "cleaned_contacts", "nCount_accessibility")}
            if any(not math.isfinite(value) or value <= 0 for value in metrics.values()):
                raise ValueError("An mESC QC row has a missing or nonpositive modality")
            cells[name] = {**metrics, "histone_modification_type": row["histone_modification_type"]}
    if not cells:
        raise ValueError("The QC table contains no mESC cells")
    return {
        "all_qc_cells": len(seen), "mesc_qc_cells": len(cells),
        "mesc_histone_labels_in_qc": dict(sorted(Counter(
            item["histone_modification_type"] for item in cells.values()
        ).items())),
    }, cells


def atac_summary(path: Path) -> tuple[dict, Counter]:
    counts: Counter = Counter()
    rows = 0
    with gzip.open(path, "rt", encoding="utf-8", newline="") as stream:
        for line in stream:
            if line.startswith("#"):
                continue
            fields = line.rstrip("\r\n").split("\t")
            if len(fields) != 7 or not fields[0] or IDENTIFIER.fullmatch(fields[6]) is None:
                raise ValueError("An ATAC fragment has invalid columns or cell identifier")
            start, end = int(fields[1]), int(fields[2])
            if start < 0 or end <= start:
                raise ValueError("An ATAC fragment has invalid genomic coordinates")
            counts[fields[6]] += 1
            rows += 1
    if not counts:
        raise ValueError("The ATAC file has no observed fragments")
    return {"fragment_rows": rows, "cells_with_fragments": len(counts)}, counts


def contact_summary(path: Path) -> tuple[dict, set[str]]:
    cells: set[str] = set()
    with tarfile.open(path, mode="r|gz") as archive:
        for member in archive:
            if member.isdir():
                continue
            entry = PurePosixPath(member.name)
            if (
                not member.isfile() or entry.is_absolute() or ".." in entry.parts
                or not entry.name.endswith(".pairs.gz")
            ):
                raise ValueError(f"Unexpected contact archive member: {member.name}")
            cell = entry.name.removesuffix(".pairs.gz")
            if IDENTIFIER.fullmatch(cell) is None or cell in cells:
                raise ValueError("The contact archive has an invalid or duplicated cell identifier")
            raw = archive.extractfile(member)
            if raw is None:
                raise ValueError("The contact archive contains an unreadable member")
            with gzip.GzipFile(fileobj=raw) as stream:
                for _ in range(1000):
                    line = stream.readline(1024 * 1024)
                    if not line:
                        raise ValueError("A contact member contains no observed contact")
                    if not line.startswith(b"#"):
                        if len(line.rstrip(b"\r\n").split(b"\t")) < 7:
                            raise ValueError("A contact member contains a malformed first record")
                        break
                else:
                    raise ValueError("A contact member has no contact after its header")
            cells.add(cell)
    if not cells:
        raise ValueError("The contact archive contains no per-cell contact files")
    return {"cells_with_contact_files": len(cells),
            "inspection": "Each inner gzip was opened through its first contact; full inner CRCs were not checked."}, cells


def audit_sources(rna: Path, qc: Path, atac: Path, contacts: Path | None) -> tuple[dict, dict]:
    paths = {"rna": rna, "qc": qc, "atac": atac}
    if contacts is not None:
        paths["contacts"] = contacts
    inputs = {name: describe_input(path) for name, path in paths.items()}
    rna_report, totals = rna_summary(rna)
    qc_report, qc_cells = qc_summary(qc)
    atac_report, atac_counts = atac_summary(atac)
    contact_report, contact_cells = contact_summary(contacts) if contacts is not None else (None, set())
    for name, path in paths.items():
        stat = path.stat()
        if (stat.st_size, stat.st_mtime_ns) != (inputs[name]["bytes"], inputs[name]["mtime_ns"]):
            raise ValueError(f"Input changed during the audit: {path.name}")
        del inputs[name]["mtime_ns"]
    rna_cells, qc_names, atac_names = set(totals), set(qc_cells), set(atac_counts)
    positive_rna = {cell for cell, total in totals.items() if total > 0}
    qc_rna = qc_names & positive_rna
    qc_rna_atac = qc_rna & atac_names
    matched = qc_rna_atac & contact_cells if contacts is not None else set()
    comparable = qc_names & atac_names
    equal_atac = sum(
        math.isclose(atac_counts[cell], qc_cells[cell]["nCount_accessibility"], abs_tol=1e-6, rel_tol=0)
        for cell in comparable
    )
    report = {
        "format": "epi-c3-charm-mesc-pairing-audit-v1",
        "state": "complete" if contacts is not None else "partial-no-contact-archive",
        "dataset": "GSE303006", "biosample": "SAMN49739633",
        "cell_line_label": "mESC", "species": "Mus musculus", "genome_assembly": "GRCm38",
        "audit_script_sha256": describe_input(Path(__file__))["sha256"],
        "inputs": inputs, "rna": rna_report, "qc": qc_report, "atac": atac_report,
        "contacts": contact_report,
        "overlap": {
            "rna_positive_and_qc": len(qc_rna), "rna_positive_qc_and_atac": len(qc_rna_atac),
            "rna_positive_qc_atac_and_contacts": len(matched) if contacts is not None else None,
            "atac_cells_in_qc": len(comparable), "atac_fragment_count_equals_qc": equal_atac,
        },
        "final_training_ready": False,
        "limitations": [
            "These are direct identifier intersections for one source-labelled mouse mESC sample, not three independently verified cell lines.",
            "The public RNA matrix uses fractional multimapping assignments; noninteger values must not be silently rounded for an integer-count model.",
            "The current shared RNA foundation uses a human gene axis; a validated mouse-specific axis or explicit cross-species mapping is required.",
            "Identifier agreement and the authors' same-cell protocol do not independently verify every physical cell or reproduce the original experiment.",
            "The mESC QC table's histone label differs from the mESC H3K27me3 sample description; the histone field was not used for pairing.",
            "Local SHA-256 values bind the examined files; no independent source-provided digest was available for these processed files.",
        ],
        "source_urls": [
            "https://doi.org/10.1038/s41586-026-10322-z",
            "https://www.ebi.ac.uk/ena/browser/api/xml/PRJNA1284811",
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=gds&id=200303006&retmode=json",
            "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE303nnn/GSE303006/suppl/GSE303006_file_format_README.txt",
            "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE303nnn/GSE303006/suppl/GSE303006_feature_README.tsv",
        ],
    }
    private = {
        "format": "epi-c3-charm-mesc-identities-private-v1",
        "rna_positive_qc_atac": sorted(qc_rna_atac),
        "rna_positive_qc_atac_contacts": sorted(matched) if contacts is not None else None,
    }
    return report, private


def write_results(outdir: Path, report: dict, private: dict) -> None:
    outdir.mkdir(mode=0o700, parents=True, exist_ok=False)
    private_path = outdir / "identities.private.json"
    descriptor = os.open(private_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(private, stream, indent=2, allow_nan=False)
        stream.write("\n")
    report["private_identity_file"] = private_path.name
    report["private_identity_sha256"] = describe_input(private_path)["sha256"]
    with (outdir / "audit.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("rna", "qc", "atac", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--contacts", type=Path, help="Omit only for an explicitly partial RNA/ATAC audit")
    args = parser.parse_args()
    report, private = audit_sources(args.rna, args.qc, args.atac, args.contacts)
    write_results(args.output, report, private)
    print(json.dumps({key: report[key] for key in ("state", "rna", "qc", "atac", "contacts", "overlap")},
                     indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
