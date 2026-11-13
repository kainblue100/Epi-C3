"""Audit source-provided scHiCAR RNA/DNA pairing on bounded public file prefixes."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import urllib.request
import zlib


BASE = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE305nnn/GSE305439/suppl/"
FILES = {
    "rna_barcodes": "GSE305439_RNA.barcodes.tsv.gz",
    "metadata": "GSE305439_mouse_brain_scHiCAR_1_RNA_metadata.txt.gz",
    "atac": "GSE305439_DNA.ATAC.tsv.gz",
    "contacts": "GSE305439_DNA.dedup.pairs.gz",
}
SMALL_FILE_LIMIT = 1024 * 1024
MAX_PREFIX_BYTES = 8 * 1024 * 1024
MAX_DECOMPRESSED_BYTES = 128 * 1024 * 1024
BARCODE = re.compile(r"[ACGT]{18}\Z")


def fetch_prefix(url: str, limit: int) -> bytes:
    request = urllib.request.Request(
        url,
        headers={"Range": f"bytes=0-{limit - 1}", "User-Agent": "EpiC3-source-qa/1.0"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        if response.status not in (200, 206):
            raise ValueError(f"Unexpected HTTP status: {response.status}")
        data = response.read(limit)
        if response.status == 206:
            content_range = response.headers.get("Content-Range", "")
            match = re.fullmatch(r"bytes 0-(\d+)/(\d+|\*)", content_range)
            if match is None or int(match.group(1)) + 1 != len(data):
                raise ValueError("The server did not return the requested initial byte range")
    if not data:
        raise ValueError("The source returned an empty compressed file")
    return data


def unpack_prefix(raw: bytes, limit: int, *, require_complete: bool) -> tuple[list[str], dict]:
    if not raw or len(raw) > limit:
        raise ValueError("Invalid compressed input length")
    decoder = zlib.decompressobj(16 + zlib.MAX_WBITS)
    decoded = decoder.decompress(raw, MAX_DECOMPRESSED_BYTES + 1)
    if len(decoded) > MAX_DECOMPRESSED_BYTES or decoder.unconsumed_tail:
        raise ValueError("The decompressed prefix exceeds the safety limit")
    if decoder.eof and decoder.unused_data:
        raise ValueError("Unexpected trailing data or additional gzip member")
    if require_complete and not decoder.eof:
        raise ValueError("The barcode and metadata files must be complete gzip streams")
    if not decoder.eof and len(raw) < limit:
        raise ValueError("The source returned a short, incomplete gzip prefix")
    discarded_partial_line = False
    if not decoder.eof and not decoded.endswith(b"\n"):
        last_newline = decoded.rfind(b"\n")
        if last_newline < 0:
            raise ValueError("The compressed prefix contains no complete line")
        decoded = decoded[: last_newline + 1]
        discarded_partial_line = True
    lines = decoded.decode("utf-8").splitlines()
    return lines, {
        "compressed_bytes": len(raw),
        "compressed_sha256": hashlib.sha256(raw).hexdigest(),
        "gzip_stream_complete": decoder.eof,
        "discarded_incomplete_final_line": discarded_partial_line,
    }


def read_pairing_map(barcode_lines: list[str], metadata_lines: list[str]) -> tuple[set[str], dict[str, str], dict]:
    rna_ids = set(barcode_lines)
    if (
        not barcode_lines or len(rna_ids) != len(barcode_lines)
        or any(BARCODE.fullmatch(item) is None for item in barcode_lines)
    ):
        raise ValueError("The RNA axis must contain unique 18-base source barcodes")
    reader = csv.DictReader(metadata_lines, delimiter="\t")
    fields = reader.fieldnames or []
    if len(fields) != len(set(fields)) or not {"RNAbarcode", "DNAbarcode"}.issubset(fields):
        raise ValueError("The source metadata must provide distinct RNAbarcode and DNAbarcode columns")
    dna_to_rna: dict[str, str] = {}
    seen_rna: set[str] = set()
    same = 0
    for row in reader:
        rna, dna = row.get("RNAbarcode"), row.get("DNAbarcode")
        if (
            None in row or not isinstance(rna, str) or not isinstance(dna, str)
            or BARCODE.fullmatch(rna) is None or BARCODE.fullmatch(dna) is None
        ):
            raise ValueError("A source metadata row has an invalid barcode or column count")
        if rna in seen_rna or dna in dna_to_rna:
            raise ValueError("The source RNA-to-DNA map must be one-to-one")
        seen_rna.add(rna)
        dna_to_rna[dna] = rna
        same += rna == dna
    if seen_rna != rna_ids:
        raise ValueError("The metadata RNA barcodes do not match the ordered RNA axis")
    dna_also_rna = sum(dna in rna_ids for dna in dna_to_rna)
    return rna_ids, dna_to_rna, {
        "rna_axis_cells": len(rna_ids),
        "mapped_rna_cells": len(seen_rna),
        "mapped_dna_cells": len(dna_to_rna),
        "same_barcode_in_source_paired_row": same,
        "dna_barcodes_also_on_rna_axis": dna_also_rna,
        "dna_barcodes_equal_to_another_cells_rna_barcode": dna_also_rna - same,
    }


def summarize_assay(
    lines: list[str], *, columns: int, barcode_column: int,
    rna_ids: set[str], dna_to_rna: dict[str, str],
) -> tuple[dict, set[str]]:
    total = mapped = direct = conflicts = direct_unmapped = 0
    mapped_ids: set[str] = set()
    for line in lines:
        if not line or line.startswith("#"):
            continue
        fields = line.split("\t")
        if len(fields) != columns or BARCODE.fullmatch(fields[barcode_column]) is None:
            raise ValueError("A complete assay-prefix row has an invalid schema or DNA barcode")
        dna = fields[barcode_column]
        total += 1
        if dna in dna_to_rna:
            mapped += 1
            mapped_ids.add(dna)
        if dna in rna_ids:
            direct += 1
            if dna not in dna_to_rna:
                direct_unmapped += 1
            elif dna_to_rna[dna] != dna:
                conflicts += 1
    if not total:
        raise ValueError("The assay prefix has no complete data rows")
    return {
        "complete_rows_examined": total,
        "rows_matched_by_source_dna_map": mapped,
        "rows_not_in_source_dna_map": total - mapped,
        "unique_mapped_dna_barcodes": len(mapped_ids),
        "rows_matching_rna_axis_directly": direct,
        "direct_rna_matches_conflicting_with_source_map": conflicts,
        "direct_rna_matches_without_source_dna_mapping": direct_unmapped,
    }, mapped_ids


def audit(fetch=fetch_prefix, *, prefix_bytes: int = 1024 * 1024) -> dict:
    if not 1 <= prefix_bytes <= MAX_PREFIX_BYTES:
        raise ValueError("The compressed assay prefix must be between 1 byte and 8 MiB")
    lines: dict[str, list[str]] = {}
    inputs = {}
    for label, name in FILES.items():
        limit = SMALL_FILE_LIMIT if label in ("rna_barcodes", "metadata") else prefix_bytes
        raw = fetch(BASE + name, limit)
        lines[label], inputs[label] = unpack_prefix(
            raw, limit, require_complete=label in ("rna_barcodes", "metadata"),
        )
        inputs[label]["name"] = name
    rna_ids, dna_to_rna, pairing = read_pairing_map(lines["rna_barcodes"], lines["metadata"])
    atac, atac_ids = summarize_assay(
        lines["atac"], columns=6, barcode_column=3, rna_ids=rna_ids, dna_to_rna=dna_to_rna,
    )
    contacts, contact_ids = summarize_assay(
        lines["contacts"], columns=8, barcode_column=0, rna_ids=rna_ids, dna_to_rna=dna_to_rna,
    )
    return {
        "format": "epi-c3-schicar-source-prefix-qa-v1",
        "status": "complete-prefix-qa",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "accession": "GSE305439",
            "species": "Mus musculus",
            "sample": "mouse frontal cortex tissue",
            "assembly_as_reported": "mm10",
            "geo": "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE305439",
            "sample_method": "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM9177145",
            "author_data_description": (
                "https://github.com/DiaoLab/scHiCAR/blob/"
                "2ece10efca321d73e21046e9c8426c0b2d3c1f3e/GEO_datasets.md"
            ),
        },
        "compressed_assay_prefix_limit": prefix_bytes,
        "inputs": inputs,
        "source_pairing_map": pairing,
        "assay_prefixes": {"atac": atac, "contacts": contacts},
        "mapped_dna_cells_seen_in_both_assay_prefixes": len(atac_ids & contact_ids),
        "final_human_cell_line_training_ready": False,
        "model_result": False,
        "limitations": [
            "This is mouse tissue, not a human cell line or a three-cell-line training collection.",
            "Only the initial compressed assay prefixes were examined; they are not random or representative samples.",
            "The RNA expression matrix was not read, and a barcode match does not establish a positive RNA count.",
            "Partial gzip streams have no verified full-file CRC; hashes bind only the bytes actually read.",
            "The source map and protocol support the declared pairing but do not independently verify each physical cell.",
            "Coordinates, assay quality, data-use permissions, cross-source leakage and model performance were not audited.",
        ],
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New aggregate JSON output file")
    parser.add_argument("--prefix-bytes", type=int, default=1024 * 1024)
    args = parser.parse_args()
    report = audit(prefix_bytes=args.prefix_bytes)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print("Completed source-prefix QA; this is not a trained model or a performance result.")


if __name__ == "__main__":
    main()
