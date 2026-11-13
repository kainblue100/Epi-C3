"""Audit every CHARM contact record without publishing cell or read identities."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import tarfile


CHUNK_SIZE = 1024 * 1024
MAX_LINE_BYTES = 1024 * 1024
MEMBER = re.compile(r"P[0-9]+\.pairs\.gz\Z")
COLUMNS = [
    b"readID", b"chr1", b"pos1", b"chr2", b"pos2",
    b"strand1", b"strand2", b"phase0", b"phase1",
]
METRICS = (
    "records",
    "records_unchecked_due_to_schema",
    "malformed_records",
    "empty_read_ids",
    "unknown_chromosome_endpoints",
    "undeclared_chromosome_endpoints",
    "unmapped_endpoints",
    "noninteger_position_endpoints",
    "out_of_bounds_endpoints",
    "missing_strand_endpoints",
    "invalid_strand_endpoints",
    "invalid_phase_endpoints",
    "headers_after_data",
)


class HashingReader:
    def __init__(self, stream):
        self.stream = stream
        self.digest = hashlib.sha256()

    def read(self, size=-1):
        data = self.stream.read(size)
        self.digest.update(data)
        return data


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_reference(path: Path, expected_sha256: str) -> dict[bytes, int]:
    if digest_file(path) != expected_sha256:
        raise ValueError("Chromosome reference SHA-256 differs from the frozen value")
    reference: dict[bytes, int] = {}
    with path.open("rb") as stream:
        for line in stream:
            columns = line.split()
            if len(columns) != 2 or columns[0] in reference:
                raise ValueError("Invalid or duplicate chromosome reference entry")
            try:
                length = int(columns[1])
            except ValueError as exc:
                raise ValueError("Invalid chromosome reference length") from exc
            if length <= 0:
                raise ValueError("Invalid chromosome reference length")
            reference[columns[0]] = length
    if not reference:
        raise ValueError("Chromosome reference is empty")
    return reference


def load_pairing(audit_path: Path, identity_path: Path) -> tuple[dict, set[str], str]:
    audit_bytes = audit_path.read_bytes()
    audit = json.loads(audit_bytes)
    if audit.get("state") != "complete" or audit.get("genome_assembly") != "GRCm38":
        raise ValueError("Expected a complete GRCm38 pairing audit")
    if digest_file(identity_path) != audit.get("private_identity_sha256"):
        raise ValueError("Private identity map differs from the frozen pairing audit")
    with identity_path.open("r", encoding="utf-8") as stream:
        identities = json.load(stream)
    matched = identities.get("rna_positive_qc_atac_contacts")
    if (
        not isinstance(matched, list)
        or any(not isinstance(value, str) or re.fullmatch(r"P[0-9]+", value) is None for value in matched)
        or len(set(matched)) != len(matched)
        or len(matched) != audit["overlap"]["rna_positive_qc_atac_and_contacts"]
    ):
        raise ValueError("Private identity map has invalid or inconsistent matched cells")
    return audit, set(matched), hashlib.sha256(audit_bytes).hexdigest()


def audit_stream(stream, reference: dict[bytes, int]) -> tuple[dict, dict]:
    metrics = dict.fromkeys(METRICS, 0)
    chromosomes: dict[bytes, int] = {}
    first_line = True
    columns = None
    header_problems = 0
    saw_data = False
    schema_valid = False
    while True:
        line = stream.readline(MAX_LINE_BYTES + 1)
        if not line:
            break
        if len(line) > MAX_LINE_BYTES:
            raise ValueError("A contact header or record exceeds the maximum line length")
        line = line.rstrip(b"\r\n")
        if first_line:
            first_line = False
            if line != b"## pairs format v1.0":
                header_problems += 1
        if line.startswith(b"#"):
            if saw_data:
                metrics["headers_after_data"] += 1
                continue
            key, separator, value = line.partition(b":")
            if separator and key in (b"#chromosome", b"#chromsize"):
                fields = value.split()
                if len(fields) != 2:
                    header_problems += 1
                    continue
                try:
                    length = int(fields[1])
                except ValueError:
                    header_problems += 1
                    continue
                if fields[0] in chromosomes:
                    header_problems += 1
                else:
                    chromosomes[fields[0]] = length
                    if reference.get(fields[0]) != length:
                        header_problems += 1
            elif separator and key == b"#columns":
                if columns is not None:
                    header_problems += 1
                columns = value.split()
            continue
        if not saw_data:
            saw_data = True
            if not chromosomes or columns != COLUMNS:
                header_problems += 1
            schema_valid = bool(chromosomes) and columns == COLUMNS and header_problems == 0
        metrics["records"] += 1
        if not schema_valid:
            metrics["records_unchecked_due_to_schema"] += 1
            continue
        fields = line.split(b"\t")
        if len(fields) != len(COLUMNS):
            metrics["malformed_records"] += 1
            continue
        if not fields[0]:
            metrics["empty_read_ids"] += 1
        for chrom, position in ((fields[1], fields[2]), (fields[3], fields[4])):
            if chrom == b"!" and position == b"0":
                metrics["unmapped_endpoints"] += 1
                continue
            if chrom not in reference:
                metrics["unknown_chromosome_endpoints"] += 1
            if chrom not in chromosomes:
                metrics["undeclared_chromosome_endpoints"] += 1
            try:
                coordinate = int(position)
            except ValueError:
                metrics["noninteger_position_endpoints"] += 1
                continue
            if chrom in reference and not 1 <= coordinate <= reference[chrom]:
                metrics["out_of_bounds_endpoints"] += 1
        for strand in (fields[5], fields[6]):
            if strand == b".":
                metrics["missing_strand_endpoints"] += 1
            elif strand not in (b"+", b"-"):
                metrics["invalid_strand_endpoints"] += 1
        for phase in (fields[7], fields[8]):
            if phase not in (b".", b"0", b"1"):
                metrics["invalid_phase_endpoints"] += 1
    if first_line or not saw_data:
        header_problems += 1
    return metrics, {
        "header_problems": header_problems,
        "declared_chromosomes": chromosomes,
        "has_records": saw_data,
    }


def add_metrics(total: dict, part: dict) -> None:
    for key in METRICS:
        total[key] += part[key]


def audit_archive(
    path: Path,
    pairing: dict,
    matched: set[str],
    reference: dict[bytes, int],
    *,
    progress_every: int = 25,
) -> dict:
    expected = pairing["inputs"]["contacts"]
    expected_members = pairing["contacts"]["cells_with_contact_files"]
    if path.name != expected["name"] or not isinstance(expected_members, int) or expected_members <= 0:
        raise ValueError("Frozen pairing audit does not describe the contact archive")
    seen: set[str] = set()
    matched_seen: set[str] = set()
    totals = dict.fromkeys(METRICS, 0)
    matched_totals = dict.fromkeys(METRICS, 0)
    header_problem_files = 0
    matched_header_problem_files = 0
    empty_files = 0
    declared_sets: set[tuple[tuple[bytes, int], ...]] = set()
    with path.open("rb") as source:
        before = os.fstat(source.fileno())
        if before.st_size != expected["bytes"]:
            raise ValueError("Archive size differs from the frozen pairing audit")
        reader = HashingReader(source)
        with tarfile.open(fileobj=reader, mode="r|gz") as archive:
            for member in archive:
                if member.isdir():
                    continue
                entry = PurePosixPath(member.name)
                if (
                    not member.isfile() or entry.is_absolute() or ".." in entry.parts
                    or MEMBER.fullmatch(entry.name) is None or entry.name in seen
                ):
                    raise ValueError("Unexpected or duplicate contact archive member")
                raw = archive.extractfile(member)
                if raw is None:
                    raise ValueError("Unreadable contact archive member")
                with gzip.GzipFile(fileobj=raw) as stream:
                    metrics, header = audit_stream(stream, reference)
                identity = entry.name.removesuffix(".pairs.gz")
                seen.add(entry.name)
                add_metrics(totals, metrics)
                header_problem_files += bool(header["header_problems"])
                empty_files += not header["has_records"]
                declared_sets.add(tuple(sorted(header["declared_chromosomes"].items())))
                if identity in matched:
                    matched_seen.add(identity)
                    add_metrics(matched_totals, metrics)
                    matched_header_problem_files += bool(header["header_problems"])
                if progress_every and len(seen) % progress_every == 0:
                    print(
                        f"Audited {len(seen)} of {expected_members} contact streams; "
                        f"{totals['records']} records read",
                        flush=True,
                    )
        for _ in iter(lambda: reader.read(CHUNK_SIZE), b""):
            pass
        after = os.fstat(source.fileno())
    current = path.stat()
    identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    if identity_before != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
        raise ValueError("The archive changed while it was being audited")
    if identity_before != (current.st_dev, current.st_ino, current.st_size, current.st_mtime_ns):
        raise ValueError("The archive path changed while it was being audited")
    archive_sha256 = reader.digest.hexdigest()
    if archive_sha256 != expected["sha256"]:
        raise ValueError("The archive SHA-256 differs from the frozen pairing audit")
    if len(seen) != expected_members or matched_seen != matched:
        raise ValueError("Contact members or matched identities differ from the frozen pairing audit")
    issue_fields = [key for key in METRICS if key != "records"]
    clean = (
        not header_problem_files and not empty_files
        and not any(totals[key] for key in issue_fields)
    )
    return {
        "format": "epi-c3-charm-contact-coordinate-audit-v1",
        "status": "complete",
        "mapped_record_check": "pass" if clean else "findings",
        "archive_name": path.name,
        "archive_bytes": before.st_size,
        "archive_sha256": archive_sha256,
        "contact_files": len(seen),
        "matched_contact_files": len(matched_seen),
        "files_with_header_problems": header_problem_files,
        "matched_files_with_header_problems": matched_header_problem_files,
        "files_without_records": empty_files,
        "distinct_declared_chromosome_sets": len(declared_sets),
        "all_contacts": totals,
        "matched_contacts": matched_totals,
        "coordinate_convention": "1-based closed point positions",
        "reference_assembly": "GRCm38/mm10",
        "limitations": [
            "The 1-based coordinate convention follows the published pairs format; "
            "a file-specific producer implementation has not been independently established.",
            "Chromosome lengths consistent with mm10 do not prove reference-sequence identity.",
            "These checks do not prove biological pairing, contact uniqueness, mapping quality, "
            "row sorting, assay completeness, or downstream model validity.",
            "Local input hashes are not independent source-provided checksums.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--pairing-audit", required=True, type=Path)
    parser.add_argument("--private-identities", required=True, type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--reference-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    reference = load_reference(args.reference, args.reference_sha256)
    pairing, matched, pairing_sha256 = load_pairing(args.pairing_audit, args.private_identities)
    args.output.mkdir(mode=0o700, parents=True, exist_ok=False)
    report = audit_archive(args.archive, pairing, matched, reference)
    report["pairing_audit_sha256"] = pairing_sha256
    report["reference_sha256"] = args.reference_sha256
    report["reference_url"] = "https://hgdownload.soe.ucsc.edu/goldenPath/mm10/bigZips/mm10.chrom.sizes"
    report["script_sha256"] = digest_file(Path(__file__))
    destination = args.output / "audit.json"
    descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(report, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
