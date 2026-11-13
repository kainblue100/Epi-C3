"""Validate and aggregate CpG observations without inventing coverage."""

from __future__ import annotations

import gzip
import io
import math
import os
import sqlite3
import tempfile
from collections import Counter
from pathlib import Path
from typing import Iterator, Protocol

from .integrity import file_sha256


class CoordinateConverter(Protocol):
    def convert_coordinate(
        self, chromosome: str, position: int, strand: str,
    ) -> list[tuple[str, int, str, int]] | None: ...


class ReferenceSequence(Protocol):
    def chroms(self) -> dict[str, int]: ...

    def sequence(self, chromosome: str, start: int, end: int) -> str | None: ...


class FastaReference:
    def __init__(self, paths: list[Path], *, expected_lengths: dict[str, int]) -> None:
        self.data = {}
        for path in paths:
            with gzip.open(path, "rt", encoding="ascii") as stream:
                header = stream.readline().strip()
                chrom = header[1:].split()[0] if header.startswith(">") else ""
                if chrom not in expected_lengths or chrom in self.data:
                    raise ValueError(f"Unexpected reference FASTA chromosome: {header}")
                sequence = "".join(line.strip() for line in stream).upper()
            if len(sequence) != expected_lengths[chrom] or set(sequence) - set("ACGTN"):
                raise ValueError(f"Unexpected reference FASTA sequence: {path}")
            self.data[chrom] = sequence

    def chroms(self) -> dict[str, int]:
        return {chrom: len(sequence) for chrom, sequence in self.data.items()}

    def sequence(self, chromosome: str, start: int, end: int) -> str:
        return self.data[chromosome][start:end]


def lifted_cpgs(
    path: Path, converter: CoordinateConverter, reference: ReferenceSequence, *,
    min_coverage: int, stats: Counter[str], source_chromosomes: set[str] | None = None,
) -> Iterator[tuple[str, int, float, int]]:
    """Yield uniquely mapped, sequence-verified canonical CpGs from an ENCODE bedMethyl."""
    if min_coverage < 1:
        raise ValueError("CpG coverage threshold must be positive")
    open_text = gzip.open if path.suffix == ".gz" else open
    lengths = reference.chroms()
    with open_text(path, "rt", encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip() or line.startswith(("#", "track ", "browser ")):
                continue
            stats["input_sites"] += 1
            fields = line.rstrip("\n").split("\t")
            if source_chromosomes is not None and fields[0] not in source_chromosomes:
                stats["outside_requested_source_chromosomes"] += 1
                continue
            if len(fields) < 11:
                raise ValueError(f"{path}:{number}: bedMethyl requires at least 11 columns")
            try:
                start, end = int(fields[1]), int(fields[2])
                coverage, percentage = int(fields[9]), float(fields[10])
            except ValueError as exc:
                raise ValueError(f"{path}:{number}: invalid bedMethyl measurement") from exc
            strand = fields[5]
            if (
                not fields[0] or start < 0 or end != start + 1 or strand not in ("+", "-")
                or coverage < 0 or not math.isfinite(percentage) or not 0 <= percentage <= 100
            ):
                raise ValueError(f"{path}:{number}: invalid CpG coordinate, strand, or fraction")
            if coverage < min_coverage:
                stats["low_coverage"] += 1
                continue
            partner = start + (1 if strand == "+" else -1)
            if partner < 0:
                stats["unmapped_or_ambiguous"] += 1
                continue
            first = converter.convert_coordinate(fields[0], start, strand)
            second = converter.convert_coordinate(fields[0], partner, strand)
            if not first or not second or len(first) != 1 or len(second) != 1:
                stats["unmapped_or_ambiguous"] += 1
                continue
            chrom, position, mapped_strand, _ = first[0]
            other_chrom, other_position, other_strand, _ = second[0]
            if (
                mapped_strand not in ("+", "-") or other_chrom != chrom
                or other_strand != mapped_strand
                or other_position != position + (1 if mapped_strand == "+" else -1)
            ):
                stats["non_contiguous"] += 1
                continue
            canonical = position if mapped_strand == "+" else other_position
            if chrom not in lengths or canonical < 0 or canonical + 2 > lengths[chrom]:
                stats["outside_reference"] += 1
                continue
            sequence = reference.sequence(chrom, canonical, canonical + 2)
            if sequence is None or sequence.upper() != "CG":
                stats["not_reference_cpg"] += 1
                continue
            stats["retained_sites"] += 1
            yield chrom, canonical, percentage / 100.0, coverage


def validated_percent_cpgs(
    path: Path, reference: ReferenceSequence, *, stats: Counter[str],
    chromosomes: set[str],
) -> Iterator[tuple[str, int, float]]:
    """Yield reference-CpG dyad coordinates from strand-specific bedGraph rows."""
    lengths = reference.chroms()
    if not chromosomes or not chromosomes <= set(lengths):
        raise ValueError("Requested CpG chromosomes must be present in the reference")
    previous: dict[str, int] = {}
    open_text = gzip.open if path.suffix == ".gz" else open
    with open_text(path, "rt", encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip() or line.startswith(("#", "track ", "browser ")):
                continue
            stats["input_rows"] += 1
            chrom, _, remainder = line.partition("\t")
            if chrom not in chromosomes:
                stats["outside_requested_chromosomes"] += 1
                continue
            fields = remainder.rstrip("\n").split("\t")
            if len(fields) != 3:
                raise ValueError(f"{path}:{number}: expected four bedGraph columns")
            try:
                start, end = int(fields[0]), int(fields[1])
                percentage = float(fields[2])
            except ValueError as exc:
                raise ValueError(f"{path}:{number}: invalid CpG percentage") from exc
            if (
                start < 0 or end != start + 1 or end > lengths[chrom]
                or not math.isfinite(percentage) or not 0 <= percentage <= 100
            ):
                raise ValueError(f"{path}:{number}: invalid CpG coordinate or percentage")
            if chrom in previous and start <= previous[chrom]:
                raise ValueError(f"{path}:{number}: duplicated or unsorted CpG position")
            previous[chrom] = start
            forward = reference.sequence(chrom, start, start + 2) if start + 2 <= lengths[chrom] else None
            if forward is not None and forward.upper() == "CG":
                canonical = start
                stats["forward_rows"] += 1
            else:
                reverse = reference.sequence(chrom, start - 1, start + 1) if start > 0 else None
                if reverse is not None and reverse.upper() == "CG":
                    canonical = start - 1
                    stats["reverse_rows"] += 1
                else:
                    stats["not_reference_cpg"] += 1
                    continue
            stats["retained_rows"] += 1
            yield chrom, canonical, percentage / 100.0


def prepare_uncovered_cpg_fractions(
    path: Path, reference: ReferenceSequence, output: Path, *, chromosomes: set[str],
) -> dict:
    """Average available strands per reference CpG; retain no read-depth claims."""
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    if not output.parent.is_dir():
        raise ValueError(f"Output directory does not exist: {output.parent}")
    stats: Counter[str] = Counter()
    with tempfile.NamedTemporaryFile(
        prefix=".cpg-pool-", suffix=".sqlite", dir=output.parent, delete=False,
    ) as stream:
        database = Path(stream.name)
    with tempfile.NamedTemporaryFile(
        prefix=".cpg-fractions-", suffix=".tmp", dir=output.parent, delete=False,
    ) as stream:
        temporary = Path(stream.name)
    try:
        with sqlite3.connect(database) as connection:
            connection.execute(
                "CREATE TABLE cpg (chrom TEXT, start INTEGER, sum REAL, n INTEGER, "
                "PRIMARY KEY(chrom, start)) WITHOUT ROWID"
            )
            batch = []
            for chrom, start, fraction in validated_percent_cpgs(
                path, reference, stats=stats, chromosomes=chromosomes,
            ):
                batch.append((chrom, start, fraction))
                if len(batch) >= 10_000:
                    connection.executemany(
                        "INSERT INTO cpg VALUES (?,?,?,1) ON CONFLICT(chrom,start) "
                        "DO UPDATE SET sum=sum+excluded.sum, n=n+1", batch,
                    )
                    batch.clear()
            if batch:
                connection.executemany(
                    "INSERT INTO cpg VALUES (?,?,?,1) ON CONFLICT(chrom,start) "
                    "DO UPDATE SET sum=sum+excluded.sum, n=n+1", batch,
                )
            connection.commit()
            counts = dict(connection.execute("SELECT n, COUNT(*) FROM cpg GROUP BY n"))
            if not counts or set(counts) - {1, 2}:
                raise ValueError("No valid CpG dyads, or more than two observations per dyad")
            observed_chromosomes = [
                row[0] for row in connection.execute("SELECT DISTINCT chrom FROM cpg ORDER BY chrom")
            ]
            with temporary.open("wb") as raw:
                with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
                    with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as stream:
                        stream.write("#chrom\tstart\tend\tmethyl_fraction\n")
                        for chrom, start, total, n in connection.execute(
                            "SELECT chrom,start,sum,n FROM cpg ORDER BY chrom,start"
                        ):
                            stream.write(f"{chrom}\t{start}\t{start + 1}\t{total / n:.8g}\n")
        digest = file_sha256(temporary)
        os.link(temporary, output)
        return {
            "unique_cpg_dyads": sum(counts.values()),
            "single_strand_dyads": counts.get(1, 0),
            "both_strands_dyads": counts.get(2, 0),
            "covered_chromosomes": observed_chromosomes,
            "source_statistics": dict(stats),
            "sha256": digest,
        }
    finally:
        temporary.unlink(missing_ok=True)
        database.unlink(missing_ok=True)
