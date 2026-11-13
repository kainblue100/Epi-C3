"""Helpers for provenance-bound Tahoe cell-identity isolation."""

from __future__ import annotations

import bisect
import hashlib
import json
from pathlib import Path

import pyarrow as pa

from c3vcell.identity import source_cell_id


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def digest_json(value) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")).hexdigest()


def check_shard(root: Path, entry: dict) -> Path:
    relative = Path(entry["path"])
    if relative.is_absolute() or ".." in relative.parts or relative.parts[:1] != ("data",):
        raise ValueError("Invalid Tahoe shard path")
    path = root / relative
    stat = path.stat()
    if stat.st_size != entry["size"] or stat.st_mtime_ns != entry["mtime_ns"]:
        raise ValueError(f"Tahoe source changed since its metadata scan: {relative}")
    return path


def row_group_may_contain(parquet, row_group: int, ordered_samples: list[str]) -> bool:
    if not ordered_samples:
        return False
    names = parquet.schema.names
    if names.count("sample") != 1:
        raise ValueError("Tahoe shard must have one scalar sample field")
    index = names.index("sample")
    column = parquet.metadata.row_group(row_group).column(index)
    field = parquet.schema_arrow.field("sample")
    if column.path_in_schema != "sample" or not (
        pa.types.is_string(field.type) or pa.types.is_large_string(field.type)
    ):
        raise ValueError("Tahoe sample field has an unexpected schema")
    stats = column.statistics
    if stats is None or not stats.has_min_max:
        return True
    lower, upper = stats.min, stats.max
    if (
        not isinstance(lower, str) or not isinstance(upper, str)
        or not lower.isascii() or not upper.isascii() or lower > upper
    ):
        return True
    position = bisect.bisect_left(ordered_samples, lower)
    return position < len(ordered_samples) and ordered_samples[position] <= upper


def identity(sample, barcode) -> str:
    if not isinstance(sample, str) or not isinstance(barcode, str):
        raise ValueError("Tahoe identity components must be strings")
    return source_cell_id("tahoe", sample, barcode)


def add_pair(pairs: dict, cell_id: str, sample: str, barcode: str) -> None:
    if identity(sample, barcode) != cell_id:
        raise ValueError("Tahoe source identity does not match its components")
    pair = (sample, barcode)
    previous = pairs.setdefault(cell_id, pair)
    if previous != pair:
        raise ValueError("Distinct Tahoe components generated the same source identity")


def grouped_barcodes(pairs: dict[str, tuple[str, str]]) -> dict[str, set[str]]:
    grouped = {}
    for cell_id, (sample, barcode) in pairs.items():
        if identity(sample, barcode) != cell_id:
            raise ValueError("Frozen Tahoe cell identity was altered")
        grouped.setdefault(sample, set()).add(barcode)
    return grouped
