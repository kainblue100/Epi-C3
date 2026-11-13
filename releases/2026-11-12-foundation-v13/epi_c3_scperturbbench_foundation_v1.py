"""Protected, resumable count streams for scPerturbBench control cells.

The frozen plans protect known within-file evaluation rows. They do not prove
that cells or donors are disjoint from other datasets.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Sequence

import h5py
import numpy as np
import scipy.sparse as sp

from epi_c3_foundation_sources_v1 import H5CsrSource, align_counts
from epi_c3_foundation_stage1_sources_v1 import windows
from epi_c3_shared_foundation_v1 import file_digest, gene_axis_digest


PRIVATE_FORMAT = "epi-c3-scperturbbench-foundation-private-v1"
PUBLIC_FORMAT = "epi-c3-scperturbbench-foundation-audit-v1"


def categorical(group: h5py.Group, name: str, length: int) -> np.ndarray:
    node = group[name]
    if not isinstance(node, h5py.Group) or set(node.keys()) != {"categories", "codes"}:
        raise ValueError(f"{name} must use categorical HDF5 encoding")
    categories = node["categories"].asstr()[:]
    codes = node["codes"][:]
    if (
        len(set(categories.tolist())) != len(categories)
        or codes.shape != (length,)
        or not np.issubdtype(codes.dtype, np.integer)
        or np.any(codes < 0)
        or np.any(codes >= len(categories))
        or any(not item for item in categories)
    ):
        raise ValueError(f"{name} contains invalid categorical metadata")
    return categories[codes]


def resolve_mapping(
    genes: Sequence[str],
    axis: Sequence[str],
    symbol_ids: dict[str, set[str]],
    direct_ids: Sequence[str] | None = None,
) -> tuple[np.ndarray, dict]:
    if (
        not genes
        or any(not isinstance(value, str) or not value.strip() for value in genes)
        or len(set(genes)) != len(genes)
        or direct_ids is not None and len(direct_ids) != len(genes)
    ):
        raise ValueError("Source gene annotations must be unique, complete and aligned")
    gene_axis_digest(axis)
    lookup = {name: index for index, name in enumerate(axis)}
    mapping = np.full(len(genes), -1, dtype=np.int64)
    status = []
    for index, symbol in enumerate(genes):
        if direct_ids is not None:
            identifier = direct_ids[index]
            if not isinstance(identifier, str) or not identifier:
                raise ValueError("Direct gene identifiers cannot be missing")
            if identifier in lookup:
                mapping[index] = lookup[identifier]
                status.append("exact_id")
            else:
                status.append("direct_id_not_in_axis")
            continue
        candidates = symbol_ids.get(symbol, set())
        if len(candidates) > 1:
            status.append("ambiguous_symbol")
        elif len(candidates) == 1 and next(iter(candidates)) in lookup:
            mapping[index] = lookup[next(iter(candidates))]
            status.append("unique_symbol")
        elif not candidates and f"HUMAN_SYMBOL:{symbol}" in lookup:
            mapping[index] = lookup[f"HUMAN_SYMBOL:{symbol}"]
            status.append("exact_symbol_namespace")
        else:
            status.append("not_in_axis")
    duplicates = {value for value, count in Counter(mapping[mapping >= 0]).items() if count > 1}
    for index, value in enumerate(mapping):
        if value in duplicates:
            mapping[index] = -1
            status[index] = "mapping_collision"
    mapped = mapping[mapping >= 0]
    if not len(mapped) or len(set(mapped.tolist())) != len(mapped):
        raise ValueError("No unambiguous one-to-one feature mapping is available")
    return mapping, {
        "source_features": len(genes),
        "mapped_features": len(mapped),
        "status_counts": dict(sorted(Counter(status).items())),
        "unmapped_features": [gene for gene, value in zip(genes, mapping, strict=True) if value < 0],
    }


def audit_sparse_counts(group: h5py.Group, mapping: np.ndarray, *, chunk_size: int = 1_000_000) -> tuple[dict, np.ndarray]:
    encoding = group.attrs.get("encoding-type")
    if encoding not in ("csr_matrix", "csc_matrix") or chunk_size < 1:
        raise ValueError("Expected a supported sparse count matrix")
    shape = tuple(map(int, group.attrs["shape"]))
    if len(shape) != 2 or min(shape) < 1 or len(mapping) != shape[1]:
        raise ValueError("Sparse count shape and source gene mapping differ")
    data, indices, raw_indptr = group["data"], group["indices"], group["indptr"][:]
    major, minor = shape if encoding == "csr_matrix" else shape[::-1]
    if (
        not np.issubdtype(indices.dtype, np.integer)
        or not np.issubdtype(raw_indptr.dtype, np.integer)
        or len(raw_indptr) != major + 1
        or raw_indptr[0] != 0
        or np.any(raw_indptr > len(data))
        or int(raw_indptr[-1]) != len(data)
        or len(indices) != len(data)
    ):
        raise ValueError("Sparse matrix pointers or indices are invalid")
    indptr = raw_indptr.astype(np.int64)
    if np.any(np.diff(indptr) < 0):
        raise ValueError("Sparse matrix pointers are not monotonic")
    total = np.zeros(shape[0], dtype=np.float64)
    mapped = np.zeros(shape[0], dtype=np.float64)
    previous = None
    explicit_zeros = 0
    nonascending_vectors: set[int] = set()
    for start in range(0, len(data), chunk_size):
        stop = min(start + chunk_size, len(data))
        values = data[start:stop]
        index = indices[start:stop].astype(np.int64)
        if (
            not np.isfinite(values).all()
            or np.any(values < 0)
            or np.any(values != np.floor(values))
            or np.any(values > 2**24)
            or np.any(index < 0)
            or np.any(index >= minor)
        ):
            raise ValueError("Count values or sparse indices are invalid")
        if previous is not None and not np.any(indptr == start) and previous >= index[0]:
            nonascending_vectors.add(int(np.searchsorted(indptr[1:], start, side="right")))
        if len(index) > 1:
            invalid = np.diff(index) <= 0
            boundaries = indptr[(indptr > start) & (indptr < stop)] - start - 1
            invalid[boundaries.astype(np.int64)] = False
            locations = np.flatnonzero(invalid) + start
            nonascending_vectors.update(np.searchsorted(indptr[1:], locations, side="right").tolist())
        if len(index):
            previous = int(index[-1])
        positions = np.arange(start, stop, dtype=np.int64)
        major_indices = np.searchsorted(indptr[1:], positions, side="right")
        rows, columns = (major_indices, index) if encoding == "csr_matrix" else (index, major_indices)
        total += np.bincount(rows, weights=values, minlength=shape[0])
        keep = mapping[columns] >= 0
        mapped += np.bincount(rows[keep], weights=values[keep], minlength=shape[0])
        explicit_zeros += int(np.count_nonzero(values == 0))
    for vector in sorted(nonascending_vectors):
        entries = indices[int(indptr[vector]):int(indptr[vector + 1])]
        if len(np.unique(entries)) != len(entries):
            raise ValueError("Sparse indices are duplicated within a row or column")
    if not np.isfinite(total).all() or not np.isfinite(mapped).all() or np.any(mapped > total):
        raise ValueError("Count totals are invalid")
    total_sum, mapped_sum = float(total.sum()), float(mapped.sum())
    if total_sum <= 0 or mapped_sum <= 0:
        raise ValueError("The source has no usable measured counts")
    return {
        "encoding": encoding,
        "shape": list(shape),
        "stored_entries": len(data),
        "explicit_zero_entries": explicit_zeros,
        "unsorted_major_vectors": len(nonascending_vectors),
        "numeric_dtype": str(data.dtype),
        "empty_total_rows": int(np.count_nonzero(total == 0)),
        "empty_mapped_rows": int(np.count_nonzero(mapped == 0)),
        "total_counts": total_sum,
        "mapped_counts": mapped_sum,
        "mapped_count_fraction": mapped_sum / total_sum,
        "numeric_integrity": "nonnegative finite integer values, bounded and unique sparse indices",
    }, mapped


def partition_controls(
    source: str,
    conditions: np.ndarray,
    cell_types: np.ndarray,
    positive_mapped: np.ndarray,
    protected: set[int],
    *,
    donors: np.ndarray | None = None,
    validation_size: int = 32,
    seed: int = 20261117,
) -> tuple[dict, dict]:
    count = len(conditions)
    if (
        source not in ("KHP", "CMO")
        or count < 3
        or len(cell_types) != count
        or positive_mapped.shape != (count,)
        or positive_mapped.dtype != bool
        or validation_size < 1
        or any(type(row) is not int or row < 0 or row >= count for row in protected)
        or any(not isinstance(value, str) or not value for value in conditions)
        or any(not isinstance(value, str) or not value for value in cell_types)
    ):
        raise ValueError("Invalid source metadata or protected row indices")
    controls = conditions == "ctrl"
    if not np.any(controls):
        raise ValueError("The source has no explicitly labelled control cells")
    excluded = ~controls | ~positive_mapped
    withheld_donor = None
    if source == "KHP":
        if any(cell_types[row] != "CD8T" for row in protected):
            raise ValueError("Frozen KHP target evaluation rows do not match CD8T metadata")
        excluded |= cell_types == "CD8T"
    else:
        if donors is None or len(donors) != count or any(not isinstance(value, str) or not value for value in donors):
            raise ValueError("CMO requires complete donor metadata")
        choices = sorted(set(donors.tolist()))
        if len(choices) < 2:
            raise ValueError("CMO requires at least two donors")
        withheld_donor = min(
            choices,
            key=lambda value: hashlib.sha256(f"cmo-donor\0{seed}\0{value}".encode()).digest(),
        )
        excluded |= donors == withheld_donor
    if protected:
        excluded[np.fromiter(protected, dtype=np.int64)] = True
    eligible = np.flatnonzero(~excluded)
    if len(eligible) < validation_size + 2:
        raise ValueError("Insufficient independent candidate controls for train and validation")
    ranked = sorted(
        eligible.tolist(),
        key=lambda row: hashlib.sha256(f"{source}\0{seed}\0{row}".encode()).digest(),
    )
    validation = np.asarray(sorted(ranked[:validation_size]), dtype=np.int64)
    training = np.asarray(sorted(ranked[validation_size:]), dtype=np.int64)
    if set(training).intersection(validation) or set(training).intersection(protected):
        raise ValueError("Protected, training and validation rows overlap")
    result = {"training_rows": training.tolist(), "validation_rows": validation.tolist()}
    summary = {
        "total_rows": count,
        "control_rows": int(np.count_nonzero(controls)),
        "noncontrol_rows_excluded": int(np.count_nonzero(~controls)),
        "empty_mapped_rows_excluded": int(np.count_nonzero(~positive_mapped)),
        "known_protected_rows": len(protected),
        "training_controls": len(training),
        "validation_controls": len(validation),
        "other_excluded_rows": count - len(training) - len(validation),
        "target_cd8_rows_excluded": int(np.count_nonzero(cell_types == "CD8T")) if source == "KHP" else None,
        "heldout_donor_rows_excluded": int(np.count_nonzero(donors == withheld_donor)) if source == "CMO" else None,
    }
    return result, summary


class BenchmarkControlStream:
    """Read frozen candidate control rows; caller must resolve cross-source isolation."""

    def __init__(
        self,
        source_path: Path,
        public_manifest: Path,
        private_manifest: Path,
        source: str,
        axis: Sequence[str],
        *,
        batch_size: int = 48,
        seed: int = 20261117,
    ) -> None:
        report = json.loads(public_manifest.read_text(encoding="utf-8"))
        if (
            report.get("format") != PUBLIC_FORMAT
            or report.get("state") != "complete"
            or report.get("private_sha256") != file_digest(private_manifest)
            or report.get("gene_axis_sha256") != gene_axis_digest(axis)
            or report.get("source_module_sha256") != file_digest(Path(__file__))
        ):
            raise ValueError("The source plan or shared gene axis does not match the audit")
        private = json.loads(private_manifest.read_text(encoding="utf-8"))
        if (
            private.get("format") != PRIVATE_FORMAT
            or private.get("gene_axis_sha256") != report["gene_axis_sha256"]
            or source not in private["sources"]
            or source not in report["sources"]
        ):
            raise ValueError("Unexpected private source plan")
        plan = private["sources"][source]
        if plan["source_sha256"] != report["sources"][source]["source_sha256"]:
            raise ValueError("Public and private benchmark source hashes differ")
        before = source_path.stat()
        if file_digest(source_path) != plan["source_sha256"]:
            raise ValueError("The benchmark source differs from the frozen source")
        after = source_path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError("The source changed while its hash was checked")
        self.path = source_path
        self.file_stat = (after.st_size, after.st_mtime_ns)
        self.rows = np.asarray(plan["training_rows"], dtype=np.int64)
        validation = np.asarray(plan["validation_rows"], dtype=np.int64)
        self.mapping = np.asarray(plan["mapping"], dtype=np.int64)
        self.gene_count = len(axis)
        shape = tuple(plan["shape"])
        if (
            len(shape) != 2 or self.mapping.shape != (shape[1],)
            or np.any(self.mapping < -1) or np.any(self.mapping >= self.gene_count)
            or len(set(self.mapping[self.mapping >= 0].tolist())) != np.count_nonzero(self.mapping >= 0)
            or len(self.rows) < 2 or len(validation) < 1
            or np.any(self.rows < 0) or np.any(self.rows >= shape[0])
            or np.any(validation < 0) or np.any(validation >= shape[0])
            or np.any(np.diff(self.rows) <= 0) or np.any(np.diff(validation) <= 0)
            or np.intersect1d(self.rows, validation).size
        ):
            raise ValueError("The frozen source rows or mapping are invalid")
        self.keep = self.mapping >= 0
        self.measured = np.zeros(self.gene_count, dtype=bool)
        self.measured[self.mapping[self.keep]] = True
        self.expected_cells = len(self.rows)
        self.intervals = windows(self.expected_cells, batch_size, seed)
        self.position = 0
        self.file = None
        self.csr = None
        self.memory = None
        try:
            if plan["encoding"] == "csr_matrix":
                self.csr = H5CsrSource(source_path, "layers/counts")
                if (self.csr.n_rows, self.csr.n_genes) != shape:
                    raise ValueError("CSR matrix shape differs from the frozen source")
            elif plan["encoding"] == "csc_matrix":
                self.file = h5py.File(source_path, "r")
                node = self.file["layers/counts"]
                if node.attrs.get("encoding-type") != "csc_matrix" or tuple(node.attrs["shape"]) != shape:
                    raise ValueError("CSC matrix differs from the frozen source")
                if len(node["data"]) > 10_000_000:
                    raise ValueError("CSC source exceeds the bounded in-memory conversion limit")
                self.memory = sp.csc_matrix(
                    (node["data"][:], node["indices"][:], node["indptr"][:]), shape=shape
                ).tocsr()
            else:
                raise ValueError("Unknown frozen sparse encoding")
            self.validation = self._read(validation)
        except Exception:
            self.close()
            raise

    def _read(self, selected: np.ndarray) -> sp.csr_matrix:
        stat = self.path.stat()
        if (stat.st_size, stat.st_mtime_ns) != self.file_stat:
            raise ValueError("The benchmark source changed during streaming")
        if self.csr is not None:
            left, right = int(selected[0]), int(selected[-1]) + 1
            local = self.csr.counts(left, right)[selected - left]
        else:
            local = self.memory[selected]
        return align_counts(local[:, self.keep], self.mapping[self.keep], self.gene_count)

    def next_batch(self) -> sp.csr_matrix | None:
        if self.position >= len(self.intervals):
            return None
        left, right = self.intervals[self.position]
        matrix = self._read(self.rows[left:right])
        self.position += 1
        return matrix

    def cursor(self) -> dict[str, int]:
        return {"position": self.position}

    def set_cursor(self, cursor: dict[str, int]) -> None:
        position = cursor.get("position", -1)
        if type(position) is not int or not 0 <= position <= len(self.intervals):
            raise ValueError("Invalid benchmark source recovery cursor")
        self.position = position

    def close(self) -> None:
        if self.csr is not None:
            self.csr.close()
            self.csr = None
        if self.file is not None:
            self.file.close()
            self.file = None
