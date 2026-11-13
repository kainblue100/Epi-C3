"""Strict human gene-axis alignment and bounded CSR reads for RNA pretraining."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Sequence

import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp


@dataclass(frozen=True)
class GeneAlignment:
    genes: tuple[str, ...]
    source_to_global: dict[str, np.ndarray]
    measured: dict[str, np.ndarray]
    notes: dict[str, int]


def h5_text(node: h5py.Dataset | h5py.Group) -> list[str]:
    if isinstance(node, h5py.Dataset):
        return list(node.asstr()[:])
    if set(node.keys()) != {"categories", "codes"}:
        raise ValueError(f"Unsupported HDF5 text encoding: {node.name}")
    categories = node["categories"].asstr()[:]
    codes = node["codes"][:]
    if np.any(codes < 0) or np.any(codes >= len(categories)):
        raise ValueError(f"Missing categorical identities: {node.name}")
    return [str(categories[code]) for code in codes]


def align_human_genes(
    tabula_ids: Sequence[str],
    tabula_symbols: Sequence[str],
    tahoe: pd.DataFrame,
    orion: pd.DataFrame,
    arc_symbols: Sequence[str],
) -> GeneAlignment:
    if len(tabula_ids) != len(tabula_symbols) or len(tabula_ids) == 0 or len(arc_symbols) == 0:
        raise ValueError("Tabula and Arc gene annotations must be nonempty and aligned")
    if len(set(tabula_ids)) != len(tabula_ids) or len(set(arc_symbols)) != len(arc_symbols):
        raise ValueError("Tabula gene IDs and Arc gene symbols must be unique")
    for frame, symbol_column in ((tahoe, "gene_symbol"), (orion, "gene_name")):
        if not {"ensembl_id", symbol_column}.issubset(frame.columns) or frame.empty:
            raise ValueError("Source gene metadata lacks required annotation columns")
        if frame["ensembl_id"].isna().any() or frame["ensembl_id"].duplicated().any():
            raise ValueError("Source Ensembl identifiers must be complete and unique")
    datasets = {
        "tabula": list(tabula_ids),
        "tahoe": tahoe["ensembl_id"].astype(str).tolist(),
        "orion": orion["ensembl_id"].astype(str).tolist(),
    }
    if any(re.fullmatch(r"ENSG[0-9]+", identifier) is None for ids in datasets.values() for identifier in ids):
        raise ValueError("The human source axis contains a non-human or non-Ensembl identifier")
    symbols_to_ids: dict[str, set[str]] = defaultdict(set)
    for symbol, identifier in zip(tabula_symbols, tabula_ids, strict=True):
        if symbol:
            symbols_to_ids[str(symbol)].add(identifier)
    for frame, column in ((tahoe, "gene_symbol"), (orion, "gene_name")):
        for symbol, identifier in zip(frame[column], frame["ensembl_id"], strict=True):
            if isinstance(symbol, str) and symbol:
                symbols_to_ids[symbol].add(str(identifier))
    resolved = {
        symbol: next(iter(ids)) for symbol, ids in symbols_to_ids.items() if len(ids) == 1
    }
    counts = Counter(resolved[symbol] for symbol in arc_symbols if symbol in resolved)
    arc_ids = [
        resolved[symbol] if symbol in resolved and counts[resolved[symbol]] == 1
        else f"HUMAN_SYMBOL:{symbol}"
        for symbol in arc_symbols
    ]
    if any(not isinstance(symbol, str) or not symbol.strip() for symbol in arc_symbols):
        raise ValueError("Arc symbols must be nonempty strings")
    if len(set(arc_ids)) != len(arc_ids):
        raise ValueError("Arc annotation does not define a one-to-one feature mapping")
    datasets["arc"] = arc_ids
    ids = set().union(*(set(values) for values in datasets.values()))
    genes = tuple(sorted(ids))
    lookup = {identifier: index for index, identifier in enumerate(genes)}
    mappings = {
        name: np.asarray([lookup[identifier] for identifier in values], dtype=np.int64)
        for name, values in datasets.items()
    }
    measured = {}
    for name, mapping in mappings.items():
        mask = np.zeros(len(genes), dtype=bool)
        mask[mapping] = True
        measured[name] = mask
    return GeneAlignment(
        genes=genes,
        source_to_global=mappings,
        measured=measured,
        notes={
            "arc_resolved_to_ensembl": sum(identifier.startswith("ENSG") for identifier in arc_ids),
            "arc_symbol_namespace": sum(identifier.startswith("HUMAN_SYMBOL:") for identifier in arc_ids),
            "arc_ambiguous_symbols": sum(
                symbol in symbols_to_ids and len(symbols_to_ids[symbol]) > 1 for symbol in arc_symbols
            ),
            "arc_unmapped_symbols": sum(symbol not in symbols_to_ids for symbol in arc_symbols),
        },
    )


def align_counts(
    source_counts: sp.spmatrix,
    source_to_global: np.ndarray,
    gene_count: int,
) -> sp.csr_matrix:
    matrix = sp.csr_matrix(source_counts)
    mapping = np.asarray(source_to_global)
    if (
        matrix.shape[1] != len(mapping) or not np.issubdtype(mapping.dtype, np.integer)
        or np.any(mapping < 0) or np.any(mapping >= gene_count)
        or len(set(mapping.tolist())) != len(mapping)
        or not np.isfinite(matrix.data).all() or np.any(matrix.data < 0)
        or np.any(matrix.data != np.floor(matrix.data))
        or np.any(matrix.data > 2**24)
        or np.any(np.asarray(matrix.sum(axis=1)).ravel() <= 0)
    ):
        raise ValueError("Counts or their one-to-one source-to-foundation mapping are invalid")
    matrix.sum_duplicates()
    if not np.isfinite(matrix.data).all() or np.any(matrix.data > 2**24):
        raise ValueError("Aggregating duplicate sparse entries overflowed")
    matrix = matrix.astype(np.float32)
    result = sp.csr_matrix(
        (matrix.data.copy(), mapping[matrix.indices], matrix.indptr.copy()),
        shape=(matrix.shape[0], gene_count), dtype=np.float32,
    )
    result.sort_indices()
    return result


class H5CsrSource:
    def __init__(self, path: Path, matrix_key: str) -> None:
        self.file = h5py.File(path, "r")
        try:
            self.matrix = self.file[matrix_key]
            if self.matrix.attrs.get("encoding-type") != "csr_matrix":
                raise ValueError("The source expression matrix must be CSR")
            self.n_rows, self.n_genes = map(int, self.matrix.attrs["shape"])
            self.indptr = self.matrix["indptr"]
            if (
                len(self.indptr) != self.n_rows + 1 or self.indptr[0] != 0
                or self.indptr[-1] != len(self.matrix["data"])
                or self.indptr[-1] != len(self.matrix["indices"])
            ):
                raise ValueError("Source CSR pointers are inconsistent")
        except Exception:
            self.file.close()
            raise

    def close(self) -> None:
        self.file.close()

    def counts(self, left: int, right: int) -> sp.csr_matrix:
        if not 0 <= left < right <= self.n_rows:
            raise ValueError("Requested rows are outside the source matrix")
        pointer = self.indptr[left:right + 1].astype(np.int64)
        if np.any(np.diff(pointer) < 0):
            raise ValueError("Source CSR row pointers are not monotonic")
        start, end = int(pointer[0]), int(pointer[-1])
        indices = self.matrix["indices"][start:end]
        data = self.matrix["data"][start:end]
        if np.any(indices < 0) or np.any(indices >= self.n_genes):
            raise ValueError("Source CSR gene indices are out of range")
        return sp.csr_matrix(
            (data, indices, pointer - start), shape=(right - left, self.n_genes),
        )
