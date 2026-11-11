"""Bounded, count-preserving reference data with explicit measurement masks."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from .data import validate_counts


@dataclass(frozen=True)
class ReferenceCounts:
    matrix: sp.csr_matrix
    measured: np.ndarray
    cell_ids: list[str]
    source_identity: str
    source_genes: int
    shared_genes: int


def reference_metadata(
    reference: ReferenceCounts | None,
    path: Path | None,
    *,
    fraction: float,
    max_cells: int,
    seed: int,
    min_coverage: float,
    model_genes: int,
) -> dict | None:
    if reference is None:
        return None
    if path is None:
        raise ValueError("Reference metadata requires its source path")
    return {
        "path": str(path.resolve()), "fraction": fraction, "max_cells": max_cells,
        "seed": seed, "min_coverage": min_coverage,
        "source_identity": reference.source_identity, "cell_ids": reference.cell_ids,
        "source_genes": reference.source_genes, "shared_genes": reference.shared_genes,
        "model_genes": model_genes,
    }


def load_reference(
    path: Path,
    genes: list[str],
    *,
    max_cells: int,
    seed: int,
    min_coverage: float,
) -> ReferenceCounts:
    if path.suffix != ".h5ad" or not path.is_file() or ".aivc-partials" in path.parts:
        raise ValueError(f"Reference must be a completed H5AD file: {path}")
    if max_cells < 1 or not 0 < min_coverage <= 1:
        raise ValueError("Reference max_cells must be positive and min_coverage in (0, 1]")
    if not genes or len(set(genes)) != len(genes):
        raise ValueError("The model gene axis must be nonempty and unique")
    source = ad.read_h5ad(path, backed="r")
    try:
        if not source.obs_names.is_unique or not source.var_names.is_unique:
            raise ValueError("Reference cell and gene identifiers must be unique")
        if source.n_obs < 1:
            raise ValueError("Reference data has no cells")
        rng = np.random.default_rng(seed)
        rows = np.sort(rng.choice(source.n_obs, min(max_cells, source.n_obs), replace=False))
        raw = sp.csr_matrix(source.X[rows, :])
        ids = source.obs_names[rows].astype(str).tolist()
        source_genes = source.var_names.astype(str).tolist()
    finally:
        source.file.close()
    sampled = ad.AnnData(
        X=raw, obs=pd.DataFrame(index=ids), var=pd.DataFrame(index=source_genes),
    )
    validate_counts(sampled)
    positions = pd.Index(source_genes).get_indexer(genes)
    model_columns = np.flatnonzero(positions >= 0)
    coverage = len(model_columns) / len(genes)
    if coverage < min_coverage:
        raise ValueError(
            f"Reference covers {coverage:.1%} of model genes, below {min_coverage:.1%}"
        )
    source_columns = positions[model_columns]
    placement = sp.csr_matrix(
        (np.ones(len(model_columns), dtype=np.float32), (np.arange(len(model_columns)), model_columns)),
        shape=(len(model_columns), len(genes)),
    )
    aligned = (raw[:, source_columns].astype(np.float32) @ placement).tocsr()
    if np.any(np.asarray(aligned.sum(axis=1)).ravel() <= 0):
        raise ValueError("Some reference cells have no counts on the measured model gene panel")
    measured = np.zeros(len(genes), dtype=np.float32)
    measured[model_columns] = 1
    canonical = raw.astype(np.int64)
    canonical.sum_duplicates()
    canonical.sort_indices()
    digest = hashlib.sha256()
    digest.update(json.dumps([ids, source_genes], ensure_ascii=False).encode("utf-8"))
    for array in (canonical.indptr, canonical.indices, canonical.data):
        digest.update(np.asarray(array, dtype="<i8").tobytes())
    return ReferenceCounts(
        aligned, measured, ids, digest.hexdigest(), len(source_genes), len(model_columns),
    )
