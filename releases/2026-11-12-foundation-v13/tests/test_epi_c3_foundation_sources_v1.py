from __future__ import annotations

import h5py
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from epi_c3_foundation_sources_v1 import H5CsrSource, align_counts, align_human_genes


def inputs():
    tabula_ids = ["ENSG1", "ENSG2", "ENSG3"]
    tabula_symbols = ["A", "B", "AMB"]
    tahoe = pd.DataFrame({"ensembl_id": ["ENSG1", "ENSG2", "ENSG4"], "gene_symbol": ["A", "B", "AMB"]})
    orion = pd.DataFrame({"ensembl_id": ["ENSG1", "ENSG5"], "gene_name": ["A", "OTHER"]})
    return tabula_ids, tabula_symbols, tahoe, orion, ["B", "AMB", "UNMAPPED"]


def test_stable_ids_and_ambiguous_symbols_stay_distinct():
    alignment = align_human_genes(*inputs())
    assert set(alignment.genes) == {"ENSG1", "ENSG2", "ENSG3", "ENSG4", "ENSG5",
                                    "HUMAN_SYMBOL:AMB", "HUMAN_SYMBOL:UNMAPPED"}
    assert alignment.notes == {
        "arc_resolved_to_ensembl": 1, "arc_symbol_namespace": 2,
        "arc_ambiguous_symbols": 1, "arc_unmapped_symbols": 1,
    }
    assert alignment.measured["arc"].sum() == 3
    assert not alignment.measured["arc"][alignment.genes.index("ENSG3")]


def test_alignment_preserves_counts_and_source_masks():
    alignment = align_human_genes(*inputs())
    original = sp.csr_matrix([[1, 2, 0], [0, 0, 3]], dtype=np.float32)
    result = align_counts(original, alignment.source_to_global["arc"], len(alignment.genes))
    assert np.array_equal(np.asarray(result.sum(axis=1)), np.asarray(original.sum(axis=1)))
    assert np.array_equal(result[:, alignment.source_to_global["arc"]].toarray(), original.toarray())
    assert np.all(result[:, ~alignment.measured["arc"]].toarray() == 0)


def test_bad_axes_and_count_semantics_fail_explicitly():
    args = list(inputs())
    args[0] = ["ENSG1", "ENSG1", "ENSG3"]
    with pytest.raises(ValueError, match="unique"):
        align_human_genes(*args)
    with pytest.raises(ValueError):
        align_counts(sp.csr_matrix([[1.1, 2]]), np.array([0, 1]), 2)
    with pytest.raises(ValueError):
        align_counts(sp.csr_matrix([[1, 2]]), np.array([0, 0]), 2)
    with pytest.raises(ValueError):
        align_counts(sp.csr_matrix([[0, 0]]), np.array([0, 1]), 2)


def test_h5_csr_reads_only_requested_contiguous_rows(tmp_path):
    source = sp.csr_matrix([[0, 2, 1], [1, 0, 4], [3, 0, 0]], dtype=np.float32)
    path = tmp_path / "source.h5"
    with h5py.File(path, "w") as file:
        node = file.create_group("raw/X")
        node.attrs["encoding-type"] = "csr_matrix"
        node.attrs["shape"] = source.shape
        node.create_dataset("data", data=source.data)
        node.create_dataset("indices", data=source.indices)
        node.create_dataset("indptr", data=source.indptr)
    reader = H5CsrSource(path, "raw/X")
    try:
        assert np.array_equal(reader.counts(1, 3).toarray(), source[1:3].toarray())
        with pytest.raises(ValueError):
            reader.counts(-1, 2)
    finally:
        reader.close()
