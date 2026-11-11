from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from c3vcell.data import CONTROL
from c3vcell.identity import check_cell_id_schemes, source_cell_id
from c3vcell.predict import predict_control_baseline


def test_source_ids_are_stable_namespaced_and_unambiguous() -> None:
    cell = source_cell_id("orion", "cell-line", "sample", "barcode")
    assert cell == source_cell_id("orion", "cell-line", "sample", "barcode")
    assert cell != source_cell_id("tahoe", "cell-line", "sample", "barcode")
    assert source_cell_id("orion", "a|b", "c") != source_cell_id("orion", "a", "b|c")
    with pytest.raises(ValueError, match="nonempty"):
        source_cell_id("orion", "")


def test_legacy_and_stable_source_ids_cannot_be_compared() -> None:
    stable = source_cell_id("orion", "line", "sample", "barcode")
    with pytest.raises(ValueError, match="Legacy and stable orion"):
        check_cell_id_schemes(["orion-000000001"], [stable])
    with pytest.warns(RuntimeWarning, match="assigned by position"):
        check_cell_id_schemes(["tahoe-000000000"], ["tahoe-000000001"])
    check_cell_id_schemes([stable], [source_cell_id("tahoe", "sample", "barcode")])


def test_control_baseline_rejects_mixed_legacy_and_stable_ids(tmp_path: Path) -> None:
    genes = pd.DataFrame(index=["G0", "G1"])
    reference = ad.AnnData(
        sp.csr_matrix(np.ones((2, 2), dtype=np.int32)),
        obs=pd.DataFrame(
            {"target_gene": [CONTROL, "G0"], "context": ["c", "c"], "batch": ["b", "b"]},
            index=["orion-000000000", "orion-000000001"],
        ),
        var=genes.copy(),
    )
    controls = ad.AnnData(
        sp.csr_matrix(np.ones((1, 2), dtype=np.int32)),
        obs=pd.DataFrame(
            {"target_gene": [CONTROL], "context": ["c"], "batch": ["b"]},
            index=[source_cell_id("orion", "c", "b", "source")],
        ),
        var=genes.copy(),
    )
    reference_path, controls_path = tmp_path / "reference.h5ad", tmp_path / "controls.h5ad"
    reference.write_h5ad(reference_path)
    controls.write_h5ad(controls_path)
    with pytest.raises(ValueError, match="Legacy and stable orion"):
        predict_control_baseline(
            reference_path, controls_path, tmp_path / "prediction.h5ad", cells_per_target=1,
        )
