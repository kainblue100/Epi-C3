import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from c3vcell.data import CONTROL
from c3vcell.external_bulk import compare_bulk_tpm
from c3vcell.integrity import file_sha256


@pytest.fixture
def inputs(tmp_path: Path) -> dict[str, Path]:
    genes = ["T", "A", "B", "C"]
    obs = pd.DataFrame(
        {"target_gene": [CONTROL] * 2, "context": ["HCT116"] * 2},
        index=["control-1", "control-2"],
    )
    controls = ad.AnnData(
        sp.csr_matrix(np.array([[10, 20, 30, 40], [12, 18, 30, 40]], dtype=np.int32)),
        obs=obs, var=pd.DataFrame(index=genes),
    )
    control_path = tmp_path / "controls.h5ad"
    controls.write_h5ad(control_path)
    prediction = controls.copy()
    prediction.obs["target_gene"] = "T"
    prediction.obs["source_control_id"] = controls.obs_names
    prediction.obs_names = ["pred-1", "pred-2"]
    prediction.uns["c3vcell_prediction"] = {
        "control_sampling": "input-controls",
        "input_controls_sha256": file_sha256(control_path),
    }
    prediction_path = tmp_path / "prediction.h5ad"
    prediction.write_h5ad(prediction_path)
    bulk = pd.DataFrame({
        "Symbol": genes + ["OUTSIDE"],
        "c1": [10, 20, 30, 40, 100],
        "c2": [10, 20, 30, 40, 100],
        "p1": [1, 25, 30, 44, 100],
        "p2": [1, 25, 30, 44, 100],
    })
    bulk_path = tmp_path / "bulk.tsv.gz"
    bulk.to_csv(bulk_path, sep="\t", index=False, compression="gzip")
    return {"controls": control_path, "prediction": prediction_path, "bulk": bulk_path}


def _compare(inputs: dict[str, Path], output: Path, **options) -> None:
    compare_bulk_tpm(
        inputs["prediction"], inputs["controls"], inputs["bulk"], output,
        target="T", context="HCT116", control_samples=["c1", "c2"],
        perturbed_samples=["p1", "p2"], **options,
    )


def test_zero_response_bulk_comparison_is_paired_and_not_a_vcc_score(
    inputs: dict[str, Path], tmp_path: Path,
) -> None:
    output = tmp_path / "result.json"
    _compare(inputs, output)
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["not_a_vcc_score"] is True
    assert result["n_shared_genes_in_normalization"] == 3
    for metric in result["panels"].values():
        assert metric["n_genes"] == 3
        assert metric["spearman_lfc"] is None
        assert metric["pearson_lfc"] is None
        assert metric["direction_agreement_among_comparable"] is None
        assert metric["predicted_nonzero_fraction"] == 0
        assert metric["relative_mse_improvement_vs_zero"] == pytest.approx(0)
    assert result["inputs_sha256"]["bulk_sha256"] == file_sha256(inputs["bulk"])
    with pytest.raises(FileExistsError):
        _compare(inputs, output)


def test_bulk_comparison_uses_fixed_expression_filter(
    inputs: dict[str, Path], tmp_path: Path,
) -> None:
    prediction = ad.read_h5ad(inputs["prediction"])
    prediction.X = sp.csr_matrix(np.array([[9, 25, 28, 38], [11, 23, 28, 38]], dtype=np.int32))
    prediction.write_h5ad(inputs["prediction"])
    output = tmp_path / "filtered.json"
    _compare(inputs, output, minimum_tpm=35)
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["panels"]["all_common_nontarget"]["n_genes"] == 3
    filtered = result["panels"]["expressed_common_nontarget"]
    assert filtered["n_genes"] == 1
    assert filtered["pearson_lfc"] is None
    assert filtered["spearman_lfc"] is None
    assert filtered["predicted_nonzero_fraction"] == 1


def test_bulk_comparison_rejects_bad_inputs(inputs: dict[str, Path], tmp_path: Path) -> None:
    prediction = ad.read_h5ad(inputs["prediction"])
    prediction.obs.loc["pred-1", "source_control_id"] = "missing"
    prediction.write_h5ad(inputs["prediction"])
    with pytest.raises(ValueError, match="missing source control"):
        _compare(inputs, tmp_path / "missing.json")
    prediction.obs.loc["pred-1", "source_control_id"] = "control-1"
    prediction.uns["c3vcell_prediction"]["input_controls_sha256"] = "wrong"
    prediction.write_h5ad(inputs["prediction"])
    with pytest.raises(ValueError, match="exact control file"):
        _compare(inputs, tmp_path / "wrong-hash.json")
    prediction.uns["c3vcell_prediction"]["input_controls_sha256"] = file_sha256(inputs["controls"])
    prediction.write_h5ad(inputs["prediction"])
    bulk = pd.read_csv(inputs["bulk"], sep="\t")
    bulk.loc[1, "Symbol"] = bulk.loc[0, "Symbol"]
    bulk.to_csv(inputs["bulk"], sep="\t", index=False, compression="gzip")
    with pytest.raises(ValueError, match="unique and nonempty"):
        _compare(inputs, tmp_path / "duplicate.json")
