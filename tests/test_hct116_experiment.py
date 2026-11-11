import importlib.util
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from c3vcell.epigenetics import NETWORK_FEATURES
from c3vcell.integrity import file_sha256
from c3vcell.scoring import SCORED_METRICS


@pytest.fixture
def experiment():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run-epi-hct116.py"
    spec = importlib.util.spec_from_file_location("hct116_experiment", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_priority_panel_uses_only_measured_fresh_targets_with_edges(
    experiment, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.h5ad"
    ad.AnnData(
        sp.csr_matrix(np.ones((2, 4), dtype=np.int32)),
        var=pd.DataFrame(index=["A", "B", "C", "D"]),
    ).write_h5ad(source)
    specification = tmp_path / "spec.json"
    specification.write_text("{}", encoding="utf-8")
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    pd.DataFrame([
        {"source": "A", "target": "C"},
        {"source": "B", "target": "D"},
    ]).to_csv(bundle / "graph.csv", index=False)
    masks = np.zeros((4, len(NETWORK_FEATURES)), dtype=np.float32)
    masks[0, NETWORK_FEATURES.index("promoter_methylation")] = 1
    masks[2, NETWORK_FEATURES.index("linked_element_methylation")] = 1
    metadata = {
        "context": "HCT116", "format_version": 2, "features": list(NETWORK_FEATURES),
        "specification_sha256": file_sha256(specification),
        "manifest_sha256": "manifest",
        "files": {"graph.csv": file_sha256(bundle / "graph.csv")},
    }
    monkeypatch.setattr(
        experiment, "load_epigenetic_bundle",
        lambda *_args: (np.zeros_like(masks), masks, metadata),
    )
    totals = pd.DataFrame({"gene_target": ["B", "A", "C"], "n_cells": [100, 80, 60]})
    selected, required, record = experiment._priority_panel(totals, source, bundle, specification, 1)
    assert selected == ["A"]
    assert required == ["C"]
    assert record["prespecified_outgoing_neighbors"] == {"A": ["C"]}
    selected, required, record = experiment.priority_panel(
        totals, source, bundle, specification, 1, additional_graph_count=1,
        loader=experiment.load_epigenetic_bundle,
    )
    assert selected == ["A", "B"]
    assert required == ["C", "D"]
    assert record["methylation_prioritized_holdouts"] == ["A"]
    assert record["graph_only_prioritized_holdouts"] == ["B"]
    with pytest.raises(ValueError, match="Not enough additional"):
        experiment.priority_panel(
            totals, source, bundle, specification, 1, additional_graph_count=2,
            loader=experiment.load_epigenetic_bundle,
        )
    with pytest.raises(ValueError, match="ordered by descending"):
        experiment.priority_panel(
            totals.iloc[::-1], source, bundle, specification, 1,
            loader=experiment.load_epigenetic_bundle,
        )
    with pytest.raises(ValueError, match="Only 1 fresh eligible"):
        experiment._priority_panel(totals, source, bundle, specification, 2)
    metadata["specification_sha256"] = "different"
    with pytest.raises(ValueError, match="same specification"):
        experiment._priority_panel(totals, source, bundle, specification, 1)


@pytest.mark.parametrize("denominator", [-0.01, 0.02])
def test_hct116_scoring_reports_undefined_expression_without_fabrication(
    experiment, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, denominator: float,
) -> None:
    calls = []

    def fake_score(_prediction: Path, _reference: Path, outdir: Path, **kwargs) -> dict:
        metrics = kwargs["metrics"]
        calls.append(metrics)
        outdir.mkdir(parents=True)
        if metrics == ("expr_distance_unbiased",):
            return {"raw_metrics": {"HCT116": {"expr_distance_unbiased": denominator}}}
        return {
            "raw_metrics": {"HCT116": {metric: 0.5 for metric in metrics}},
            "undefined_metrics": {},
        }

    monkeypatch.setattr(experiment, "score_local", fake_score)
    out = tmp_path / "score"
    result = experiment._score_observable(tmp_path / "pred", tmp_path / "ref", out)
    assert calls[0] == ("expr_distance_unbiased",)
    assert (out / "evaluation-summary.json").is_file()
    assert json.loads((out / "evaluation-summary.json").read_text()) == result
    assert result["official_leaderboard_score"] is False
    if denominator <= 0:
        assert calls[1] == tuple(m for m in SCORED_METRICS if m != experiment.OBJECTIVE)
        assert result["raw_metrics"]["HCT116"][experiment.OBJECTIVE] is None
        assert result["undefined_metrics"]["HCT116"] == [experiment.OBJECTIVE]
    else:
        assert calls[1] == SCORED_METRICS
        assert result["raw_metrics"]["HCT116"][experiment.OBJECTIVE] == 0.5


def test_hct116_scoring_rejects_missing_denominator(
    experiment, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        experiment, "score_local",
        lambda *_args, **_kwargs: {"raw_metrics": {"HCT116": {"expr_distance_unbiased": None}}},
    )
    with pytest.raises(ValueError, match="denominator diagnostic is unavailable"):
        experiment._score_observable(tmp_path / "pred", tmp_path / "ref", tmp_path / "score")
