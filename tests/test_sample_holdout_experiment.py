import importlib.util
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import torch

from c3vcell.data import CONTROL
from c3vcell.integrity import file_sha256
from c3vcell.scoring import SCORED_METRICS


@pytest.fixture
def experiment():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run-epi-sample-holdout.py"
    spec = importlib.util.spec_from_file_location("sample_holdout_experiment", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_sample_holdout_panels_are_disjoint_and_deterministic(experiment) -> None:
    training, test = experiment._samples(["D", "C", "A", "B", "A"], 2, 2)
    assert training == ["A", "B"]
    assert test == ["C", "D"]
    with pytest.raises(ValueError, match="Not enough distinct samples"):
        experiment._samples(["A", "B"], 1, 2)

    totals = pd.DataFrame({
        "gene_target": ["A", "B", "C", "D", "E", "F", "SP1", "CTCF"],
        "n_cells": [200, 190, 180, 170, 160, 150, 30, 20],
    })
    genes = set(totals["gene_target"])
    train = experiment._training_targets(
        totals, genes, {"A"}, ("SP1", "CTCF"), count=4, minimum_cells=100,
    )
    assert train == ["B", "C", "CTCF", "SP1"]
    first = experiment._general_targets(totals, genes, {"A", "SP1"}, 3, 100, seed=3)
    assert first == experiment._general_targets(totals, genes, {"A", "SP1"}, 3, 100, seed=3)
    assert len(set(first)) == 3 and not set(first) & {"A", "SP1"}
    with pytest.raises(ValueError, match="required prior-linked"):
        experiment._training_targets(totals, genes, {"SP1"}, ("SP1",), 3, 100)
    with pytest.raises(ValueError, match="Not enough fresh targets"):
        experiment._general_targets(totals, genes, set(), 20, 100, seed=3)


def test_priority_selection_samples_must_come_from_training(
    experiment, tmp_path: Path,
) -> None:
    data = ad.AnnData(
        sp.csr_matrix(np.ones((2, 1), dtype=np.int32)),
        obs=pd.DataFrame(
            {"context": ["HCT116", "HCT116"], "batch": ["sample-1", "sample-2"]},
            index=["c1", "c2"],
        ),
        var=pd.DataFrame(index=["G"]),
    )
    path = tmp_path / "selection.h5ad"
    data.write_h5ad(path)
    assert experiment._selection_samples(path, ["sample-1", "sample-2", "sample-3"]) == [
        "sample-1", "sample-2",
    ]
    with pytest.raises(ValueError, match="only selected HCT116 training samples"):
        experiment._selection_samples(path, ["sample-1", "sample-3"])
    data.obs["context"] = ["HCT116", "other"]
    data.write_h5ad(path)
    with pytest.raises(ValueError, match="only selected HCT116 training samples"):
        experiment._selection_samples(path, ["sample-1", "sample-2"])
    del data.obs["batch"]
    data.write_h5ad(path)
    with pytest.raises(ValueError, match="context and batch metadata"):
        experiment._selection_samples(path, ["sample-1", "sample-2"])


def test_development_evidence_is_training_only_and_reproducible(
    experiment, tmp_path: Path,
) -> None:
    source = tmp_path / "source.h5ad"
    ad.AnnData(
        sp.csr_matrix(np.ones((2, 1), dtype=np.int32)),
        obs=pd.DataFrame(
            {"context": ["HCT116", "HCT116"], "batch": ["sample-1", "sample-2"]},
            index=["c1", "c2"],
        ),
        var=pd.DataFrame(index=["G"]),
    ).write_h5ad(source)
    checkpoint = tmp_path / "model.pt"
    torch.save({"training_cell_ids": ["c1", "c2"], "reference": None}, checkpoint)
    plan_path = tmp_path / "experiment-plan.json"
    plan_path.write_text(json.dumps({
        "objective": "expr_mse_unbiased_capped",
        "flow_targets": ["paired", "population-shift"],
        "folds": 2,
        "strengths": [0.0, 0.5],
        "inputs_sha256": {str(source): file_sha256(source), str(checkpoint): file_sha256(checkpoint)},
    }))
    arm = {
        "folds": [{"values": [1.0, 0.8]}, {"values": [0.8, 0.7]}],
        "mean_objective_by_strength": [0.9, 0.75],
        "selected_strength": 0.5,
    }
    evidence_path = tmp_path / "result.json"
    evidence = {
        "experiment_plan_sha256": file_sha256(plan_path),
        "results": {"no-prior-population-shift": arm, "hct116-epigenome-population-shift": arm},
    }
    evidence_path.write_text(json.dumps(evidence))
    record = experiment._development_record(evidence_path, ["sample-1", "sample-2"])
    assert record["samples"] == ["sample-1", "sample-2"]
    assert record["result_sha256"] == file_sha256(evidence_path)
    with pytest.raises(ValueError, match="only selected HCT116 training samples"):
        experiment._development_record(evidence_path, ["sample-1"])
    evidence["results"]["no-prior-population-shift"] = {**arm, "selected_strength": 0.0}
    evidence_path.write_text(json.dumps(evidence))
    with pytest.raises(ValueError, match="does not support population-shift"):
        experiment._development_record(evidence_path, ["sample-1", "sample-2"])
    torch.save({"training_cell_ids": ["other"]}, checkpoint)
    with pytest.raises(ValueError, match="input changed"):
        experiment._development_record(evidence_path, ["sample-1", "sample-2"])
    plan = json.loads(plan_path.read_text())
    plan["inputs_sha256"][str(checkpoint)] = file_sha256(checkpoint)
    plan_path.write_text(json.dumps(plan))
    evidence["experiment_plan_sha256"] = file_sha256(plan_path)
    evidence["results"]["no-prior-population-shift"] = arm
    evidence_path.write_text(json.dumps(evidence))
    with pytest.raises(ValueError, match="unverified training cells"):
        experiment._development_record(evidence_path, ["sample-1", "sample-2"])


def test_sample_holdout_history_reads_prior_targets_and_rejects_bad_manifests(
    experiment, tmp_path: Path,
) -> None:
    prior = tmp_path / "old" / "split.json"
    prior.parent.mkdir()
    prior.write_text(json.dumps({"train_targets": ["A"], "test_targets": ["B"]}))
    (tmp_path / "orion-targets.txt").write_text("C\nD\n")
    targets, records = experiment._history(tmp_path)
    assert targets == {"A", "B", "C", "D"}
    assert {Path(item["path"]).name for item in records} == {"split.json", "orion-targets.txt"}
    assert all(len(item["sha256"]) == 64 for item in records)
    prior.write_text(json.dumps({"train_targets": "not-a-list"}))
    with pytest.raises(ValueError, match="Invalid target history"):
        experiment._history(tmp_path)


def test_panel_scoring_keeps_controls_and_records_undefined_metrics(
    experiment, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    def dataset(labels: list[str], prefix: str) -> ad.AnnData:
        return ad.AnnData(
            sp.csr_matrix(np.ones((len(labels), 4), dtype=np.int32)),
            obs=pd.DataFrame(
                {"target_gene": labels, "context": "HCT116"},
                index=[f"{prefix}-{i}" for i in range(len(labels))],
            ),
            var=pd.DataFrame(index=["A", "B", "C", "D"]),
        )

    prediction = tmp_path / "prediction.h5ad"
    reference = tmp_path / "reference.h5ad"
    dataset(["A", "A", "B", "B", "C", "C"], "p").write_h5ad(prediction)
    dataset([CONTROL, CONTROL, "A", "B", "C"], "r").write_h5ad(reference)
    seen = {}

    def fake_score(pred: Path, ref: Path, outdir: Path, **_kwargs) -> dict:
        pred_data, ref_data = ad.read_h5ad(pred), ad.read_h5ad(ref)
        seen[outdir.name] = (
            set(pred_data.obs["target_gene"]), set(ref_data.obs["target_gene"]),
        )
        metrics = {name: 0.5 for name in SCORED_METRICS}
        metrics["de_wilcoxon_lfc_nmae"] = None
        return {
            "raw_metrics": {"HCT116": metrics},
            "undefined_metrics": {"HCT116": ["de_wilcoxon_lfc_nmae"]},
            "official_leaderboard_score": False,
        }

    monkeypatch.setattr(experiment, "score_single_context_observable", fake_score)
    panels = {"all": ["A", "B", "C"], "general": ["A", "B"], "prior-enriched": ["C"]}
    results = experiment._score_panels(prediction, reference, tmp_path / "scores", panels)
    assert seen["all"] == ({"A", "B", "C"}, {CONTROL, "A", "B", "C"})
    assert seen["general"] == ({"A", "B"}, {CONTROL, "A", "B"})
    assert seen["prior-enriched"] == ({"C"}, {CONTROL, "C"})
    assert results["general"]["undefined_metrics"]["HCT116"] == ["de_wilcoxon_lfc_nmae"]
    with pytest.raises(ValueError, match="Missing prediction targets"):
        experiment._score_panels(
            prediction, reference, tmp_path / "bad-scores", {"missing": ["A", "D"]},
        )
