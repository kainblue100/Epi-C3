import json
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import torch

from c3vcell.calibration import OBJECTIVE
from c3vcell.cli import main
from c3vcell.data import CONTROL, write_split
from c3vcell.integrity import file_sha256
from c3vcell.mean_response import _generate_shared_counts, fit_mean_response, predict_mean_response
from c3vcell.shared_calibration import calibrate_shared_residual
from c3vcell.shared_export import _read_profile_mapping, export_shared_vcc_h5ad
from c3vcell.training import train


def test_shared_rounding_probabilities_do_not_depend_on_control_layout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    counts = np.tile([4, 5, 9, 3, 6, 2, 6, 4, 5, 5], (400, 1)).astype(np.float32)
    logfold = torch.tensor(np.tile([
        0.69882274, -0.12020586, -0.09357969, -0.14289193, -0.02309253,
        -0.15504180, -0.19973269, -0.05785440, -0.12037552, -0.20640703,
    ], (400, 1)), dtype=torch.float32)
    probabilities = []

    def capture(values: torch.Tensor) -> torch.Tensor:
        probabilities.append(values.clone())
        return torch.zeros_like(values)

    monkeypatch.setattr(torch, "bernoulli", capture)
    for array in (counts, np.asfortranarray(counts)):
        _generate_shared_counts(
            torch.from_numpy(array), logfold, effect_strength=1, conditional_strength=0,
            model=None, target_index=None, anchor=None, graph=None, steps=1,
        )
    assert torch.equal(*probabilities)


@pytest.mark.parametrize("prefer_null", [False, True])
def test_shared_export_matches_local_predictor_and_enforces_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, prefer_null: bool,
) -> None:
    rng = np.random.default_rng(42)
    labels = [CONTROL] * 60 + sum(([f"G{i}"] * 8 for i in range(8)), [])
    genes = [f"G{i}" for i in range(10)]
    data = ad.AnnData(
        sp.csr_matrix(rng.poisson(3, size=(len(labels), len(genes))).astype(np.int32) + 1),
        obs=pd.DataFrame(
            {"target_gene": labels, "context": "c", "batch": "b"},
            index=[f"train-source-{i}" for i in range(len(labels))],
        ),
        var=pd.DataFrame(index=genes),
    )
    split = tmp_path / "outer"
    write_split(data, split, control=CONTROL, holdout_fraction=0.25, seed=4, holdout_targets=["G6", "G7"])
    source = split / "train.h5ad"
    checkpoint = train(
        source, tmp_path / "model", latent_dim=4, hidden_dim=6,
        pretrain_steps=2, flow_steps=2, batch_size=4, seed=3,
    )
    profile = tmp_path / "profile.h5ad"
    fit_mean_response(source, profile)

    def fake_score(prediction_path: Path, *_args, **_kwargs) -> dict:
        if prediction_path.name == "null-control.h5ad":
            value = 0.1 if prefer_null else 2.0
        else:
            strength = ad.read_h5ad(prediction_path).uns["c3vcell_mean_response_prediction"][
                "conditional_strength"
            ]
            value = 1.0 + (strength - 0.5) ** 2
        return {"raw_metrics": {"c": {OBJECTIVE: value}}}

    monkeypatch.setattr("c3vcell.shared_calibration.score_local", fake_score)
    calibration = calibrate_shared_residual(
        source, checkpoint, profile, tmp_path / "calibration", folds=2,
        strengths=(0, 0.5), include_no_response=True, control_sampling="input-controls",
        cells_per_target=400, prediction_steps=2,
    )
    calibration_record = json.loads(calibration.read_text(encoding="utf-8"))
    assert calibration_record["selected_shared_strength"] == (0.0 if prefer_null else 1.0)
    assert calibration_record["selected_conditional_strength"] == (0.0 if prefer_null else 0.5)

    challenge_genes = genes + ["EXTRA"]
    gene_file, pert_file = tmp_path / "genes.csv", tmp_path / "targets.csv"
    pd.DataFrame({"gene_name": challenge_genes}).to_csv(gene_file, index=False)
    pd.DataFrame({"target_gene": ["G6", "G7"], "n_cells": 400}).to_csv(pert_file, index=False)
    monkeypatch.setattr("c3vcell.shared_export._read_gene_list", lambda _: challenge_genes)
    monkeypatch.setattr("c3vcell.shared_export._read_targets", lambda _: ["G6", "G7"])
    monkeypatch.setattr("c3vcell.shared_export._check_disk_reserve", lambda _: None)
    monkeypatch.setattr("c3vcell.predict._check_disk_reserve", lambda _: None)
    source_controls = ad.read_h5ad(split / "input_controls.h5ad")
    controls = {}
    mapping = []
    for context in ("A", "B", "C"):
        obs = source_controls.obs.copy()
        obs.index = [f"{context}-{index}" for index in obs.index]
        obs["context"] = context
        obs["batch"] = "challenge-batch"
        matrix = sp.hstack(
            [source_controls.X, sp.csr_matrix(np.arange(source_controls.n_obs)[:, None] + 1)],
            format="csr",
        )
        path = tmp_path / f"controls-{context}.h5ad"
        ad.AnnData(matrix, obs=obs, var=pd.DataFrame(index=challenge_genes)).write_h5ad(path)
        controls[context] = path
        mapping.append({
            "input_context": context, "input_batch": "challenge-batch",
            "profile_context": "c", "profile_batch": "b",
        })
    mapping_path = tmp_path / "mapping.json"
    mapping_path.write_text(json.dumps(mapping), encoding="utf-8")
    kwargs = {
        "profile_mapping_path": None if prefer_null else mapping_path,
        "steps": 2, "seed": 17,
    }
    output = tmp_path / "export.h5ad"
    export_shared_vcc_h5ad(
        checkpoint, profile, calibration, controls, gene_file, pert_file, output, **kwargs,
    )
    prediction = ad.read_h5ad(output)
    provenance = json.loads(output.with_suffix(".h5ad.json").read_text(encoding="utf-8"))
    assert prediction.shape == (2400, len(challenge_genes))
    assert set(prediction.obs["context"]) == {"A", "B", "C"}
    assert provenance["control_sampling"] == "input-controls"
    assert provenance["no_response_selected"] is prefer_null
    assert provenance["challenge_genes_not_in_model"] == ["EXTRA"]
    assert provenance["conditional_strength"] == (0.0 if prefer_null else 0.5)
    assert provenance["output_sha256"] == file_sha256(output)

    challenge_controls = ad.read_h5ad(controls["A"])
    local_controls = challenge_controls[:, genes].copy()
    local_controls.obs["context"] = "c"
    local_controls.obs["batch"] = "b"
    local_controls_path = tmp_path / "local-controls.h5ad"
    local_controls.write_h5ad(local_controls_path)
    local_output = tmp_path / "local.h5ad"
    predict_mean_response(
        profile, split / "reference.h5ad", local_controls_path, local_output,
        checkpoint_path=checkpoint, conditional_calibration_path=calibration,
        cells_per_target=400, steps=2, seed=17, control_sampling="input-controls",
    )
    local = ad.read_h5ad(local_output)
    context_a = prediction[prediction.obs["context"] == "A"].copy()
    assert (context_a[:, genes].X != local.X).nnz == 0
    selected = challenge_controls.obs_names.get_indexer(local.obs["source_control_id"])
    assert np.all(selected >= 0)
    np.testing.assert_array_equal(
        context_a[:, ["EXTRA"]].X.toarray(), challenge_controls.X[selected, -1].toarray(),
    )

    legacy = dict(calibration_record)
    legacy["control_sampling"] = "reference-batch"
    wrong_calibration = tmp_path / "wrong-calibration.json"
    wrong_calibration.write_text(json.dumps(legacy), encoding="utf-8")
    with pytest.raises(ValueError, match="prediction settings differ"):
        export_shared_vcc_h5ad(
            checkpoint, profile, wrong_calibration, controls, gene_file, pert_file,
            tmp_path / "wrong-calibration.h5ad", **kwargs,
        )

    if prefer_null:
        short_genes = [gene for gene in challenge_genes if gene != "G9"]
        monkeypatch.setattr("c3vcell.shared_export._read_gene_list", lambda _: short_genes)
        short_controls = {}
        for context, path in controls.items():
            short_controls[context] = tmp_path / f"short-{context}.h5ad"
            ad.read_h5ad(path)[:, short_genes].copy().write_h5ad(short_controls[context])
        short_output = tmp_path / "short-null.h5ad"
        export_shared_vcc_h5ad(
            checkpoint, profile, calibration, short_controls, gene_file, pert_file,
            short_output, steps=2, seed=17,
        )
        short = ad.read_h5ad(short_output)
        assert short.shape == (2400, len(short_genes))
        assert "G9" in json.loads(short_output.with_suffix(".h5ad.json").read_text())[
            "model_genes_not_in_challenge"
        ]
    else:
        with pytest.raises(ValueError, match="requires an explicit profile mapping"):
            export_shared_vcc_h5ad(
                checkpoint, profile, calibration, controls, gene_file, pert_file,
                tmp_path / "missing-map.h5ad", steps=2,
            )
        unavailable = tmp_path / "unavailable-map.json"
        unavailable.write_text(
            json.dumps([dict(entry, profile_batch="absent") for entry in mapping]), encoding="utf-8",
        )
        with pytest.raises(ValueError, match="Mapped profile groups do not exist"):
            export_shared_vcc_h5ad(
                checkpoint, profile, calibration, controls, gene_file, pert_file,
                tmp_path / "unavailable.h5ad", profile_mapping_path=unavailable, steps=2,
            )
        incomplete = tmp_path / "incomplete-map.json"
        incomplete.write_text(json.dumps(mapping[:2]), encoding="utf-8")
        incomplete_output = tmp_path / "incomplete.h5ad"
        with pytest.raises(ValueError, match="mapping is missing input groups"):
            export_shared_vcc_h5ad(
                checkpoint, profile, calibration, controls, gene_file, pert_file,
                incomplete_output, profile_mapping_path=incomplete, steps=2,
            )
        assert not incomplete_output.exists()
        unused = tmp_path / "unused-map.json"
        unused.write_text(
            json.dumps([*mapping, dict(mapping[0], input_batch="unused")]), encoding="utf-8",
        )
        with pytest.raises(ValueError, match="mapping contains unused input groups"):
            export_shared_vcc_h5ad(
                checkpoint, profile, calibration, controls, gene_file, pert_file,
                tmp_path / "unused.h5ad", profile_mapping_path=unused, steps=2,
            )
        monkeypatch.setattr(
            "c3vcell.shared_export._read_gene_list", lambda _: [gene for gene in challenge_genes if gene != "G9"],
        )
        with pytest.raises(ValueError, match="model genes are absent"):
            export_shared_vcc_h5ad(
                checkpoint, profile, calibration, controls, gene_file, pert_file,
                tmp_path / "missing-gene.h5ad", profile_mapping_path=mapping_path, steps=2,
            )
        monkeypatch.setattr("c3vcell.shared_export._read_gene_list", lambda _: challenge_genes)
        monkeypatch.setattr("c3vcell.shared_export._read_targets", lambda _: ["EXTRA"])
        with pytest.raises(ValueError, match="targets are absent"):
            export_shared_vcc_h5ad(
                checkpoint, profile, calibration, controls, gene_file, pert_file,
                tmp_path / "missing-target.h5ad", profile_mapping_path=mapping_path, steps=2,
            )
        fallback = tmp_path / "missing-target-fallback.h5ad"
        export_shared_vcc_h5ad(
            checkpoint, profile, calibration, controls, gene_file, pert_file, fallback,
            profile_mapping_path=mapping_path, allow_missing_target=True, steps=2,
        )
        assert json.loads(fallback.with_suffix(".h5ad.json").read_text())["control_baseline_targets"] == ["EXTRA"]
        for context in ("A", "B", "C"):
            rows = ad.read_h5ad(fallback)
            rows = rows[rows.obs["context"] == context].X.toarray()
            available = {tuple(row) for row in ad.read_h5ad(controls[context]).X.toarray()}
            assert all(tuple(row) in available for row in rows)


def test_profile_mapping_rejects_duplicate_and_invalid_entries(tmp_path: Path) -> None:
    entry = {
        "input_context": "A", "input_batch": "",
        "profile_context": "c", "profile_batch": "b",
    }
    path = tmp_path / "map.json"
    path.write_text(json.dumps([entry, entry]), encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate profile mapping"):
        _read_profile_mapping(path)
    path.write_text(json.dumps([dict(entry, input_batch=None)]), encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid profile mapping"):
        _read_profile_mapping(path)


def test_shared_export_cli_passes_contexts_and_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = {}

    def fake_export(*args, **kwargs) -> None:
        captured["args"] = args
        captured["kwargs"] = kwargs

    monkeypatch.setattr("c3vcell.shared_export.export_shared_vcc_h5ad", fake_export)
    monkeypatch.setattr(sys, "argv", [
        "c3-vcell", "export-shared-vcc", "--checkpoint", "model.pt", "--profile", "profile.h5ad",
        "--calibration", "calibration.json", "--controls", "A=a.h5ad", "B=b.h5ad", "C=c.h5ad",
        "--gene-names", "genes.csv", "--pert-counts", "targets.csv", "--output", "output.h5ad",
        "--profile-mapping", "map.json", "--steps", "4", "--allow-missing-target",
    ])
    main()
    assert captured["args"][3] == {"A": Path("a.h5ad"), "B": Path("b.h5ad"), "C": Path("c.h5ad")}
    assert captured["kwargs"]["profile_mapping_path"] == Path("map.json")
    assert captured["kwargs"]["steps"] == 4
    assert captured["kwargs"]["allow_missing_target"] is True
