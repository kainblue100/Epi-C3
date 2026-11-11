from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from c3vcell import scoring
from c3vcell.cli import build_parser
from c3vcell.integrity import file_sha256


def test_scorer_is_found_beside_active_interpreter(tmp_path: Path, monkeypatch) -> None:
    scorer = tmp_path / ("cell-eval2.exe" if scoring.os.name == "nt" else "cell-eval2")
    scorer.touch(mode=0o755)
    monkeypatch.setattr(scoring.sys, "executable", str(tmp_path / "python"))
    monkeypatch.setattr(scoring.shutil, "which", lambda _: None)
    assert scoring._find_scorer() == str(scorer)


def test_score_cli_accepts_an_explicit_metric_subset() -> None:
    args = build_parser().parse_args([
        "score", "--prediction", "p.h5ad", "--reference", "r.h5ad",
        "--outdir", "out", "--metrics", "pds_cosine", "de_wilcoxon_sig_jaccard",
    ])
    assert args.metrics == ["pds_cosine", "de_wilcoxon_sig_jaccard"]
    diagnostic = build_parser().parse_args([
        "score", "--prediction", "p.h5ad", "--reference", "r.h5ad",
        "--outdir", "out", "--metrics", "expr_mse_unbiased_capped",
    ])
    assert diagnostic.metrics == ["expr_mse_unbiased_capped"]
    with pytest.raises(SystemExit):
        build_parser().parse_args([
            "score", "--prediction", "p.h5ad", "--reference", "r.h5ad",
            "--outdir", "out", "--metrics", "not-a-metric",
        ])


def test_scoring_uses_safe_context_directories(tmp_path: Path, monkeypatch) -> None:
    genes = pd.DataFrame(index=["G0", "G1"])
    obs = pd.DataFrame(
        {"target_gene": ["G0", "G1"], "context": ["../outside", "../outside"]},
        index=["a", "b"],
    )
    pred = ad.AnnData(sp.csr_matrix(np.ones((2, 2))), obs=obs, var=genes)
    ref_obs = pd.concat([
        obs,
        pd.DataFrame(
            {"target_gene": ["non-targeting"], "context": ["../outside"]}, index=["ctrl"]
        ),
    ])
    ref = ad.AnnData(sp.csr_matrix(np.ones((3, 2))), obs=ref_obs, var=genes)
    pred.write_h5ad(tmp_path / "pred.h5ad")
    ref.write_h5ad(tmp_path / "ref.h5ad")
    commands = []
    monkeypatch.setattr(scoring.shutil, "which", lambda _: "cell-eval2")

    def fake_run(command):
        commands.append(command)
        scorer_pred = ad.read_h5ad(command[command.index("-ap") + 1])
        assert scorer_pred.obs["target_gene"].astype(str).tolist() == [
            "G0", "G1", "non-targeting"
        ]
        assert scorer_pred.obs_names.is_unique
        run = Path(command[command.index("-o") + 1])
        run.mkdir()
        pd.DataFrame(
            [{"statistic": "mean", **{metric: 0.5 for metric in scoring.SCORED_METRICS}}]
        ).to_csv(run / "agg_results.csv", index=False)

    monkeypatch.setattr(scoring, "_run", fake_run)
    out = tmp_path / "score"
    metadata = scoring.score_local(tmp_path / "pred.h5ad", tmp_path / "ref.h5ad", out)
    assert metadata["prediction_sha256"] == file_sha256(tmp_path / "pred.h5ad")
    assert metadata["reference_sha256"] == file_sha256(tmp_path / "ref.h5ad")
    assert (out / "context-000" / "prediction.h5ad").is_file()
    assert not (tmp_path / "outside").exists()
    assert commands[0][1] == "run"
    assert "vcc2026" in commands[0]
    assert "de.backend=pdex" in commands[0]
    assert "num_threads=2" in commands[0]
    assert ad.read_h5ad(tmp_path / "pred.h5ad").n_obs == 2
    drug_out = tmp_path / "drug-score"
    scoring.score_local(
        tmp_path / "pred.h5ad", tmp_path / "ref.h5ad", drug_out, no_target_genes=True,
    )
    assert "--no-target-genes" in commands[-1]
    assert "no_target_genes" in (drug_out / "scoring-provenance.json").read_text()


def test_scoring_rejects_prediction_controls(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(scoring.shutil, "which", lambda _: "cell-eval2")
    obs = pd.DataFrame({"target_gene": ["non-targeting"], "context": ["c"]}, index=["a"])
    data = ad.AnnData(sp.csr_matrix(np.ones((1, 1))), obs=obs)
    pred = tmp_path / "pred.h5ad"
    ref = tmp_path / "ref.h5ad"
    data.write_h5ad(pred)
    data.write_h5ad(ref)
    with pytest.raises(ValueError, match="without control rows"):
        scoring.score_local(pred, ref, tmp_path / "score")


def test_scoring_rejects_invalid_counts_missing_genes_and_control_leakage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(scoring.shutil, "which", lambda _: "cell-eval2")
    genes = pd.DataFrame(index=["G0", "G1"])
    pred_obs = pd.DataFrame(
        {"target_gene": ["G0"], "context": ["c"], "source_control_id": ["input"]},
        index=["prediction"],
    )
    ref_obs = pd.DataFrame(
        {"target_gene": ["G0", "non-targeting"], "context": ["c", "c"]},
        index=["heldout", "reference-control"],
    )
    prediction = ad.AnnData(sp.csr_matrix([[2.0, 3.0]]), obs=pred_obs, var=genes)
    reference = ad.AnnData(sp.csr_matrix([[2.0, 3.0], [1.0, 2.0]]), obs=ref_obs, var=genes)
    pred = tmp_path / "prediction.h5ad"
    ref = tmp_path / "reference.h5ad"
    reference.write_h5ad(ref)
    prediction.X.data[0] = 0.5
    prediction.write_h5ad(pred)
    with pytest.raises(ValueError, match="raw integer"):
        scoring.score_local(pred, ref, tmp_path / "fractional")
    prediction.X.data[0] = 2.0
    prediction.obs["source_control_id"] = "reference-control"
    prediction.write_h5ad(pred)
    with pytest.raises(ValueError, match="overlap scoring"):
        scoring.score_local(pred, ref, tmp_path / "leaked")
    prediction.obs["source_control_id"] = "input"
    prediction.obs["target_gene"] = "missing"
    reference.obs.loc["heldout", "target_gene"] = "missing"
    prediction.write_h5ad(pred)
    reference.write_h5ad(ref)
    with pytest.raises(ValueError, match="absent from the gene axis"):
        scoring.score_local(pred, ref, tmp_path / "missing-gene")


def test_undefined_metrics_are_reported(tmp_path: Path) -> None:
    aggregate = tmp_path / "agg_results.csv"
    values = {metric: 0.5 for metric in scoring.SCORED_METRICS}
    values["de_wilcoxon_lfc_nmae"] = np.nan
    pd.DataFrame([{"statistic": "mean", **values}]).to_csv(aggregate, index=False)
    raw, undefined = scoring._read_raw_metrics(aggregate)
    assert undefined == ["de_wilcoxon_lfc_nmae"]
    assert raw["de_wilcoxon_lfc_nmae"] is None


def test_diagnostic_metric_is_explicitly_labeled(tmp_path: Path, monkeypatch) -> None:
    genes = pd.DataFrame(index=["G0", "G1"])
    prediction = ad.AnnData(
        sp.csr_matrix([[2, 3]]),
        obs=pd.DataFrame({"target_gene": ["G0"], "context": ["c"]}, index=["prediction"]),
        var=genes,
    )
    reference = ad.AnnData(
        sp.csr_matrix([[2, 3], [1, 2]]),
        obs=pd.DataFrame(
            {"target_gene": ["G0", "non-targeting"], "context": ["c", "c"]},
            index=["heldout", "reference-control"],
        ),
        var=genes,
    )
    pred = tmp_path / "prediction.h5ad"
    ref = tmp_path / "reference.h5ad"
    prediction.write_h5ad(pred)
    reference.write_h5ad(ref)
    monkeypatch.setattr(scoring.shutil, "which", lambda _: "cell-eval2")
    metric = "expr_mse_unbiased_capped"

    def fake_run(command: list[str]) -> None:
        assert f'metrics=["{metric}"]' in command
        run = Path(command[command.index("-o") + 1])
        run.mkdir()
        pd.DataFrame([{"statistic": "mean", metric: -0.01}]).to_csv(
            run / "agg_results.csv", index=False,
        )

    monkeypatch.setattr(scoring, "_run", fake_run)
    result = scoring.score_local(pred, ref, tmp_path / "score", metrics=(metric,))
    assert result["diagnostic_metrics"] == [metric]
    assert result["raw_metrics"]["c"][metric] == -0.01
    assert result["official_leaderboard_score"] is False
    assert "diagnostic metrics are not scored" in result["notes"]
    with pytest.raises(ValueError, match="Six-metric score calibration"):
        scoring.score_local(pred, ref, tmp_path / "bad", metrics=(metric,), calibrate=True)


def test_scoring_detects_input_changes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    genes = pd.DataFrame(index=["G0", "G1"])
    pred_obs = pd.DataFrame({"target_gene": ["G0"], "context": ["c"]}, index=["prediction"])
    ref_obs = pd.DataFrame(
        {"target_gene": ["G0", "non-targeting"], "context": ["c", "c"]},
        index=["heldout", "reference-control"],
    )
    pred = tmp_path / "prediction.h5ad"
    ref = tmp_path / "reference.h5ad"
    ad.AnnData(sp.csr_matrix([[2, 3]]), obs=pred_obs, var=genes).write_h5ad(pred)
    ad.AnnData(sp.csr_matrix([[2, 3], [1, 2]]), obs=ref_obs, var=genes).write_h5ad(ref)
    monkeypatch.setattr(scoring.shutil, "which", lambda _: "cell-eval2")

    def fake_run(command: list[str]) -> None:
        run = Path(command[command.index("-o") + 1])
        run.mkdir()
        pd.DataFrame([{
            "statistic": "mean", **{metric: 0.5 for metric in scoring.SCORED_METRICS},
        }]).to_csv(run / "agg_results.csv", index=False)
        with pred.open("ab") as stream:
            stream.write(b"changed")

    monkeypatch.setattr(scoring, "_run", fake_run)
    out = tmp_path / "score"
    with pytest.raises(RuntimeError, match="inputs changed"):
        scoring.score_local(pred, ref, out)
    assert not (out / "scoring-provenance.json").exists()


def test_scoring_allows_individual_zero_cells_but_rejects_empty_groups_and_overflow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    genes = pd.DataFrame(index=["G0", "G1"])
    pred_obs = pd.DataFrame(
        {"target_gene": ["G0", "G0"], "context": ["c", "c"]}, index=["prediction1", "prediction2"],
    )
    ref_obs = pd.DataFrame(
        {"target_gene": ["G0", "non-targeting"], "context": ["c", "c"]},
        index=["heldout", "reference-control"],
    )
    pred = tmp_path / "prediction.h5ad"
    ref = tmp_path / "reference.h5ad"
    prediction = ad.AnnData(sp.csr_matrix([[0, 0], [1, 2]]), obs=pred_obs, var=genes)
    reference = ad.AnnData(sp.csr_matrix([[2, 3], [1, 2]]), obs=ref_obs, var=genes)
    reference.write_h5ad(ref)
    monkeypatch.setattr(scoring.shutil, "which", lambda _: "cell-eval2")

    def fake_run(command: list[str]) -> None:
        run = Path(command[command.index("-o") + 1])
        run.mkdir()
        pd.DataFrame([{
            "statistic": "mean", **{metric: 0.5 for metric in scoring.SCORED_METRICS},
        }]).to_csv(run / "agg_results.csv", index=False)

    monkeypatch.setattr(scoring, "_run", fake_run)
    prediction.write_h5ad(pred)
    scoring.score_local(pred, ref, tmp_path / "valid")
    prediction.X = sp.csr_matrix((2, 2), dtype=np.int32)
    prediction.write_h5ad(pred)
    with pytest.raises(ValueError, match="entire context/perturbation"):
        scoring.score_local(pred, ref, tmp_path / "empty-group")
    prediction.X = sp.csr_matrix([[0, 0], [1_000_000, 1]])
    prediction.write_h5ad(pred)
    with pytest.raises(ValueError, match="1,000,000"):
        scoring.score_local(pred, ref, tmp_path / "overflow")


def test_batch_stratified_scoring_uses_only_matched_controls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    obs = pd.DataFrame(
        {
            "target_gene": ["drug-a", "drug-b", "drug-a", "drug-b"],
            "context": ["line"] * 4,
            "batch": ["plate1", "plate1", "plate2", "plate2"],
        },
        index=["a1", "b1", "a2", "b2"],
    )
    controls = pd.DataFrame(
        {
            "target_gene": ["non-targeting", "non-targeting"],
            "context": ["line", "line"],
            "batch": ["plate1", "plate2"],
        },
        index=["c1", "c2"],
    )
    ad.AnnData(sp.csr_matrix(np.ones((4, 2))), obs=obs).write_h5ad(tmp_path / "pred.h5ad")
    ad.AnnData(
        sp.csr_matrix(np.ones((6, 2))), obs=pd.concat([obs, controls]),
    ).write_h5ad(tmp_path / "ref.h5ad")
    monkeypatch.setattr(scoring.shutil, "which", lambda _: "cell-eval2")
    seen = []

    def fake_run(command: list[str]) -> None:
        prediction = ad.read_h5ad(command[command.index("-ap") + 1])
        reference = ad.read_h5ad(command[command.index("-ar") + 1])
        assert prediction.obs["batch"].nunique() == reference.obs["batch"].nunique() == 1
        assert set(prediction.obs["batch"]) == set(reference.obs["batch"])
        seen.append(reference.obs["batch"].iloc[0])
        run = Path(command[command.index("-o") + 1])
        run.mkdir()
        pd.DataFrame([{
            "statistic": "mean", **{metric: 0.5 for metric in scoring.SCORED_METRICS}
        }]).to_csv(run / "agg_results.csv", index=False)

    monkeypatch.setattr(scoring, "_run", fake_run)
    result = scoring.score_local(
        tmp_path / "pred.h5ad", tmp_path / "ref.h5ad", tmp_path / "score",
        no_target_genes=True, stratify_by_batch=True,
    )
    assert set(seen) == {"plate1", "plate2"}
    assert len(result["raw_metrics"]) == 2
    assert result["stratify_by_batch"]


def test_batch_stratified_scoring_requires_batch_labels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(scoring.shutil, "which", lambda _: "cell-eval2")
    obs = pd.DataFrame({"target_gene": ["drug"], "context": ["line"]}, index=["a"])
    data = ad.AnnData(sp.csr_matrix(np.ones((1, 1))), obs=obs)
    data.write_h5ad(tmp_path / "pred.h5ad")
    data.write_h5ad(tmp_path / "ref.h5ad")
    with pytest.raises(ValueError, match="non-null batch"):
        scoring.score_local(
            tmp_path / "pred.h5ad", tmp_path / "ref.h5ad", tmp_path / "score",
            stratify_by_batch=True,
        )
