import json
from pathlib import Path

import pytest

from c3vcell.research_review import _check_score_provenance, compare_scores
from c3vcell.scoring import SCORED_METRICS


def _metrics(mse: float, pds: float) -> dict:
    return {
        "pds_cosine": pds,
        "expr_mse_unbiased_capped_norm": mse,
        "de_wilcoxon_direction_fidelity_yield_raw": 0.4,
        "de_wilcoxon_direction_reach_raw": None,
        "de_wilcoxon_sig_jaccard": 0.0,
        "de_wilcoxon_lfc_nmae": None,
    }


def test_score_comparison_preserves_direction_and_undefined_values() -> None:
    candidate = {"HCT116": _metrics(12, 0.6)}
    baseline = {"HCT116": _metrics(10, 0.5)}
    rows = compare_scores(candidate, {"no_response": baseline})["HCT116"]
    assert rows["expr_mse_unbiased_capped_norm"]["improvement_vs_no_response"] == -2
    assert rows["pds_cosine"]["improvement_vs_no_response"] == pytest.approx(0.1)
    assert rows["de_wilcoxon_direction_reach_raw"]["improvement_vs_no_response"] is None


@pytest.mark.parametrize("bad", [
    {"HCT116": {"pds_cosine": 0.5}},
    {"HCT116": {**_metrics(10, 0.5), "pds_cosine": float("nan")}},
    {"HCT116": {**_metrics(10, 0.5), "pds_cosine": True}},
])
def test_score_comparison_rejects_invalid_metrics(bad: dict) -> None:
    with pytest.raises(ValueError):
        compare_scores(bad, {"no_response": {"HCT116": _metrics(10, 0.5)}})
    with pytest.raises(ValueError, match="contexts differ"):
        compare_scores({"HCT116": _metrics(10, 0.5)}, {"other": {"K562": _metrics(10, 0.5)}})


def test_score_provenance_rejects_official_or_undefined_mislabeling(tmp_path: Path) -> None:
    raw = {"HCT116": _metrics(10, 0.5)}
    undefined = {"HCT116": ["de_wilcoxon_direction_reach_raw", "de_wilcoxon_lfc_nmae"]}
    metadata = {
        "implementation": "cell-eval2", "required_version": "0.22.0", "preset": "vcc2026",
        "metrics": list(SCORED_METRICS), "no_target_genes": False, "official_leaderboard_score": False,
        "local_reference_calibration": False, "prediction_sha256": "prediction",
        "reference_sha256": "reference", "raw_metrics": raw, "undefined_metrics": undefined,
    }
    path = tmp_path / "scoring-provenance.json"
    path.write_text(json.dumps(metadata), encoding="utf-8")
    _check_score_provenance(
        path, prediction_sha256="prediction", reference_sha256="reference",
        raw_metrics=raw, undefined_metrics=undefined,
    )
    metadata["official_leaderboard_score"] = True
    path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="Scoring provenance"):
        _check_score_provenance(
            path, prediction_sha256="prediction", reference_sha256="reference",
            raw_metrics=raw, undefined_metrics=undefined,
        )
    metadata["official_leaderboard_score"] = False
    metadata["undefined_metrics"] = {"HCT116": []}
    path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="Undefined-metric"):
        _check_score_provenance(
            path, prediction_sha256="prediction", reference_sha256="reference",
            raw_metrics=raw, undefined_metrics=None,
        )
