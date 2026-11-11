import json
import os
import subprocess
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import torch

from c3vcell.integrity import file_sha256


@pytest.mark.parametrize(
    ("include_null_control", "control_sampling"),
    [(False, "reference-batch"), (True, "input-controls")],
)
def test_nested_arc_script_with_prespecified_exclusions(
    tmp_path: Path, include_null_control: bool, control_sampling: str,
) -> None:
    root = Path(__file__).resolve().parents[1]
    rng = np.random.default_rng(42)
    labels = ["non-targeting"] * 80 + sum(([f"G{i}"] * 24 for i in range(8)), [])
    counts = rng.poisson(12, size=(len(labels), 16)).astype(np.int32) + 1
    for row, label in enumerate(labels):
        if label != "non-targeting":
            target = int(label[1:])
            counts[row, target] = rng.poisson(1)
            counts[row, (target + 1) % 8] += rng.poisson(15)
            counts[row, 10] += rng.poisson(8)
    data = ad.AnnData(
        sp.csr_matrix(counts),
        obs=pd.DataFrame(
            {"target_gene": labels, "context": "local", "batch": "batch1"},
            index=[f"cell-{i}" for i in range(len(labels))],
        ),
        var=pd.DataFrame(index=[f"G{i}" for i in range(16)]),
    )
    source = tmp_path / "source.h5ad"
    data.write_h5ad(source)
    exclusion = tmp_path / "prior.json"
    exclusion.write_text(
        json.dumps({"test_targets": ["G7"], "train_targets": ["G6"]}), encoding="utf-8",
    )
    prior_training = tmp_path / "prior-training.json"
    prior_training.write_text(json.dumps({"training_targets": ["G0", "G1"]}), encoding="utf-8")
    output = tmp_path / "experiment"
    command = [
        sys.executable, str(root / "scripts" / "run_nested_arc.py"),
        "--source", str(source), "--source-sha256", file_sha256(source),
        "--outdir", str(output), "--exclude-manifests", str(exclusion),
        "--exclude-training-targets-manifests", str(exclusion),
        "--exclude-test-training-targets-manifests", str(prior_training),
        "--selection-evidence", str(exclusion), "--selection-note", "Synthetic engineering check",
        "--train-targets", "4", "--test-targets", "2", "--max-per-target", "20",
        "--max-controls", "48", "--cells-per-target", "12", "--folds", "2",
        "--pretrain-steps", "2", "--flow-steps", "2", "--batch-size", "4",
        "--latent-dim", "4", "--hidden-dim", "6", "--variance-weight", "2",
        "--covariance-weight", "0.1", "--regularization-view", "unmasked",
        "--prediction-steps", "2", "--seed", "12", "--threads", "1",
        "--control-sampling", control_sampling,
    ]
    if include_null_control:
        command.append("--include-null-control")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root / "src") + os.pathsep + env.get("PYTHONPATH", "")
    completed = subprocess.run(command, env=env, capture_output=True, text=True, check=True)
    assert "Verified result" in completed.stdout
    protocol = json.loads((output / "protocol.json").read_text(encoding="utf-8"))
    result = json.loads((output / "result.json").read_text(encoding="utf-8"))
    assert protocol["excluded_prior_evaluation_targets"] == ["G7"]
    assert protocol["excluded_prior_training_targets"] == ["G6"]
    assert protocol["excluded_from_test_prior_training_targets"] == ["G0", "G1"]
    assert protocol["eligible_test_targets"] == 4
    assert set(protocol["test_targets"]).isdisjoint({"G0", "G1"})
    assert {"G0", "G1"}.issubset(protocol["training_targets"])
    assert protocol["include_null_control"] is include_null_control
    assert protocol["control_sampling"] == control_sampling
    calibration_command = next(item for item in protocol["commands"] if item["stage"] == "calibration")
    assert ("--include-null-control" in calibration_command["argv"]) is include_null_control
    for stage in ("calibration", "baseline", "mean", "uncalibrated", "calibrated"):
        argv = next(item["argv"] for item in protocol["commands"] if item["stage"] == stage)
        assert argv[argv.index("--control-sampling") + 1] == control_sampling
    assert set(protocol["test_targets"]).isdisjoint(protocol["training_targets"])
    assert set(protocol["test_targets"] + protocol["training_targets"]).isdisjoint({"G6", "G7"})
    assert protocol["code_snapshot_sha256"] == file_sha256(output / "code-snapshot.zip")
    assert set(result["metrics"]) == {"baseline", "mean", "uncalibrated", "calibrated"}
    calibration = json.loads(
        (output / "calibration" / "shared-residual-calibration.json").read_text(encoding="utf-8")
    )
    assert result["selected_shared_strength"] == calibration.get(
        "selected_shared_strength", calibration["shared_strength"]
    )
    for fold in range(2):
        checkpoint = torch.load(
            output / "calibration" / f"fold-{fold:02d}" / "model" / "model.pt",
            map_location="cpu", weights_only=True,
        )
        assert checkpoint["model_config"]["variance_weight"] == 2
        assert checkpoint["model_config"]["covariance_weight"] == 0.1
        assert checkpoint["model_config"]["regularization_view"] == "unmasked"
