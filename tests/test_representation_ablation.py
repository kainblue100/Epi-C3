import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch


def test_representation_ablation_is_reproducible_and_refuses_overwrite(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts" / "run_representation_ablation.py"
    source = tmp_path / "source.h5ad"
    matrix = np.random.default_rng(42).poisson(3, size=(12, 6)).astype(np.int32) + 1
    data = ad.AnnData(
        sp.csr_matrix(matrix),
        obs=pd.DataFrame(
            {"target_gene": [f"target-{i % 3}" for i in range(len(matrix))]},
            index=[f"cell-{i}" for i in range(len(matrix))],
        ),
        var=pd.DataFrame(index=[f"gene-{i}" for i in range(matrix.shape[1])]),
    )
    data.write_h5ad(source)
    output = tmp_path / "ablation"
    command = [
        sys.executable, str(script), "--source", str(source), "--outdir", str(output),
        "--weights", "0.1", "1", "--covariance-weights", "0", "0.05",
        "--regularization-views", "unmasked", "--pretrain-steps", "2", "--batch-size", "4",
        "--latent-dim", "4", "--hidden-dim", "6", "--validation-cells", "4",
        "--train-diagnostic-cells", "4", "--mask-cells", "2", "--mask-repeats", "2",
        "--threads", "1", "--seed", "7",
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root / "src") + os.pathsep + env.get("PYTHONPATH", "")
    completed = subprocess.run(command, env=env, capture_output=True, text=True, check=True)
    assert "completed_configuration" in completed.stdout
    protocol = json.loads((output / "protocol.json").read_text(encoding="utf-8"))
    result = json.loads((output / "result.json").read_text(encoding="utf-8"))
    assert len(set(protocol["validation_cell_ids"])) == 4
    with zipfile.ZipFile(output / "code-snapshot.zip") as snapshot:
        assert "src/c3vcell/model.py" in snapshot.namelist()
    assert [
        (row["variance_weight"], row["covariance_weight"], row["regularization_view"])
        for row in result["results"]
    ] == [(0.1, 0.0, "unmasked"), (0.1, 0.05, "unmasked"),
          (1.0, 0.0, "unmasked"), (1.0, 0.05, "unmasked")]
    for index, row in enumerate(result["results"]):
        state = torch.load(
            output / f"weight-{index:02d}" / "representation-only.pt",
            map_location="cpu", weights_only=True,
        )
        assert state["kind"] == "representation-only-v1"
        assert state["model_config"]["variance_weight"] == row["variance_weight"]
        assert state["model_config"]["covariance_weight"] == row["covariance_weight"]
        assert state["model_config"]["regularization_view"] == row["regularization_view"]
        assert row["validation_diagnostic"]["sampled_cells"] == 4
        assert row["mask_diagnostic"]["independent_masks_per_cell"] == 2
        assert np.isfinite(row["validation_unmasked_nb"])
    audit_path = tmp_path / "audit.json"
    subprocess.run(
        [
            sys.executable, str(root / "scripts" / "audit_representation_ablation.py"),
            "--source", str(source), "--run", str(output), "--output", str(audit_path),
            "--batch-size", "4", "--permutations", "8", "--seed", "7",
        ],
        env=env, capture_output=True, text=True, check=True,
    )
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    assert audit["diagnostic_cells"] == 4
    for row in audit["results"]:
        assert 0 <= row["first_component_variance_fraction"] <= 1
        assert row["participation_rank"] >= 1
        assert row["associations"]["pc1_by_target_gene"]["categories"] >= 1
    repeated = subprocess.run(command, env=env, capture_output=True, text=True, check=False)
    assert repeated.returncode != 0
    assert "Refusing to overwrite" in repeated.stderr
