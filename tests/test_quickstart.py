import ast
import json
import os
import subprocess
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import scipy.sparse as sp


ROOT = Path(__file__).resolve().parents[1]


def test_toy_data_is_deterministic_and_does_not_overwrite(tmp_path: Path) -> None:
    script = ROOT / "examples" / "make_toy_data.py"
    first, second = tmp_path / "first.h5ad", tmp_path / "second.h5ad"
    for output in (first, second):
        subprocess.run(
            [sys.executable, str(script), "--output", str(output)],
            check=True, capture_output=True, text=True,
        )
    a, b = ad.read_h5ad(first), ad.read_h5ad(second)
    assert a.shape == (360, 64)
    assert a.obs_names.is_unique and a.var_names.is_unique
    assert set(a.obs["target_gene"]) == {"non-targeting", *(f"G{i}" for i in range(6))}
    assert {"SP1", "USF1"} <= set(a.var_names)
    assert a.var["ensembl_ids"].is_unique and a.uns["c3vcell_genome_assembly"] == "GRCh38"
    assert (sp.csr_matrix(a.X) != sp.csr_matrix(b.X)).nnz == 0
    assert np.isfinite(a.X.data).all() and (a.X.data >= 0).all()
    assert np.equal(a.X.data, np.floor(a.X.data)).all()
    refused = subprocess.run(
        [sys.executable, str(script), "--output", str(first)],
        capture_output=True, text=True,
    )
    assert refused.returncode != 0 and "Refusing to overwrite" in refused.stderr


def test_quickstart_notebook_executes_without_jupyter(tmp_path: Path) -> None:
    notebook = json.loads((ROOT / "examples" / "quickstart.ipynb").read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    code = []
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            assert cell["outputs"] == [] and cell["execution_count"] is None
            source = "".join(cell["source"])
            ast.parse(source)
            code.append(source)
    work = tmp_path / "notebook-run"
    env = {
        **os.environ,
        "C3_TUTORIAL_RUN_DIR": str(work),
        "OMP_NUM_THREADS": "2",
        "MKL_NUM_THREADS": "2",
        "OPENBLAS_NUM_THREADS": "2",
    }
    completed = subprocess.run(
        [sys.executable, "-c", "\n\n".join(code)], cwd=ROOT / "examples", env=env,
        capture_output=True, text=True, timeout=300,
    )
    assert completed.returncode == 0, completed.stdout + "\n" + completed.stderr
    assert ad.read_h5ad(work / "prediction.h5ad").shape == (40, 64)
    for name in ("prediction", "control"):
        provenance = json.loads((work / f"{name}-score" / "scoring-provenance.json").read_text())
        assert provenance["preset"] == "vcc2026"
        assert provenance["official_leaderboard_score"] is False
    manifest = json.loads((work / "epigenome" / "manifest.json").read_text())
    assert manifest["format_version"] == 2 and manifest["edge_count"] > 0
    assert (work / "epigenome" / "features.csv").read_bytes() == (
        work / "graph-shuffle" / "features.csv"
    ).read_bytes()
    assert (work / "epi-model" / "model.pt").is_file()
