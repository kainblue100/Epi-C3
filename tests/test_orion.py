import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from c3vcell.data import CONTROL
from c3vcell.orion import inspect_shards, prepare_orion


def test_orion_sampling_and_symbol_aggregation(tmp_path: Path) -> None:
    metadata = pd.DataFrame({
        "gene_token_id": [1, 2, 3, 4],
        "gene_name": ["G0", "G0", "G1", "G2"],
        "ensembl_id": ["E1", "E2", "E3", "E4"],
    })
    meta = tmp_path / "metadata.parquet"
    metadata.to_parquet(meta)
    rows = []
    for label in ["control", "G0", "G1", "G2"]:
        for i in range(4):
            rows.append({
                "gene_token_id": [1, 2, 3, 4],
                "gene_expression": [1, 2, 3, 4],
                "cell_barcode": f"{label}-{i}",
                "sample": "S1",
                "guide_target": label,
                "gene_target": label,
                "pass_guide_filter": int(i != 3),
            })
    shard = tmp_path / "HCT116_part.parquet"
    pd.DataFrame(rows).to_parquet(shard)
    inspected = inspect_shards([shard])
    assert inspected["n_cells"].sum() == 12
    output = tmp_path / "orion.h5ad"
    prepare_orion(
        [shard], meta, output, targets=["G0", "G1", "G2"],
        control_label="control", max_per_target=2, max_controls=3,
    )
    result = ad.read_h5ad(output)
    assert result.shape == (9, 3)
    assert result.var_names.tolist() == ["G0", "G1", "G2"]
    assert result.var.loc["G0", "ensembl_ids"] == "E1|E2"
    assert (result.X.toarray()[0] == [3, 3, 4]).all()
    assert set(result.obs["target_gene"]) == {CONTROL, "G0", "G1", "G2"}
    assert result.obs_names.str.startswith("orion-v1-").all()
    assert result.uns["c3vcell_orion"]["cell_identity"].startswith("orion-v1")
    reordered = tmp_path / "reordered.h5ad"
    prepare_orion(
        [shard], meta, reordered, targets=["G2", "G1", "G0"],
        control_label="control", max_per_target=3, max_controls=3,
    )
    other = ad.read_h5ad(reordered)
    original_by_barcode = result.obs.reset_index().set_index("cell_barcode")["cell_id"]
    reordered_by_barcode = other.obs.reset_index().set_index("cell_barcode")["cell_id"]
    for barcode, cell_id in original_by_barcode.items():
        assert reordered_by_barcode[barcode] == cell_id
    with pytest.raises(ValueError, match="more than once"):
        prepare_orion(
            [shard, shard], meta, tmp_path / "duplicate.h5ad",
            targets=["G0"], control_label="control",
        )
    copied_shard = tmp_path / "HCT116_copy.parquet"
    pd.DataFrame(rows).to_parquet(copied_shard)
    with pytest.raises(ValueError, match="Duplicate Orion cell identifier"):
        prepare_orion(
            [shard, copied_shard], meta, tmp_path / "duplicate-cell.h5ad",
            targets=["G0"], control_label="control", max_per_target=10, max_controls=10,
        )


def test_batched_reservoir_matches_original_rowwise_sampling(tmp_path: Path) -> None:
    metadata = tmp_path / "metadata.parquet"
    pd.DataFrame({
        "gene_token_id": [1, 2, 3],
        "gene_name": ["G0", "G1", "G2"],
        "ensembl_id": ["E1", "E2", "E3"],
    }).to_parquet(metadata)
    labels = ["control", "G0", "G1", "G2"]
    rows = [
        {
            "gene_token_id": [1, 2, 3], "gene_expression": [1, 2, 3],
            "cell_barcode": f"cell-{i:04d}", "sample": "S1",
            "guide_target": labels[i % 4], "gene_target": labels[i % 4],
            "pass_guide_filter": int(i % 11 != 0),
        }
        for i in range(2500)
    ]
    shard = tmp_path / "HCT116_part.parquet"
    pd.DataFrame(rows).to_parquet(shard)
    seed = 12
    expected: dict[str, list[str]] = {label: [] for label in ["G0", "G1", "G2", "control"]}
    seen: Counter[str] = Counter()
    rng = np.random.default_rng(seed)
    for row in rows:
        if not row["pass_guide_filter"]:
            continue
        label = row["gene_target"]
        seen[label] += 1
        limit = 11 if label == "control" else 7
        if len(expected[label]) < limit:
            expected[label].append(row["cell_barcode"])
        else:
            position = int(rng.integers(seen[label]))
            if position < limit:
                expected[label][position] = row["cell_barcode"]
    output = tmp_path / "sample.h5ad"
    prepare_orion(
        [shard], metadata, output, targets=["G0", "G1", "G2"], control_label="control",
        max_per_target=7, max_controls=11, seed=seed,
    )
    result = ad.read_h5ad(output)
    for label, barcodes in expected.items():
        actual = CONTROL if label == "control" else label
        assert result.obs.loc[result.obs.target_gene == actual, "cell_barcode"].tolist() == barcodes


def test_target_selection_uses_counts_and_refuses_overwrite(tmp_path: Path) -> None:
    groups = pd.DataFrame([
        ("HCT116", "control", "S1", 100),
        ("HCT116", "G0", "S1", 50),
        ("HCT116", "G1", "S1", 30),
        ("HCT116", "G2", "S1", 25),
        ("HCT116", "not-a-gene", "S1", 1000),
    ], columns=["context", "gene_target", "sample", "n_cells"])
    group_path = tmp_path / "groups.csv"
    groups.to_csv(group_path, index=False)
    metadata = tmp_path / "metadata.parquet"
    pd.DataFrame({"gene_name": ["G0", "G1", "G2"]}).to_parquet(metadata)
    output = tmp_path / "targets.txt"
    command = [
        sys.executable, str(Path(__file__).parents[1] / "scripts" / "select-orion-targets.py"),
        "--groups", str(group_path), "--gene-metadata", str(metadata),
        "--control-label", "control", "--targets", "3", "--min-cells", "25",
        "--output", str(output),
    ]
    subprocess.run(command, check=True, capture_output=True, text=True)
    assert output.read_text().splitlines() == ["G0", "G1", "G2"]
    summary = json.loads(output.with_suffix(".json").read_text())
    assert summary["target_cells_with_context_sample_controls"] == 105
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode != 0
    assert "Refusing to overwrite" in result.stderr


def test_target_selection_excludes_previous_panel(tmp_path: Path) -> None:
    groups = pd.DataFrame([
        ("HCT116", "control", "S1", 100),
        ("HCT116", "G0", "S1", 50),
        ("HCT116", "G1", "S1", 40),
        ("HCT116", "G2", "S1", 30),
        ("HCT116", "G3", "S1", 20),
    ], columns=["context", "gene_target", "sample", "n_cells"])
    group_path = tmp_path / "groups.csv"
    groups.to_csv(group_path, index=False)
    metadata = tmp_path / "metadata.parquet"
    pd.DataFrame({"gene_name": ["G0", "G1", "G2", "G3"]}).to_parquet(metadata)
    excluded = tmp_path / "old.txt"
    excluded.write_text("G0\n", encoding="utf-8")
    output = tmp_path / "new.txt"
    script = Path(__file__).parents[1] / "scripts" / "select-orion-targets.py"
    subprocess.run([
        sys.executable, str(script), "--groups", str(group_path),
        "--gene-metadata", str(metadata), "--control-label", "control",
        "--targets", "3", "--min-cells", "10",
        "--exclude-targets", str(excluded), "--output", str(output),
    ], check=True, capture_output=True, text=True)
    assert output.read_text().splitlines() == ["G1", "G2", "G3"]
    assert json.loads(output.with_suffix(".json").read_text())["excluded_targets"] == ["G0"]
