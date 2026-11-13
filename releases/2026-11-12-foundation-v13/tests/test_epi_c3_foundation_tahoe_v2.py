import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import scipy.sparse as sp

import epi_c3_foundation_tahoe_v2 as adapter
import epi_c3_tahoe_isolation_v1 as isolation


spec = importlib.util.spec_from_file_location(
    "audit_tahoe", Path(__file__).parent.parent / "audit-tahoe-foundation-isolation-v1.py",
)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class FakeHelper:
    __file__ = __file__

    @staticmethod
    def create_token_map(tokens):
        result = np.full(max(tokens) + 1, -1, dtype=np.int32)
        for token, value in tokens.items():
            result[token] = value
        return result


class FakeTahoe:
    scanner = SimpleNamespace(COLUMNS=["sample", "BARCODE_SUB_LIB_ID", "value"])
    COLUMNS = scanner.COLUMNS

    @staticmethod
    def treatment_split(metadata, frame):
        return {}, set(), {"eligible_rows": metadata["rows"]}

    @staticmethod
    def select_training(batch, metadata, structures, aliases, seed):
        return np.flatnonzero(np.asarray(batch.column("value").to_pylist()) > 0)

    @staticmethod
    def counts_from_batch(batch, selected, mapping, n_genes):
        values = np.asarray(batch.column("value").to_pylist())[selected]
        result = np.zeros((len(selected), n_genes), dtype=np.float32)
        result[:, mapping[5]] = values
        return sp.csr_matrix(result)


def build_stream(tmp_path, monkeypatch):
    root = tmp_path / "source"
    run_root = tmp_path / "runs" / "current"
    protected_dir = run_root / "protected"
    audit_dir = run_root / "audit"
    for path in (root / "data", root / "metadata", protected_dir, audit_dir):
        path.mkdir(parents=True)
    frame = pd.DataFrame({"ensembl_id": ["ENSG1"], "token_id": [5]})
    frame.to_parquet(root / "metadata/gene_metadata.parquet")
    pd.DataFrame({"drug": ["dummy"]}).to_parquet(root / "metadata/drug_metadata.parquet")
    for name, cells in (("a.parquet", list(range(40))), ("b.parquet", [0, 4])):
        pq.write_table(pa.table({
            "sample": ["smp_1"] * len(cells),
            "BARCODE_SUB_LIB_ID": [f"cell-{index}" for index in cells],
            "value": [index + 1 for index in cells],
        }), root / "data" / name, row_group_size=4)
    entries = []
    for path in sorted((root / "data").glob("*.parquet")):
        stat = path.stat()
        entries.append({"path": f"data/{path.name}", "size": stat.st_size, "mtime_ns": stat.st_mtime_ns})
    metadata = {"state": "complete", "rows": 42, "config": {"seed": 17}, "provenance": {"files": entries}}
    metadata_path = run_root / "metadata-plan.json"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    (run_root / "audit-tahoe-foundation-isolation-v1.py").write_text("frozen", encoding="utf-8")
    (protected_dir / "manifest.json").write_text("frozen", encoding="utf-8")
    protected_id = isolation.identity("smp_1", "cell-4")
    protected_record = {
        "cell_id": protected_id, "sample": "smp_1", "barcode": "cell-4",
        "legacy_source_files": ["data/a.parquet", "data/b.parquet"],
    }
    protected_records = {protected_id: protected_record}
    monkeypatch.setattr(adapter, "load_protected", lambda *args: (
        {"protected_unique_identities": 1}, protected_records,
    ))
    monkeypatch.setattr(adapter, "source_dependencies", lambda *args: {})
    ordered = adapter.ordered_entries(metadata, 5)
    records, _ = adapter.choose_validation(
        root, metadata, FakeTahoe, ordered, {"smp_1": {"cell-4"}}, {}, set(), 4,
    )
    assert len(records) == 32
    assert all(record["cell_id"] != protected_id for record in records)
    assert isolation.identity("smp_1", "cell-0") in {record["cell_id"] for record in records}
    validation_path = audit_dir / "validation.private.json"
    validation_path.write_text(json.dumps({
        "format": "tahoe-foundation-validation-private-v1", "records": records,
    }), encoding="utf-8")
    result = {
        "format": "tahoe-foundation-isolation-audit-v1", "state": "complete",
        "metadata_plan_sha256": isolation.sha256(metadata_path),
        "protected_manifest_sha256": isolation.sha256(protected_dir / "manifest.json"),
        "source_entries_sha256": isolation.digest_json(entries),
        "ordered_source_paths_sha256": isolation.digest_json([item["path"] for item in ordered]),
        "batch_size": 4, "order_seed": 5, "files_scanned": 2, "source_rows_checked": 42,
        "legacy_source_provenance_verified": True,
        "dependencies": {"source_helper_sha256": isolation.sha256(Path(FakeHelper.__file__))},
        "audit_script_sha256": isolation.sha256(run_root / "audit-tahoe-foundation-isolation-v1.py"),
        "validation_private_file": validation_path.name,
        "validation_private_sha256": isolation.sha256(validation_path),
        "eligible_rows_before_exclusions": 42, "protected_eligible_rows": 2,
        "validation_eligible_rows": 33, "eligible_training_rows_after_exclusions": 7,
    }
    (audit_dir / "audit.json").write_text(json.dumps(result), encoding="utf-8")
    return root, run_root, metadata_path, protected_dir, audit_dir, frame, metadata, protected_records, records


def instantiate(parts, *, seed=5):
    root, run_root, metadata_path, protected_dir, audit_dir, frame, *_ = parts
    return adapter.TahoeStream(
        root, run_root, metadata_path, protected_dir, audit_dir, frame, ["ENSG1", "ENSG2"],
        FakeHelper, FakeTahoe, batch_size=4, seed=seed,
    )


def test_full_shard_audit_counts_duplicate_identity_rows(tmp_path, monkeypatch):
    parts = build_stream(tmp_path, monkeypatch)
    root, _, _, _, _, _, metadata, protected, validation = parts
    pairs = {key: (value["sample"], value["barcode"]) for key, value in protected.items()}
    validation_pairs = {row["cell_id"]: (row["sample"], row["barcode"]) for row in validation}
    lookup = {pair: (cell_id, "protected") for cell_id, pair in pairs.items()}
    lookup.update({pair: (cell_id, "validation") for cell_id, pair in validation_pairs.items()})
    progress = {
        "source_rows_checked": 0, "row_groups_scanned": 0, "row_groups_pruned": 0,
        "protected_seen": {}, "validation_seen": {}, "protected_eligible": {},
        "validation_eligible": {}, "legacy_seen": {},
    }
    expected_sources = {(next(iter(protected)), "data/a.parquet"), (next(iter(protected)), "data/b.parquet")}
    for entry in metadata["provenance"]["files"]:
        audit.scan_one_file(root, entry, metadata, FakeTahoe, {}, set(), lookup, ["smp_1"], expected_sources, progress)
    assert progress["source_rows_checked"] == 42
    assert sum(progress["protected_eligible"].values()) == 2
    assert sum(progress["validation_eligible"].values()) == 33
    assert {(cell_id, path) for cell_id, paths in progress["legacy_seen"].items() for path in paths} == expected_sources


def test_stream_excludes_all_duplicate_rows_and_replays_cursor(tmp_path, monkeypatch):
    stream = instantiate(build_stream(tmp_path, monkeypatch))
    assert stream.validation.shape == (32, 2)
    assert stream.expected_cells == 7
    first = stream.next_batch()
    cursor = stream.cursor()
    following = stream.next_batch()
    stream.set_cursor(cursor)
    replay = stream.next_batch()
    assert following is not None
    assert (following != replay).nnz == 0
    total = first.shape[0] + replay.shape[0]
    while True:
        batch = stream.next_batch()
        if batch is None:
            break
        total += batch.shape[0]
    assert total == 7
    with pytest.raises(ValueError, match="cursor"):
        stream.set_cursor({"file_position": 2, "batch_position": 0, "returned_cells": 1})


def test_stream_rejects_stale_audit_and_changed_shards(tmp_path, monkeypatch):
    parts = build_stream(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="stale"):
        instantiate(parts, seed=6)
    source = parts[0] / "data/a.parquet"
    with source.open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="source changed"):
        instantiate(parts)
