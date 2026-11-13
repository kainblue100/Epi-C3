"""Verify Tahoe foundation stream replay and exclusion on an actual protected row."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import epi_c3_foundation_tahoe_v2 as adapter
import epi_c3_tahoe_isolation_v1 as isolation
from epi_c3_shared_foundation_v1 import gene_axis_digest


def main(args) -> None:
    if args.output.exists():
        raise FileExistsError(args.output)
    audit_path = args.run_root / "tahoe-foundation-isolation-v1/audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    plan_path = args.run_root / "rna-foundation-stage1-v1/plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    genes = plan["genes"]
    if gene_axis_digest(genes) != plan["gene_axis_sha256"]:
        raise ValueError("Shared foundation gene axis changed")
    spec = importlib.util.spec_from_file_location(
        "tahoe_smoke_audit", args.run_root / "audit-tahoe-foundation-isolation-v1.py",
    )
    if spec is None or spec.loader is None:
        raise ImportError("Cannot load the audited Tahoe source")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    tahoe = module.load_tahoe(args.run_root)
    protected_dir = args.run_root / "tahoe-protected-identities-v1"
    audit_dir = args.run_root / "tahoe-foundation-isolation-v1"
    metadata_path = args.run_root / "tahoe-full-metadata-v2/metadata-plan.json"
    frame = pd.read_parquet(args.root / "metadata/gene_metadata.parquet")
    stream = adapter.TahoeStream(
        args.root, args.run_root, metadata_path, protected_dir, audit_dir, frame, genes,
        tahoe.support, tahoe, batch_size=audit["batch_size"], seed=audit["order_seed"],
    )
    first = stream.next_batch()
    cursor = stream.cursor()
    second = stream.next_batch()
    stream.set_cursor(cursor)
    replay = stream.next_batch()
    if first is None or second is None or replay is None or (second != replay).nnz:
        raise ValueError("Real Tahoe stream checkpoint replay changed the next batch")

    _, protected_records = adapter.load_protected(args.run_root, protected_dir, metadata_path)
    progress = json.loads((audit_dir / "progress.private.json").read_text(encoding="utf-8"))
    protected_ids = set(protected_records)
    validation_ids = stream.validation_ids
    candidate = next(
        (item for cell_id, item in sorted(protected_records.items())
         if cell_id in progress["protected_eligible"] and item["legacy_source_files"]),
        None,
    )
    if candidate is None:
        raise ValueError("No protected eligible legacy identity is available for the real-data smoke test")
    relative = candidate["legacy_source_files"][0]
    file_index = next(index for index, item in enumerate(stream.entries) if item["path"] == relative)
    path = isolation.check_shard(args.root, stream.entries[file_index])
    found = None
    for batch_index, batch in enumerate(pq.ParquetFile(path).iter_batches(
        batch_size=audit["batch_size"], columns=tahoe.scanner.COLUMNS,
    )):
        samples = batch.column("sample").to_pylist()
        barcodes = batch.column("BARCODE_SUB_LIB_ID").to_pylist()
        selected = tahoe.select_training(
            batch, stream.metadata, stream.structures, stream.aliases, stream.metadata["config"]["seed"],
        )
        ids = [isolation.identity(samples[index], barcodes[index]) for index in selected]
        if candidate["cell_id"] in ids:
            kept = np.asarray([
                int(index) for index, cell_id in zip(selected, ids, strict=True)
                if cell_id not in protected_ids and cell_id not in validation_ids
            ], dtype=np.int64)
            if not len(kept):
                raise ValueError("The chosen protected batch has no remaining training rows")
            found = batch_index, len(selected), kept, sum(cell_id in protected_ids for cell_id in ids)
            break
    if found is None:
        raise ValueError("The audited eligible protected row was not found in its declared shard")
    batch_index, selected_before, kept, protected_in_batch = found
    full_iterator = pq.ParquetFile(path).iter_batches(batch_size=audit["batch_size"], columns=tahoe.COLUMNS)
    for _ in range(batch_index):
        next(full_iterator)
    full_batch = next(full_iterator)
    expected = tahoe.counts_from_batch(full_batch, kept, stream.mapping, stream.gene_count)
    stream.set_cursor({"file_position": file_index, "batch_position": batch_index, "returned_cells": 0})
    actual = stream.next_batch()
    if actual is None or actual.shape != expected.shape or (actual != expected).nnz:
        raise ValueError("A protected real Tahoe identity was not excluded exactly")
    stream.close()
    args.output.mkdir()
    result = {
        "format": "tahoe-foundation-stream-smoke-v2", "state": "complete",
        "audit_sha256": isolation.sha256(audit_path),
        "foundation_plan_sha256": isolation.sha256(plan_path),
        "gene_count": len(genes), "measured_genes": int(stream.measured.sum()),
        "validation_shape": list(stream.validation.shape),
        "audited_training_rows": stream.expected_cells,
        "first_batch_rows": first.shape[0], "replayed_batch_rows": replay.shape[0],
        "cursor_replay_exact": True,
        "targeted_batch_eligible_before_exclusions": selected_before,
        "targeted_batch_protected_rows": protected_in_batch,
        "targeted_batch_emitted_rows": actual.shape[0],
        "targeted_exclusion_matches_independent_identity_filter": True,
        "scope": "Real-data loader smoke test only; no model training was performed.",
    }
    with (args.output / "smoke.json").open("x", encoding="utf-8") as file:
        json.dump(result, file, indent=2, allow_nan=False)
        file.write("\n")
    print(json.dumps(result))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args())
