"""Audit all Tahoe shards for protected and validation identity overlaps."""

from __future__ import annotations

import argparse
import fcntl
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

import epi_c3_foundation_tahoe_v2 as adapter
import epi_c3_tahoe_isolation_v1 as isolation


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + ".part")
    descriptor = os.open(temporary, os.O_CREAT | os.O_TRUNC | os.O_WRONLY, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def load_tahoe(run_root: Path):
    path = run_root / "stream-tahoe-representation-v1.py"
    spec = importlib.util.spec_from_file_location("tahoe_isolation_source", path)
    if spec is None or spec.loader is None:
        raise ImportError("Cannot load the frozen Tahoe source code")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def scan_one_file(
    root: Path, entry: dict, metadata: dict, tahoe, structures: dict, aliases: set[str],
    lookup: dict, ordered_samples: list[str], legacy_expected: set[tuple[str, str]], progress: dict,
) -> None:
    parquet = pq.ParquetFile(isolation.check_shard(root, entry))
    progress["source_rows_checked"] += parquet.metadata.num_rows
    for row_group in range(parquet.metadata.num_row_groups):
        if not isolation.row_group_may_contain(parquet, row_group, ordered_samples):
            progress["row_groups_pruned"] += 1
            continue
        identities = parquet.read_row_group(row_group, columns=["sample", "BARCODE_SUB_LIB_ID"])
        progress["row_groups_scanned"] += 1
        matched = []
        for index, pair in enumerate(zip(
            identities.column("sample").to_pylist(), identities.column("BARCODE_SUB_LIB_ID").to_pylist(),
            strict=True,
        )):
            found = lookup.get(pair)
            if found is None:
                continue
            cell_id, kind = found
            matched.append((index, cell_id, kind))
            seen = progress["protected_seen"] if kind == "protected" else progress["validation_seen"]
            seen[cell_id] = seen.get(cell_id, 0) + 1
            if (cell_id, entry["path"]) in legacy_expected:
                progress["legacy_seen"].setdefault(cell_id, [])
                if entry["path"] not in progress["legacy_seen"][cell_id]:
                    progress["legacy_seen"][cell_id].append(entry["path"])
        if not matched:
            continue
        rows = np.asarray([item[0] for item in matched], dtype=np.int64)
        metadata_table = parquet.read_row_group(row_group, columns=tahoe.scanner.COLUMNS)
        subset = metadata_table.take(pa.array(rows, type=pa.int64()))
        if (
            subset.column("sample").to_pylist() != [identities.column("sample")[i].as_py() for i in rows]
            or subset.column("BARCODE_SUB_LIB_ID").to_pylist()
            != [identities.column("BARCODE_SUB_LIB_ID")[i].as_py() for i in rows]
        ):
            raise ValueError("Tahoe row-group identity columns changed during the scan")
        selected = tahoe.select_training(
            subset, metadata, structures, aliases, metadata["config"]["seed"],
        )
        for selected_index in selected:
            _, cell_id, kind = matched[int(selected_index)]
            field = "protected_eligible" if kind == "protected" else "validation_eligible"
            progress[field][cell_id] = progress[field].get(cell_id, 0) + 1


def merge_file_progress(progress: dict, delta: dict) -> None:
    for field in ("source_rows_checked", "row_groups_scanned", "row_groups_pruned"):
        progress[field] += delta[field]
    for field in ("protected_seen", "validation_seen", "protected_eligible", "validation_eligible"):
        for cell_id, count in delta[field].items():
            progress[field][cell_id] = progress[field].get(cell_id, 0) + count
    for cell_id, paths in delta["legacy_seen"].items():
        existing = progress["legacy_seen"].setdefault(cell_id, [])
        existing.extend(path for path in paths if path not in existing)


def audit(args) -> None:
    root, run_root, metadata_path = args.root, args.run_root, args.metadata
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    tahoe = load_tahoe(run_root)
    _, protected_records = adapter.load_protected(run_root, args.protected, metadata_path)
    dependencies = adapter.source_dependencies(root, metadata, tahoe)
    dependencies["source_helper_sha256"] = isolation.sha256(Path(tahoe.support.__file__))
    entries = adapter.ordered_entries(metadata, args.seed)
    structures, aliases, split = tahoe.treatment_split(
        metadata, pd.read_parquet(root / "metadata/drug_metadata.parquet"),
    )
    protected_pairs = {key: (value["sample"], value["barcode"]) for key, value in protected_records.items()}
    protected = isolation.grouped_barcodes(protected_pairs)
    validation, _ = adapter.choose_validation(
        root, metadata, tahoe, entries, protected, structures, aliases, args.batch_size,
    )
    validation_pairs = {item["cell_id"]: (item["sample"], item["barcode"]) for item in validation}
    if len(validation_pairs) != adapter.VALIDATION_SIZE or set(validation_pairs) & set(protected_pairs):
        raise ValueError("Validation identities overlap protected identities")
    isolation.grouped_barcodes(validation_pairs)
    lookup = {pair: (cell_id, "protected") for cell_id, pair in protected_pairs.items()}
    if len(lookup) != len(protected_pairs):
        raise ValueError("Protected Tahoe source components are not unique")
    for cell_id, pair in validation_pairs.items():
        if pair in lookup:
            raise ValueError("Validation source components overlap protected identities")
        lookup[pair] = (cell_id, "validation")
    ordered_samples = sorted({sample for sample, _ in lookup})
    if any(not sample.isascii() for sample in ordered_samples):
        raise ValueError("Sample-statistics pruning requires ASCII source sample names")
    legacy_expected = {
        (cell_id, source_file) for cell_id, item in protected_records.items()
        for source_file in item["legacy_source_files"]
    }
    known_files = {item["path"] for item in entries}
    if any(source_file not in known_files for _, source_file in legacy_expected):
        raise ValueError("A protected legacy source shard is missing from the metadata plan")
    validation_data = {"format": "tahoe-foundation-validation-private-v1", "records": validation}
    plan = {
        "format": "tahoe-foundation-isolation-plan-v1",
        "metadata_plan_sha256": isolation.sha256(metadata_path),
        "protected_manifest_sha256": isolation.sha256(args.protected / "manifest.json"),
        "source_entries_sha256": isolation.digest_json(metadata["provenance"]["files"]),
        "ordered_source_paths_sha256": isolation.digest_json([item["path"] for item in entries]),
        "validation_records_sha256": isolation.digest_json(validation_data),
        "order_seed": args.seed, "batch_size": args.batch_size,
        "dependencies": dependencies, "audit_script_sha256": isolation.sha256(Path(__file__)),
        "eligible_rows_before_exclusions": split["eligible_rows"],
    }
    plan_digest = isolation.digest_json(plan)
    if not args.resume:
        args.output.mkdir(mode=0o700)
        atomic_json(args.output / "plan.json", plan)
        atomic_json(args.output / "validation.private.json", validation_data)
        progress = {
            "format": "tahoe-foundation-isolation-progress-private-v1", "state": "running",
            "plan_sha256": plan_digest, "next_file": 0, "source_rows_checked": 0,
            "row_groups_scanned": 0, "row_groups_pruned": 0, "protected_seen": {},
            "validation_seen": {}, "protected_eligible": {}, "validation_eligible": {}, "legacy_seen": {},
        }
        atomic_json(args.output / "progress.private.json", progress)
    elif not args.output.is_dir():
        raise FileNotFoundError("Cannot resume a missing Tahoe isolation audit")

    with (args.output / ".lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            if json.loads((args.output / "plan.json").read_text(encoding="utf-8")) != plan:
                raise ValueError("Tahoe isolation audit plan changed")
            if json.loads((args.output / "validation.private.json").read_text(encoding="utf-8")) != validation_data:
                raise ValueError("Tahoe validation selection changed after audit creation")
            progress_path = args.output / "progress.private.json"
            progress = json.loads(progress_path.read_text(encoding="utf-8"))
            if (
                progress.get("plan_sha256") != plan_digest or progress.get("state") not in ("running", "failed")
                or not 0 <= progress.get("next_file", -1) <= len(entries)
            ):
                raise ValueError("Tahoe isolation progress is complete, invalid, or incompatible")
            progress["state"] = "running"
            progress.pop("error", None)
            for position in range(progress["next_file"], len(entries)):
                delta = {
                    "source_rows_checked": 0, "row_groups_scanned": 0, "row_groups_pruned": 0,
                    "protected_seen": {}, "validation_seen": {}, "protected_eligible": {},
                    "validation_eligible": {}, "legacy_seen": {},
                }
                scan_one_file(
                    root, entries[position], metadata, tahoe, structures, aliases,
                    lookup, ordered_samples, legacy_expected, delta,
                )
                merge_file_progress(progress, delta)
                progress["next_file"] = position + 1
                if progress["next_file"] % args.checkpoint_every == 0:
                    atomic_json(progress_path, progress)
                    print(json.dumps({
                        "files_scanned": progress["next_file"], "files_total": len(entries),
                        "row_groups_scanned": progress["row_groups_scanned"],
                    }), flush=True)
                if args.stop_after_files and progress["next_file"] >= args.stop_after_files:
                    atomic_json(progress_path, progress)
                    print(json.dumps({"state": "paused", "files_scanned": progress["next_file"]}), flush=True)
                    return
            if progress["source_rows_checked"] != metadata["rows"]:
                raise ValueError("Scanned Tahoe source rows differ from the frozen metadata plan")
            if set(progress["protected_seen"]) != set(protected_pairs):
                raise ValueError("Not every protected identity was found in the complete source snapshot")
            if set(progress["validation_eligible"]) != set(validation_pairs):
                raise ValueError("Not every validation identity was found as an eligible source row")
            legacy_seen = {(cell_id, path) for cell_id, paths in progress["legacy_seen"].items() for path in paths}
            if not legacy_expected.issubset(legacy_seen):
                raise ValueError("A legacy evaluation identity is missing from its declared source shard")
            protected_rows = sum(progress["protected_eligible"].values())
            validation_rows = sum(progress["validation_eligible"].values())
            expected_rows = split["eligible_rows"] - protected_rows - validation_rows
            if expected_rows < 2:
                raise ValueError("Too few audited Tahoe training rows remain")
            audit_result = {
                **plan,
                "format": "tahoe-foundation-isolation-audit-v1", "state": "complete",
                "files_scanned": progress["next_file"], "source_rows_checked": progress["source_rows_checked"],
                "row_groups_scanned": progress["row_groups_scanned"], "row_groups_pruned": progress["row_groups_pruned"],
                "protected_unique_identities": len(protected_pairs),
                "protected_found_unique": len(progress["protected_seen"]),
                "protected_occurrence_rows": sum(progress["protected_seen"].values()),
                "protected_eligible_unique": len(progress["protected_eligible"]),
                "protected_eligible_rows": protected_rows,
                "validation_unique_identities": len(validation_pairs),
                "validation_occurrence_rows": sum(progress["validation_seen"].values()),
                "validation_eligible_rows": validation_rows,
                "eligible_training_rows_after_exclusions": expected_rows,
                "legacy_source_provenance_verified": True,
                "validation_private_file": "validation.private.json",
                "validation_private_sha256": isolation.sha256(args.output / "validation.private.json"),
                "limitations": [
                    "This audit protects the explicitly listed historical Tahoe evaluation artifacts only.",
                    "Source integrity relies on original LFS verification plus frozen file size and modification time.",
                    "Sample-row-group statistics are used only for ASCII sample names; absent statistics are scanned.",
                    "Rows are not assumed to be unique biological cells.",
                ],
            }
            atomic_json(args.output / "audit.json", audit_result)
            progress["state"] = "complete"
            atomic_json(progress_path, progress)
            print(json.dumps({key: audit_result[key] for key in (
                "files_scanned", "source_rows_checked", "protected_eligible_rows",
                "validation_eligible_rows", "eligible_training_rows_after_exclusions",
            )}), flush=True)
        except Exception as exc:
            if "progress" in locals() and progress.get("state") != "complete":
                progress["state"] = "failed"
                progress["error"] = f"{type(exc).__name__}: {exc}"
                atomic_json(args.output / "progress.private.json", progress)
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--protected", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--stop-after-files", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 2 or args.checkpoint_every < 1 or args.stop_after_files < 0:
        raise ValueError("Invalid Tahoe isolation scan settings")
    audit(args)
