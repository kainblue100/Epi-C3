"""Build a compound-disjoint metadata plan for all Tahoe expression train shards."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import sys
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from c3vcell.features import gene_token_mapping
from c3vcell.tahoe import TAHOE_CONTROL, _smiles, read_samples


RESERVE = 200 * 1024**3
COLUMNS = ["drug", "sample", "plate", "cell_line_id", "BARCODE_SUB_LIB_ID", "canonical_smiles"]


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + ".part")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


@contextmanager
def locked_scan(outdir: Path):
    with (outdir / ".lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        except Exception as exc:
            progress_path = outdir / "scan-progress.json"
            if progress_path.exists():
                try:
                    progress = json.loads(progress_path.read_text())
                    if progress.get("state") != "complete":
                        progress.update(state="failed", error=f"{type(exc).__name__}: {exc}", updated_at_utc=now())
                        write_json(progress_path, progress)
                except (OSError, ValueError) as status_error:
                    print(f"Could not record Tahoe scan failure: {status_error}", file=sys.stderr)
            raise


def control_role(sample: str, barcode: str, seed: int) -> str:
    payload = json.dumps(["tahoe-control-v1", seed, sample, barcode], separators=(",", ":")).encode()
    bucket = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % 10
    return "reference" if bucket == 0 else "input_controls" if bucket == 1 else "training"


def normalized_smiles(value, drug: str):
    if pd.isna(value):
        return None
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
    return _smiles(value, drug)


def selected_holdouts(counts: dict[str, int], seed: int, number: int, minimum: int):
    eligible = [smiles for smiles, count in counts.items() if count >= minimum]
    ordered = sorted(
        eligible, key=lambda value: (
            hashlib.sha256(f"tahoe-compound-v1:{seed}:{value}".encode()).hexdigest(), value,
        ),
    )
    if len(ordered) < number:
        raise ValueError(f"Only {len(ordered)} compounds satisfy the matched-control holdout threshold")
    return ordered[:number], len(ordered)


def validate_files(root: Path, manifest_path: Path):
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("repo") != "tahoebio/Tahoe-100M":
        raise ValueError("Unexpected Tahoe source repository")
    items = [
        item for item in manifest["files"]
        if item["rfilename"].startswith("data/train-") and item["rfilename"].endswith(".parquet")
    ]
    names = {item["rfilename"] for item in items}
    actual = {str(path.relative_to(root)) for path in (root / "data").glob("*.parquet")}
    if not items or names != actual:
        raise ValueError("Tahoe expression shards do not match the fixed train snapshot")
    files = []
    for item in sorted(items, key=lambda record: record["rfilename"]):
        path = root / item["rfilename"]
        stat = path.stat()
        lfs = item.get("lfs", {})
        if stat.st_size != item["size"] or stat.st_size != lfs.get("size"):
            raise ValueError(f"Tahoe shard size differs from the verified manifest: {path}")
        source_hash = lfs.get("sha256")
        if not isinstance(source_hash, str) or len(source_hash) != 64:
            raise ValueError(f"Tahoe shard lacks an LFS SHA-256: {path}")
        files.append({
            "path": item["rfilename"], "size": stat.st_size, "mtime_ns": stat.st_mtime_ns,
            "source_sha256_verified_at_download": source_hash,
        })
    for relative in (
        "metadata/gene_metadata.parquet", "metadata/sample_metadata.parquet",
        "metadata/drug_metadata.parquet", "metadata/obs_metadata.parquet",
    ):
        records = [item for item in manifest["files"] if item["rfilename"] == relative]
        if len(records) != 1 or (root / relative).stat().st_size != records[0]["size"]:
            raise ValueError(f"Tahoe metadata file differs from the verified manifest: {relative}")
    return manifest, files


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261115)
    parser.add_argument("--holdout-compounds", type=int, default=10)
    parser.add_argument("--min-matched-cells", type=int, default=100)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--stop-after-files", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if (
        args.holdout_compounds < 1 or args.min_matched_cells < 1 or args.checkpoint_every < 1
        or args.stop_after_files < 0 or args.resume and args.check_only
    ):
        raise ValueError("Invalid Tahoe metadata-plan arguments")
    args.outdir.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(args.outdir.parent).free < RESERVE + 1024**3:
        raise OSError("Insufficient headroom above the 200 GiB reserve")
    manifest, files = validate_files(args.root, args.manifest)
    samples = read_samples(args.root / "metadata/sample_metadata.parquet")
    genes, tokens, _ = gene_token_mapping(
        pd.read_parquet(args.root / "metadata/gene_metadata.parquet"),
        token_column="token_id", symbol_column="gene_symbol",
    )
    drugs = pd.read_parquet(args.root / "metadata/drug_metadata.parquet")
    if drugs["drug"].isna().any() or drugs["drug"].duplicated().any():
        raise ValueError("Tahoe drug metadata needs unique nonempty drug identifiers")
    smiles_by_drug = {}
    drugs_without_smiles = []
    for row in drugs.itertuples(index=False):
        value = normalized_smiles(row.canonical_smiles, str(row.drug))
        if value is None:
            drugs_without_smiles.append(str(row.drug))
        else:
            smiles_by_drug[str(row.drug)] = value
    config = {
        "seed": args.seed, "holdout_compounds": args.holdout_compounds,
        "min_matched_cells": args.min_matched_cells, "checkpoint_every": args.checkpoint_every,
    }
    provenance = {
        "source_repo": manifest["repo"], "revision": manifest["revision"],
        "manifest_sha256": sha256(args.manifest),
        "gene_metadata_sha256": sha256(args.root / "metadata/gene_metadata.parquet"),
        "sample_metadata_sha256": sha256(args.root / "metadata/sample_metadata.parquet"),
        "drug_metadata_sha256": sha256(args.root / "metadata/drug_metadata.parquet"),
        "script_sha256": sha256(Path(__file__)), "files": files,
        "source_sha256_note": "LFS digests were verified during download; this metadata scan does not rehash shards.",
    }
    if args.check_only:
        first = pq.ParquetFile(args.root / files[0]["path"])
        if not set(COLUMNS).issubset(first.schema_arrow.names):
            raise ValueError("The first Tahoe shard lacks required treatment fields")
        print(json.dumps({
            "config": config, "revision": manifest["revision"], "train_shards": len(files),
            "samples": len(samples), "unique_gene_symbols": len(genes), "gene_tokens": len(tokens),
            "drug_metadata_rows": len(drugs), "drugs_with_smiles": len(smiles_by_drug),
            "drugs_without_smiles": drugs_without_smiles,
            "obs_metadata_rows": pq.ParquetFile(args.root / "metadata/obs_metadata.parquet").metadata.num_rows,
        }, indent=2))
        return
    if not args.resume and args.outdir.exists():
        raise FileExistsError("Refusing to overwrite an existing Tahoe metadata scan")
    if args.resume and not args.outdir.is_dir():
        raise FileNotFoundError("Cannot resume a missing Tahoe metadata scan")
    if not args.resume:
        args.outdir.mkdir()
    with locked_scan(args.outdir):
        partial = args.outdir / "scan-progress.json"
        result = args.outdir / "metadata-plan.json"
        if result.exists():
            raise FileExistsError("The Tahoe metadata scan is already complete")
        if args.resume:
            progress = json.loads(partial.read_text())
            if progress["config"] != config or progress["provenance"] != provenance:
                raise ValueError("Tahoe files, code, or configuration changed since the scan")
        else:
            progress = {
                "state": "running", "started_at_utc": now(), "config": config,
                "provenance": provenance, "next_file": 0, "rows": 0, "groups": {},
                "control_groups": {}, "missing_smiles_rows": 0,
            }
            write_json(partial, progress)
        progress.pop("error", None)
        progress.update(state="running", updated_at_utc=now())
        write_json(partial, progress)
        groups = progress["groups"]
        control_groups = progress["control_groups"]
        invocation_start = progress["next_file"]
        for position in range(progress["next_file"], len(files)):
            item = files[position]
            path = args.root / item["path"]
            stat = path.stat()
            if stat.st_size != item["size"] or stat.st_mtime_ns != item["mtime_ns"]:
                raise ValueError(f"Tahoe source changed during metadata scan: {path}")
            for batch in pq.ParquetFile(path).iter_batches(batch_size=65_536, columns=COLUMNS):
                frame = batch.to_pandas()
                if frame[[name for name in COLUMNS if name != "canonical_smiles"]].isna().any().any():
                    raise ValueError(f"Tahoe cells have missing treatment or identity metadata in {path}")
                progress["rows"] += len(frame)
                group_columns = ["sample", "drug", "plate", "cell_line_id", "canonical_smiles"]
                for values, count in frame.groupby(group_columns, dropna=False, sort=False).size().items():
                    sample, drug, plate, line, smiles = values
                    sample, drug, plate, line = map(str, (sample, drug, plate, line))
                    if any(not value or "\t" in value for value in (sample, drug, plate, line)):
                        raise ValueError(f"Invalid Tahoe treatment identifier in {path}")
                    if sample not in samples:
                        raise ValueError(f"Sample {sample!r} is absent from Tahoe sample metadata")
                    expected_drug, expected_plate, dose = samples[sample]
                    if (drug, plate) != (expected_drug, expected_plate):
                        raise ValueError(f"Tahoe sample metadata disagrees for {sample!r}")
                    if drug == TAHOE_CONTROL:
                        canonical = ""
                    else:
                        canonical = normalized_smiles(smiles, drug)
                        if canonical is None:
                            progress["missing_smiles_rows"] += int(count)
                        else:
                            expected_smiles = smiles_by_drug.get(drug)
                            if expected_smiles is not None and canonical != expected_smiles:
                                raise ValueError(f"Tahoe SMILES disagree for drug {drug!r}")
                    key = f"{sample}\t{line}"
                    details = {
                        "sample": sample, "drug": drug, "plate": plate, "cell_line_id": line,
                        "dose_um": dose, "canonical_smiles": canonical, "cells": 0,
                    }
                    current = groups.setdefault(key, details)
                    if any(current[field] != details[field] for field in details if field != "cells"):
                        raise ValueError(f"Inconsistent Tahoe condition metadata for {key}")
                    current["cells"] += int(count)
                control_frame = frame.loc[frame["drug"] == TAHOE_CONTROL, [
                    "sample", "BARCODE_SUB_LIB_ID", "cell_line_id", "plate",
                ]]
                for sample, barcode, line, plate in control_frame.itertuples(index=False, name=None):
                    key = f"{line}\t{plate}"
                    details = control_groups.setdefault(key, {
                        "training": 0, "reference": 0, "input_controls": 0,
                    })
                    details[control_role(str(sample), str(barcode), args.seed)] += 1
            progress.update(next_file=position + 1, updated_at_utc=now())
            if shutil.disk_usage(args.outdir).free < RESERVE:
                raise OSError("Free disk space reached the 200 GiB reserve")
            pause = args.stop_after_files and position + 1 - invocation_start >= args.stop_after_files
            if (position + 1) % args.checkpoint_every == 0 or pause or position + 1 == len(files):
                progress["state"] = "paused" if pause else "running"
                write_json(partial, progress)
            if position % 25 == 0 or position + 1 == len(files):
                print(f"Scanned {position + 1}/{len(files)} Tahoe shards, {progress['rows']} rows", flush=True)
            if pause:
                return
        matched_counts = Counter()
        unmatched_rows = 0
        for group in groups.values():
            if group["drug"] == TAHOE_CONTROL:
                continue
            control = control_groups.get(f"{group['cell_line_id']}\t{group['plate']}")
            if control is None or min(control.values()) < 1:
                unmatched_rows += group["cells"]
            elif group["canonical_smiles"] is not None:
                matched_counts[group["canonical_smiles"]] += group["cells"]
        holdouts, eligible = selected_holdouts(
            matched_counts, args.seed, args.holdout_compounds, args.min_matched_cells,
        )
        heldout_rows = sum(
            group["cells"] for group in groups.values() if group["canonical_smiles"] in holdouts
        )
        control_counts = {
            role: sum(values[role] for values in control_groups.values())
            for role in ("training", "reference", "input_controls")
        }
        progress.update(
            state="complete", finished_at_utc=now(), heldout_compounds=holdouts,
            eligible_holdout_compounds=eligible, matched_compound_cells=dict(matched_counts),
            heldout_rows=heldout_rows, control_counts=control_counts,
            eligible_representation_rows=progress["rows"] - heldout_rows
            - control_counts["reference"] - control_counts["input_controls"],
            treatment_rows_without_all_control_partitions=unmatched_rows,
            drugs_without_smiles_in_metadata=drugs_without_smiles,
            unique_gene_symbols=len(genes), gene_tokens=len(tokens),
            obs_metadata_rows=pq.ParquetFile(args.root / "metadata/obs_metadata.parquet").metadata.num_rows,
            notes=[
                "Holdouts are selected by canonical SMILES; all aliases and doses for each held-out compound are excluded.",
                "Controls are split by sample and barcode before training and matching.",
                "All expression train shards are scanned; pseudobulk metadata is not treated as independent cells.",
                "Leading/trailing whitespace is stripped from source SMILES; missing structures are retained only for representation.",
                "QC categories from obs_metadata are not joined, so no additional QC filtering is claimed.",
                "The expression-shard row count may differ from the separate obs_metadata row count.",
                "This is a research split, not an official challenge score.",
            ],
        )
        write_json(result, progress)
        write_json(partial, progress)
        print(f"Completed Tahoe metadata plan: {result}", flush=True)


if __name__ == "__main__":
    main()
