"""Build a target-disjoint metadata plan for the complete Orion parquet snapshot."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from c3vcell.features import gene_token_mapping
from c3vcell.identity import source_cell_id


CONTROL = "Non-Targeting"
RESERVE = 200 * 1024**3
CONTEXTS = ("HCT116", "HEK293T")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    tmp = path.with_name(path.name + ".part")
    with tmp.open("w") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def control_role(context: str, sample: str, barcode: str, seed: int) -> str:
    cell_id = source_cell_id("orion", context, sample, barcode)
    digest = hashlib.sha256(f"orion-control-v1:{seed}:{cell_id}".encode()).digest()
    bucket = int.from_bytes(digest[:8], "big") % 10
    return "reference" if bucket == 0 else "input_controls" if bucket == 1 else "training"


def select_targets(target_counts, known_genes, seed, number, minimum):
    eligible = [
        target for target in known_genes
        if target != CONTROL and all(target_counts[context].get(target, 0) >= minimum for context in CONTEXTS)
    ]
    ordered = sorted(
        eligible, key=lambda target: (hashlib.sha256(f"orion-target-v1:{seed}:{target}".encode()).hexdigest(), target)
    )
    if len(ordered) < number:
        raise ValueError(f"Only {len(ordered)} targets satisfy the cross-context held-out threshold")
    return ordered[:number], len(ordered)


def validate_files(root: Path, manifest_path: Path):
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("repo") != "Xaira-Therapeutics/X-Atlas-Orion":
        raise ValueError("Unexpected Orion source repository")
    items = [
        item for item in manifest["files"]
        if item["rfilename"].startswith("data/")
        and item["rfilename"].endswith(".parquet")
        and any(Path(item["rfilename"]).name.startswith(context + "_") for context in CONTEXTS)
    ]
    if not items:
        raise ValueError("No matching complete Orion shards in the manifest")
    expected_names = {item["rfilename"] for item in items}
    actual_names = {
        str(path.relative_to(root)) for path in (root / "data").glob("*.parquet")
        if any(path.name.startswith(context + "_") for context in CONTEXTS)
    }
    if expected_names != actual_names:
        raise ValueError("Orion source shards do not match the fixed snapshot manifest")
    files = []
    for item in sorted(items, key=lambda value: value["rfilename"]):
        relative = item["rfilename"]
        path = root / relative
        stat = path.stat()
        lfs = item.get("lfs", {})
        if stat.st_size != item["size"] or lfs.get("size") != stat.st_size:
            raise ValueError(f"Orion shard size differs from the verified manifest: {path}")
        source_hash = lfs.get("sha256")
        if not isinstance(source_hash, str) or len(source_hash) != 64:
            raise ValueError(f"Orion shard has no source SHA-256: {path}")
        files.append({
            "path": relative, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns,
            "source_sha256_verified_at_download": source_hash,
        })
    metadata = root / "metadata/gene_metadata.parquet"
    records = [item for item in manifest["files"] if item["rfilename"] == "metadata/gene_metadata.parquet"]
    if len(records) != 1 or metadata.stat().st_size != records[0]["size"]:
        raise ValueError("Orion gene metadata differs from the verified manifest")
    return manifest, files, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261115)
    parser.add_argument("--holdout-targets", type=int, default=12)
    parser.add_argument("--min-target-cells-per-context", type=int, default=50)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--stop-after-files", type=int, default=0)
    args = parser.parse_args()
    if (
        args.holdout_targets < 1 or args.min_target_cells_per_context < 1
        or args.stop_after_files < 0 or args.resume and args.check_only
    ):
        raise ValueError("Invalid metadata-plan arguments")
    args.outdir.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(args.outdir.parent).free < RESERVE + 1024**3:
        raise OSError("Insufficient headroom above the 200 GiB reserve")
    manifest, files, gene_metadata = validate_files(args.root, args.manifest)
    symbols, tokens, _ = gene_token_mapping(
        pd.read_parquet(gene_metadata), token_column="gene_token_id", symbol_column="gene_name",
    )
    known = set(symbols)
    config = {
        "seed": args.seed, "holdout_targets": args.holdout_targets,
        "min_target_cells_per_context": args.min_target_cells_per_context,
    }
    provenance = {
        "source_repo": manifest["repo"], "revision": manifest["revision"],
        "manifest_sha256": sha256(args.manifest), "gene_metadata_sha256": sha256(gene_metadata),
        "script_sha256": sha256(Path(__file__)), "files": files,
        "source_sha256_note": "LFS digests were verified during download; this metadata scan does not rehash shards.",
    }
    if args.check_only:
        print(json.dumps({
            "config": config, "revision": manifest["revision"], "shards": len(files),
            "shards_per_context": {
                context: sum(Path(item["path"]).name.startswith(context + "_") for item in files)
                for context in CONTEXTS
            }, "unique_gene_symbols": len(symbols), "gene_tokens": len(tokens),
        }, indent=2))
        return
    if not args.resume and args.outdir.exists():
        raise FileExistsError("Refusing to overwrite an existing metadata scan")
    if args.resume and not args.outdir.is_dir():
        raise FileNotFoundError("Cannot resume a missing metadata scan")
    if not args.resume:
        args.outdir.mkdir()
    with (args.outdir / ".lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        partial = args.outdir / "scan-progress.json"
        result = args.outdir / "metadata-plan.json"
        if result.exists():
            raise FileExistsError("The Orion metadata scan is already complete")
        if args.resume:
            progress = json.loads(partial.read_text())
            if progress["config"] != config or progress["provenance"] != provenance:
                raise ValueError("Orion files, code, or configuration changed since the scan")
        else:
            progress = {
                "state": "running", "started_at_utc": datetime.now(timezone.utc).isoformat(),
                "config": config, "provenance": provenance, "next_file": 0,
                "target_counts": {context: {} for context in CONTEXTS},
                "samples": {}, "totals": {context: {} for context in CONTEXTS},
            }
            write_json(partial, progress)
        target_counts = {context: Counter(progress["target_counts"][context]) for context in CONTEXTS}
        totals = {context: Counter(progress["totals"][context]) for context in CONTEXTS}
        samples = progress["samples"]
        progress.update(state="running", updated_at_utc=datetime.now(timezone.utc).isoformat())
        write_json(partial, progress)
        invocation_start = progress["next_file"]
        for position in range(progress["next_file"], len(files)):
            item = files[position]
            path = args.root / item["path"]
            context = path.name.split("_", 1)[0]
            columns = ["gene_target", "guide_target", "sample", "cell_barcode", "pass_guide_filter"]
            seen = set()
            local = {}
            for batch in pq.ParquetFile(path).iter_batches(batch_size=8192, columns=columns):
                frame = batch.to_pandas()
                qc = frame["pass_guide_filter"]
                if not qc.dropna().isin([0, 1, False, True]).all():
                    raise ValueError(f"Invalid pass_guide_filter values in {path}")
                passing = qc.fillna(False).astype(bool)
                totals[context]["all_rows"] += len(frame)
                totals[context]["qc_failed_or_missing"] += int((~passing).sum())
                frame = frame.loc[passing]
                if frame[["sample", "cell_barcode", "guide_target"]].isna().any().any():
                    raise ValueError(f"Passing Orion cells have missing identity or guide metadata in {path}")
                for sample, barcode in zip(frame["sample"], frame["cell_barcode"], strict=True):
                    key = (str(sample), str(barcode))
                    if not all(key) or key in seen:
                        raise ValueError(f"Duplicate or empty sample/barcode in {path}: {key}")
                    seen.add(key)
                for sample, group in frame.groupby("sample", sort=False):
                    sample = str(sample)
                    key = f"{context}\t{sample}"
                    if key in samples and samples[key]["source_file"] != item["path"]:
                        raise ValueError(f"Sample {key} appears in multiple source shards")
                    details = local.setdefault(key, {
                        "context": context, "sample": sample, "source_file": item["path"],
                        "training_controls": 0, "reference_controls": 0, "input_controls": 0,
                        "passing_rows": 0,
                    })
                    details["passing_rows"] += len(group)
                    labels = group["gene_target"]
                    totals[context]["unlabeled"] += int(labels.isna().sum())
                    names = labels.dropna().astype(str)
                    for target, count in names.value_counts().items():
                        if target == CONTROL:
                            totals[context]["control_rows"] += int(count)
                        elif target in known:
                            target_counts[context][target] += int(count)
                        else:
                            totals[context]["targets_absent_from_gene_axis"] += int(count)
                    controls = group.loc[labels == CONTROL, "cell_barcode"]
                    for barcode in controls:
                        role = control_role(context, sample, str(barcode), args.seed)
                        details[f"{role}_controls" if role != "input_controls" else "input_controls"] += 1
            samples.update(local)
            progress.update(
                next_file=position + 1, target_counts={k: dict(v) for k, v in target_counts.items()},
                totals={k: dict(v) for k, v in totals.items()}, samples=samples,
                updated_at_utc=datetime.now(timezone.utc).isoformat(),
            )
            if shutil.disk_usage(args.outdir).free < RESERVE:
                raise OSError("Free disk space reached the 200 GiB reserve")
            write_json(partial, progress)
            if position % 10 == 0 or position + 1 == len(files):
                print(f"Scanned {position + 1}/{len(files)} Orion shards", flush=True)
            if args.stop_after_files and position + 1 - invocation_start >= args.stop_after_files:
                progress.update(state="paused", updated_at_utc=datetime.now(timezone.utc).isoformat())
                write_json(partial, progress)
                return
        holdout, eligible = select_targets(
            target_counts, known, args.seed, args.holdout_targets, args.min_target_cells_per_context,
        )
        poor_controls = [
            key for key, value in samples.items()
            if min(value["training_controls"], value["reference_controls"], value["input_controls"]) < 1
        ]
        if poor_controls:
            raise ValueError(f"Samples lack a distinct control partition: {poor_controls[:10]}")
        progress.update(
            state="complete", finished_at_utc=datetime.now(timezone.utc).isoformat(),
            heldout_targets=holdout, eligible_holdout_targets=eligible, control_label=CONTROL,
            unique_gene_symbols=len(symbols), tokens=len(tokens),
            notes=[
                "Holdout targets are selected from metadata only and excluded from both contexts' training.",
                "Control cells are deterministically split into training/reference/input groups by stable ID.",
                "Dataset is CC BY-NC-SA 4.0; this is a research split, not an official challenge score.",
            ],
        )
        write_json(result, progress)
        write_json(partial, progress)
        print(f"Completed Orion metadata plan: {result}", flush=True)


if __name__ == "__main__":
    main()
