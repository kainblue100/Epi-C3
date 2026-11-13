"""Audit historical eligible identities against held-out Tahoe contexts, privately."""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path


CONTEXTS = ("C32", "HOP62", "HepG2/C3A", "Hs 766T", "PANC-1")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value) -> str:
    data = json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def identity(sample, barcode) -> str:
    if not isinstance(sample, str) or not sample or not isinstance(barcode, str) or not barcode:
        raise ValueError("Invalid source identity components")
    data = json.dumps(["tahoe", sample, barcode], ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return "tahoe-v1-" + hashlib.sha256(data).hexdigest()


def candidate_sources(protected, validation, progress, entries):
    shards = defaultdict(dict)
    totals = Counter()
    seen_ids, seen_pairs = set(), set()
    for kind, records in (("protected", protected), ("validation", validation)):
        if not isinstance(records, list):
            raise ValueError("A private identity list is invalid")
        by_id = {}
        for record in records:
            cell_id = record["cell_id"]
            pair = (record["sample"], record["barcode"])
            if identity(*pair) != cell_id or cell_id in seen_ids or pair in seen_pairs:
                raise ValueError("An audited identity is invalid or duplicated")
            by_id[cell_id] = record
            seen_ids.add(cell_id)
            seen_pairs.add(pair)
        eligible = progress[f"{kind}_eligible"]
        counts = progress[f"{kind}_seen"]
        if not set(eligible).issubset(by_id):
            raise ValueError("An eligible identity is absent from its private list")
        for cell_id, occurrence in eligible.items():
            if type(occurrence) is not int or occurrence != 1 or counts.get(cell_id) != 1:
                raise ValueError("Targeted shard lookup requires exactly one source occurrence")
            record = by_id[cell_id]
            if kind == "protected":
                sources = record["legacy_source_files"]
                if not isinstance(sources, list) or len(sources) > 1:
                    raise ValueError("Unexpected legacy source-shard provenance")
                if sources and sources[0] not in progress["legacy_seen"].get(cell_id, []):
                    raise ValueError("A legacy source shard was not verified in the prior audit")
                source = sources[0] if sources else None
            else:
                source = record.get("source_file")
                source_row = record.get("source_row")
                if type(source_row) is not int or source_row < 0:
                    raise ValueError("A validation source row is invalid")
            totals[f"{kind}_eligible"] += 1
            if source is None:
                totals[f"{kind}_unresolved"] += 1
                continue
            if not isinstance(source, str) or source not in entries:
                raise ValueError("An eligible identity points outside the frozen source files")
            pair = (record["sample"], record["barcode"])
            if pair in shards[source]:
                raise ValueError("Duplicate identity components in a target shard")
            shards[source][pair] = kind
            totals[f"{kind}_located"] += 1
    return shards, totals


def group_may_contain(parquet, index, samples):
    import pyarrow as pa

    names = parquet.schema.names
    if names.count("sample") != 1:
        raise ValueError("A shard has an ambiguous sample field")
    column = parquet.metadata.row_group(index).column(names.index("sample"))
    field = parquet.schema_arrow.field("sample")
    if column.path_in_schema != "sample" or not (
        pa.types.is_string(field.type) or pa.types.is_large_string(field.type)
    ):
        raise ValueError("A shard sample field is not scalar")
    stats = column.statistics
    if stats is None or not stats.has_min_max:
        return True
    low, high = stats.min, stats.max
    if (
        not isinstance(low, str) or not isinstance(high, str) or not low.isascii()
        or not high.isascii() or low > high
    ):
        return True
    position = bisect.bisect_left(samples, low)
    return position < len(samples) and samples[position] <= high


def check_shard(root: Path, entry: dict) -> Path:
    relative = Path(entry["path"])
    if relative.is_absolute() or ".." in relative.parts or relative.parts[:1] != ("data",):
        raise ValueError("An invalid shard path appears in the frozen source plan")
    path = root / relative
    current = path.stat()
    if current.st_size != entry["size"] or current.st_mtime_ns != entry["mtime_ns"]:
        raise ValueError("A frozen source shard changed")
    return path


def scan(root, entries, groups, shards, blocked_lines):
    import pyarrow.parquet as pq

    totals = Counter()
    for source, targets in sorted(shards.items()):
        parquet = pq.ParquetFile(check_shard(root, entries[source]))
        try:
            samples = sorted({sample for sample, _ in targets})
            if any(not sample.isascii() for sample in samples):
                raise ValueError("The frozen sample-statistics pruning requires ASCII sample names")
            found = Counter()
            for group_index in range(parquet.metadata.num_row_groups):
                if not group_may_contain(parquet, group_index, samples):
                    totals["row_groups_pruned"] += 1
                    continue
                table = parquet.read_row_group(
                    group_index, columns=["sample", "BARCODE_SUB_LIB_ID", "cell_line_id"],
                )
                totals["row_groups_scanned"] += 1
                for sample, barcode, line in zip(
                    table.column("sample").to_pylist(),
                    table.column("BARCODE_SUB_LIB_ID").to_pylist(),
                    table.column("cell_line_id").to_pylist(),
                    strict=True,
                ):
                    pair = (sample, barcode)
                    kind = targets.get(pair)
                    if kind is None:
                        continue
                    group = groups.get(f"{sample}\t{line}")
                    if group is None or group["sample"] != sample or group["cell_line_id"] != line:
                        raise ValueError("A matched identity differs from the frozen source groups")
                    found[pair] += 1
                    totals[f"{kind}_{'heldout' if line in blocked_lines else 'retained'}"] += 1
            if set(found) != set(targets) or any(count != 1 for count in found.values()):
                raise ValueError("A target identity was missing or duplicated in its verified source shard")
            totals["source_files_scanned"] += 1
        finally:
            parquet.close()
    return totals


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "metadata-plan", "isolation-dir", "protected-dir", "cell-line-metadata"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    for name in ("metadata-sha256", "isolation-sha256", "cell-line-sha256"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--coverage-only", action="store_true")
    args = parser.parse_args()
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    if os.name == "posix":
        import resource

        soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        limit = 12 * 1024**3
        hard = limit if hard == resource.RLIM_INFINITY or hard > limit else hard
        soft = hard if soft == resource.RLIM_INFINITY or soft > hard else soft
        resource.setrlimit(resource.RLIMIT_AS, (soft, hard))

    audit_path = args.isolation_dir / "audit.json"
    for path, expected in (
        (args.metadata_plan, args.metadata_sha256), (audit_path, args.isolation_sha256),
        (args.cell_line_metadata, args.cell_line_sha256),
    ):
        if sha256(path) != expected:
            raise ValueError("A frozen source or audit digest changed")
    metadata = json.loads(args.metadata_plan.read_text(encoding="utf-8"))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    progress_path = args.isolation_dir / "progress.private.json"
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    old_plan = json.loads((args.isolation_dir / "plan.json").read_text(encoding="utf-8"))
    manifest_path = args.protected_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        metadata.get("state") != "complete" or audit.get("state") != "complete"
        or progress.get("state") != "complete"
        or manifest.get("state") != "complete"
        or manifest.get("format") != "tahoe-protected-identities-v1"
        or audit["metadata_plan_sha256"] != args.metadata_sha256
        or audit["protected_manifest_sha256"] != sha256(manifest_path)
        or progress["plan_sha256"] != canonical_sha256(old_plan)
        or any(audit.get(key) != value for key, value in old_plan.items() if key != "format")
        or not audit.get("legacy_source_provenance_verified")
        or audit["source_rows_checked"] != metadata["rows"]
        or progress["source_rows_checked"] != metadata["rows"]
        or progress["next_file"] != audit["files_scanned"]
    ):
        raise ValueError("The frozen source or historical audit is incomplete or stale")
    private_name, validation_name = manifest["private_file"], audit["validation_private_file"]
    if Path(private_name).name != private_name or Path(validation_name).name != validation_name:
        raise ValueError("An invalid private input filename was recorded")
    private_path, validation_path = args.protected_dir / private_name, args.isolation_dir / validation_name
    if sha256(private_path) != manifest["private_sha256"] or sha256(validation_path) != audit[
        "validation_private_sha256"
    ]:
        raise ValueError("A private identity input differs from its manifest")
    protected_data = json.loads(private_path.read_text(encoding="utf-8"))
    validation_data = json.loads(validation_path.read_text(encoding="utf-8"))
    if (
        protected_data.get("format") != "tahoe-protected-identities-private-v1"
        or validation_data.get("format") != "tahoe-foundation-validation-private-v1"
    ):
        raise ValueError("An unexpected private identity format was supplied")
    protected, validation = protected_data["records"], validation_data["records"]
    if (
        len(protected) != manifest["protected_unique_identities"]
        or len(validation) != audit["validation_unique_identities"]
        or set(progress["protected_seen"]) != {item["cell_id"] for item in protected}
        or set(progress["validation_seen"]) != {item["cell_id"] for item in validation}
        or sum(progress["protected_seen"].values()) != audit["protected_occurrence_rows"]
        or sum(progress["validation_seen"].values()) != audit["validation_occurrence_rows"]
        or sum(progress["protected_eligible"].values()) != audit["protected_eligible_rows"]
        or sum(progress["validation_eligible"].values()) != audit["validation_eligible_rows"]
    ):
        raise ValueError("The historical identity totals changed")
    entries = {item["path"]: item for item in metadata["provenance"]["files"]}
    if (
        len(entries) != len(metadata["provenance"]["files"])
        or canonical_sha256(metadata["provenance"]["files"]) != audit["source_entries_sha256"]
    ):
        raise ValueError("The frozen source file list contains duplicates")
    shards, coverage = candidate_sources(protected, validation, progress, entries)
    if (
        coverage["protected_eligible"] != audit["protected_eligible_rows"]
        or coverage["validation_eligible"] != audit["validation_eligible_rows"]
    ):
        raise ValueError("The targeted source coverage does not match the prior audit")
    result = {
        "format": "epi-c3-tahoe-context-identity-intersection-v1",
        "status": "coverage-only" if args.coverage_only else "complete",
        "protected_eligible": coverage["protected_eligible"],
        "validation_eligible": coverage["validation_eligible"],
        "protected_with_verified_shard": coverage["protected_located"],
        "validation_with_verified_shard": coverage["validation_located"],
        "protected_unresolved": coverage["protected_unresolved"],
        "validation_unresolved": coverage["validation_unresolved"],
        "target_shards": len(shards),
        "metadata_plan_sha256": args.metadata_sha256,
        "isolation_audit_sha256": args.isolation_sha256,
        "cell_line_metadata_sha256": args.cell_line_sha256,
        "protected_manifest_sha256": sha256(manifest_path),
        "protected_private_sha256": sha256(private_path),
        "validation_private_sha256": sha256(validation_path),
        "historical_progress_private_sha256": sha256(progress_path),
    }
    if not args.coverage_only:
        if result["protected_unresolved"] or result["validation_unresolved"]:
            raise ValueError("Some eligible identities lack individually verified source-shard provenance")
        import pyarrow.parquet as pq

        lines = pq.read_table(
            args.cell_line_metadata, columns=["cell_name", "Cell_ID_Cellosaur"],
        ).to_pylist()
        mapped = {
            name: {row["Cell_ID_Cellosaur"] for row in lines if row["cell_name"] == name}
            for name in CONTEXTS
        }
        if any(len(values) != 1 or None in values for values in mapped.values()):
            raise ValueError("A STATE context does not map to exactly one cell line")
        blocked_lines = {next(iter(values)) for values in mapped.values()}
        if len(blocked_lines) != len(CONTEXTS):
            raise ValueError("STATE contexts do not map to distinct cell lines")
        counted = scan(args.root, entries, metadata["groups"], shards, blocked_lines)
        for kind in ("protected", "validation"):
            if counted[f"{kind}_heldout"] + counted[f"{kind}_retained"] != coverage[f"{kind}_eligible"]:
                raise ValueError("The targeted shard scan did not account for every eligible identity")
        result.update({
            "protected_heldout": counted["protected_heldout"],
            "protected_retained": counted["protected_retained"],
            "validation_heldout": counted["validation_heldout"],
            "validation_retained": counted["validation_retained"],
            "source_files_scanned": counted["source_files_scanned"],
            "row_groups_scanned": counted["row_groups_scanned"],
            "row_groups_pruned": counted["row_groups_pruned"],
        })
    result["limitations"] = [
        "Historical identity counts and source-shard provenance depend on the frozen prior audit.",
        "Current historical progress is checked against aggregate audit totals, not bound by a prior progress hash.",
        "Source shards are checked using frozen size and mtime; download-time LFS hashes are not recomputed.",
        "Results are Tahoe row occurrences, not deduplicated biological cells or proof of cross-source independence.",
        "This audit does not alter or enforce the candidate holdout.",
    ]
    print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(json.dumps({
            "format": "epi-c3-tahoe-context-identity-intersection-v1",
            "status": "failed",
            "error_type": type(error).__name__,
        }), flush=True)
        raise
