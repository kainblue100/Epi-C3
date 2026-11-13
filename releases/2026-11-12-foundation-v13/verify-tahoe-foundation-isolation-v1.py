"""Independently verify per-identity Tahoe holdout overlap counts."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from c3vcell.tahoe import TAHOE_CONTROL
import epi_c3_tahoe_isolation_v1 as isolation


def stable_identity(sample: str, barcode: str) -> str:
    payload = json.dumps(["tahoe", sample, barcode], ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return "tahoe-v1-" + hashlib.sha256(payload).hexdigest()


def control_is_training(sample: str, barcode: str, seed: int) -> bool:
    payload = json.dumps(["tahoe-control-v1", seed, sample, barcode], separators=(",", ":")).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % 10 >= 2


def possibly_contains(parquet, index: int, candidates: set[str]) -> bool:
    names = parquet.schema.names
    if names.count("sample") != 1:
        raise ValueError("A Tahoe shard has an ambiguous sample column")
    column = parquet.metadata.row_group(index).column(names.index("sample"))
    if column.path_in_schema != "sample":
        raise ValueError("A Tahoe sample statistics column is not scalar")
    stats = column.statistics
    if stats is None or not stats.has_min_max:
        return True
    lower, upper = stats.min, stats.max
    if (
        not isinstance(lower, str) or not isinstance(upper, str)
        or not lower.isascii() or not upper.isascii() or lower > upper
    ):
        return True
    return any(lower <= sample <= upper for sample in candidates)


def eligible(row: dict, metadata: dict, structures: dict, aliases: set[str], normalize) -> bool:
    for name in ("drug", "sample", "plate", "cell_line_id", "BARCODE_SUB_LIB_ID"):
        if not isinstance(row[name], str) or not row[name]:
            raise ValueError("A matched Tahoe row has incomplete identity metadata")
    group = metadata["groups"].get(f"{row['sample']}\t{row['cell_line_id']}")
    if group is None or (row["drug"], row["plate"]) != (group["drug"], group["plate"]):
        raise ValueError("A matched Tahoe row differs from its frozen sample group")
    if row["drug"] == TAHOE_CONTROL:
        return control_is_training(row["sample"], row["BARCODE_SUB_LIB_ID"], metadata["config"]["seed"])
    if normalize(row["canonical_smiles"], row["drug"]) != group["canonical_smiles"]:
        raise ValueError("A matched Tahoe row has changed compound metadata")
    return structures.get(row["drug"]) is not None and row["drug"] not in aliases


def run(args) -> None:
    if args.output.exists():
        raise FileExistsError(args.output)
    meta_path = args.run_root / "tahoe-full-metadata-v2/metadata-plan.json"
    protected_dir = args.run_root / "tahoe-protected-identities-v1"
    audit_dir = args.run_root / "tahoe-foundation-isolation-v1"
    audit_path = audit_dir / "audit.json"
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    manifest = json.loads((protected_dir / "manifest.json").read_text(encoding="utf-8"))
    result = json.loads(audit_path.read_text(encoding="utf-8"))
    progress = json.loads((audit_dir / "progress.private.json").read_text(encoding="utf-8"))
    if (
        result.get("state") != "complete" or progress.get("state") != "complete"
        or result["metadata_plan_sha256"] != isolation.sha256(meta_path)
        or result["protected_manifest_sha256"] != isolation.sha256(protected_dir / "manifest.json")
        or manifest["private_sha256"] != isolation.sha256(protected_dir / manifest["private_file"])
        or result["validation_private_sha256"] != isolation.sha256(audit_dir / result["validation_private_file"])
    ):
        raise ValueError("A frozen Tahoe isolation input changed")
    records = json.loads((protected_dir / manifest["private_file"]).read_text(encoding="utf-8"))["records"]
    validation = json.loads((audit_dir / result["validation_private_file"]).read_text(encoding="utf-8"))["records"]
    pairs = {}
    for kind, collection in (("protected", records), ("validation", validation)):
        for item in collection:
            sample, barcode, cell_id = item["sample"], item["barcode"], item["cell_id"]
            if stable_identity(sample, barcode) != cell_id or (sample, barcode) in pairs:
                raise ValueError("Frozen Tahoe identities are invalid or overlap")
            pairs[(sample, barcode)] = (kind, cell_id)
    samples = {sample for sample, _ in pairs}
    if any(not sample.isascii() for sample in samples):
        raise ValueError("This verification requires ASCII Tahoe sample names")
    scanner_path = args.run_root / "scan-tahoe-stream-v1.py"
    if isolation.sha256(scanner_path) != metadata["provenance"]["script_sha256"]:
        raise ValueError("Tahoe chemistry normalization changed")
    spec = importlib.util.spec_from_file_location("tahoe_verify_scanner", scanner_path)
    if spec is None or spec.loader is None:
        raise ImportError("Cannot load Tahoe chemistry normalization")
    scanner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scanner)
    drugs = pd.read_parquet(args.root / "metadata/drug_metadata.parquet")
    if isolation.sha256(args.root / "metadata/drug_metadata.parquet") != metadata["provenance"]["drug_metadata_sha256"]:
        raise ValueError("Tahoe drug metadata changed")
    structures = {str(row.drug): scanner.normalized_smiles(row.canonical_smiles, str(row.drug))
                  for row in drugs.itertuples(index=False)}
    for group in metadata["groups"].values():
        drug, structure = group["drug"], group["canonical_smiles"]
        if drug != TAHOE_CONTROL and structure is not None:
            previous = structures.get(drug)
            if previous is not None and previous != structure:
                raise ValueError("Tahoe drug structure is inconsistent")
            structures[drug] = structure
    heldout = set(metadata["heldout_compounds"])
    aliases = {drug for drug, structure in structures.items() if structure in heldout}
    if not heldout.issubset({structures[drug] for drug in aliases}):
        raise ValueError("A held-out Tahoe compound has no source alias")
    columns = ["drug", "sample", "plate", "cell_line_id", "BARCODE_SUB_LIB_ID", "canonical_smiles"]
    seen = {"protected": Counter(), "validation": Counter()}
    selected = {"protected": Counter(), "validation": Counter()}
    source_rows = groups_scanned = groups_pruned = 0
    for item in metadata["provenance"]["files"]:
        parquet = pq.ParquetFile(isolation.check_shard(args.root, item))
        source_rows += parquet.metadata.num_rows
        for index in range(parquet.metadata.num_row_groups):
            if not possibly_contains(parquet, index, samples):
                groups_pruned += 1
                continue
            groups_scanned += 1
            table = parquet.read_row_group(index, columns=columns)
            values = {name: table.column(name).to_pylist() for name in columns}
            for row_index, (sample, barcode) in enumerate(zip(
                values["sample"], values["BARCODE_SUB_LIB_ID"], strict=True,
            )):
                match = pairs.get((sample, barcode))
                if match is None:
                    continue
                kind, cell_id = match
                row = {name: values[name][row_index] for name in columns}
                seen[kind][cell_id] += 1
                if eligible(row, metadata, structures, aliases, scanner.normalized_smiles):
                    selected[kind][cell_id] += 1
    for kind in ("protected", "validation"):
        if dict(seen[kind]) != progress[f"{kind}_seen"] or dict(selected[kind]) != progress[f"{kind}_eligible"]:
            raise ValueError(f"Independent {kind} per-identity counts differ from the original audit")
    if source_rows != metadata["rows"] or sum(selected["protected"].values()) != result["protected_eligible_rows"]:
        raise ValueError("Independent Tahoe source-row totals differ from the original audit")
    if sum(selected["validation"].values()) != result["validation_eligible_rows"]:
        raise ValueError("Independent Tahoe validation-row totals differ from the original audit")
    if (
        result["eligible_rows_before_exclusions"] - sum(selected["protected"].values())
        - sum(selected["validation"].values()) != result["eligible_training_rows_after_exclusions"]
    ):
        raise ValueError("Independent Tahoe eligible-row arithmetic differs")
    args.output.mkdir()
    verification = {
        "format": "tahoe-foundation-isolation-verification-v1", "state": "complete",
        "audit_sha256": isolation.sha256(audit_path),
        "script_sha256": isolation.sha256(Path(__file__)),
        "files_checked": len(metadata["provenance"]["files"]),
        "source_rows_checked": source_rows, "row_groups_scanned": groups_scanned,
        "row_groups_pruned": groups_pruned,
        "protected_unique": len(seen["protected"]),
        "protected_eligible_rows": sum(selected["protected"].values()),
        "validation_unique": len(seen["validation"]),
        "validation_eligible_rows": sum(selected["validation"].values()),
        "eligible_training_rows": result["eligible_training_rows_after_exclusions"],
        "per_identity_counts_match": True,
        "limitations": [
            "The same frozen source plan and chemistry normalizer are used as in the original audit.",
            "Parquet sample statistics and frozen file size/mtime are trusted as described by the original audit.",
            "Only explicitly inventoried historical evaluation artifacts are protected.",
        ],
    }
    with (args.output / "verification.json").open("x", encoding="utf-8") as file:
        json.dump(verification, file, indent=2, allow_nan=False)
        file.write("\n")
    print(json.dumps({key: verification[key] for key in (
        "files_checked", "source_rows_checked", "protected_unique", "protected_eligible_rows",
        "validation_eligible_rows", "eligible_training_rows", "per_identity_counts_match",
    )}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
