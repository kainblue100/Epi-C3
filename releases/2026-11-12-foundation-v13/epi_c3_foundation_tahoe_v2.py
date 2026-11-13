"""Tahoe foundation stream gated by a complete, provenance-bound isolation audit."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import scipy.sparse as sp

import c3vcell.identity as source_identity
import epi_c3_tahoe_isolation_v1 as isolation


VALIDATION_SIZE = 32


def load_protected(run_root: Path, directory: Path, metadata_path: Path) -> tuple[dict, dict]:
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("state") != "complete"
        or manifest.get("format") != "tahoe-protected-identities-v1"
        or manifest.get("metadata_plan_sha256") != isolation.sha256(metadata_path)
        or manifest.get("identity_helper_sha256") != isolation.sha256(Path(isolation.__file__))
        or manifest.get("script_sha256") != isolation.sha256(run_root / "freeze-tahoe-protected-identities-v1.py")
    ):
        raise ValueError("Protected Tahoe identity manifest is incomplete or stale")
    expected_artifacts = {
        f"{name}/{role}" for name in ("tahoe-demo-multidose", "tahoe-demo-plate4")
        for role in ("reference", "input_controls")
    } | {f"{kind}/{role}" for kind in ("interim", "smoke") for role in ("reference", "input_controls")}
    if {item["label"] for item in manifest["evaluation_artifacts"]} != expected_artifacts:
        raise ValueError("Protected Tahoe manifest lacks a required evaluation artifact")
    for artifact in manifest["evaluation_artifacts"]:
        name, role = artifact["label"].split("/")
        if name.startswith("tahoe-demo-"):
            path = run_root.parent / name / "split" / f"{role}.h5ad"
        else:
            path = run_root / (
                "tahoe-interim-eval-v1" if name == "interim" else "tahoe-real-evaluation-smoke-v1"
            ) / f"{role}.h5ad"
        if isolation.sha256(path) != artifact["sha256"]:
            raise ValueError("A protected Tahoe evaluation artifact changed")
        if name == "smoke" and isolation.sha256(
            run_root / "tahoe-parent-clean-real-smoke-v1" / path.name
        ) != artifact["sha256"]:
            raise ValueError("A mirrored Tahoe smoke evaluation artifact changed")
    for relative, field in (
        ("tahoe-interim-evaluation-protocol-v1.json", "interim_protocol_sha256"),
        ("tahoe-interim-eval-v1/state.json", "interim_state_sha256"),
        ("tahoe-smoke-metadata-v1/metadata-plan.json", "smoke_metadata_sha256"),
    ):
        if isolation.sha256(run_root / relative) != manifest[field]:
            raise ValueError("Protected Tahoe evaluation provenance changed")
    private_name = manifest["private_file"]
    if Path(private_name).name != private_name:
        raise ValueError("Invalid private identity filename")
    private_path = directory / private_name
    if isolation.sha256(private_path) != manifest["private_sha256"]:
        raise ValueError("Protected Tahoe identity file changed")
    private = json.loads(private_path.read_text(encoding="utf-8"))
    if private.get("format") != "tahoe-protected-identities-private-v1":
        raise ValueError("Unexpected protected identity format")
    records = private["records"]
    pairs = {item["cell_id"]: (item["sample"], item["barcode"]) for item in records}
    if len(pairs) != len(records) or len(pairs) != manifest["protected_unique_identities"]:
        raise ValueError("Protected identity count or uniqueness changed")
    isolation.grouped_barcodes(pairs)
    return manifest, {item["cell_id"]: item for item in records}


def source_dependencies(root: Path, metadata: dict, tahoe) -> dict:
    if metadata.get("state") != "complete":
        raise ValueError("Tahoe metadata scan is incomplete")
    provenance = metadata["provenance"]
    scanner_sha = isolation.sha256(Path(tahoe.scanner.__file__))
    if scanner_sha != provenance["script_sha256"]:
        raise ValueError("Tahoe source split code changed after metadata scanning")
    for name in ("gene", "sample", "drug"):
        if isolation.sha256(root / f"metadata/{name}_metadata.parquet") != provenance[f"{name}_metadata_sha256"]:
            raise ValueError(f"Tahoe {name} metadata changed after its source scan")
    return {
        "representation_script_sha256": isolation.sha256(Path(tahoe.__file__)),
        "scanner_script_sha256": scanner_sha,
        "identity_implementation_sha256": isolation.sha256(Path(source_identity.__file__)),
        "isolation_helper_sha256": isolation.sha256(Path(isolation.__file__)),
        "adapter_sha256": isolation.sha256(Path(__file__)),
    }


def ordered_entries(metadata: dict, seed: int) -> list[dict]:
    entries = metadata["provenance"]["files"]
    if not entries or len({item["path"] for item in entries}) != len(entries):
        raise ValueError("Tahoe metadata has missing or duplicated source shards")
    return [entries[index] for index in np.random.default_rng(seed).permutation(len(entries))]


def choose_validation(
    root: Path, metadata: dict, tahoe, entries: list[dict], protected: dict[str, set[str]],
    structures: dict, aliases: set[str], batch_size: int, mapping=None, gene_count=None,
) -> tuple[list[dict], sp.csr_matrix | None]:
    if batch_size < 2:
        raise ValueError("Tahoe requires batches of at least two rows")
    records, pieces, seen = [], [], set()
    columns = tahoe.scanner.COLUMNS if mapping is None else tahoe.COLUMNS
    for entry in entries:
        parquet = pq.ParquetFile(isolation.check_shard(root, entry))
        offset = 0
        for batch in parquet.iter_batches(batch_size=batch_size, columns=columns):
            selected = tahoe.select_training(batch, metadata, structures, aliases, metadata["config"]["seed"])
            samples = batch.column("sample").to_pylist()
            barcodes = batch.column("BARCODE_SUB_LIB_ID").to_pylist()
            chosen = []
            for index in selected:
                index = int(index)
                sample, barcode = samples[index], barcodes[index]
                cell_id = isolation.identity(sample, barcode)
                if barcode in protected.get(sample, ()) or cell_id in seen:
                    continue
                seen.add(cell_id)
                records.append({
                    "cell_id": cell_id, "sample": sample, "barcode": barcode,
                    "source_file": entry["path"], "source_row": offset + index,
                })
                chosen.append(index)
                if len(records) == VALIDATION_SIZE:
                    break
            if chosen and mapping is not None:
                pieces.append(tahoe.counts_from_batch(
                    batch, np.asarray(chosen, dtype=np.int64), mapping, gene_count,
                ))
            offset += batch.num_rows
            if len(records) == VALIDATION_SIZE:
                break
        if len(records) == VALIDATION_SIZE:
            break
    if len(records) != VALIDATION_SIZE:
        raise ValueError("Tahoe requires 32 distinct, non-protected validation identities")
    return records, sp.vstack(pieces, format="csr") if mapping is not None else None


class TahoeStream:
    def __init__(
        self, root: Path, run_root: Path, metadata_path: Path, protected_dir: Path, audit_dir: Path,
        frame: pd.DataFrame, genes: list[str], helper, tahoe, *, batch_size: int, seed: int,
    ) -> None:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        manifest, protected_records = load_protected(run_root, protected_dir, metadata_path)
        dependencies = source_dependencies(root, metadata, tahoe)
        dependencies["source_helper_sha256"] = isolation.sha256(Path(helper.__file__))
        entries = ordered_entries(metadata, seed)
        audit_path = audit_dir / "audit.json"
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if (
            audit.get("format") != "tahoe-foundation-isolation-audit-v1"
            or audit.get("state") != "complete"
            or audit.get("metadata_plan_sha256") != isolation.sha256(metadata_path)
            or audit.get("protected_manifest_sha256") != isolation.sha256(protected_dir / "manifest.json")
            or audit.get("source_entries_sha256") != isolation.digest_json(metadata["provenance"]["files"])
            or audit.get("ordered_source_paths_sha256") != isolation.digest_json([item["path"] for item in entries])
            or audit.get("batch_size") != batch_size or audit.get("order_seed") != seed
            or audit.get("files_scanned") != len(entries)
            or audit.get("source_rows_checked") != metadata["rows"]
            or audit.get("legacy_source_provenance_verified") is not True
            or audit.get("dependencies") != dependencies
            or audit.get("audit_script_sha256") != isolation.sha256(run_root / "audit-tahoe-foundation-isolation-v1.py")
        ):
            raise ValueError("Tahoe isolation audit is incomplete, stale, or incompatible")
        validation_name = audit["validation_private_file"]
        if Path(validation_name).name != validation_name:
            raise ValueError("Invalid Tahoe validation identity filename")
        validation_path = audit_dir / validation_name
        if isolation.sha256(validation_path) != audit["validation_private_sha256"]:
            raise ValueError("Tahoe validation identities changed after the audit")
        validation_file = json.loads(validation_path.read_text(encoding="utf-8"))
        if validation_file.get("format") != "tahoe-foundation-validation-private-v1":
            raise ValueError("Unexpected Tahoe validation identity format")
        frozen_validation = validation_file["records"]
        for entry in entries:
            isolation.check_shard(root, entry)
        expected_frame = pd.read_parquet(root / "metadata/gene_metadata.parquet", columns=["ensembl_id", "token_id"])
        if not {"ensembl_id", "token_id"}.issubset(frame.columns) or not frame[
            ["ensembl_id", "token_id"]
        ].reset_index(drop=True).equals(expected_frame.reset_index(drop=True)):
            raise ValueError("Tahoe gene metadata differs from the frozen source")
        if (
            len(set(genes)) != len(genes) or frame["ensembl_id"].isna().any()
            or frame["ensembl_id"].duplicated().any()
        ):
            raise ValueError("Tahoe gene identifiers must be unique and complete")
        tokens = pd.to_numeric(frame["token_id"], errors="raise").to_numpy(dtype=np.float64)
        if (
            not np.isfinite(tokens).all() or np.any(tokens < 0) or np.any(tokens != np.floor(tokens))
            or len(set(tokens.tolist())) != len(tokens)
        ):
            raise ValueError("Tahoe gene tokens must be distinct nonnegative integers")
        lookup = {name: index for index, name in enumerate(genes)}
        local_genes = frame["ensembl_id"].astype(str).tolist()
        if any(name not in lookup for name in local_genes):
            raise ValueError("A Tahoe gene is absent from the shared foundation axis")
        columns = np.asarray([lookup[name] for name in local_genes], dtype=np.int64)
        self.measured = np.zeros(len(genes), dtype=bool)
        self.measured[columns] = True
        self.mapping = helper.create_token_map({
            int(token): int(column) for token, column in zip(tokens, columns, strict=True)
        })
        self.root, self.metadata, self.helper, self.tahoe = root, metadata, helper, tahoe
        self.entries, self.batch_size, self.gene_count = entries, batch_size, len(genes)
        self.structures, self.aliases, split = tahoe.treatment_split(
            metadata, pd.read_parquet(root / "metadata/drug_metadata.parquet"),
        )
        if split["eligible_rows"] != audit["eligible_rows_before_exclusions"]:
            raise ValueError("Tahoe eligible-row count changed after the audit")
        protected_pairs = {
            key: (value["sample"], value["barcode"]) for key, value in protected_records.items()
        }
        self.protected = isolation.grouped_barcodes(protected_pairs)
        records, self.validation = choose_validation(
            root, metadata, tahoe, entries, self.protected, self.structures, self.aliases,
            batch_size, self.mapping, self.gene_count,
        )
        if records != frozen_validation:
            raise ValueError("Tahoe validation selection changed after the isolation audit")
        validation_pairs = {item["cell_id"]: (item["sample"], item["barcode"]) for item in records}
        if len(validation_pairs) != VALIDATION_SIZE or set(validation_pairs) & set(protected_pairs):
            raise ValueError("Tahoe validation identities are duplicated or protected")
        self.validation_ids = set(validation_pairs)
        self.validation_pairs = isolation.grouped_barcodes(validation_pairs)
        self.expected_cells = audit["eligible_training_rows_after_exclusions"]
        if (
            self.expected_cells < 2
            or self.expected_cells != split["eligible_rows"] - audit["protected_eligible_rows"]
            - audit["validation_eligible_rows"]
            or self.validation.shape != (VALIDATION_SIZE, len(genes))
        ):
            raise ValueError("Tahoe audited row counts or validation shape are inconsistent")
        self.file_position = self.batch_position = self.returned_cells = 0
        self.iterator = None
        self.notes = {
            "isolation_audit_sha256": isolation.sha256(audit_path),
            "protected_manifest_sha256": isolation.sha256(protected_dir / "manifest.json"),
            "protected_unique_identities": manifest["protected_unique_identities"],
            "validation_unique_identities": len(records),
            "protected_eligible_rows": audit["protected_eligible_rows"],
            "validation_eligible_rows": audit["validation_eligible_rows"],
            "eligible_rows_are_not_proven_unique_cells": True,
            "validation_is_donor_disjoint": False,
        }

    def next_batch(self) -> sp.csr_matrix | None:
        while self.file_position < len(self.entries):
            if self.iterator is None:
                path = isolation.check_shard(self.root, self.entries[self.file_position])
                self.iterator = iter(pq.ParquetFile(path).iter_batches(
                    batch_size=self.batch_size, columns=self.tahoe.COLUMNS,
                ))
                for _ in range(self.batch_position):
                    try:
                        next(self.iterator)
                    except StopIteration as exc:
                        raise ValueError("Tahoe recovery cursor exceeds a source file") from exc
            try:
                batch = next(self.iterator)
            except StopIteration:
                self.file_position += 1
                self.batch_position = 0
                self.iterator = None
                continue
            self.batch_position += 1
            selected = self.tahoe.select_training(
                batch, self.metadata, self.structures, self.aliases, self.metadata["config"]["seed"],
            )
            samples = batch.column("sample").to_pylist()
            barcodes = batch.column("BARCODE_SUB_LIB_ID").to_pylist()
            selected = np.asarray([
                int(index) for index in selected
                if barcodes[index] not in self.protected.get(samples[index], ())
                and barcodes[index] not in self.validation_pairs.get(samples[index], ())
            ], dtype=np.int64)
            if len(selected):
                self.returned_cells += len(selected)
                if self.returned_cells > self.expected_cells:
                    raise ValueError("Tahoe stream exceeded the audited eligible-row count")
                return self.tahoe.counts_from_batch(batch, selected, self.mapping, self.gene_count)
        if self.returned_cells != self.expected_cells:
            raise ValueError("Tahoe stream ended before its audited eligible-row count")
        return None

    def cursor(self) -> dict[str, int]:
        return {
            "file_position": self.file_position, "batch_position": self.batch_position,
            "returned_cells": self.returned_cells,
        }

    def set_cursor(self, cursor: dict[str, int]) -> None:
        file_position = cursor.get("file_position", -1)
        batch_position = cursor.get("batch_position", -1)
        returned = cursor.get("returned_cells", -1)
        if (
            type(file_position) is not int or type(batch_position) is not int or type(returned) is not int
            or not 0 <= file_position <= len(self.entries) or batch_position < 0
            or not 0 <= returned <= self.expected_cells
            or (file_position == len(self.entries) and (batch_position != 0 or returned != self.expected_cells))
        ):
            raise ValueError("Invalid Tahoe recovery cursor")
        self.file_position, self.batch_position, self.returned_cells = file_position, batch_position, returned
        self.iterator = None

    def close(self) -> None:
        self.iterator = None
