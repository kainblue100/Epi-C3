"""Protected, resumable Tahoe stream for a future shared RNA foundation stage."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import scipy.sparse as sp

from c3vcell.identity import source_cell_id
from epi_c3_foundation_stage1_sources_v1 import digest_json


class TahoeStream:
    def __init__(
        self, root: Path, metadata: dict, frame: pd.DataFrame, genes: list[str], helper, tahoe,
        protected_ids: set[str], *, batch_size: int, seed: int,
    ) -> None:
        if metadata.get("state") != "complete" or batch_size < 2:
            raise ValueError("A completed Tahoe split and batches of at least two rows are required")
        if len(set(genes)) != len(genes) or not set(frame.columns) >= {"ensembl_id", "token_id"}:
            raise ValueError("Tahoe gene metadata or the shared axis is invalid")
        if frame["ensembl_id"].isna().any() or frame["ensembl_id"].duplicated().any():
            raise ValueError("Tahoe requires unique Ensembl identifiers")
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
        self.measured = np.zeros(len(genes), dtype=bool)
        columns = np.asarray([lookup[name] for name in local_genes], dtype=np.int64)
        self.measured[columns] = True
        self.mapping = helper.create_token_map({
            int(token): int(column) for token, column in zip(tokens, columns, strict=True)
        })
        self.root, self.metadata, self.helper, self.tahoe = root, metadata, helper, tahoe
        self.protected_ids = set(protected_ids)
        self.batch_size, self.gene_count = batch_size, len(genes)
        self.structures, self.aliases, split = tahoe.treatment_split(
            metadata, pd.read_parquet(root / "metadata/drug_metadata.parquet"),
        )
        entries = metadata["provenance"]["files"]
        if not entries:
            raise ValueError("No frozen Tahoe expression files")
        self.entries = [entries[index] for index in np.random.default_rng(seed).permutation(len(entries))]
        self.file_position = 0
        self.batch_position = 0
        self.iterator = None
        pieces, validation_ids = [], []
        for item in self.entries:
            helper.check_files(root, [item])
            for batch in pq.ParquetFile(root / item["path"]).iter_batches(batch_size=batch_size, columns=tahoe.COLUMNS):
                selected = self.select(batch)
                if len(selected):
                    selected = selected[:32 - len(validation_ids)]
                    validation_ids.extend(self.identities(batch, selected))
                    pieces.append(tahoe.counts_from_batch(batch, selected, self.mapping, self.gene_count))
                if len(validation_ids) == 32:
                    break
            if len(validation_ids) == 32:
                break
        if len(validation_ids) != 32 or len(set(validation_ids)) != 32:
            raise ValueError("Tahoe requires 32 distinct validation source identities")
        self.validation = sp.vstack(pieces, format="csr")
        self.validation_ids = set(validation_ids)
        self.expected_cells = int(split["eligible_rows"] - len(validation_ids))
        if self.expected_cells < 2:
            raise ValueError("Too few Tahoe rows remain after validation exclusions")
        self.notes = {
            "eligible_rows_before_validation": int(split["eligible_rows"]),
            "validation_cells": len(validation_ids),
            "validation_identity_sha256": digest_json(validation_ids),
            "protected_ids": len(self.protected_ids),
            "protected_identity_sha256": digest_json(sorted(self.protected_ids)),
            "ordered_files_sha256": digest_json([item["path"] for item in self.entries]),
            "validation_is_donor_disjoint": False,
            "eligible_rows_are_not_proven_unique_cells": True,
            "frozen_split": split,
        }

    @staticmethod
    def identities(batch, selected: np.ndarray) -> list[str]:
        samples = batch.column("sample").to_pylist()
        barcodes = batch.column("BARCODE_SUB_LIB_ID").to_pylist()
        return [source_cell_id("tahoe", str(samples[index]), str(barcodes[index])) for index in selected]

    def select(self, batch) -> np.ndarray:
        selected = self.tahoe.select_training(
            batch, self.metadata, self.structures, self.aliases, self.metadata["config"]["seed"],
        )
        if len(selected) and self.protected_ids.intersection(self.identities(batch, selected)):
            raise ValueError("A frozen Tahoe evaluation identity would enter the foundation training split")
        return selected

    def next_batch(self) -> sp.csr_matrix | None:
        while self.file_position < len(self.entries):
            if self.iterator is None:
                item = self.entries[self.file_position]
                self.helper.check_files(self.root, [item])
                self.iterator = iter(pq.ParquetFile(self.root / item["path"]).iter_batches(
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
            selected = self.select(batch)
            if len(selected):
                keep = [identity not in self.validation_ids for identity in self.identities(batch, selected)]
                selected = selected[np.asarray(keep, dtype=bool)]
            if len(selected):
                return self.tahoe.counts_from_batch(batch, selected, self.mapping, self.gene_count)
        return None

    def cursor(self) -> dict[str, int]:
        return {"file_position": self.file_position, "batch_position": self.batch_position}

    def set_cursor(self, cursor: dict[str, int]) -> None:
        file_position, batch_position = cursor.get("file_position", -1), cursor.get("batch_position", -1)
        if (
            not isinstance(file_position, int) or not isinstance(batch_position, int)
            or not 0 <= file_position <= len(self.entries) or batch_position < 0
            or (file_position == len(self.entries) and batch_position != 0)
        ):
            raise ValueError("Invalid Tahoe recovery cursor")
        self.file_position, self.batch_position = file_position, batch_position
        self.iterator = None

    def close(self) -> None:
        self.iterator = None
