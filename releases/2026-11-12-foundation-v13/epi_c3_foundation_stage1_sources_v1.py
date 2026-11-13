"""Resumable Arc, Tabula and Orion streams on one shared RNA gene axis."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import scipy.sparse as sp

from epi_c3_foundation_sources_v1 import GeneAlignment, H5CsrSource, align_counts


def digest_json(value) -> str:
    data = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def windows(length: int, batch_size: int, seed: int) -> list[tuple[int, int]]:
    if length < 2 or batch_size < 2:
        raise ValueError("A source must contain at least two rows and use batches of at least two")
    intervals = [(left, min(left + batch_size, length)) for left in range(0, length, batch_size)]
    if len(intervals) > 1 and intervals[-1][1] - intervals[-1][0] == 1:
        intervals[-2] = (intervals[-2][0], intervals[-1][1])
        intervals.pop()
    blocks = [intervals[index:index + 64] for index in range(0, len(intervals), 64)]
    order = np.random.default_rng(seed).permutation(len(blocks))
    return [interval for block in order for interval in blocks[block]]


class ArcStream:
    def __init__(
        self, path: Path, arc, alignment: GeneAlignment, original_split: Path,
        protected_h5ads: list[Path], heldout_targets: set[str], *, batch_size: int, seed: int,
    ) -> None:
        self.source = arc.ArcSource(path)
        try:
            old_split = json.loads(original_split.read_text())
            original = arc.make_split(self.source, sorted(old_split["test_targets"]), 20261114)["train"]
            protected = set()
            for artifact in protected_h5ads:
                with h5py.File(artifact, "r") as file:
                    protected.update(map(str, arc.index_values(file["obs"])))
            selected = np.asarray([
                int(row) for row in original
                if self.source.labels[row] not in heldout_targets and self.source.ids[row] not in protected
            ], dtype=np.int64)
            if len(selected) < 64:
                raise ValueError("Arc has insufficient non-heldout training cells")
            validation = np.random.default_rng(seed).choice(selected, size=32, replace=False)
            self.validation = align_counts(
                self.source.matrix(np.sort(validation)), alignment.source_to_global["arc"], len(alignment.genes),
            )
            self.rows = selected[~np.isin(selected, validation)]
            self.expected_cells = len(self.rows)
            self.measured = alignment.measured["arc"]
            self.mapping = alignment.source_to_global["arc"]
            self.gene_count = len(alignment.genes)
            self.intervals = windows(self.expected_cells, batch_size, seed)
            self.position = 0
            self.notes = {
                "prior_training_cells": len(original), "protected_evaluation_ids": len(protected),
                "protected_evaluation_identity_sha256": digest_json(sorted(protected)),
                "after_exclusions_before_validation": len(selected),
                "validation_cells": len(validation),
                "validation_identity_sha256": digest_json(self.source.ids[np.sort(validation)].tolist()),
                "training_identity_sha256": digest_json(self.source.ids[self.rows].tolist()),
            }
        except Exception:
            self.source.close()
            raise

    def next_batch(self) -> sp.csr_matrix | None:
        if self.position >= len(self.intervals):
            return None
        left, right = self.intervals[self.position]
        self.position += 1
        matrix = self.source.matrix(self.rows[left:right])
        return align_counts(matrix, self.mapping, self.gene_count)

    def cursor(self) -> dict[str, int]:
        return {"position": self.position}

    def set_cursor(self, cursor: dict[str, int]) -> None:
        position = cursor.get("position", -1)
        if not isinstance(position, int) or not 0 <= position <= len(self.intervals):
            raise ValueError("Invalid Arc recovery cursor")
        self.position = position

    def close(self) -> None:
        self.source.close()


class TabulaStream:
    def __init__(
        self, path: Path, alignment: GeneAlignment, *, batch_size: int, seed: int,
    ) -> None:
        self.source = H5CsrSource(path, "raw/X")
        try:
            donor = self.source.file["obs/donor_id"]
            if not isinstance(donor, h5py.Group) or set(donor.keys()) != {"categories", "codes"}:
                raise ValueError("Tabula donor metadata must be categorical")
            categories = donor["categories"].asstr()[:]
            codes = donor["codes"][:]
            if (
                len(codes) != self.source.n_rows or len(categories) < 2
                or np.any(codes < 0) or np.any(codes >= len(categories))
            ):
                raise ValueError("Tabula has invalid donor identifiers")
            heldout = min(
                range(len(categories)),
                key=lambda index: hashlib.sha256(f"{seed}\0{categories[index]}".encode()).digest(),
            )
            self.training_rows = codes != heldout
            self.expected_cells = int(self.training_rows.sum())
            validation = np.flatnonzero(codes == heldout)[:32]
            if self.expected_cells < 2 or len(validation) < 32:
                raise ValueError("Tabula donor split has insufficient training or validation cells")
            self.validation = align_counts(
                sp.vstack([self.source.counts(int(row), int(row) + 1) for row in validation], format="csr"),
                alignment.source_to_global["tabula"], len(alignment.genes),
            )
            self.measured = alignment.measured["tabula"]
            self.mapping = alignment.source_to_global["tabula"]
            self.gene_count = len(alignment.genes)
            self.intervals = windows(self.source.n_rows, batch_size, seed)
            self.position = 0
            self.notes = {
                "source_cells": self.source.n_rows,
                "heldout_donor_cells": int(np.count_nonzero(codes == heldout)),
                "heldout_donor_identity_sha256": hashlib.sha256(str(categories[heldout]).encode()).hexdigest(),
                "validation_cells": len(validation),
                "validation_rows_sha256": digest_json(validation.tolist()),
                "validation_is_donor_disjoint": True,
            }
        except Exception:
            self.source.close()
            raise

    def next_batch(self) -> sp.csr_matrix | None:
        while self.position < len(self.intervals):
            left, right = self.intervals[self.position]
            self.position += 1
            selected = self.training_rows[left:right]
            if np.any(selected):
                matrix = self.source.counts(left, right)[selected]
                return align_counts(matrix, self.mapping, self.gene_count)
        return None

    def cursor(self) -> dict[str, int]:
        return {"position": self.position}

    def set_cursor(self, cursor: dict[str, int]) -> None:
        position = cursor.get("position", -1)
        if not isinstance(position, int) or not 0 <= position <= len(self.intervals):
            raise ValueError("Invalid Tabula recovery cursor")
        self.position = position

    def close(self) -> None:
        self.source.close()


class OrionStream:
    def __init__(
        self, root: Path, metadata: dict, frame: pd.DataFrame, alignment: GeneAlignment,
        helper, context: str, heldout_targets: set[str], *, batch_size: int, seed: int,
    ) -> None:
        self.root, self.metadata, self.helper, self.context = root, metadata, helper, context
        self.heldout_targets = heldout_targets
        self.batch_size = batch_size
        self.measured = alignment.measured["orion"]
        self.gene_count = len(alignment.genes)
        raw_tokens = pd.to_numeric(frame["gene_token_id"], errors="raise").to_numpy(dtype=np.float64)
        if (
            not np.isfinite(raw_tokens).all() or np.any(raw_tokens < 0)
            or np.any(raw_tokens != np.floor(raw_tokens)) or len(set(raw_tokens.tolist())) != len(raw_tokens)
        ):
            raise ValueError("Orion gene tokens must be distinct nonnegative integers")
        self.mapping = helper.create_token_map({
            int(token): int(column)
            for token, column in zip(raw_tokens, alignment.source_to_global["orion"], strict=True)
        })
        self.known = set(frame["gene_name"].astype(str))
        entries = [item for item in metadata["provenance"]["files"]
                   if Path(item["path"]).name.startswith(context + "_")]
        if not entries:
            raise ValueError(f"No Orion files for {context}")
        order = np.random.default_rng(seed).permutation(len(entries))
        self.entries = [entries[index] for index in order]
        self.file_position = 0
        self.batch_position = 0
        self.iterator = None
        validation_pieces, validation_ids = [], []
        for item in self.entries:
            helper.check_files(root, [item])
            for batch in pq.ParquetFile(root / item["path"]).iter_batches(batch_size=batch_size, columns=helper.COLUMNS):
                selected = self.select(batch)
                if not len(selected):
                    continue
                selected = selected[:32 - len(validation_ids)]
                validation_ids.extend(self.identities(batch, selected))
                validation_pieces.append(helper.counts_from_batch(batch, selected, self.mapping, self.gene_count))
                if len(validation_ids) == 32:
                    break
            if len(validation_ids) == 32:
                break
        if len(validation_ids) != 32 or len(set(validation_ids)) != 32:
            raise ValueError(f"Orion {context} needs 32 distinct validation cells")
        self.validation = sp.vstack(validation_pieces, format="csr")
        self.validation_ids = set(validation_ids)
        targets = metadata["target_counts"][context]
        target_count = sum(count for target, count in targets.items() if target not in heldout_targets)
        control_count = sum(
            item["training_controls"] for item in metadata["samples"].values() if item["context"] == context
        )
        self.expected_cells = int(target_count + control_count - len(validation_ids))
        if self.expected_cells < 2:
            raise ValueError(f"Too few Orion {context} cells remain after exclusions")
        self.notes = {
            "target_training_cells_before_validation": int(target_count),
            "control_training_cells_before_validation": int(control_count),
            "validation_cells": len(validation_ids),
            "validation_identity_sha256": digest_json(validation_ids),
            "files": len(self.entries),
            "ordered_files_sha256": digest_json([item["path"] for item in self.entries]),
            "validation_is_donor_disjoint": False,
        }

    def select(self, batch) -> np.ndarray:
        return self.helper.select_training(
            batch, self.context, self.heldout_targets, self.known, self.metadata["config"]["seed"],
        )

    @staticmethod
    def identities(batch, selected: np.ndarray) -> list[str]:
        samples = batch.column("sample").to_pylist()
        barcodes = batch.column("cell_barcode").to_pylist()
        return [f"{samples[index]}\0{barcodes[index]}" for index in selected]

    def next_batch(self) -> sp.csr_matrix | None:
        while self.file_position < len(self.entries):
            if self.iterator is None:
                item = self.entries[self.file_position]
                self.helper.check_files(self.root, [item])
                self.iterator = iter(pq.ParquetFile(self.root / item["path"]).iter_batches(
                    batch_size=self.batch_size, columns=self.helper.COLUMNS,
                ))
                for _ in range(self.batch_position):
                    try:
                        next(self.iterator)
                    except StopIteration as exc:
                        raise ValueError("Orion recovery cursor exceeds a source file") from exc
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
                return self.helper.counts_from_batch(batch, selected, self.mapping, self.gene_count)
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
            raise ValueError("Invalid Orion recovery cursor")
        self.file_position, self.batch_position = file_position, batch_position
        self.iterator = None

    def close(self) -> None:
        self.iterator = None
