"""Stream bounded, verified Orion parquet shards into a count AnnData."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import scipy.sparse as sp

from .data import CONTROL, validate_counts
from .features import gene_token_mapping
from .identity import source_cell_id


def inspect_shards(paths: list[Path], *, batch_size: int = 1024) -> pd.DataFrame:
    counts: Counter[tuple[str, str, str]] = Counter()
    for path in paths:
        if path.suffix != ".parquet" or not path.is_file():
            raise ValueError(f"Only complete parquet files are accepted: {path}")
        context = path.name.split("_", 1)[0]
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(
            batch_size=batch_size, columns=["gene_target", "sample", "pass_guide_filter"]
        ):
            frame = batch.to_pandas()
            qc = frame["pass_guide_filter"]
            if not qc.dropna().isin([0, 1, False, True]).all():
                raise ValueError(f"Invalid pass_guide_filter values in {path}")
            frame = frame[qc.fillna(False).astype(bool)]
            for (target, sample), count in frame.groupby(["gene_target", "sample"], dropna=False).size().items():
                counts[(context, str(target), str(sample))] += int(count)
    rows = [
        {"context": context, "gene_target": target, "sample": sample, "n_cells": count}
        for (context, target, sample), count in counts.items()
    ]
    return pd.DataFrame(rows, columns=["context", "gene_target", "sample", "n_cells"])


def _reservoir_updates(
    labels: list[str],
    sizes: dict[str, int],
    seen: Counter[str],
    limits: dict[str, int],
    rng: np.random.Generator,
) -> list[tuple[int, str, int]]:
    updates: dict[tuple[str, int], int] = {}
    for row, target in enumerate(labels):
        seen[target] += 1
        if sizes[target] < limits[target]:
            position = sizes[target]
            sizes[target] += 1
        else:
            position = int(rng.integers(seen[target]))
            if position >= limits[target]:
                continue
        updates[(target, position)] = row
    return sorted(
        ((row, target, position) for (target, position), row in updates.items()),
        key=lambda item: (item[1], item[2]),
    )


def prepare_orion(
    paths: list[Path],
    gene_metadata: Path,
    output: Path,
    *,
    targets: list[str],
    control_label: str,
    max_per_target: int = 100,
    max_controls: int = 500,
    seed: int = 0,
) -> None:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    if not paths or max_per_target < 1 or max_controls < 1:
        raise ValueError("Provide complete parquet shards and positive sampling limits")
    if len({path.resolve() for path in paths}) != len(paths):
        raise ValueError("The same Orion shard was supplied more than once")
    if not targets or len(set(targets)) != len(targets) or control_label in targets:
        raise ValueError("Provide distinct gene targets excluding the control label")
    symbols, tokens, ensembls = gene_token_mapping(
        pd.read_parquet(gene_metadata), token_column="gene_token_id", symbol_column="gene_name",
    )
    absent = sorted(set(targets) - set(symbols))
    if absent:
        raise ValueError(f"Target genes absent from Orion metadata: {absent}")
    rng = np.random.default_rng(seed)
    selected: dict[str, list[dict]] = {target: [] for target in [*targets, control_label]}
    sizes = {target: 0 for target in selected}
    limits = {
        target: max_controls if target == control_label else max_per_target
        for target in selected
    }
    seen: Counter[str] = Counter()
    columns = [
        "gene_token_id", "gene_expression", "cell_barcode", "sample",
        "guide_target", "gene_target", "pass_guide_filter",
    ]
    requested_labels = pa.array(list(selected))
    for path in paths:
        if path.suffix != ".parquet" or not path.is_file():
            raise ValueError(f"Only complete parquet files are accepted: {path}")
        context = path.name.split("_", 1)[0]
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=1024, columns=columns):
            requested = pc.is_in(batch.column("gene_target"), value_set=requested_labels)
            qc = batch.column("pass_guide_filter")
            if pa.types.is_boolean(qc.type):
                passing = pc.fill_null(qc, False)
            elif pa.types.is_integer(qc.type):
                valid = pc.fill_null(pc.or_(pc.equal(qc, 0), pc.equal(qc, 1)), True)
                if not pc.all(valid).as_py():
                    raise ValueError(f"Invalid pass_guide_filter values in {path}")
                passing = pc.fill_null(pc.equal(qc, 1), False)
            else:
                raise ValueError(f"Unsupported pass_guide_filter type {qc.type} in {path}")
            filtered = batch.filter(pc.and_(requested, passing))
            labels = filtered.column("gene_target").to_pylist()
            updates = _reservoir_updates(labels, sizes, seen, limits, rng)
            if not updates:
                continue
            chosen = filtered.take(pa.array([item[0] for item in updates], type=pa.int64()))
            for (_, target, position), record in zip(updates, chosen.to_pylist(), strict=True):
                record["gene_token_id"] = np.asarray(record["gene_token_id"], dtype=np.int64)
                record["gene_expression"] = np.asarray(record["gene_expression"], dtype=np.float64)
                record["context"] = context
                record["source_file"] = path.name
                if position == len(selected[target]):
                    selected[target].append(record)
                elif position < len(selected[target]):
                    selected[target][position] = record
                else:
                    raise RuntimeError("Reservoir updates are not contiguous")
            if any(len(selected[target]) != sizes[target] for target in selected):
                raise RuntimeError("Reservoir sample sizes became inconsistent")

    missing = [target for target, records in selected.items() if not records]
    if missing:
        raise ValueError(f"No passing cells for requested labels: {missing}")
    indices_parts, value_parts, indptr = [], [], [0]
    obs_rows = []
    seen_cells: set[tuple[str, str, str]] = set()
    for target, records in selected.items():
        for record in records:
            ids = record["gene_token_id"]
            expression = np.asarray(record["gene_expression"], dtype=np.float64)
            if ids is None or len(ids) != len(expression):
                raise ValueError("Mismatched Orion gene token and expression arrays")
            if not np.isfinite(expression).all() or np.any(expression < 0):
                raise ValueError("Invalid Orion count values")
            if not np.equal(expression, np.floor(expression)).all():
                raise ValueError("Orion values are not raw integer counts")
            if np.any(expression > np.iinfo(np.int32).max):
                raise ValueError("An Orion count is too large for int32")
            try:
                gene_indices = np.fromiter(
                    (tokens[int(token)] for token in ids), dtype=np.int32, count=len(ids)
                )
            except KeyError as exc:
                raise ValueError(f"Gene token {exc.args[0]} is missing from metadata") from exc
            indices_parts.append(gene_indices)
            value_parts.append(expression.astype(np.int32))
            indptr.append(indptr[-1] + len(gene_indices))
            if record["sample"] is None or record["cell_barcode"] is None:
                raise ValueError("Orion cells require sample and barcode metadata")
            cell_key = (
                record["context"], str(record["sample"]), str(record["cell_barcode"]),
            )
            if cell_key in seen_cells:
                raise ValueError(f"Duplicate Orion cell identifier: {cell_key}")
            seen_cells.add(cell_key)
            obs_rows.append({
                "cell_id": source_cell_id("orion", *cell_key),
                "cell_barcode": str(record["cell_barcode"]),
                "target_gene": CONTROL if target == control_label else target,
                "context": record["context"],
                "batch": str(record["sample"]),
                "guide_id": str(record["guide_target"]),
                "source_file": record["source_file"],
            })
    matrix = sp.csr_matrix(
        (np.concatenate(value_parts), np.concatenate(indices_parts), np.asarray(indptr)),
        shape=(len(obs_rows), len(symbols)),
    )
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    obs = pd.DataFrame(obs_rows).set_index("cell_id")
    var = pd.DataFrame(index=pd.Index(symbols, name="gene"))
    var["ensembl_ids"] = [ensembls.loc[name] for name in symbols]
    data = ad.AnnData(X=matrix, obs=obs, var=var)
    data.uns["c3vcell_orion"] = {
        "source": "X-Atlas-Orion",
        "cell_identity": "orion-v1: context, sample, and cell_barcode",
        "reference_annotation": "10x Genomics 2024-A",
        "sampling_seed": seed,
        "complete_shards": [path.name for path in paths],
    }
    data.uns["c3vcell_genome_assembly"] = "GRCh38"
    validate_counts(data)
    output.parent.mkdir(parents=True, exist_ok=True)
    data.write_h5ad(output, compression="gzip")
    print(f"Saved {data.n_obs} cells from {len(paths)} complete Orion shards", flush=True)
