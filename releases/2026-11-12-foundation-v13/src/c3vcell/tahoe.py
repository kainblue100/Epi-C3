"""Prepare bounded Tahoe drug-response counts from completed parquet shards."""

from __future__ import annotations

import ast
import json
import math
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

TAHOE_CONTROL = "DMSO_TF"
_UM_PER_UNIT = {"nM": 0.001, "uM": 1.0, "mM": 1000.0, "M": 1_000_000.0}


def _validate_shards(paths: list[Path]) -> None:
    if not paths or len({path.resolve() for path in paths}) != len(paths):
        raise ValueError("Provide distinct, completed Tahoe parquet shards")
    for path in paths:
        if path.suffix != ".parquet" or not path.is_file() or ".aivc-partials" in path.parts:
            raise ValueError(f"Only complete Tahoe parquet files are accepted: {path}")


def inspect_tahoe(paths: list[Path]) -> pd.DataFrame:
    _validate_shards(paths)
    columns = ["drug", "sample", "plate", "cell_line_id"]
    counts: Counter[tuple[str, str, str, str]] = Counter()
    for path in paths:
        for batch in pq.ParquetFile(path).iter_batches(batch_size=65_536, columns=columns):
            frame = batch.to_pandas()
            if frame[columns].isna().any().any():
                raise ValueError(f"Missing Tahoe treatment metadata in {path}")
            groups = frame.groupby(columns, sort=False, observed=True).size()
            for key, count in groups.items():
                counts[tuple(map(str, key))] += int(count)
    return pd.DataFrame(
        [(*key, value) for key, value in counts.items()],
        columns=[*columns, "n_cells"],
    )


def read_samples(path: Path) -> dict[str, tuple[str, str, float]]:
    metadata = pd.read_parquet(path)
    required = {"sample", "plate", "drug", "drugname_drugconc"}
    if not required.issubset(metadata.columns):
        raise ValueError(f"Tahoe sample metadata must contain {sorted(required)}")
    if metadata[list(required)].isna().any().any() or metadata["sample"].duplicated().any():
        raise ValueError("Tahoe samples must have unique identifiers and complete dose metadata")
    result = {}
    for row in metadata.itertuples(index=False):
        sample = str(row.sample)
        try:
            ingredients = ast.literal_eval(row.drugname_drugconc)
        except (ValueError, SyntaxError) as exc:
            raise ValueError(f"Invalid dose metadata for sample {sample}") from exc
        if not isinstance(ingredients, (list, tuple)) or len(ingredients) != 1:
            raise ValueError(f"Sample {sample} is not a single-drug treatment")
        ingredient = ingredients[0]
        if not isinstance(ingredient, (list, tuple)) or len(ingredient) != 3:
            raise ValueError(f"Sample {sample} needs a drug, concentration, and unit")
        name, dose, unit = ingredient
        if not isinstance(name, str) or name.strip() != str(row.drug).strip():
            raise ValueError(f"Sample {sample} has inconsistent drug names")
        if unit not in _UM_PER_UNIT or isinstance(dose, bool):
            raise ValueError(f"Sample {sample} has an unsupported dose or unit")
        try:
            dose_um = float(dose) * _UM_PER_UNIT[unit]
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Sample {sample} has an invalid concentration") from exc
        if not math.isfinite(dose_um) or dose_um < 0:
            raise ValueError(f"Sample {sample} has a non-finite or negative concentration")
        drug = str(row.drug).strip()
        if (drug == TAHOE_CONTROL) != (dose_um == 0):
            raise ValueError(f"Sample {sample} has an inconsistent control concentration")
        result[sample] = (drug, str(row.plate), dose_um)
    return result


def _smiles(value: object, drug: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"Drug {drug!r} lacks a usable canonical SMILES")
    if len(value) > 512 or not value.isascii() or any(
        char.isspace() or not char.isprintable() for char in value
    ):
        raise ValueError(f"Drug {drug!r} has an unsupported canonical SMILES")
    return value


def prepare_tahoe(
    paths: list[Path],
    gene_metadata: Path,
    sample_metadata: Path,
    output: Path,
    *,
    drugs: list[str],
    cell_lines: list[str],
    plates: list[str],
    max_per_condition: int = 200,
    max_controls_per_group: int = 1_000,
    seed: int = 0,
) -> None:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    _validate_shards(paths)
    for values, name in ((drugs, "drugs"), (cell_lines, "cell lines"), (plates, "plates")):
        if not values or len(set(values)) != len(values) or any(not value for value in values):
            raise ValueError(f"Provide unique, nonempty {name}")
    if TAHOE_CONTROL in drugs or min(max_per_condition, max_controls_per_group) < 1:
        raise ValueError("Drug selection must exclude DMSO_TF and sampling limits must be positive")
    symbols, tokens, ensembls = gene_token_mapping(
        pd.read_parquet(gene_metadata), token_column="token_id", symbol_column="gene_symbol",
    )
    samples = read_samples(sample_metadata)
    rng = np.random.default_rng(seed)
    requested = pa.array([*drugs, TAHOE_CONTROL])
    requested_lines = pa.array(cell_lines)
    requested_plates = pa.array(plates)
    seen: Counter[tuple[str, str, str, float]] = Counter()
    selected: dict[tuple[str, str, str, float], list[dict | None]] = {}
    smiles_by_drug: dict[str, str] = {}
    columns = [
        "drug", "sample", "plate", "cell_line_id", "BARCODE_SUB_LIB_ID", "canonical_smiles",
    ]
    for path in paths:
        parquet = pq.ParquetFile(path)
        for group in range(parquet.num_row_groups):
            metadata = parquet.read_row_group(group, columns=columns)
            mask = pc.and_(
                pc.and_(
                    pc.is_in(metadata.column("drug"), value_set=requested),
                    pc.is_in(metadata.column("cell_line_id"), value_set=requested_lines),
                ),
                pc.is_in(metadata.column("plate"), value_set=requested_plates),
            )
            positions = pc.indices_nonzero(pc.fill_null(mask, False)).to_pylist()
            if not positions:
                continue
            metadata_rows = metadata.take(pa.array(positions, type=pa.int64())).to_pylist()
            accepted = []
            for position, row in zip(positions, metadata_rows):
                if any(row.get(name) is None for name in columns if name != "canonical_smiles"):
                    raise ValueError(f"Selected Tahoe cell has missing metadata in {path}")
                drug, sample, plate, line = (
                    str(row["drug"]), str(row["sample"]), str(row["plate"]),
                    str(row["cell_line_id"]),
                )
                if sample not in samples:
                    raise ValueError(f"Tahoe sample {sample!r} is missing from sample metadata")
                meta_drug, meta_plate, dose_um = samples[sample]
                if drug.strip() != meta_drug or plate != meta_plate:
                    raise ValueError(f"Treatment metadata disagrees for sample {sample!r}")
                smile = "" if drug == TAHOE_CONTROL else _smiles(row["canonical_smiles"], drug)
                if drug != TAHOE_CONTROL:
                    old = smiles_by_drug.setdefault(drug, smile)
                    if old != smile:
                        raise ValueError(f"Drug {drug!r} has inconsistent canonical SMILES")
                key = (drug, line, plate, dose_um)
                seen[key] += 1
                bucket = selected.setdefault(key, [])
                limit = max_controls_per_group if drug == TAHOE_CONTROL else max_per_condition
                if len(bucket) < limit:
                    slot = len(bucket)
                    bucket.append(None)
                else:
                    slot = int(rng.integers(seen[key]))
                    if slot >= limit:
                        continue
                row.update({"smiles": smile, "dose_um": dose_um, "source_file": path.name})
                accepted.append((position, key, slot, row))
            if not accepted:
                continue
            arrays = parquet.read_row_group(group, columns=["genes", "expressions"]).take(
                pa.array([item[0] for item in accepted], type=pa.int64())
            ).to_pylist()
            for (_, key, slot, row), values in zip(accepted, arrays):
                row["genes"] = np.asarray(values["genes"], dtype=np.int64)
                row["expressions"] = np.asarray(values["expressions"], dtype=np.float64)
                selected[key][slot] = row

    present = {key[0] for key, records in selected.items() if records}
    missing = sorted((set(drugs) | {TAHOE_CONTROL}) - present)
    if missing:
        raise ValueError(f"No Tahoe cells found for requested drugs: {missing}")
    control_groups = {(line, plate) for drug, line, plate, _ in selected if drug == TAHOE_CONTROL}
    target_groups = {(line, plate) for drug, line, plate, _ in selected if drug != TAHOE_CONTROL}
    if target_groups - control_groups:
        raise ValueError(f"Missing matched DMSO controls for: {sorted(target_groups-control_groups)}")

    indices_parts, value_parts, indptr, obs_rows = [], [], [0], []
    seen_cells: set[tuple[str, str]] = set()
    for (drug, line, plate, dose_um), records in sorted(selected.items()):
        for row in records:
            if row is None:
                raise RuntimeError("Tahoe reservoir contains an unfilled sample")
            ids, expression = row["genes"], row["expressions"]
            if len(ids) != len(expression) or not len(ids) or ids[0] != 1 or expression[0] != -2:
                raise ValueError("Tahoe gene and expression arrays need the expected CLS prefix")
            ids, expression = ids[1:], expression[1:]
            if (
                not np.isfinite(expression).all() or np.any(expression < 0)
                or not np.equal(expression, np.floor(expression)).all()
                or np.any(expression > np.iinfo(np.int32).max)
            ):
                raise ValueError("Tahoe expression must contain finite nonnegative integer counts")
            try:
                indices = np.fromiter((tokens[int(token)] for token in ids), np.int32, len(ids))
            except KeyError as exc:
                raise ValueError(f"Gene token {exc.args[0]} is missing from Tahoe metadata") from exc
            indices_parts.append(indices)
            value_parts.append(expression.astype(np.int64))
            indptr.append(indptr[-1] + len(indices))
            cell_key = (str(row["sample"]), str(row["BARCODE_SUB_LIB_ID"]))
            if cell_key in seen_cells:
                raise ValueError(f"Duplicate Tahoe cell identifier: {cell_key}")
            seen_cells.add(cell_key)
            control = drug == TAHOE_CONTROL
            obs_rows.append({
                "cell_id": source_cell_id("tahoe", *cell_key),
                "source_cell_id": "|".join(cell_key),
                "target_gene": CONTROL if control else f"{drug}::{dose_um:.12g}uM",
                "compound": CONTROL if control else row["smiles"],
                "drug": drug,
                "canonical_smiles": row["smiles"],
                "dose_um": dose_um,
                "context": line,
                "batch": plate,
                "sample": str(row["sample"]),
                "source_file": row["source_file"],
            })
    matrix = sp.csr_matrix(
        (np.concatenate(value_parts), np.concatenate(indices_parts), np.asarray(indptr)),
        shape=(len(obs_rows), len(symbols)),
    )
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    if matrix.data.max(initial=0) > np.iinfo(np.int32).max:
        raise ValueError("Aggregated Tahoe gene counts exceed int32")
    matrix = matrix.astype(np.int32)
    var = pd.DataFrame(index=pd.Index(symbols, name="gene"))
    var["ensembl_ids"] = [ensembls.loc[name] for name in symbols]
    data = ad.AnnData(matrix, obs=pd.DataFrame(obs_rows).set_index("cell_id"), var=var)
    data.uns["c3vcell_tahoe"] = {
        "source": "Tahoe-100M", "license": "CC0", "control": TAHOE_CONTROL,
        "counts": "raw UMI counts after removing the CLS token",
        "cell_identity": "tahoe-v1: sample and BARCODE_SUB_LIB_ID",
        "sampling_seed": seed, "complete_shards": [path.name for path in paths],
        "available_group_counts": json.dumps({str(key): value for key, value in seen.items()}),
    }
    validate_counts(data)
    output.parent.mkdir(parents=True, exist_ok=True)
    data.write_h5ad(output, compression="gzip")
    print(f"Saved {data.n_obs} Tahoe cells from {len(paths)} completed shards", flush=True)
