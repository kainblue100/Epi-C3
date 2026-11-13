"""Shared, versioned data guards for the KHP transfer experiment."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import h5py
import numpy as np
import scipy.sparse as sp


def canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_plan(path: Path) -> dict:
    with path.open(encoding="utf-8") as source:
        plan = json.load(source)
    if plan.get("format") != "epi-c3-khp-cd8-transfer-plan-v1":
        raise ValueError("Unexpected frozen-plan format")
    unsigned = dict(plan)
    claimed = unsigned.pop("plan_sha256")
    actual = hashlib.sha256(canonical_bytes(unsigned)).hexdigest()
    if claimed != actual:
        raise ValueError("Frozen plan content does not match its SHA-256")
    return plan


def case_groups(plan: dict, condition: str) -> tuple[list[dict], list[dict]]:
    if condition not in plan["design"]["conditions"]:
        raise ValueError(f"Unknown condition {condition!r}")
    selected = plan["selection"]
    control = plan["design"]["control_label"]
    train = []
    for cell_type in plan["design"]["source_cell_types"]:
        train.extend([
            {
                "name": f"source_control_{cell_type}",
                "cell_type": cell_type,
                "condition": control,
                "rows": selected["source_controls"][cell_type],
            },
            {
                "name": f"source_treatment_{cell_type}",
                "cell_type": cell_type,
                "condition": condition,
                "rows": selected["source_treatments"][condition][cell_type],
            },
        ])
    target = plan["design"]["target_cell_type"]
    train.append({
        "name": "target_input_control",
        "cell_type": target,
        "condition": control,
        "rows": selected["target_input_control"],
    })
    reference = [
        {
            "name": "target_reference_control",
            "cell_type": target,
            "condition": control,
            "rows": selected["target_reference_control"],
        },
        {
            "name": "target_reference_treatment",
            "cell_type": target,
            "condition": condition,
            "rows": selected["target_reference_treatments"][condition],
        },
    ]
    train_rows = [row for group in train for row in group["rows"]]
    ref_rows = [row for group in reference for row in group["rows"]]
    if len(set(train_rows)) != len(train_rows) or len(set(ref_rows)) != len(ref_rows):
        raise ValueError("A case contains duplicate row indices")
    if set(train_rows).intersection(ref_rows):
        raise ValueError("Training/input and reference rows overlap")
    return train, reference


def group_ranges(groups: list[dict]) -> dict:
    result = {}
    offset = 0
    for group in groups:
        if group["name"] in result or not group["rows"]:
            raise ValueError("Case groups must have unique names and nonempty row lists")
        end = offset + len(group["rows"])
        result[group["name"]] = {
            "start": offset,
            "stop": end,
            "cell_type": group["cell_type"],
            "condition": group["condition"],
        }
        offset = end
    return result


def read_csr_rows(group: h5py.Group, rows: list[int], block_rows: int = 256) -> sp.csr_matrix:
    n_rows, n_genes = map(int, group.attrs["shape"])
    row_array = np.asarray(rows, dtype=np.int64)
    if (
        row_array.ndim != 1
        or not len(row_array)
        or np.any(row_array < 0)
        or np.any(row_array >= n_rows)
        or np.any(np.diff(row_array) <= 0)
        or block_rows < 1
    ):
        raise ValueError("Selected CSR rows must be sorted, unique, and in bounds")
    indptr = np.asarray(group["indptr"][:], dtype=np.int64)
    if indptr.shape != (n_rows + 1,) or indptr[0] != 0 or np.any(np.diff(indptr) < 0):
        raise ValueError("Invalid CSR row pointer")
    if indptr[-1] != len(group["data"]) or indptr[-1] != len(group["indices"]):
        raise ValueError("CSR data and indices do not match the row pointer")
    values = []
    columns = []
    new_ptr = [0]
    offsets = row_array // block_rows
    for block in np.unique(offsets):
        block_start = int(block * block_rows)
        block_stop = min(block_start + block_rows, n_rows)
        start, stop = int(indptr[block_start]), int(indptr[block_stop])
        block_values = group["data"][start:stop]
        block_columns = group["indices"][start:stop]
        for row in row_array[offsets == block]:
            left = int(indptr[row]) - start
            right = int(indptr[row + 1]) - start
            values.append(block_values[left:right])
            columns.append(block_columns[left:right])
            new_ptr.append(new_ptr[-1] + right - left)
    data = np.concatenate(values) if values else np.asarray([], dtype=np.float64)
    indices = np.concatenate(columns) if columns else np.asarray([], dtype=np.int64)
    if (
        not np.isfinite(data).all()
        or np.any(data < 0)
        or np.any(data != np.floor(data))
        or np.any(data > np.iinfo(np.int32).max)
        or np.any(indices < 0)
        or np.any(indices >= n_genes)
    ):
        raise ValueError("Count layer contains invalid or nonintegral values")
    matrix = sp.csr_matrix(
        (data.astype(np.int32), indices, np.asarray(new_ptr, dtype=np.int64)),
        shape=(len(rows), n_genes),
    )
    if not matrix.has_canonical_format:
        raise ValueError("Count matrix has duplicate or unsorted gene entries")
    if np.any(np.asarray(matrix.sum(axis=1)).ravel() <= 0):
        raise ValueError("Selected cells must have nonzero measured-gene library size")
    return matrix


def log_cpm(counts: sp.csr_matrix) -> sp.csr_matrix:
    matrix = counts.astype(np.float32, copy=True)
    libraries = np.asarray(matrix.sum(axis=1), dtype=np.float64).ravel()
    if not np.isfinite(libraries).all() or np.any(libraries <= 0):
        raise ValueError("Normalization requires positive finite library sizes")
    matrix = sp.diags((10000.0 / libraries).astype(np.float32)).dot(matrix).tocsr()
    np.log1p(matrix.data, out=matrix.data)
    if not np.isfinite(matrix.data).all():
        raise ValueError("Nonfinite normalized expression")
    return matrix


def write_npz_new(path: Path, matrix: sp.csr_matrix) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    created = False
    try:
        with temporary.open("xb") as out:
            created = True
            sp.save_npz(out, matrix, compressed=True)
            out.flush()
            os.fsync(out.fileno())
        os.link(temporary, path)
    finally:
        if created and temporary.exists():
            temporary.unlink()


def write_json_new(path: Path, value: object) -> None:
    payload = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    created = False
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as out:
            created = True
            out.write(payload)
            out.flush()
            os.fsync(out.fileno())
        os.link(temporary, path)
    finally:
        if created and temporary.exists():
            temporary.unlink()
