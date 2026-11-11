"""Diagnose representation collapse on a model's training cells."""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import anndata as ad
import numpy as np
import scipy.sparse as sp
import torch

from .data import validate_counts
from .identity import check_cell_id_schemes
from .integrity import file_sha256
from .training import load_model, representation_diagnostics


def diagnose_checkpoint(
    checkpoint_path: Path,
    source_path: Path,
    output_path: Path,
    *,
    max_cells: int = 512,
    batch_size: int = 64,
    seed: int = 0,
    device_name: str = "cpu",
) -> None:
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite {output_path}")
    if max_cells < 2 or batch_size < 1:
        raise ValueError("Use at least two cells and a positive batch size")
    checkpoint_digest = file_sha256(checkpoint_path)
    source_digest = file_sha256(source_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("format_version") != 1:
        raise ValueError("Unsupported model checkpoint")
    expected = checkpoint.get("training_source_sha256")
    if expected is None:
        warnings.warn(
            "The checkpoint predates training-file fingerprints; matching cell IDs "
            "cannot verify the training counts.",
            RuntimeWarning, stacklevel=2,
        )
    elif expected != source_digest:
        raise ValueError("Diagnostic source differs from the checkpoint training-count file")
    source = ad.read_h5ad(source_path, backed="r")
    try:
        if not source.obs_names.is_unique or not source.var_names.is_unique:
            raise ValueError("Diagnostic source requires unique cell and gene identifiers")
        check_cell_id_schemes(checkpoint["training_cell_ids"], source.obs_names)
        if (
            set(source.obs_names.astype(str)) != set(checkpoint["training_cell_ids"])
            or source.var_names.astype(str).tolist() != checkpoint["genes"]
        ):
            raise ValueError("Diagnostic source cell or gene identifiers differ from the checkpoint")
        rows = np.sort(
            np.random.default_rng(seed).choice(source.n_obs, min(max_cells, source.n_obs), replace=False)
        )
        matrix = sp.csr_matrix(source.X[rows, :], dtype=np.float32)
    finally:
        source.file.close()
    validate_counts(ad.AnnData(matrix))
    if checkpoint.get("kind") == "tahoe-drug-smiles-dose":
        from .drug import load_drug_model

        model, _, _ = load_drug_model(checkpoint_path, device_name)
    else:
        model, _, _ = load_model(checkpoint_path, device_name)
    result = representation_diagnostics(
        model, matrix, device=torch.device(device_name), batch_size=batch_size,
        seed=seed, max_cells=max_cells,
    )
    result.update({
        "format_version": 1,
        "checkpoint_sha256": checkpoint_digest,
        "training_source_sha256": source_digest,
        "checkpoint_has_source_fingerprint": expected is not None,
    })
    if file_sha256(checkpoint_path) != checkpoint_digest or file_sha256(source_path) != source_digest:
        raise RuntimeError("A diagnostic input changed during evaluation")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
