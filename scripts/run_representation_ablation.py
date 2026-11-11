"""Compare representation objectives without using outer held-out perturbations."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import anndata as ad
import numpy as np
import scipy.sparse as sp
import torch
from torch.nn import functional as F

from c3vcell.data import validate_counts
from c3vcell.integrity import file_sha256, snapshot_files
from c3vcell.model import C3VCell, ModelConfig, negative_binomial_loss
from c3vcell.training import _batch, fit_representation, representation_diagnostics


def _write_new(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


@torch.no_grad()
def _reconstruction(
    model: C3VCell, matrix: sp.csr_matrix, *, batch_size: int, device: torch.device,
) -> float:
    model.eval()
    total = 0.0
    for start in range(0, matrix.shape[0], batch_size):
        rows = np.arange(start, min(start + batch_size, matrix.shape[0]))
        counts = _batch(matrix, rows, device)
        latent = model.encode(counts)
        mean = model.probabilities(latent) * counts.sum(dim=1, keepdim=True)
        theta = F.softplus(model.log_theta) + 1e-4
        total += float(negative_binomial_loss(counts, mean, theta)) * len(rows)
    return total / matrix.shape[0]


@torch.no_grad()
def _mask_variation(
    model: C3VCell, matrix: sp.csr_matrix, *, batch_size: int, seed: int,
    cells: int, repeats: int, device: torch.device,
) -> dict:
    selected = np.sort(np.random.default_rng(seed).choice(matrix.shape[0], cells, replace=False))
    repeated = np.repeat(selected, repeats)
    parts = []
    model.eval()
    torch.manual_seed(seed)
    for start in range(0, len(repeated), batch_size):
        counts = _batch(matrix, repeated[start:start + batch_size], device)
        visible = (torch.rand_like(counts) >= model.config.mask_probability).to(counts.dtype)
        parts.append(model.encoder(counts, visible).cpu().numpy())
    encoded = np.concatenate(parts).reshape(cells, repeats, model.config.latent_dim)
    if not np.isfinite(encoded).all():
        raise RuntimeError("Mask variation produced non-finite encodings")
    return {
        "cells": cells,
        "independent_masks_per_cell": repeats,
        "mean_within_cell_dimension_std": float(encoded.astype(np.float64).std(axis=1).mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--weights", type=float, nargs="+", default=[0.1, 1.0, 5.0])
    parser.add_argument("--covariance-weights", type=float, nargs="+", default=[0.0])
    parser.add_argument(
        "--regularization-views", choices=["masked", "unmasked"], nargs="+", default=["masked"],
    )
    parser.add_argument("--pretrain-steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--latent-dim", type=int, default=96)
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--validation-cells", type=int, default=512)
    parser.add_argument("--train-diagnostic-cells", type=int, default=512)
    parser.add_argument("--mask-cells", type=int, default=64)
    parser.add_argument("--mask-repeats", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20261110)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.outdir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.outdir}")
    if (
        not args.weights or len(set(args.weights)) != len(args.weights)
        or any(not math.isfinite(w) or w < 0 for w in args.weights)
        or not args.covariance_weights
        or len(set(args.covariance_weights)) != len(args.covariance_weights)
        or any(not math.isfinite(w) or w < 0 for w in args.covariance_weights)
        or len(set(args.regularization_views)) != len(args.regularization_views)
        or not math.isfinite(args.learning_rate) or args.learning_rate <= 0
        or min(
            args.pretrain_steps, args.batch_size, args.latent_dim, args.hidden_dim,
            args.threads, args.mask_cells,
        ) < 1
        or min(args.validation_cells, args.train_diagnostic_cells, args.mask_repeats) < 2
        or args.mask_cells > args.validation_cells
        or (any(w > 0 for w in args.covariance_weights) and args.batch_size < 2)
    ):
        raise ValueError("Invalid weight, learning rate, or ablation dimensions")
    torch.set_num_threads(args.threads)
    device = torch.device("cpu")
    source_digest = file_sha256(args.source)
    data = ad.read_h5ad(args.source)
    validate_counts(data)
    if data.n_obs <= args.validation_cells:
        raise ValueError("The training source must contain more cells than the diagnostic split")
    rng = np.random.default_rng(args.seed)
    validation_rows = np.sort(rng.choice(data.n_obs, args.validation_cells, replace=False))
    training_mask = np.ones(data.n_obs, dtype=bool)
    training_mask[validation_rows] = False
    if training_mask.sum() < 2:
        raise ValueError("At least two cells must remain for representation fitting")
    matrix = sp.csr_matrix(data.X, dtype=np.float32)
    training_matrix = matrix[training_mask]
    validation_matrix = matrix[validation_rows]
    training_ids = data.obs_names[training_mask].astype(str).tolist()
    validation_ids = data.obs_names[validation_rows].astype(str).tolist()
    if set(training_ids) & set(validation_ids):
        raise ValueError("Training and diagnostic cells must be distinct")
    source_root = Path(__file__).resolve().parents[1]
    code = sorted((source_root / "src" / "c3vcell").rglob("*.py"))
    code.extend([source_root / "pyproject.toml", Path(__file__).resolve()])
    configurations = [
        {"variance_weight": variance, "covariance_weight": covariance, "regularization_view": view}
        for variance, covariance, view in itertools.product(
            args.weights, args.covariance_weights, args.regularization_views,
        )
    ]
    args.outdir.mkdir(parents=True)
    code_sha256 = {str(path.relative_to(source_root)): file_sha256(path) for path in code}
    snapshot_sha256 = snapshot_files(source_root, code_sha256, args.outdir / "code-snapshot.zip")
    protocol = {
        "kind": "within-training-cell-representation-ablation-v2",
        "source": str(args.source.resolve()),
        "source_sha256": source_digest,
        "source_cells": data.n_obs,
        "genes": data.n_vars,
        "training_cells": int(training_mask.sum()),
        "validation_cell_ids": validation_ids,
        "same_targets_note": (
            "The diagnostic cells are held out only from representation fitting; "
            "they can share perturbation targets with training cells."
        ),
        "outer_holdout_note": "No outer held-out perturbation expression is read by this script.",
        "parameters": {
            "weights": args.weights,
            "covariance_weights": args.covariance_weights,
            "regularization_views": args.regularization_views,
            "configurations": configurations,
            "pretrain_steps": args.pretrain_steps,
            "batch_size": args.batch_size,
            "latent_dim": args.latent_dim,
            "hidden_dim": args.hidden_dim,
            "learning_rate": args.learning_rate,
            "validation_cells": args.validation_cells,
            "train_diagnostic_cells": args.train_diagnostic_cells,
            "mask_cells": args.mask_cells,
            "mask_repeats": args.mask_repeats,
            "seed": args.seed,
            "threads": args.threads,
        },
        "code_sha256": code_sha256,
        "code_snapshot_sha256": snapshot_sha256,
        "python": sys.version,
        "torch": torch.__version__,
    }
    _write_new(args.outdir / "protocol.json", protocol)
    results = []
    for index, configuration in enumerate(configurations):
        directory = args.outdir / f"weight-{index:02d}"
        directory.mkdir()
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        config = ModelConfig(
            data.n_vars, args.latent_dim, args.hidden_dim, **configuration,
        )
        model = C3VCell(config).to(device)
        started = time.monotonic()
        with (directory / "training.jsonl").open("x", encoding="utf-8") as log:
            fit_representation(
                model, training_matrix, rng=np.random.default_rng(args.seed), device=device,
                steps=args.pretrain_steps, batch_size=args.batch_size,
                learning_rate=args.learning_rate, log=log,
            )
        train_diagnostic = representation_diagnostics(
            model, training_matrix, device=device, batch_size=args.batch_size,
            seed=args.seed, max_cells=args.train_diagnostic_cells,
        )
        validation_diagnostic = representation_diagnostics(
            model, validation_matrix, device=device, batch_size=args.batch_size,
            seed=args.seed, max_cells=args.validation_cells,
        )
        mask_diagnostic = _mask_variation(
            model, validation_matrix, batch_size=args.batch_size, seed=args.seed + 1,
            cells=args.mask_cells, repeats=args.mask_repeats, device=device,
        )
        mask_diagnostic["ratio_to_cross_cell_std"] = (
            mask_diagnostic["mean_within_cell_dimension_std"]
            / max(validation_diagnostic["mean_dimension_std"], 1e-12)
        )
        record = {
            **configuration,
            "training_diagnostic": train_diagnostic,
            "validation_diagnostic": validation_diagnostic,
            "validation_unmasked_nb": _reconstruction(
                model, validation_matrix, batch_size=args.batch_size, device=device,
            ),
            "mask_diagnostic": mask_diagnostic,
            "elapsed_seconds": time.monotonic() - started,
            "note": "The saved state is representation-only; its population flow was not trained.",
        }
        torch.save(
            {"kind": "representation-only-v1", "model_config": model.configuration(),
             "model_state": model.state_dict(), "source_sha256": source_digest},
            directory / "representation-only.pt",
        )
        _write_new(directory / "result.json", record)
        results.append(record)
        print(json.dumps({"completed_configuration": configuration, **record}), flush=True)
    if file_sha256(args.source) != source_digest:
        raise RuntimeError("Training source changed during the ablation")
    if {str(path.relative_to(source_root)): file_sha256(path) for path in code} != protocol["code_sha256"]:
        raise RuntimeError("Code changed during the ablation")
    if file_sha256(args.outdir / "code-snapshot.zip") != snapshot_sha256:
        raise RuntimeError("Source snapshot changed during the ablation")
    _write_new(args.outdir / "result.json", {
        "protocol_sha256": file_sha256(args.outdir / "protocol.json"),
        "results": results,
        "note": (
            "Diagnostic reconstruction and variation are not evidence of perturbation "
            "prediction accuracy or an official challenge score."
        ),
    })


if __name__ == "__main__":
    main()
