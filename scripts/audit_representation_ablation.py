"""Explore latent rank and nuisance associations in a completed representation ablation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import scipy.sparse as sp
import torch

from c3vcell.data import validate_counts
from c3vcell.integrity import file_sha256
from c3vcell.model import C3VCell, ModelConfig
from c3vcell.training import _batch


def _correlation(left: np.ndarray, right: np.ndarray) -> float | None:
    if np.std(left) <= 1e-12 or np.std(right) <= 1e-12:
        return None
    return abs(float(np.corrcoef(left, right)[0, 1]))


def _group_association(values: np.ndarray, labels: np.ndarray, *, seed: int, n: int) -> dict:
    unique, groups = np.unique(labels, return_inverse=True)
    if len(unique) < 2:
        return {"categories": len(unique), "eta_squared": None, "permutation_mean": None}

    def eta(indices: np.ndarray) -> float:
        center = values.mean()
        denominator = float(np.square(values - center).sum())
        if denominator <= 1e-24:
            return 0.0
        sizes = np.bincount(indices, minlength=len(unique))
        sums = np.bincount(indices, weights=values, minlength=len(unique))
        return float(np.sum(sizes * np.square(sums / sizes - center)) / denominator)

    rng = np.random.default_rng(seed)
    observed = eta(groups)
    shuffled = np.asarray([eta(rng.permutation(groups)) for _ in range(n)])
    return {
        "categories": len(unique),
        "eta_squared": observed,
        "permutation_mean": float(shuffled.mean()),
        "permutation_95th_percentile": float(np.quantile(shuffled, 0.95)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--permutations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20261110)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    if args.batch_size < 1 or args.permutations < 1:
        raise ValueError("Batch size and permutation count must be positive")
    protocol_path = args.run / "protocol.json"
    result_path = args.run / "result.json"
    protocol_digest = file_sha256(protocol_path)
    result_digest = file_sha256(result_path)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if protocol.get("kind") not in {
        "within-training-cell-representation-ablation-v1",
        "within-training-cell-representation-ablation-v2",
    }:
        raise ValueError("Unsupported ablation protocol")
    if result.get("protocol_sha256") != protocol_digest:
        raise ValueError("Ablation result does not match the protocol")
    source_digest = file_sha256(args.source)
    if source_digest != protocol["source_sha256"]:
        raise ValueError("Audit source does not match the ablation training-count file")
    ids = protocol["validation_cell_ids"]
    if len(ids) < 2 or len(ids) != len(set(ids)):
        raise ValueError("Ablation diagnostic cell IDs must be unique")
    data = ad.read_h5ad(args.source, backed="r")
    try:
        if not data.obs_names.is_unique or not data.var_names.is_unique:
            raise ValueError("Audit source requires unique cell and gene identifiers")
        positions = data.obs_names.get_indexer(ids)
        if np.any(positions < 0):
            raise ValueError("Ablation diagnostic cells are missing from the source")
        positions = np.sort(positions)
        matrix = sp.csr_matrix(data.X[positions, :], dtype=np.float32)
        obs = data.obs.iloc[positions].copy()
    finally:
        data.file.close()
    validate_counts(ad.AnnData(matrix))
    matrix.eliminate_zeros()
    log_library = np.log1p(np.asarray(matrix.sum(axis=1)).ravel())
    log_detected = np.log1p(np.diff(matrix.indptr))
    configurations = protocol["parameters"].get("configurations")
    if configurations is None:
        configurations = [
            {"variance_weight": weight, "covariance_weight": 0.0, "regularization_view": "masked"}
            for weight in protocol["parameters"]["weights"]
        ]
    actual = [
        {
            "variance_weight": row["variance_weight"],
            "covariance_weight": row.get("covariance_weight", 0.0),
            "regularization_view": row.get("regularization_view", "masked"),
        }
        for row in result["results"]
    ]
    if actual != configurations:
        raise ValueError("Ablation configurations do not match the protocol")
    records = []
    for index, configuration in enumerate(configurations):
        state_path = args.run / f"weight-{index:02d}" / "representation-only.pt"
        state_digest = file_sha256(state_path)
        state = torch.load(state_path, map_location="cpu", weights_only=True)
        if (
            state.get("kind") != "representation-only-v1"
            or state.get("source_sha256") != source_digest
            or any(
                state["model_config"].get(key, default) != configuration[key]
                for key, default in (
                    ("variance_weight", 0.1), ("covariance_weight", 0.0),
                    ("regularization_view", "masked"),
                )
            )
            or state["model_config"]["n_genes"] != matrix.shape[1]
        ):
            raise ValueError(f"Representation state does not match ablation configuration {configuration}")
        model = C3VCell(ModelConfig(**state["model_config"]))
        model.load_state_dict(state["model_state"])
        model.eval()
        parts = []
        with torch.no_grad():
            for start in range(0, matrix.shape[0], args.batch_size):
                rows = np.arange(start, min(start + args.batch_size, matrix.shape[0]))
                parts.append(model.encode(_batch(matrix, rows, torch.device("cpu"))).numpy())
        latent = np.concatenate(parts).astype(np.float64)
        if not np.isfinite(latent).all():
            raise RuntimeError("Representation state produced non-finite encodings")
        centered = latent - latent.mean(axis=0, keepdims=True)
        left, singular, _ = np.linalg.svd(centered, full_matrices=False)
        eigenvalues = np.square(singular)
        total = float(eigenvalues.sum())
        if total <= 1e-24:
            top_fraction = None
            participation_rank = 0.0
            first_component = np.zeros(len(centered))
        else:
            top_fraction = float(eigenvalues[0] / total)
            participation_rank = float(total**2 / np.square(eigenvalues).sum())
            first_component = left[:, 0] * singular[0]
        associations = {
            "absolute_pc1_log_library_correlation": _correlation(first_component, log_library),
            "absolute_pc1_log_detected_genes_correlation": _correlation(first_component, log_detected),
        }
        for name in ("batch", "target_gene"):
            if name in obs:
                if obs[name].isna().any():
                    raise ValueError(f"Audit metadata column {name!r} contains missing values")
                associations[f"pc1_by_{name}"] = _group_association(
                    first_component, obs[name].astype(str).to_numpy(),
                    seed=args.seed, n=args.permutations,
                )
        if file_sha256(state_path) != state_digest:
            raise RuntimeError("Representation state changed during the audit")
        records.append({
            **configuration,
            "state_sha256": state_digest,
            "mean_dimension_std": float(centered.std(axis=0).mean()),
            "first_component_variance_fraction": top_fraction,
            "participation_rank": participation_rank,
            "associations": associations,
        })
    if (
        file_sha256(args.source) != source_digest
        or file_sha256(protocol_path) != protocol_digest
        or file_sha256(result_path) != result_digest
    ):
        raise RuntimeError("An ablation input changed during the audit")
    output = {
        "kind": "exploratory-ablation-representation-audit-v1",
        "source_sha256": source_digest,
        "protocol_sha256": protocol_digest,
        "ablation_result_sha256": result_digest,
        "audit_script_sha256": file_sha256(Path(__file__)),
        "diagnostic_cells": len(ids),
        "permutations": args.permutations,
        "seed": args.seed,
        "results": records,
        "note": (
            "PC1 associations are exploratory and can be confounded by cell composition, "
            "batch and sampling depth. They are not causal or held-out prediction results."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(output, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
