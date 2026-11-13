"""Reproducible two-stage training on a bounded AnnData subset."""

from __future__ import annotations

import json
import os
import random
import warnings
from pathlib import Path
from typing import Literal, TextIO

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

from .data import CONTROL, validate_counts
from .epigenetics import feature_names, load_epigenetic_bundle, validate_prior_data
from .identity import check_cell_id_schemes
from .integrity import file_sha256
from .model import C3VCell, ModelConfig
from .reference import ReferenceCounts, load_reference, reference_metadata


def _batch(matrix: sp.csr_matrix, rows: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(matrix[rows].toarray(), dtype=torch.float32, device=device)


def fit_representation(
    model: C3VCell,
    matrix: sp.csr_matrix,
    *,
    rng: np.random.Generator,
    device: torch.device,
    steps: int,
    batch_size: int,
    learning_rate: float,
    log: TextIO,
    reference: ReferenceCounts | None = None,
    reference_fraction: float = 0.5,
) -> None:
    optimizer = torch.optim.AdamW(
        list(model.encoder.parameters())
        + list(model.predictor.parameters())
        + list(model.decoder.parameters())
        + [model.log_theta],
        lr=learning_rate,
    )
    model.train()
    for step in range(1, steps + 1):
        use_reference = reference is not None and rng.random() < reference_fraction
        current = reference.matrix if use_reference and reference is not None else matrix
        rows = rng.integers(0, current.shape[0], size=batch_size)
        counts = _batch(current, rows, device)
        measured = None
        if use_reference and reference is not None:
            measured = torch.as_tensor(
                np.broadcast_to(reference.measured, counts.shape).copy(),
                dtype=counts.dtype, device=device,
            )
        optimizer.zero_grad(set_to_none=True)
        loss, metrics = model.representation_loss(counts, measured=measured)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite representation loss at step {step}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        model.update_teacher()
        if step == 1 or step % 25 == 0 or step == steps:
            record = {"phase": "representation", "step": step, **metrics}
            if reference is not None:
                record["source"] = "reference" if use_reference else "training"
            log.write(json.dumps(record) + "\n")
            log.flush()
            print(json.dumps(record), flush=True)


@torch.no_grad()
def representation_diagnostics(
    model: C3VCell,
    matrix: sp.csr_matrix,
    *,
    device: torch.device,
    batch_size: int,
    seed: int,
    max_cells: int = 512,
) -> dict:
    if matrix.shape[0] < 2 or batch_size < 1 or max_cells < 2:
        raise ValueError("Representation diagnostics need at least two cells and positive batch size")
    rows = np.sort(
        np.random.default_rng(seed).choice(matrix.shape[0], min(max_cells, matrix.shape[0]), replace=False)
    )
    was_training = model.training
    model.eval()
    try:
        parts = [
            model.encode(_batch(matrix, rows[start:start + batch_size], device)).cpu().numpy()
            for start in range(0, len(rows), batch_size)
        ]
    finally:
        model.train(was_training)
    latent = np.concatenate(parts).astype(np.float64)
    if not np.isfinite(latent).all():
        raise RuntimeError("Representation diagnostics found non-finite latent values")
    per_dimension_std = latent.std(axis=0)
    mean_std = float(per_dimension_std.mean())
    rms = float(np.sqrt(np.mean(np.square(latent))))
    relative_std = mean_std / max(rms, 1e-12)
    centered = latent - latent.mean(axis=0, keepdims=True)
    eigenvalues = np.square(np.linalg.svd(centered, full_matrices=False, compute_uv=False))
    total_variance = float(eigenvalues.sum())
    if total_variance > 1e-24:
        top_fraction = float(eigenvalues[0] / total_variance)
        participation_rank = float(total_variance**2 / np.square(eigenvalues).sum())
    else:
        top_fraction = None
        participation_rank = 0.0
    low_variation = relative_std < 0.05
    low_rank = (
        len(rows) >= 10 and latent.shape[1] > 1
        and (participation_rank < 2 or (top_fraction is not None and top_fraction > 0.9))
    )
    return {
        "sampled_cells": len(rows),
        "sample_seed": seed,
        "mean_dimension_std": mean_std,
        "median_dimension_std": float(np.median(per_dimension_std)),
        "max_dimension_std": float(per_dimension_std.max()),
        "latent_rms": rms,
        "relative_mean_std": relative_std,
        "first_component_variance_fraction": top_fraction,
        "participation_rank": participation_rank,
        "low_variation_warning": low_variation,
        "low_rank_warning": low_rank,
        "collapse_warning": low_variation or low_rank,
        "note": "Low variation and rank are heuristic warnings, not biological validity tests.",
    }


def read_graph(path: Path | None, genes: list[str], device: torch.device) -> torch.Tensor | None:
    if path is None:
        return None
    edges = pd.read_csv(path)
    if not {"source", "target", "weight"}.issubset(edges.columns):
        raise ValueError("Graph CSV must contain source,target,weight columns")
    lookup = {name: index for index, name in enumerate(genes)}
    rows = list(range(len(genes)))
    cols = list(range(len(genes)))
    weights = [1.0] * len(genes)
    discarded = 0
    for edge in edges.itertuples(index=False):
        source, target, weight = str(edge.source), str(edge.target), float(edge.weight)
        if not np.isfinite(weight):
            raise ValueError("Graph weights must be finite")
        if source not in lookup or target not in lookup:
            discarded += 1
            continue
        rows.append(lookup[source])
        cols.append(lookup[target])
        weights.append(weight)
    if len(rows) == len(genes):
        raise ValueError("No graph edges match the model gene axis")
    if discarded:
        print(f"Graph: skipped {discarded} edges with genes absent from this assay", flush=True)
    matrix = sp.coo_matrix((weights, (rows, cols)), shape=(len(genes), len(genes))).tocsr()
    normalizer = np.asarray(abs(matrix).sum(axis=1)).ravel()
    matrix = (sp.diags(1.0 / normalizer) @ matrix).tocoo()
    indices = torch.tensor(np.vstack((matrix.row, matrix.col)), dtype=torch.long, device=device)
    values = torch.tensor(matrix.data, dtype=torch.float32, device=device)
    return torch.sparse_coo_tensor(indices, values, matrix.shape, device=device).coalesce()


def _graph_payload(graph: torch.Tensor | None) -> dict | None:
    if graph is None:
        return None
    graph = graph.coalesce().cpu()
    return {"indices": graph.indices(), "values": graph.values(), "shape": list(graph.shape)}


def _restore_graph(payload: dict | None, device: torch.device) -> torch.Tensor | None:
    if payload is None:
        return None
    graph = torch.sparse_coo_tensor(
        payload["indices"].to(device),
        payload["values"].to(device),
        tuple(payload["shape"]),
        device=device,
    ).coalesce()
    if (
        len(graph.shape) != 2 or graph.shape[0] != graph.shape[1]
        or not torch.isfinite(graph.values()).all()
    ):
        raise ValueError("Saved graph must be a finite square sparse matrix")
    rows = torch.zeros(graph.shape[0], dtype=torch.float64, device=device)
    rows.index_add_(0, graph.indices()[0], graph.values().abs().to(dtype=torch.float64))
    if not torch.allclose(rows, torch.ones_like(rows), rtol=1e-5, atol=1e-6):
        raise ValueError("Saved graph rows must already be normalized")
    return graph


def _contexts(obs: pd.DataFrame, column: str, default: str) -> np.ndarray:
    if column not in obs:
        return np.full(len(obs), default, dtype=object)
    if obs[column].isna().any():
        raise ValueError(f"{column} contains missing values")
    return obs[column].astype(str).to_numpy()


def _control_groups(
    obs: pd.DataFrame, control: str
) -> tuple[dict[tuple[str, str], np.ndarray], dict[str, np.ndarray]]:
    labels = obs["target_gene"].astype(str).to_numpy()
    contexts = _contexts(obs, "context", "local")
    batches = _contexts(obs, "batch", "")
    exact: dict[tuple[str, str], list[int]] = {}
    by_context: dict[str, list[int]] = {}
    for row in np.flatnonzero(labels == control):
        exact.setdefault((contexts[row], batches[row]), []).append(row)
        by_context.setdefault(contexts[row], []).append(row)
    if not by_context:
        raise ValueError(f"No training controls match {control!r}")
    return (
        {key: np.asarray(value) for key, value in exact.items()},
        {key: np.asarray(value) for key, value in by_context.items()},
    )


@torch.no_grad()
def _population_shift_groups(
    model: C3VCell,
    matrix: sp.csr_matrix,
    labels: np.ndarray,
    contexts: np.ndarray,
    batches: np.ndarray,
    exact: dict[tuple[str, str], np.ndarray],
    by_context: dict[str, np.ndarray],
    *,
    control: str,
    min_group_cells: int,
    batch_size: int,
    device: torch.device,
) -> tuple[dict[str, list[tuple[np.ndarray, np.ndarray]]], dict]:
    latent_parts = []
    model.eval()
    for start in range(0, matrix.shape[0], batch_size):
        rows = np.arange(start, min(start + batch_size, matrix.shape[0]))
        latent_parts.append(model.encode(_batch(matrix, rows, device)).cpu().numpy())
    latent = np.concatenate(latent_parts)
    perturbed: dict[tuple[str, str, str], list[int]] = {}
    for row in np.flatnonzero(labels != control):
        perturbed.setdefault((labels[row], contexts[row], batches[row]), []).append(int(row))
    control_means: dict[tuple[str, str], np.ndarray] = {}
    groups: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
    excluded = 0
    for (target, context, batch), rows in perturbed.items():
        key = (context, batch)
        controls = exact.get(key)
        if controls is None:
            controls = by_context[context]
        if len(rows) < min_group_cells or len(controls) < min_group_cells:
            excluded += 1
            continue
        if key not in control_means:
            control_means[key] = latent[controls].mean(axis=0)
        shift = latent[rows].mean(axis=0) - control_means[key]
        groups.setdefault(target, []).append((controls, shift.astype(np.float32)))
    if not groups:
        raise ValueError(
            "No context/batch/target groups have enough perturbed cells and matched controls"
        )
    omitted = sorted((set(labels) - {control}) - set(groups))
    details = {
        "eligible_groups": sum(map(len, groups.values())),
        "excluded_groups": excluded,
        "targets_without_eligible_groups": omitted,
        "min_group_cells": min_group_cells,
    }
    if excluded or omitted:
        print(f"Population-shift group selection: {json.dumps(details)}", flush=True)
    return groups, details


def train(
    train_path: Path,
    outdir: Path,
    *,
    control: str = CONTROL,
    latent_dim: int = 128,
    hidden_dim: int = 256,
    pretrain_steps: int = 300,
    flow_steps: int = 500,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    variance_weight: float = 0.1,
    covariance_weight: float = 0.0,
    regularization_view: Literal["masked", "unmasked"] = "masked",
    seed: int = 0,
    device_name: str = "cpu",
    graph_path: Path | None = None,
    epigenetic_bundle: Path | None = None,
    allow_unmatched_batch: bool = False,
    reference_path: Path | None = None,
    reference_fraction: float = 0.5,
    reference_max_cells: int = 10_000,
    reference_seed: int | None = None,
    reference_min_coverage: float = 0.2,
    expected_reference_identity: str | None = None,
    flow_target: str = "paired",
    min_group_cells: int = 5,
    _prior_checkpoint: dict | None = None,
) -> Path:
    if min(pretrain_steps, flow_steps, batch_size) < 1:
        raise ValueError("Training steps and batch size must be positive")
    if learning_rate <= 0:
        raise ValueError("Learning rate must be positive")
    if not np.isfinite(variance_weight) or variance_weight < 0:
        raise ValueError("variance_weight must be finite and nonnegative")
    if not np.isfinite(covariance_weight) or covariance_weight < 0:
        raise ValueError("covariance_weight must be finite and nonnegative")
    if regularization_view not in {"masked", "unmasked"}:
        raise ValueError("regularization_view must be masked or unmasked")
    if covariance_weight > 0 and batch_size < 2:
        raise ValueError("Covariance regularization requires at least two cells per batch")
    if flow_target not in {"paired", "population-shift"} or min_group_cells < 1:
        raise ValueError("Use a supported flow target and positive min_group_cells")
    if reference_path is not None and not 0 < reference_fraction < 1:
        raise ValueError("Reference fraction must be strictly between zero and one")
    if expected_reference_identity is not None and reference_path is None:
        raise ValueError("An expected reference identity requires reference data")
    if (
        (graph_path is not None and epigenetic_bundle is not None)
        or (_prior_checkpoint is not None and (graph_path is not None or epigenetic_bundle is not None))
    ):
        raise ValueError("Choose only one external graph, epigenomic bundle, or saved prior")
    if outdir.exists() and any(outdir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty directory: {outdir}")
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"Requested {device_name}, but CUDA is unavailable")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    rng = np.random.default_rng(seed)

    training_source_sha256 = file_sha256(train_path)
    data = ad.read_h5ad(train_path)
    validate_counts(data)
    if "target_gene" not in data.obs or data.obs["target_gene"].isna().any():
        raise ValueError("Every training cell needs a perturbation label")
    matrix = sp.csr_matrix(data.X, dtype=np.float32)
    genes = data.var_names.astype(str).tolist()
    reference = None
    ref_seed = seed if reference_seed is None else reference_seed
    if reference_path is not None:
        reference = load_reference(
            reference_path, genes, max_cells=reference_max_cells, seed=ref_seed,
            min_coverage=reference_min_coverage,
        )
        check_cell_id_schemes(data.obs_names, reference.cell_ids)
        if expected_reference_identity is not None and reference.source_identity != expected_reference_identity:
            raise ValueError("Reference data changed since the final checkpoint was trained")
    gene_index = {gene: idx for idx, gene in enumerate(genes)}
    labels = data.obs["target_gene"].astype(str).to_numpy()
    contexts = _contexts(data.obs, "context", "local")
    batches = _contexts(data.obs, "batch", "")
    target_rows: dict[str, np.ndarray] = {}
    for name in sorted(set(labels) - {control}):
        if name not in gene_index:
            raise ValueError(f"Target {name!r} is missing from the gene axis")
        target_rows[name] = np.flatnonzero(labels == name)
    if not target_rows:
        raise ValueError("No perturbed training cells")
    exact, by_context = _control_groups(data.obs, control)
    unmatched = sorted({
        (contexts[row], batches[row])
        for rows in target_rows.values()
        for row in rows
        if (contexts[row], batches[row]) not in exact
    })
    if unmatched and not allow_unmatched_batch:
        raise ValueError(
            f"{len(unmatched)} perturbed context/batch groups have no matching controls; "
            "supply matching controls or explicitly allow context-only fallback"
        )
    if unmatched:
        print(f"WARNING: {len(unmatched)} groups use context-only control matching", flush=True)
    for context, _ in unmatched:
        if context not in by_context:
            raise ValueError(f"No controls for context {context!r}")

    epigenetics = None
    epi_values = None
    epi_masks = None
    if _prior_checkpoint is not None:
        if _prior_checkpoint.get("format_version") != 1 or _prior_checkpoint["genes"] != genes:
            raise ValueError("Saved prior must use the exact training gene axis")
        graph = _restore_graph(_prior_checkpoint.get("graph"), device)
        epigenetics = _prior_checkpoint.get("epigenetics")
        if bool(_prior_checkpoint["model_config"].get("epi_feature_dim", 0)) != (
            epigenetics is not None
        ):
            raise ValueError("Saved prior epigenomic configuration and provenance differ")
        if epigenetics is not None:
            validate_prior_data(data, epigenetics)
            names = feature_names(epigenetics)
            if (
                _prior_checkpoint["model_config"].get("epi_feature_dim") != len(names)
                or _prior_checkpoint["model_config"].get("epi_graph_propagation", False)
                != (epigenetics["format_version"] == 2)
            ):
                raise ValueError("Saved prior has an unsupported epigenomic feature dimension")
            epi_values = _prior_checkpoint["model_state"]["epi_values"]
            epi_masks = _prior_checkpoint["model_state"]["epi_masks"]
    elif epigenetic_bundle is not None:
        values, masks, epigenetics = load_epigenetic_bundle(epigenetic_bundle, data)
        epi_values = torch.from_numpy(values)
        epi_masks = torch.from_numpy(masks)
        bundle_graph = epigenetic_bundle / "graph.csv"
        graph = read_graph(bundle_graph, genes, device) if epigenetics["edge_count"] else None
        if file_sha256(bundle_graph) != epigenetics["files"]["graph.csv"]:
            raise RuntimeError("Epigenomic graph changed during loading")
    else:
        graph = read_graph(graph_path, genes, device)
    if graph is not None and graph.shape != (len(genes), len(genes)):
        raise ValueError("Saved graph does not match the training gene axis")

    outdir.mkdir(parents=True, exist_ok=True)
    model = C3VCell(ModelConfig(
        len(genes), latent_dim, hidden_dim, variance_weight=variance_weight,
        covariance_weight=covariance_weight, regularization_view=regularization_view,
        epi_feature_dim=len(feature_names(epigenetics)) if epigenetics is not None else 0,
        epi_graph_propagation=epigenetics is not None and epigenetics["format_version"] == 2,
    )).to(device)
    if epigenetics is not None:
        if epi_values is None or epi_masks is None:
            raise ValueError("Epigenomic prior features are missing")
        model.set_epigenetic_features(epi_values, epi_masks)
    log_path = outdir / "training.jsonl"
    with log_path.open("w", encoding="utf-8") as log:
        fit_representation(
            model, matrix, rng=rng, device=device, steps=pretrain_steps,
            batch_size=batch_size, learning_rate=learning_rate, log=log,
            reference=reference, reference_fraction=reference_fraction,
        )
        diagnostics = representation_diagnostics(
            model, matrix, device=device, batch_size=batch_size, seed=seed,
        )
        log.write(json.dumps({"phase": "representation-diagnostics", **diagnostics}) + "\n")
        log.flush()
        if diagnostics["collapse_warning"]:
            warnings.warn(
                "Representation encodings have very low variation or rank across sampled training cells; "
                "target-conditioned predictions may be unreliable.",
                RuntimeWarning, stacklevel=2,
            )
        model.encoder.requires_grad_(False)
        model.decoder.requires_grad_(False)
        model.predictor.requires_grad_(False)
        flow_parameters = list(model.velocity.parameters())
        if model.epi_adapter is not None:
            flow_parameters.extend(model.epi_adapter.parameters())
        optimizer = torch.optim.AdamW(flow_parameters, lr=learning_rate)
        shift_groups = None
        shift_details = None
        if flow_target == "population-shift":
            shift_groups, shift_details = _population_shift_groups(
                model, matrix, labels, contexts, batches, exact, by_context, control=control,
                min_group_cells=min_group_cells, batch_size=batch_size, device=device,
            )
        targets = np.array(sorted(target_rows if shift_groups is None else shift_groups), dtype=object)
        model.train()
        for step in range(1, flow_steps + 1):
            chosen = rng.choice(targets, size=batch_size)
            if shift_groups is None:
                pert_rows = np.array([rng.choice(target_rows[name]) for name in chosen])
                control_rows = []
                for row in pert_rows:
                    key = (contexts[row], batches[row])
                    pool = exact.get(key)
                    if pool is None:
                        pool = by_context[contexts[row]]
                    control_rows.append(rng.choice(pool))
            else:
                selected_groups = [
                    shift_groups[name][int(rng.integers(len(shift_groups[name])))]
                    for name in chosen
                ]
                control_rows = [rng.choice(group[0]) for group in selected_groups]
            counts0 = _batch(matrix, np.asarray(control_rows), device)
            indices = torch.tensor(
                [gene_index[name] for name in chosen], dtype=torch.long, device=device
            )
            optimizer.zero_grad(set_to_none=True)
            if shift_groups is None:
                counts1 = _batch(matrix, pert_rows, device)
                loss, metrics = model.flow_loss(counts0, counts1, indices, graph)
            else:
                shifts = torch.as_tensor(
                    np.stack([group[1] for group in selected_groups]),
                    dtype=torch.float32, device=device,
                )
                loss, metrics = model.flow_loss_shift(counts0, shifts, indices, graph)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite flow loss at step {step}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(flow_parameters, 5.0)
            optimizer.step()
            if step == 1 or step % 25 == 0 or step == flow_steps:
                record = {"phase": "flow", "step": step, **metrics}
                log.write(json.dumps(record) + "\n")
                log.flush()
                print(json.dumps(record), flush=True)

    if file_sha256(train_path) != training_source_sha256:
        raise RuntimeError("Training data changed while the model was being fitted")
    checkpoint = {
        "format_version": 1,
        "model_config": model.configuration(),
        "model_state": model.state_dict(),
        "genes": genes,
        "control": control,
        "graph": _graph_payload(graph),
        "epigenetics": epigenetics,
        "seed": seed,
        "training_targets": sorted(target_rows),
        "training_cells": data.n_obs,
        "training_cell_ids": data.obs_names.astype(str).tolist(),
        "training_source_sha256": training_source_sha256,
        "representation_diagnostics": diagnostics,
        "training_parameters": {
            "pretrain_steps": pretrain_steps,
            "flow_steps": flow_steps,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "variance_weight": variance_weight,
            "covariance_weight": covariance_weight,
            "regularization_view": regularization_view,
            "allow_unmatched_batch": allow_unmatched_batch,
            "flow_target": flow_target,
            "min_group_cells": min_group_cells,
        },
        "population_shift_groups": shift_details,
        "reference": reference_metadata(
            reference, reference_path, fraction=reference_fraction, max_cells=reference_max_cells,
            seed=ref_seed, min_coverage=reference_min_coverage, model_genes=len(genes),
        ),
        "unmatched_batch_groups": unmatched,
    }
    temporary = outdir / "model.pt.tmp"
    torch.save(checkpoint, temporary)
    final = outdir / "model.pt"
    os.replace(temporary, final)
    return final


def load_model(
    checkpoint_path: Path, device_name: str = "cpu"
) -> tuple[C3VCell, dict, torch.Tensor | None]:
    device = torch.device(device_name)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if checkpoint.get("format_version") != 1:
        raise ValueError("Unsupported model checkpoint")
    warn_on_representation_collapse(checkpoint, checkpoint_path)
    config = ModelConfig(**checkpoint["model_config"])
    if bool(config.epi_feature_dim) != (checkpoint.get("epigenetics") is not None):
        raise ValueError("Checkpoint epigenomic configuration and provenance differ")
    if checkpoint.get("epigenetics") is not None:
        metadata = checkpoint["epigenetics"]
        if (
            config.epi_feature_dim != len(feature_names(metadata))
            or config.epi_graph_propagation != (metadata["format_version"] == 2)
        ):
            raise ValueError("Checkpoint epigenomic feature schema and configuration differ")
    model = C3VCell(config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    graph = _restore_graph(checkpoint.get("graph"), device)
    if graph is not None and graph.shape != (config.n_genes, config.n_genes):
        raise ValueError("Checkpoint graph does not match the model gene axis")
    return model, checkpoint, graph


def warn_on_representation_collapse(checkpoint: dict, path: Path) -> None:
    diagnostics = checkpoint.get("representation_diagnostics")
    if diagnostics is not None and diagnostics.get("collapse_warning"):
        warnings.warn(
            f"Checkpoint {path} has a low-variation or low-rank representation warning; "
            "conditioned predictions may be unreliable.",
            RuntimeWarning, stacklevel=2,
        )
