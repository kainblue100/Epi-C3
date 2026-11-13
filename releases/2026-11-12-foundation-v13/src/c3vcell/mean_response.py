"""Training-only, batch-matched empirical mean-response comparator."""

from __future__ import annotations

import os
import tempfile
import warnings
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

from .calibration import file_sha256
from .data import CONTROL, validate_counts
from .epigenetics import validate_checkpoint_context
from .identity import check_cell_id_schemes
from .model import C3VCell
from .predict import (
    ControlSampling, _batches, _contexts, _labels, _sample_controls, _validate_control_sampling,
)
from .training import load_model


def _write_new(data: ad.AnnData, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=".c3-mean-response-", suffix=".h5ad", dir=path.parent, delete=False,
    ) as stream:
        temporary = Path(stream.name)
    try:
        data.write_h5ad(temporary, compression="gzip")
        os.link(temporary, path)
    finally:
        temporary.unlink()


@torch.no_grad()
def _conditional_logfold(
    model: C3VCell,
    counts: torch.Tensor,
    target_index: torch.Tensor,
    anchor: torch.Tensor,
    *,
    steps: int,
    graph: torch.Tensor | None,
) -> torch.Tensor:
    origin = model.encode(counts)
    target = model.target_embedding(target_index, graph)
    conditions = torch.cat((target, anchor.expand_as(target)), dim=0)
    origins = torch.cat((origin, origin), dim=0)
    latent = origins.clone()
    for step in range(steps):
        time = torch.full((len(latent), 1), step / steps, device=counts.device)
        latent = latent + model.flow(latent, time, origins, conditions, origins) / steps
    predicted, anchored = model.probabilities(latent).chunk(2, dim=0)
    return (torch.log(predicted + 1e-8) - torch.log(anchored + 1e-8)).clamp(-2, 2)


def _load_response_profile(
    path: Path,
) -> tuple[ad.AnnData, dict, list[tuple[str, str]], np.ndarray, str]:
    digest = file_sha256(path)
    profile = ad.read_h5ad(path)
    metadata = profile.uns.get("c3vcell_mean_response", {})
    if (
        metadata.get("format_version") != 1
        or metadata.get("kind") != "training-target-uniform-batch-mean-logfold"
    ):
        raise ValueError("Unsupported mean-response profile")
    if not {"context", "batch"}.issubset(profile.obs.columns):
        raise ValueError("Mean-response profile requires context and batch labels")
    keys = list(zip(profile.obs["context"].astype(str), profile.obs["batch"].astype(str)))
    if len(set(keys)) != len(keys):
        raise ValueError("Mean-response profile groups must be unique")
    effects = np.asarray(profile.X, dtype=np.float32)
    if not np.isfinite(effects).all():
        raise ValueError("Mean-response profiles must be finite")
    return profile, metadata, keys, effects, digest


def _load_shared_model(
    path: Path, profile: ad.AnnData, metadata: dict, device_name: str,
) -> tuple[C3VCell, dict, torch.Tensor | None, torch.Tensor, str]:
    digest = file_sha256(path)
    model, checkpoint, graph = load_model(path, device_name)
    if checkpoint.get("kind") == "tahoe-drug-smiles-dose":
        raise ValueError("The shared-residual predictor requires a genetic checkpoint")
    if checkpoint["genes"] != profile.var_names.tolist():
        raise ValueError("Checkpoint and mean-response gene axes must match")
    if set(checkpoint["training_cell_ids"]) != set(metadata["training_cell_ids"].astype(str)):
        raise ValueError("Checkpoint and mean-response training cells differ")
    if set(checkpoint["training_targets"]) != set(metadata["training_targets"].astype(str)):
        raise ValueError("Checkpoint and mean-response training targets differ")
    training_source_sha256 = checkpoint.get("training_source_sha256")
    if training_source_sha256 is None:
        warnings.warn(
            "The checkpoint predates training-file fingerprints; matching cell IDs "
            "cannot verify that model and profile used the same counts.",
            RuntimeWarning, stacklevel=2,
        )
    elif training_source_sha256 != str(metadata["source_sha256"]):
        raise ValueError("Checkpoint and mean-response training count files differ")
    gene_index = {gene: i for i, gene in enumerate(checkpoint["genes"])}
    indices = torch.tensor(
        [gene_index[target] for target in checkpoint["training_targets"]],
        dtype=torch.long, device=torch.device(device_name),
    )
    with torch.no_grad():
        anchor = model.target_embedding(indices, graph).mean(dim=0, keepdim=True)
    return model, checkpoint, graph, anchor, digest


def _generate_shared_counts(
    counts: torch.Tensor,
    logfold: torch.Tensor,
    *,
    effect_strength: float,
    conditional_strength: float,
    model: C3VCell | None,
    target_index: torch.Tensor | None,
    anchor: torch.Tensor | None,
    graph: torch.Tensor | None,
    steps: int,
) -> torch.Tensor:
    # Keep local and challenge-export rounding consistent despite different array strides.
    counts = counts.contiguous()
    logfold = logfold.contiguous()
    if conditional_strength:
        if model is None or anchor is None or target_index is None:
            raise RuntimeError("Conditional model, anchor, and target indices are required")
        logfold = logfold + conditional_strength * _conditional_logfold(
            model, counts, target_index, anchor, steps=steps, graph=graph,
        )
    if effect_strength == 0 and not conditional_strength:
        generated = counts
    else:
        library = counts.sum(dim=1, keepdim=True)
        mean = counts * torch.exp(logfold)
        mean = mean * library / mean.sum(dim=1, keepdim=True).clamp(min=1e-8)
        integral = torch.floor(mean)
        generated = integral + torch.bernoulli((mean - integral).clamp(0, 1))
    if not torch.isfinite(generated).all() or torch.any(generated.sum(dim=1) > 1_000_000):
        raise ValueError("Mean-response counts are invalid or exceed 1,000,000 per cell")
    return generated.to(torch.int32)


def fit_mean_response(
    train_path: Path,
    output_path: Path,
    *,
    control: str = CONTROL,
    min_group_cells: int = 3,
    pseudocount: float = 0.1,
    logfold_limit: float = 2.0,
) -> None:
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite {output_path}")
    if min_group_cells < 1:
        raise ValueError("min_group_cells must be positive")
    if not np.isfinite(pseudocount) or pseudocount <= 0:
        raise ValueError("pseudocount must be finite and positive")
    if not np.isfinite(logfold_limit) or logfold_limit <= 0:
        raise ValueError("logfold_limit must be finite and positive")
    source_digest = file_sha256(train_path)
    data = ad.read_h5ad(train_path)
    validate_counts(data)
    labels = _labels(data.obs)
    contexts = _contexts(data.obs)
    batches = _batches(data.obs)
    matrix = sp.csr_matrix(data.X, dtype=np.float32)
    library = np.asarray(matrix.sum(axis=1, dtype=np.float64)).ravel()
    normalized = matrix.multiply((10_000.0 / library).astype(np.float32)[:, None]).tocsr()
    controls: dict[tuple[str, str], list[int]] = {}
    treated: dict[tuple[str, str, str], list[int]] = {}
    for row, (label, context, batch) in enumerate(zip(labels, contexts, batches)):
        if label == control:
            controls.setdefault((context, batch), []).append(row)
        else:
            treated.setdefault((context, batch, label), []).append(row)
    if not controls or not treated:
        raise ValueError("Training data needs both controls and perturbed cells")
    missing_controls = sorted({key[:2] for key in treated} - set(controls))
    if missing_controls:
        raise ValueError(f"Training perturbations lack matched controls: {missing_controls[:10]}")

    def mean(rows: list[int]) -> np.ndarray:
        return np.asarray(normalized[rows].sum(axis=0, dtype=np.float64)).ravel() / len(rows)

    profiles = []
    obs = []
    eligible_targets = set()
    excluded = 0
    for context, batch in sorted(controls):
        control_rows = controls[(context, batch)]
        group_targets = sorted(
            (target, rows) for (ctx, bch, target), rows in treated.items()
            if ctx == context and bch == batch
        )
        if len(control_rows) < min_group_cells:
            excluded += len(group_targets)
            continue
        eligible = [(target, rows) for target, rows in group_targets if len(rows) >= min_group_cells]
        excluded += len(group_targets) - len(eligible)
        if not eligible:
            continue
        control_mean = mean(control_rows)
        response_mean = np.zeros(data.n_vars, dtype=np.float64)
        for target, rows in eligible:
            response_mean += mean(rows)
            eligible_targets.add(target)
        response_mean /= len(eligible)
        fold = np.log((response_mean + pseudocount) / (control_mean + pseudocount))
        profiles.append(np.clip(fold, -logfold_limit, logfold_limit).astype(np.float32))
        obs.append({
            "context": context,
            "batch": batch,
            "control_cells": len(control_rows),
            "training_targets": len(eligible),
        })
    if not profiles:
        raise ValueError("No matched training groups have enough cells")
    all_targets = sorted(set(labels) - {control})
    omitted = sorted(set(all_targets) - eligible_targets)
    if excluded or omitted:
        print(
            f"Mean-response fitting excluded {excluded} underpopulated groups; "
            f"targets without eligible groups: {omitted}",
            flush=True,
        )
    if file_sha256(train_path) != source_digest:
        raise RuntimeError("Training data changed while fitting the mean-response comparator")
    profile = ad.AnnData(
        X=np.stack(profiles),
        obs=pd.DataFrame(obs, index=[f"group-{i:06d}" for i in range(len(obs))]),
        var=data.var.copy(),
    )
    profile.uns["c3vcell_mean_response"] = {
        "format_version": 1,
        "kind": "training-target-uniform-batch-mean-logfold",
        "control": control,
        "source_sha256": source_digest,
        "training_cell_ids": np.asarray(data.obs_names.astype(str).tolist()),
        "training_targets": np.asarray(all_targets),
        "targets_without_eligible_groups": np.asarray(omitted, dtype=str),
        "min_group_cells": min_group_cells,
        "pseudocount": pseudocount,
        "logfold_limit": logfold_limit,
        "excluded_groups": excluded,
    }
    _write_new(profile, output_path)


def predict_mean_response(
    profile_path: Path,
    reference_path: Path,
    controls_path: Path,
    output_path: Path,
    *,
    cells_per_target: int = 400,
    seed: int = 0,
    effect_strength: float = 1.0,
    allow_unmatched_batch: bool = False,
    control_sampling: ControlSampling = "reference-batch",
    device_name: str = "cpu",
    checkpoint_path: Path | None = None,
    conditional_strength: float | None = None,
    conditional_calibration_path: Path | None = None,
    steps: int = 8,
) -> None:
    _validate_control_sampling(control_sampling)
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite {output_path}")
    if cells_per_target < 1:
        raise ValueError("cells_per_target must be positive")
    if not np.isfinite(effect_strength) or not 0 <= effect_strength <= 1:
        raise ValueError("effect_strength must be finite and between zero and one")
    if conditional_strength is not None and conditional_calibration_path is not None:
        raise ValueError("Use either a conditional strength or a calibration, not both")
    if conditional_strength is not None and (
        not np.isfinite(conditional_strength) or not 0 <= conditional_strength <= 1
    ):
        raise ValueError("conditional_strength must be finite and between zero and one")
    if checkpoint_path is None and (
        conditional_calibration_path is not None or conditional_strength not in (None, 0)
    ):
        raise ValueError("A model checkpoint is required for a conditional response")
    if conditional_calibration_path is None and conditional_strength is None:
        conditional_strength = 0.0
    if steps < 1:
        raise ValueError("steps must be positive")
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"Requested {device_name}, but CUDA is unavailable")
    profile, metadata, keys, effects, digest = _load_response_profile(profile_path)
    reference = ad.read_h5ad(reference_path)
    controls = ad.read_h5ad(controls_path)
    validate_counts(reference)
    validate_counts(controls)
    if not profile.var_names.equals(reference.var_names) or not profile.var_names.equals(
        controls.var_names
    ):
        raise ValueError("Profile, reference, and control gene axes must match")
    train_ids = set(metadata["training_cell_ids"].astype(str))
    check_cell_id_schemes(train_ids, reference.obs_names, controls.obs_names)
    if set(reference.obs_names).intersection(controls.obs_names):
        raise ValueError("Inference controls overlap the held-out reference")
    if train_ids.intersection(reference.obs_names) or train_ids.intersection(controls.obs_names):
        raise ValueError("Training cells overlap inference controls or the held-out reference")
    model = None
    graph = None
    anchor = None
    checkpoint_digest = None
    calibration_digest = None
    if checkpoint_path is not None:
        model, checkpoint, graph, anchor, checkpoint_digest = _load_shared_model(
            checkpoint_path, profile, metadata, device_name,
        )
        validate_checkpoint_context(checkpoint, reference)
        validate_checkpoint_context(checkpoint, controls)
        reference_ids = set((checkpoint.get("reference") or {}).get("cell_ids", []))
        check_cell_id_schemes(train_ids, reference_ids, reference.obs_names, controls.obs_names)
        if reference_ids.intersection(reference.obs_names) or reference_ids.intersection(
            controls.obs_names
        ):
            raise ValueError("Representation reference cells overlap inference or held-out cells")
        gene_index = {gene: i for i, gene in enumerate(checkpoint["genes"])}
        if conditional_calibration_path is not None:
            from .shared_calibration import resolve_shared_strength

            calibration_digest = file_sha256(conditional_calibration_path)
            effect_strength, conditional_strength = resolve_shared_strength(
                conditional_calibration_path, checkpoint_path, checkpoint,
                profile_path, metadata, cells_per_target=cells_per_target,
                shared_strength=effect_strength, prediction_steps=steps,
                allow_unmatched_batch=allow_unmatched_batch, control_sampling=control_sampling,
            )
    control = str(metadata["control"])
    if set(_labels(controls.obs)) != {control}:
        raise ValueError("Input control file contains perturbed cells")
    labels = _labels(reference.obs)
    ref_context = _contexts(reference.obs)
    ref_batch = (
        _batches(reference.obs) if control_sampling == "reference-batch"
        else np.full(reference.n_obs, "", dtype=object)
    )
    control_context = _contexts(controls.obs)
    control_batch = _batches(controls.obs)
    targets = sorted({(context, target) for context, target in zip(ref_context, labels) if target != control})
    if not targets:
        raise ValueError("No held-out perturbations to predict")
    seen = {target for _, target in targets}.intersection(metadata["training_targets"].astype(str))
    if seen:
        raise ValueError(f"Held-out perturbations appeared during training: {sorted(seen)}")
    lookup = {key: i for i, key in enumerate(keys)}
    by_context = {
        context: np.mean(effects[[i for i, (ctx, _) in enumerate(keys) if ctx == context]], axis=0)
        for context in sorted({context for context, _ in keys})
    }
    matrix = sp.csr_matrix(controls.X)
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    chunks = []
    rows = []
    fallback = set()
    for context, target in targets:
        sampled, batches = _sample_controls(
            ref_context, ref_batch, labels, control_context, control_batch,
            context, target, cells_per_target=cells_per_target, rng=rng,
            allow_unmatched_batch=allow_unmatched_batch, control_sampling=control_sampling,
        )
        counts = torch.as_tensor(matrix[sampled].toarray(), dtype=torch.float32, device=device)
        if effect_strength:
            group_effects = []
            for batch in batches:
                key = (context, str(batch))
                if key in lookup:
                    group_effects.append(effects[lookup[key]])
                elif allow_unmatched_batch and context in by_context:
                    group_effects.append(by_context[context])
                    fallback.add(key)
                else:
                    raise ValueError(f"Missing mean-response profile for context/batch {key!r}")
            logfold = torch.as_tensor(np.stack(group_effects), dtype=torch.float32, device=device)
            logfold = logfold * effect_strength
        else:
            logfold = torch.zeros_like(counts)
        target_index = None
        if model is not None and conditional_strength:
            if target not in gene_index:
                raise ValueError(f"Target {target!r} is missing from the model gene axis")
            target_index = torch.full(
                (len(counts),), gene_index[target], dtype=torch.long, device=device,
            )
        generated = _generate_shared_counts(
            counts, logfold, effect_strength=effect_strength,
            conditional_strength=float(conditional_strength or 0), model=model,
            target_index=target_index, anchor=anchor, graph=graph, steps=steps,
        )
        chunks.append(sp.csr_matrix(generated.cpu().numpy()))
        start = len(rows)
        rows.extend({
            "target_gene": target,
            "context": context,
            "batch": str(batch),
            "source_control_id": str(controls.obs_names[source]),
            "cell_id": f"mean-response-{start + i:09d}",
        } for i, (source, batch) in enumerate(zip(sampled, batches)))
    if fallback:
        print(f"WARNING: context-only mean-response fallback for groups: {sorted(fallback)}", flush=True)
    result = ad.AnnData(
        sp.vstack(chunks, format="csr"),
        obs=pd.DataFrame(rows).set_index("cell_id"),
        var=controls.var.copy(),
    )
    result_metadata = {
        "profile_sha256": digest,
        "effect_strength": effect_strength,
        "seed": seed,
        "allow_unmatched_batch": allow_unmatched_batch,
        "control_sampling": control_sampling,
        "profile_fallback_groups": [f"{context}/{batch}" for context, batch in sorted(fallback)],
        "note": "A training-only empirical comparator, not the official challenge calibration baseline.",
    }
    if checkpoint_digest is not None:
        result_metadata.update({
            "checkpoint_sha256": checkpoint_digest,
            "conditional_strength": conditional_strength,
            "conditional_steps": steps,
            "conditional_anchor": "mean training-target embedding",
            "note": (
                "Unmodified input controls; no shared or conditional response was applied."
                if effect_strength == 0 and not conditional_strength
                else "Shared empirical response plus a target-conditioned model residual."
            ),
        })
        if conditional_calibration_path is not None:
            result_metadata["conditional_calibration_sha256"] = calibration_digest
    result.uns["c3vcell_mean_response_prediction"] = result_metadata
    validate_counts(result)
    if file_sha256(profile_path) != digest:
        raise RuntimeError("Mean-response profile changed during prediction")
    if checkpoint_path is not None and file_sha256(checkpoint_path) != checkpoint_digest:
        raise RuntimeError("Model checkpoint changed during prediction")
    if conditional_calibration_path is not None and (
        file_sha256(conditional_calibration_path) != calibration_digest
    ):
        raise RuntimeError("Conditional calibration changed during prediction")
    _write_new(result, output_path)
