"""Generate local holdout predictions and challenge-shaped count matrices."""

from __future__ import annotations

import json
import math
import os
import shutil
import tempfile
from pathlib import Path
from typing import Literal

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

from .data import CONTROL, validate_counts
from .epigenetics import validate_checkpoint_context
from .identity import check_cell_id_schemes
from .integrity import file_sha256
from .training import load_model

DISK_RESERVE_BYTES = 200 * 1024**3
ControlSampling = Literal["reference-batch", "input-controls"]


def _validate_control_sampling(control_sampling: str) -> None:
    if control_sampling not in ("reference-batch", "input-controls"):
        raise ValueError("control_sampling must be 'reference-batch' or 'input-controls'")


def _resolve_effect_strength(
    checkpoint: dict,
    checkpoint_path: Path,
    sampling: str,
    effect_strength: float | None,
    calibration_path: Path | None,
    *,
    calibration_kind: Literal["genetic", "drug"] = "genetic",
    prediction_steps: int | None = None,
    control_sampling: ControlSampling = "reference-batch",
    cells_per_target: int | None = None,
) -> float:
    _validate_control_sampling(control_sampling)
    if calibration_path is not None:
        if effect_strength is not None:
            raise ValueError("Choose either an explicit effect strength or a calibration file")
        if sampling != "residual-rounding":
            raise ValueError("Calibrated strengths require residual-rounding sampling")
        from .calibration import CALIBRATION_OBJECTIVES, file_sha256, training_identity

        calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
        method = (
            "target-disjoint-training-folds-v1" if calibration_kind == "genetic"
            else "compound-disjoint-training-folds-v1"
        )
        if calibration.get("method") != method:
            raise ValueError("Unsupported or incomplete calibration file")
        if calibration.get("objective") not in CALIBRATION_OBJECTIVES or calibration.get("sampling") != sampling:
            raise ValueError("Calibration objective or sampling method does not match")
        if calibration.get("control_sampling", "reference-batch") != control_sampling:
            raise ValueError("Calibration control sampling mode does not match prediction")
        if calibration_kind == "drug" and (
            calibration.get("no_target_genes") is not True
            or calibration.get("stratify_by_batch") is not True
        ):
            raise ValueError("Drug calibration must use non-genetic, batch-stratified scoring")
        if prediction_steps is not None and calibration.get("prediction_steps") != prediction_steps:
            raise ValueError("Prediction steps must match the calibrated solver steps")
        if cells_per_target is not None and calibration.get("cells_per_target") != cells_per_target:
            raise ValueError("Prediction cells per target must match calibration")
        groups = (
            checkpoint["training_targets"] if calibration_kind == "genetic"
            else checkpoint["training_compounds"]
        )
        identity = training_identity(
            checkpoint["training_cell_ids"], groups
        )
        if calibration.get("training_identity") != identity:
            raise ValueError("Calibration was not fitted on this checkpoint's training cells and targets")
        if calibration.get("checkpoint_sha256") != file_sha256(checkpoint_path):
            raise ValueError("Calibration was not fitted for this exact checkpoint")
        effect_strength = calibration["selected_effect_strength"]
    value = 1.0 if effect_strength is None else float(effect_strength)
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("effect_strength must be finite and between zero and one")
    return value


def _check_disk_reserve(path: Path) -> None:
    free = shutil.disk_usage(path).free
    if free < DISK_RESERVE_BYTES:
        raise OSError(
            f"Only {free / 1024**3:.1f} GiB free at {path}; "
            "at least 200 GiB must remain available"
        )


def _concat_bounded(shards: list[Path], directory: Path, batch_size: int = 32) -> Path:
    """Merge files in bounded fan-in stages so open-file usage stays predictable."""
    generation = 0
    while len(shards) > 1:
        next_shards = []
        for start in range(0, len(shards), batch_size):
            batch = shards[start : start + batch_size]
            if len(batch) == 1:
                next_shards.append(batch[0])
                continue
            merged = directory / f"merge-{generation:02d}-{start // batch_size:04d}.h5ad"
            _check_disk_reserve(directory)
            ad.experimental.concat_on_disk(
                batch, merged, axis=0, join="inner", merge="same"
            )
            for path in batch:
                path.unlink()
            next_shards.append(merged)
        shards = next_shards
        generation += 1
    return shards[0]


def _contexts(obs: pd.DataFrame) -> np.ndarray:
    if "context" not in obs or obs["context"].isna().any():
        raise ValueError("Every input cell needs an explicit context")
    return obs["context"].astype(str).to_numpy()


def _batches(obs: pd.DataFrame) -> np.ndarray:
    if "batch" not in obs:
        return np.full(len(obs), "", dtype=object)
    if obs["batch"].isna().any():
        raise ValueError("Batch labels cannot be missing")
    return obs["batch"].astype(str).to_numpy()


def _labels(obs: pd.DataFrame) -> np.ndarray:
    if "target_gene" not in obs or obs["target_gene"].isna().any():
        raise ValueError("Every cell needs an explicit target_gene label")
    return obs["target_gene"].astype(str).to_numpy()


def _sample_controls(
    reference_context: np.ndarray,
    reference_batch: np.ndarray,
    labels: np.ndarray,
    control_context: np.ndarray,
    control_batch: np.ndarray,
    context: str,
    target: str,
    *,
    cells_per_target: int,
    rng: np.random.Generator,
    allow_unmatched_batch: bool,
    control_sampling: ControlSampling = "reference-batch",
) -> tuple[np.ndarray, np.ndarray]:
    _validate_control_sampling(control_sampling)
    context_pool = np.flatnonzero(control_context == context)
    if not len(context_pool):
        raise ValueError(f"Missing input controls for context {context!r}")
    target_rows = np.flatnonzero((reference_context == context) & (labels == target))
    if not len(target_rows):
        raise ValueError(f"No reference cells for {context!r}/{target!r}")
    if control_sampling == "input-controls":
        sampled = rng.choice(
            context_pool, size=cells_per_target, replace=len(context_pool) < cells_per_target,
        )
        return sampled, control_batch[sampled]
    batches = rng.choice(reference_batch[target_rows], size=cells_per_target, replace=True)
    sampled = []
    unmatched = set()
    for batch in batches:
        pool = np.flatnonzero((control_context == context) & (control_batch == batch))
        if not len(pool):
            if not allow_unmatched_batch:
                raise ValueError(f"No input controls match context {context!r} and batch {batch!r}")
            unmatched.add(batch)
            pool = context_pool
        sampled.append(rng.choice(pool))
    if unmatched:
        print(
            f"WARNING: {context}/{target} uses context-only controls for batches "
            f"{sorted(unmatched)}",
            flush=True,
        )
    return np.asarray(sampled), batches


def _write_predictions(
    model,
    graph,
    controls: ad.AnnData,
    genes: list[str],
    targets: list[tuple[str, str]],
    reference_context: np.ndarray,
    reference_batch: np.ndarray,
    labels: np.ndarray,
    output_path: Path,
    *,
    checkpoint_path: Path,
    cells_per_target: int,
    device_name: str,
    seed: int,
    steps: int,
    allow_unmatched_batch: bool,
    sampling: Literal["negative-binomial", "residual-rounding"],
    strength: float,
    calibration_path: Path | None,
    control_sampling: ControlSampling,
    provenance: dict | None = None,
) -> None:
    control_context = _contexts(controls.obs)
    control_batch = _batches(controls.obs)
    gene_index = {gene: idx for idx, gene in enumerate(genes)}
    matrix = sp.csr_matrix(controls.X)
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    device = torch.device(device_name)
    chunks = []
    obs_rows = []
    for context, target in targets:
        if target not in gene_index:
            raise ValueError(f"Target {target!r} is absent from the model gene axis")
        sampled, batches = _sample_controls(
            reference_context, reference_batch, labels, control_context, control_batch,
            context, target, cells_per_target=cells_per_target, rng=rng,
            allow_unmatched_batch=allow_unmatched_batch, control_sampling=control_sampling,
        )
        counts = torch.as_tensor(matrix[sampled].toarray(), dtype=torch.float32, device=device)
        target_ids = torch.full((len(sampled),), gene_index[target], dtype=torch.long, device=device)
        generated = model.predict_counts(
            counts, target_ids, steps=steps, graph=graph, sampling=sampling,
            effect_strength=strength,
        )
        chunks.append(sp.csr_matrix(generated.cpu().numpy()))
        start = len(obs_rows)
        obs_rows.extend(
            {
                "target_gene": target,
                "context": context,
                "batch": str(batches[i]),
                "source_control_id": str(controls.obs_names[sampled[i]]),
                "cell_id": f"pred-{start + i:09d}",
            }
            for i in range(len(sampled))
        )
    obs = pd.DataFrame(obs_rows).set_index("cell_id")
    prediction = ad.AnnData(
        X=sp.vstack(chunks, format="csr"),
        obs=obs,
        var=pd.DataFrame(index=pd.Index(genes, name="gene")),
    )
    prediction.uns["c3vcell_prediction"] = {
        "sampling": sampling,
        "effect_strength": strength,
        "effect_calibration": str(calibration_path) if calibration_path else "",
        "seed": seed,
        "steps": steps,
        "checkpoint": str(checkpoint_path),
        "allow_unmatched_batch": allow_unmatched_batch,
        "control_sampling": control_sampling,
    }
    if provenance is not None:
        prediction.uns["c3vcell_prediction"].update(provenance)
    validate_counts(prediction)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    prediction.write_h5ad(output_path, compression="gzip")


def predict_holdout(
    checkpoint_path: Path,
    reference_path: Path,
    input_control_path: Path,
    output_path: Path,
    *,
    cells_per_target: int = 100,
    device_name: str = "cpu",
    seed: int = 0,
    steps: int = 16,
    allow_unmatched_batch: bool = False,
    sampling: Literal["negative-binomial", "residual-rounding"] = "negative-binomial",
    effect_strength: float | None = None,
    calibration_path: Path | None = None,
    control_sampling: ControlSampling = "reference-batch",
) -> None:
    _validate_control_sampling(control_sampling)
    if cells_per_target < 1:
        raise ValueError("cells_per_target must be positive")
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite {output_path}")
    model, checkpoint, graph = load_model(checkpoint_path, device_name)
    strength = _resolve_effect_strength(
        checkpoint, checkpoint_path, sampling, effect_strength, calibration_path,
        prediction_steps=steps, control_sampling=control_sampling, cells_per_target=cells_per_target,
    )
    reference = ad.read_h5ad(reference_path)
    controls = ad.read_h5ad(input_control_path)
    validate_counts(reference)
    validate_counts(controls)
    validate_checkpoint_context(checkpoint, reference)
    validate_checkpoint_context(checkpoint, controls)
    genes = checkpoint["genes"]
    if reference.var_names.tolist() != genes or controls.var_names.tolist() != genes:
        raise ValueError("Holdout, input controls, and checkpoint gene axes must match exactly")
    training_ids = set(checkpoint["training_cell_ids"])
    reference_ids = set((checkpoint.get("reference") or {}).get("cell_ids", []))
    check_cell_id_schemes(training_ids, reference_ids, reference.obs_names, controls.obs_names)
    training_ids.update(reference_ids)
    if set(reference.obs_names).intersection(controls.obs_names):
        raise ValueError("Inference controls overlap the held-out reference")
    if training_ids.intersection(controls.obs_names) or training_ids.intersection(reference.obs_names):
        raise ValueError("Training cells overlap inference controls or the held-out reference")
    control = checkpoint["control"]
    if set(_labels(controls.obs)) != {control}:
        raise ValueError("Input control file contains perturbed cells")
    reference_context = _contexts(reference.obs)
    reference_batch = (
        _batches(reference.obs) if control_sampling == "reference-batch"
        else np.full(reference.n_obs, "", dtype=object)
    )
    labels = _labels(reference.obs)
    targets = sorted({(context, target) for context, target in zip(reference_context, labels) if target != control})
    if not targets:
        raise ValueError("No held-out perturbations to predict")
    seen_targets = {target for _, target in targets}.intersection(checkpoint["training_targets"])
    if seen_targets:
        raise ValueError(f"Held-out perturbations appeared during training: {sorted(seen_targets)}")
    _write_predictions(
        model, graph, controls, genes, targets, reference_context, reference_batch, labels,
        output_path, checkpoint_path=checkpoint_path, cells_per_target=cells_per_target,
        device_name=device_name, seed=seed, steps=steps,
        allow_unmatched_batch=allow_unmatched_batch, sampling=sampling, strength=strength,
        calibration_path=calibration_path, control_sampling=control_sampling,
    )


def _read_target_requests(path: Path) -> list[tuple[str, str]]:
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    columns = ["context", "target_gene"]
    if not set(columns) <= set(frame.columns):
        raise ValueError("Target requests need context and target_gene columns")
    frame = frame[columns]
    if frame.empty or frame.duplicated().any() or any(
        not value or value != value.strip() for value in frame.to_numpy().flat
    ):
        raise ValueError("Target requests must contain unique, nonempty context/target pairs")
    return sorted(map(tuple, frame.itertuples(index=False, name=None)))


def predict_targets(
    checkpoint_path: Path,
    targets_path: Path,
    input_control_path: Path,
    output_path: Path,
    *,
    cells_per_target: int = 100,
    device_name: str = "cpu",
    seed: int = 0,
    steps: int = 16,
    sampling: Literal["negative-binomial", "residual-rounding"] = "negative-binomial",
    effect_strength: float | None = None,
    calibration_path: Path | None = None,
) -> None:
    """Predict requested unseen targets using only independent input controls."""
    if cells_per_target < 1:
        raise ValueError("cells_per_target must be positive")
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite {output_path}")
    input_paths = {
        "request_source_sha256": targets_path,
        "input_controls_sha256": input_control_path,
        "checkpoint_sha256": checkpoint_path,
    }
    if calibration_path is not None:
        input_paths["calibration_sha256"] = calibration_path
    hashes = {name: file_sha256(path) for name, path in input_paths.items()}
    targets = _read_target_requests(targets_path)
    model, checkpoint, graph = load_model(checkpoint_path, device_name)
    strength = _resolve_effect_strength(
        checkpoint, checkpoint_path, sampling, effect_strength, calibration_path,
        prediction_steps=steps, control_sampling="input-controls",
        cells_per_target=cells_per_target,
    )
    controls = ad.read_h5ad(input_control_path)
    validate_counts(controls)
    validate_checkpoint_context(checkpoint, controls)
    genes = checkpoint["genes"]
    if controls.var_names.tolist() != genes:
        raise ValueError("Input controls and checkpoint gene axes must match exactly")
    training_ids = set(checkpoint["training_cell_ids"])
    reference_ids = set((checkpoint.get("reference") or {}).get("cell_ids", []))
    check_cell_id_schemes(training_ids, reference_ids, controls.obs_names)
    if (training_ids | reference_ids).intersection(controls.obs_names):
        raise ValueError("Training cells overlap inference controls")
    control = checkpoint["control"]
    if set(_labels(controls.obs)) != {control}:
        raise ValueError("Input control file contains perturbed cells")
    requested_genes = {target for _, target in targets}
    if control in requested_genes:
        raise ValueError("Target requests cannot include the control label")
    seen_targets = requested_genes.intersection(checkpoint["training_targets"])
    if seen_targets:
        raise ValueError(f"Requested perturbations appeared during training: {sorted(seen_targets)}")
    missing = requested_genes - set(genes)
    if missing:
        raise ValueError(f"Requested targets are absent from the model gene axis: {sorted(missing)}")
    missing_contexts = {context for context, _ in targets} - set(_contexts(controls.obs))
    if missing_contexts:
        raise ValueError(f"Missing input controls for contexts: {sorted(missing_contexts)}")
    if any(file_sha256(path) != hashes[name] for name, path in input_paths.items()):
        raise RuntimeError("Prediction input changed while it was being read")
    contexts = np.array([context for context, _ in targets], dtype=object)
    labels = np.array([target for _, target in targets], dtype=object)
    _write_predictions(
        model, graph, controls, genes, targets, contexts,
        np.full(len(targets), "", dtype=object), labels, output_path,
        checkpoint_path=checkpoint_path, cells_per_target=cells_per_target,
        device_name=device_name, seed=seed, steps=steps,
        allow_unmatched_batch=False, sampling=sampling, strength=strength,
        calibration_path=calibration_path, control_sampling="input-controls",
        provenance={
            "request_source": str(targets_path),
            **hashes,
        },
    )


def predict_control_baseline(
    reference_path: Path,
    input_control_path: Path,
    output_path: Path,
    *,
    control: str = CONTROL,
    cells_per_target: int = 100,
    seed: int = 0,
    allow_unmatched_batch: bool = False,
    control_sampling: ControlSampling = "reference-batch",
) -> None:
    _validate_control_sampling(control_sampling)
    if cells_per_target < 1:
        raise ValueError("cells_per_target must be positive")
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite {output_path}")
    reference = ad.read_h5ad(reference_path)
    controls = ad.read_h5ad(input_control_path)
    validate_counts(reference)
    validate_counts(controls)
    if reference.var_names.tolist() != controls.var_names.tolist():
        raise ValueError("Reference and input controls must have identical ordered genes")
    check_cell_id_schemes(reference.obs_names, controls.obs_names)
    if set(reference.obs_names).intersection(controls.obs_names):
        raise ValueError("Inference controls overlap the held-out reference")
    if set(_labels(controls.obs)) != {control}:
        raise ValueError("Input control file contains perturbed cells")
    reference_context = _contexts(reference.obs)
    control_context = _contexts(controls.obs)
    reference_batch = (
        _batches(reference.obs) if control_sampling == "reference-batch"
        else np.full(reference.n_obs, "", dtype=object)
    )
    control_batch = _batches(controls.obs)
    labels = _labels(reference.obs)
    targets = sorted({(context, target) for context, target in zip(reference_context, labels) if target != control})
    if not targets:
        raise ValueError("No held-out perturbations to predict")
    matrix = sp.csr_matrix(controls.X)
    maximum = float(matrix.data.max(initial=0))
    if maximum >= float(2**63):
        raise ValueError("Control counts exceed the supported integer range")
    matrix = matrix.astype(np.int32 if maximum <= np.iinfo(np.int32).max else np.int64)
    rng = np.random.default_rng(seed)
    chunks = []
    obs_rows = []
    for context, target in targets:
        sampled, batches = _sample_controls(
            reference_context, reference_batch, labels, control_context, control_batch,
            context, target, cells_per_target=cells_per_target, rng=rng,
            allow_unmatched_batch=allow_unmatched_batch, control_sampling=control_sampling,
        )
        chunks.append(matrix[sampled])
        start = len(obs_rows)
        obs_rows.extend(
            {
                "target_gene": target,
                "context": context,
                "batch": str(batches[i]),
                "source_control_id": str(controls.obs_names[sampled[i]]),
                "cell_id": f"control-baseline-{start + i:09d}",
            }
            for i in range(len(sampled))
        )
    obs = pd.DataFrame(obs_rows).set_index("cell_id")
    prediction = ad.AnnData(
        X=sp.vstack(chunks, format="csr"), obs=obs, var=controls.var.copy()
    )
    prediction.uns["c3vcell_control_baseline"] = {
        "control_sampling": control_sampling, "seed": seed,
    }
    validate_counts(prediction)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    prediction.write_h5ad(output_path, compression="gzip")


def _read_gene_list(path: Path) -> list[str]:
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    if "gene_name" not in frame:
        raise ValueError("gene_names.csv must have a gene_name column")
    genes = frame["gene_name"].tolist()
    if len(genes) != 18_533 or len(set(genes)) != len(genes) or not all(genes):
        raise ValueError("VCC 2026 requires 18,533 unique, nonempty gene symbols")
    return genes


def _read_targets(path: Path) -> list[str]:
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    if "target_gene" not in frame:
        raise ValueError("pert_counts.csv must have a target_gene column")
    targets = frame["target_gene"].tolist()
    if len(targets) != 300 or len(set(targets)) != 300 or CONTROL in targets or not all(targets):
        raise ValueError("VCC 2026 requires 300 unique non-control target genes")
    if "n_cells" in frame and not (pd.to_numeric(frame["n_cells"]) == 400).all():
        raise ValueError("VCC 2026 requires 400 cells for every target")
    return targets


def export_vcc_h5ad(
    checkpoint_path: Path,
    controls_by_context: dict[str, Path],
    gene_file: Path,
    pert_file: Path,
    output_path: Path,
    *,
    device_name: str = "cpu",
    seed: int = 0,
    steps: int = 16,
    allow_missing_target: bool = False,
    sampling: Literal["negative-binomial", "residual-rounding"] = "negative-binomial",
    effect_strength: float | None = None,
    calibration_path: Path | None = None,
) -> None:
    contexts = set(controls_by_context)
    if contexts not in ({"A", "B", "C"}, {"D", "E", "F"}):
        raise ValueError("Use exactly A/B/C or exactly D/E/F control contexts")
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite {output_path}")
    provenance_path = output_path.with_suffix(output_path.suffix + ".json")
    if provenance_path.exists():
        raise FileExistsError(f"Refusing to overwrite {provenance_path}")
    genes = _read_gene_list(gene_file)
    targets = _read_targets(pert_file)
    model, checkpoint, graph = load_model(checkpoint_path, device_name)
    if checkpoint.get("epigenetics") is not None:
        raise ValueError(
            "Epigenomic checkpoints cannot be exported to anonymized challenge contexts "
            "without verified matching reference metadata"
        )
    strength = _resolve_effect_strength(
        checkpoint, checkpoint_path, sampling, effect_strength, calibration_path,
        prediction_steps=steps, control_sampling="input-controls", cells_per_target=400,
    )
    model_genes = checkpoint["genes"]
    training_ids = set(checkpoint["training_cell_ids"])
    reference_ids = set((checkpoint.get("reference") or {}).get("cell_ids", []))
    check_cell_id_schemes(training_ids, reference_ids)
    training_ids.update(reference_ids)
    model_index = {gene: index for index, gene in enumerate(model_genes)}
    challenge_index = {gene: index for index, gene in enumerate(genes)}
    missing_targets = sorted(set(targets) - (set(model_genes) & set(genes)))
    if missing_targets and not allow_missing_target:
        raise ValueError(
            f"{len(missing_targets)} targets are absent from the model gene axis; "
            "train on a compatible gene axis or explicitly opt in to a control baseline"
        )
    if missing_targets:
        print(
            f"WARNING: {len(missing_targets)} missing targets use unmodified controls: "
            f"{missing_targets}",
            flush=True,
        )
    common = [gene for gene in model_genes if gene in challenge_index]
    if not common:
        raise ValueError("No shared genes between model and challenge controls")
    model_columns = np.array([model_index[gene] for gene in common])
    challenge_columns = np.array([challenge_index[gene] for gene in common])
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    device = torch.device(device_name)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _check_disk_reserve(output_path.parent)
    stored_entries = 0
    with tempfile.TemporaryDirectory(prefix="c3vcell-vcc-", dir=output_path.parent) as tmp:
        temporary = Path(tmp)
        shards: list[Path] = []
        for context in sorted(contexts):
            controls = ad.read_h5ad(controls_by_context[context])
            validate_counts(controls)
            if controls.var_names.tolist() != genes:
                raise ValueError(f"Control gene order does not match gene_names.csv for {context}")
            if set(_contexts(controls.obs)) != {context}:
                raise ValueError(f"Control file contains unexpected context labels for {context}")
            if "target_gene" in controls.obs and set(controls.obs["target_gene"].astype(str)) != {CONTROL}:
                raise ValueError(f"Input for context {context} contains perturbed cells")
            check_cell_id_schemes(training_ids, controls.obs_names)
            if training_ids.intersection(controls.obs_names):
                raise ValueError(f"Input controls overlap training cells in context {context}")
            matrix = sp.csr_matrix(controls.X)
            for target in targets:
                _check_disk_reserve(output_path.parent)
                selected = rng.choice(controls.n_obs, size=400, replace=controls.n_obs < 400)
                if target in missing_targets:
                    sparse = matrix[selected].copy()
                    if np.any(np.asarray(sparse.sum(axis=1)).ravel() > 1_000_000):
                        raise ValueError("Predicted cell exceeds the challenge count limit")
                    sparse = sparse.astype(np.int32)
                else:
                    raw = matrix[selected].toarray().astype(np.float32, copy=False)
                    model_counts = np.zeros((400, len(model_genes)), dtype=np.float32)
                    model_counts[:, model_columns] = raw[:, challenge_columns]
                    measured = np.zeros_like(model_counts)
                    measured[:, model_columns] = 1
                    if np.any(model_counts.sum(axis=1) == 0):
                        raise ValueError("A control cell has no counts on the shared model gene panel")
                    counts = torch.as_tensor(model_counts, device=device)
                    mask = torch.as_tensor(measured, device=device)
                    ids = torch.full((400,), model_index[target], dtype=torch.long, device=device)
                    generated = model.predict_counts(
                        counts, ids, steps=steps, graph=graph, measured=mask, sampling=sampling,
                        effect_strength=strength,
                    )
                    prediction = raw.copy()
                    prediction[:, challenge_columns] = generated.cpu().numpy()[:, model_columns]
                    if not np.isfinite(prediction).all() or np.any(prediction.sum(axis=1) > 1_000_000):
                        raise ValueError("Predicted cell exceeds the challenge count limit")
                    sparse = sp.csr_matrix(prediction.astype(np.int32))
                sparse.eliminate_zeros()
                stored_entries += sparse.nnz
                if stored_entries > 4_750_000_000:
                    raise ValueError("Predictions exceed the VCC stored-entry limit")
                obs = pd.DataFrame(
                    {"target_gene": [target] * 400, "context": [context] * 400},
                    index=[f"{context}:{target}:{i}" for i in range(400)],
                )
                shard = temporary / f"part-{len(shards):04d}.h5ad"
                ad.AnnData(sparse, obs=obs, var=pd.DataFrame(index=genes)).write_h5ad(
                    shard, compression="gzip"
                )
                shards.append(shard)
                print(f"Generated {context}/{target}: 400 cells", flush=True)
        assembled = _concat_bounded(shards, temporary)
        _check_disk_reserve(output_path.parent)
        os.replace(assembled, output_path)
    provenance = {
        "format": "VCC-2026 h5ad; use vcc prep for official validation",
        "contexts": sorted(contexts),
        "perturbations": len(targets),
        "cells_per_perturbation_and_context": 400,
        "genes": len(genes),
        "stored_entries": stored_entries,
        "model_genes_not_in_challenge": sorted(set(model_genes) - set(genes)),
        "challenge_genes_not_in_model": sorted(set(genes) - set(model_genes)),
        "control_baseline_targets": missing_targets,
        "targets_seen_during_training": sorted(set(targets) & set(checkpoint["training_targets"])),
        "sampling": sampling,
        "effect_strength": strength,
        "effect_calibration": str(calibration_path) if calibration_path else "",
        "seed": seed,
    }
    provenance_path.write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
