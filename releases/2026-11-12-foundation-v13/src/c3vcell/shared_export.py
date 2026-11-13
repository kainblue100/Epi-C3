"""Export a calibrated shared-residual predictor using challenge controls only."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

from .data import CONTROL, validate_counts
from .identity import check_cell_id_schemes
from .integrity import file_sha256
from .mean_response import _generate_shared_counts, _load_response_profile, _load_shared_model
from .predict import _batches, _check_disk_reserve, _concat_bounded, _contexts, _read_gene_list, _read_targets
from .shared_calibration import resolve_shared_strength


def _read_profile_mapping(path: Path) -> dict[tuple[str, str], tuple[str, str]]:
    entries = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(entries, list) or not entries:
        raise ValueError("Profile mapping must be a nonempty JSON list")
    mapping = {}
    fields = {"input_context", "input_batch", "profile_context", "profile_batch"}
    for entry in entries:
        if (
            not isinstance(entry, dict) or set(entry) != fields
            or not all(isinstance(entry[name], str) for name in fields)
            or not entry["input_context"] or not entry["profile_context"]
        ):
            raise ValueError(f"Invalid profile mapping entry: {entry!r}")
        source = (entry["input_context"], entry["input_batch"])
        if source in mapping:
            raise ValueError(f"Duplicate profile mapping for input group {source!r}")
        mapping[source] = (entry["profile_context"], entry["profile_batch"])
    return mapping


def export_shared_vcc_h5ad(
    checkpoint_path: Path,
    profile_path: Path,
    calibration_path: Path,
    controls_by_context: dict[str, Path],
    gene_file: Path,
    pert_file: Path,
    output_path: Path,
    *,
    profile_mapping_path: Path | None = None,
    shared_strength: float = 1.0,
    allow_unmatched_batch: bool = False,
    allow_missing_target: bool = False,
    device_name: str = "cpu",
    seed: int = 0,
    steps: int = 8,
) -> None:
    contexts = set(controls_by_context)
    if contexts not in ({"A", "B", "C"}, {"D", "E", "F"}):
        raise ValueError("Use exactly A/B/C or exactly D/E/F control contexts")
    if steps < 1:
        raise ValueError("steps must be positive")
    if not np.isfinite(shared_strength) or not 0 <= shared_strength <= 1:
        raise ValueError("shared_strength must be finite and between zero and one")
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"Requested {device_name}, but CUDA is unavailable")
    provenance_path = output_path.with_suffix(output_path.suffix + ".json")
    if output_path.exists() or provenance_path.exists():
        raise FileExistsError(f"Refusing to overwrite {output_path} or its provenance")

    input_paths = {
        "checkpoint": checkpoint_path,
        "profile": profile_path,
        "calibration": calibration_path,
        "gene_names": gene_file,
        "pert_counts": pert_file,
        **{f"controls_{context}": path for context, path in controls_by_context.items()},
    }
    if profile_mapping_path is not None:
        input_paths["profile_mapping"] = profile_mapping_path
    hashes = {name: file_sha256(path) for name, path in input_paths.items()}
    genes = _read_gene_list(gene_file)
    targets = _read_targets(pert_file)
    if set(targets) - set(genes):
        raise ValueError("Challenge targets must occur in the challenge gene axis")
    profile, metadata, keys, effects, profile_digest = _load_response_profile(profile_path)
    if str(metadata["control"]) != CONTROL:
        raise ValueError("The profile must use the challenge non-targeting control label")
    model, checkpoint, graph, anchor, checkpoint_digest = _load_shared_model(
        checkpoint_path, profile, metadata, device_name,
    )
    if checkpoint.get("epigenetics") is not None:
        raise ValueError(
            "Epigenomic checkpoints cannot be exported to anonymized challenge contexts "
            "without verified matching reference metadata"
        )
    selected_shared, conditional = resolve_shared_strength(
        calibration_path, checkpoint_path, checkpoint, profile_path, metadata,
        cells_per_target=400, shared_strength=shared_strength, prediction_steps=steps,
        allow_unmatched_batch=allow_unmatched_batch, control_sampling="input-controls",
    )
    if (profile_digest, checkpoint_digest) != (hashes["profile"], hashes["checkpoint"]):
        raise RuntimeError("Model or profile changed while loading")
    no_response = selected_shared == 0 and conditional == 0
    mapping = _read_profile_mapping(profile_mapping_path) if profile_mapping_path is not None else {}
    if selected_shared and not mapping:
        raise ValueError("A nonzero shared response requires an explicit profile mapping")

    model_genes = checkpoint["genes"]
    model_index = {gene: index for index, gene in enumerate(model_genes)}
    challenge_index = {gene: index for index, gene in enumerate(genes)}
    model_genes_missing = sorted(set(model_genes) - set(genes))
    if model_genes_missing and not no_response:
        raise ValueError(
            f"{len(model_genes_missing)} model genes are absent from the challenge axis; "
            "a nonzero calibrated response requires the full model gene panel"
        )
    model_columns = np.array([challenge_index[g] for g in model_genes if g in challenge_index])
    missing_targets = sorted(set(targets) - set(model_genes)) if conditional else []
    if missing_targets and not allow_missing_target:
        raise ValueError(
            f"{len(missing_targets)} targets are absent from the model gene axis; "
            "explicitly opt in to an unmodified-control fallback"
        )
    if missing_targets:
        print(f"WARNING: missing targets use unmodified controls: {missing_targets}", flush=True)
    seen_targets = sorted(set(targets) & set(checkpoint["training_targets"]))
    if seen_targets:
        print(f"WARNING: challenge targets seen during training: {seen_targets}", flush=True)
    training_ids = set(checkpoint["training_cell_ids"])
    reference_ids = set((checkpoint.get("reference") or {}).get("cell_ids", []))
    check_cell_id_schemes(training_ids, reference_ids)
    protected_ids = training_ids | reference_ids
    lookup = {key: i for i, key in enumerate(keys)}
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _check_disk_reserve(output_path.parent)
    stored_entries = 0
    used_groups = set()
    with tempfile.TemporaryDirectory(prefix="c3vcell-shared-vcc-", dir=output_path.parent) as tmp:
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
            check_cell_id_schemes(training_ids, reference_ids, controls.obs_names)
            if protected_ids.intersection(controls.obs_names):
                raise ValueError(f"Input controls overlap training or reference cells in context {context}")
            batches = _batches(controls.obs)
            groups = {(context, str(batch)) for batch in batches}
            used_groups.update(groups)
            if selected_shared:
                missing = sorted(groups - set(mapping))
                if missing:
                    raise ValueError(f"Profile mapping is missing input groups: {missing}")
                unavailable = sorted({mapping[group] for group in groups} - set(lookup))
                if unavailable:
                    raise ValueError(f"Mapped profile groups do not exist: {unavailable}")
            matrix = sp.csr_matrix(controls.X)
            for target in targets:
                _check_disk_reserve(output_path.parent)
                selected = rng.choice(controls.n_obs, size=400, replace=controls.n_obs < 400)
                if no_response or target in missing_targets:
                    sparse = matrix[selected].copy()
                    if np.any(np.asarray(sparse.sum(axis=1)).ravel() > 1_000_000):
                        raise ValueError("Predicted cell exceeds the challenge count limit")
                    sparse = sparse.astype(np.int32)
                else:
                    raw = matrix[selected].toarray().astype(np.float32, copy=False)
                    model_counts = raw[:, model_columns]
                    if np.any(model_counts.sum(axis=1) == 0):
                        raise ValueError("A control cell has no counts on the model gene panel")
                    counts = torch.as_tensor(model_counts, dtype=torch.float32, device=device)
                    if selected_shared:
                        indices = [lookup[mapping[(context, str(batch))]] for batch in batches[selected]]
                        logfold = torch.as_tensor(effects[indices], device=device) * selected_shared
                    else:
                        logfold = torch.zeros_like(counts)
                    target_index = (
                        torch.full((400,), model_index[target], dtype=torch.long, device=device)
                        if conditional else None
                    )
                    generated = _generate_shared_counts(
                        counts, logfold, effect_strength=selected_shared,
                        conditional_strength=conditional, model=model, target_index=target_index,
                        anchor=anchor, graph=graph, steps=steps,
                    )
                    raw[:, model_columns] = generated.cpu().numpy()
                    if not np.isfinite(raw).all() or np.any(raw.sum(axis=1) > 1_000_000):
                        raise ValueError("Predicted cell exceeds the challenge count limit")
                    sparse = sp.csr_matrix(raw.astype(np.int32))
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
                    shard, compression="gzip",
                )
                shards.append(shard)
                print(f"Generated {context}/{target}: 400 cells", flush=True)
        if selected_shared and set(mapping) != used_groups:
            raise ValueError(f"Profile mapping contains unused input groups: {sorted(set(mapping) - used_groups)}")
        assembled = _concat_bounded(shards, temporary)
        _check_disk_reserve(output_path.parent)
        for name, path in input_paths.items():
            if file_sha256(path) != hashes[name]:
                raise RuntimeError(f"Input changed during export: {name}")
        provenance = {
            "format": "VCC-2026 h5ad; use vcc prep for official validation",
            "predictor": "calibrated-shared-residual",
            "contexts": sorted(contexts),
            "perturbations": len(targets),
            "cells_per_perturbation_and_context": 400,
            "genes": len(genes),
            "stored_entries": stored_entries,
            "control_sampling": "input-controls",
            "shared_strength": selected_shared,
            "conditional_strength": conditional,
            "no_response_selected": no_response,
            "profile_mapping": [
                {"input_context": source[0], "input_batch": source[1],
                 "profile_context": destination[0], "profile_batch": destination[1]}
                for source, destination in sorted(mapping.items())
            ],
            "model_genes_not_in_challenge": model_genes_missing,
            "challenge_genes_not_in_model": sorted(set(genes) - set(model_genes)),
            "control_baseline_targets": targets if no_response else missing_targets,
            "targets_seen_during_training": seen_targets,
            "seed": seed,
            "steps": steps,
            "input_sha256": hashes,
            "output_sha256": file_sha256(assembled),
            "note": (
                "Explicit profile mapping is not evidence that training and challenge contexts "
                "are biologically equivalent; local calibration may not transfer."
            ),
        }
        temporary_provenance = temporary / "provenance.json"
        temporary_provenance.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
        os.link(assembled, output_path)
        os.link(temporary_provenance, provenance_path)
