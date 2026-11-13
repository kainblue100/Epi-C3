from __future__ import annotations

import json
import os
import tempfile
import warnings
from dataclasses import asdict
from pathlib import Path
from typing import Literal

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

from .calibration import file_sha256 as _sha256_file
from .data import validate_counts
from .model import ModelConfig
from .predict import ControlSampling, _validate_control_sampling, predict_holdout


def _signature(checkpoint: dict) -> dict:
    required = (
        "format_version", "genes", "control", "model_config", "training_parameters",
        "training_targets", "training_cell_ids", "seed",
    )
    missing = [key for key in required if key not in checkpoint]
    if missing:
        raise ValueError(f"Checkpoint lacks required ensemble metadata: {missing}")
    training_cells = checkpoint["training_cell_ids"]
    if len(training_cells) != len(set(training_cells)):
        raise ValueError("Checkpoint training cell IDs must be unique")
    config = ModelConfig(**checkpoint["model_config"])
    model_config = asdict(config)
    training_parameters = dict(checkpoint["training_parameters"])
    for key in ("variance_weight", "covariance_weight", "regularization_view"):
        if key in training_parameters and training_parameters[key] != model_config[key]:
            raise ValueError(f"Checkpoint has inconsistent {key} metadata")
        training_parameters[key] = model_config[key]
    reference = checkpoint.get("reference")
    if reference is not None:
        reference = {key: value for key, value in reference.items() if key != "path"}
        if "cell_ids" not in reference or "source_identity" not in reference:
            raise ValueError("Checkpoint reference provenance is incomplete")
        reference["cell_ids"] = sorted(reference["cell_ids"])
    return {
        "format_version": checkpoint["format_version"],
        "genes": checkpoint["genes"],
        "control": checkpoint["control"],
        "model_config": model_config,
        "training_parameters": training_parameters,
        "training_targets": sorted(checkpoint["training_targets"]),
        "training_cell_ids": sorted(training_cells),
        "training_source_sha256": checkpoint.get("training_source_sha256"),
        "reference": reference,
        "epigenetics": checkpoint.get("epigenetics"),
    }


def _same_graph(left: dict | None, right: dict | None) -> bool:
    if left is None or right is None:
        return left is None and right is None
    return (
        left["shape"] == right["shape"]
        and torch.equal(left["indices"], right["indices"])
        and torch.equal(left["values"], right["values"])
    )


def _same_epigenetic_features(left: dict, right: dict) -> bool:
    for key in ("epi_values", "epi_masks"):
        a = left["model_state"].get(key)
        b = right["model_state"].get(key)
        if a is None or b is None:
            if a is not None or b is not None:
                return False
        elif not torch.equal(a, b):
            return False
    return True


def _check_members(paths: list[Path]) -> tuple[list[str], list[int]]:
    if len(paths) < 2:
        raise ValueError("Ensemble prediction needs at least two checkpoints")
    digests = [_sha256_file(path) for path in paths]
    if len(set(digests)) != len(digests):
        raise ValueError("Ensemble checkpoints must be distinct")
    signature = None
    graph = None
    first_checkpoint = None
    seeds = []
    for path in paths:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        candidate = _signature(checkpoint)
        if signature is None:
            signature = candidate
            graph = checkpoint.get("graph")
            first_checkpoint = checkpoint
        elif (
            candidate != signature or not _same_graph(graph, checkpoint.get("graph"))
            or first_checkpoint is None or not _same_epigenetic_features(first_checkpoint, checkpoint)
        ):
            raise ValueError(
                "Ensemble checkpoints must use the same genes, training cells, targets, "
                "training counts, reference, model configuration, training parameters, graph, "
                "and epigenomic prior"
            )
        seeds.append(int(checkpoint["seed"]))
    if len(set(seeds)) != len(seeds):
        raise ValueError("Ensemble checkpoints must use different training seeds")
    if signature is not None and signature["training_source_sha256"] is None:
        warnings.warn(
            "These checkpoints predate training-file fingerprints; matching cell IDs "
            "cannot verify that ensemble members used the same counts.",
            RuntimeWarning, stacklevel=2,
        )
    return digests, seeds


def _check_alignment(members: list[ad.AnnData]) -> None:
    first = members[0]
    columns = ["context", "target_gene", "batch", "source_control_id"]
    for member in members[1:]:
        if (
            not member.obs_names.equals(first.obs_names)
            or not member.var_names.equals(first.var_names)
            or not member.obs[columns].equals(first.obs[columns])
        ):
            raise ValueError("Ensemble predictions do not use identical cells and controls")


def predict_ensemble(
    checkpoint_paths: list[Path],
    reference_path: Path,
    input_control_path: Path,
    output_path: Path,
    disagreement_path: Path,
    *,
    cells_per_target: int = 400,
    device_name: str = "cpu",
    seed: int = 0,
    steps: int = 16,
    allow_unmatched_batch: bool = False,
    sampling: Literal["negative-binomial", "residual-rounding"] = "residual-rounding",
    effect_strength: float = 1.0,
    control_sampling: ControlSampling = "reference-batch",
) -> None:
    _validate_control_sampling(control_sampling)
    if cells_per_target < len(checkpoint_paths):
        raise ValueError("cells_per_target must be at least the number of ensemble members")
    manifest_path = output_path.with_suffix(output_path.suffix + ".ensemble.json")
    outputs = [output_path, disagreement_path, manifest_path]
    if len({path.resolve() for path in outputs}) != len(outputs):
        raise ValueError("Prediction, disagreement, and manifest paths must be different")
    if len({path.parent.resolve() for path in outputs}) != 1:
        raise ValueError("Prediction and disagreement files must share an output directory")
    for path in outputs:
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}")
    digests, seeds = _check_members(checkpoint_paths)
    input_hashes = {
        "reference": _sha256_file(reference_path),
        "controls": _sha256_file(input_control_path),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".c3-ensemble-", dir=output_path.parent) as temp:
        temporary = Path(temp)
        members = []
        for index, checkpoint in enumerate(checkpoint_paths):
            member_path = temporary / f"member-{index}.h5ad"
            predict_holdout(
                checkpoint, reference_path, input_control_path, member_path,
                cells_per_target=cells_per_target, device_name=device_name, seed=seed,
                steps=steps, allow_unmatched_batch=allow_unmatched_batch,
                sampling=sampling, effect_strength=effect_strength, control_sampling=control_sampling,
            )
            members.append(ad.read_h5ad(member_path))
        _check_alignment(members)
        first = members[0]
        groups = sorted(set(zip(
            first.obs["context"].astype(str), first.obs["target_gene"].astype(str),
        )))
        rng = np.random.default_rng(seed)
        matrix_parts = []
        obs_parts = []
        group_obs = []
        means = []
        for context, target in groups:
            mask = (
                (first.obs["context"].astype(str).to_numpy() == context)
                & (first.obs["target_gene"].astype(str).to_numpy() == target)
            )
            rows = np.flatnonzero(mask)
            if len(rows) != cells_per_target:
                raise ValueError("Ensemble group has an unexpected number of cells")
            group_means = np.stack([
                np.asarray(member.X[rows].mean(axis=0)).ravel() for member in members
            ])
            means.append(group_means)
            group_obs.append({"context": context, "target_gene": target})
            assignment = rng.permutation(np.arange(len(rows)) % len(members))
            for index, member in enumerate(members):
                selected = rows[assignment == index]
                matrix_parts.append(sp.csr_matrix(member.X[selected]))
                obs = member.obs.iloc[selected].copy()
                obs["ensemble_member"] = index
                obs_parts.append(obs)
        member_means = np.stack(means)
        prediction = ad.AnnData(
            X=sp.vstack(matrix_parts, format="csr"),
            obs=pd.concat(obs_parts),
            var=first.var.copy(),
        )
        prediction.uns["c3vcell_ensemble"] = {
            "checkpoint_sha256": np.asarray(digests),
            "training_seeds": np.asarray(seeds),
            "seed": seed,
            "steps": steps,
            "sampling": sampling,
            "effect_strength": effect_strength,
            "control_sampling": control_sampling,
        }
        validate_counts(prediction)
        disagreement = ad.AnnData(
            X=member_means.mean(axis=1).astype(np.float32),
            obs=pd.DataFrame(
                group_obs, index=[f"group-{index:06d}" for index in range(len(groups))],
            ),
            var=first.var.copy(),
        )
        disagreement.layers["between_model_std"] = member_means.std(
            axis=1, ddof=1,
        ).astype(np.float32)
        for index in range(len(members)):
            disagreement.layers[f"member_{index}_mean"] = member_means[:, index, :].astype(
                np.float32
            )
        disagreement.uns["c3vcell_ensemble"] = {
            "checkpoint_sha256": np.asarray(digests),
            "training_seeds": np.asarray(seeds),
            "description": (
                "Sample standard deviation of each model's generated pseudobulk count mean; "
                "includes finite-cell sampling noise and is not a calibrated confidence interval."
            ),
        }
        prediction_temp = temporary / "prediction.h5ad"
        disagreement_temp = temporary / "disagreement.h5ad"
        prediction.write_h5ad(prediction_temp, compression="gzip")
        disagreement.write_h5ad(disagreement_temp, compression="gzip")
        if digests != [_sha256_file(path) for path in checkpoint_paths] or input_hashes != {
            "reference": _sha256_file(reference_path),
            "controls": _sha256_file(input_control_path),
        }:
            raise RuntimeError("An ensemble checkpoint or input changed during prediction")
        manifest = {
            "format_version": 1,
            "checkpoints": [
                {"path": str(path.resolve()), "sha256": digest, "training_seed": member_seed}
                for path, digest, member_seed in zip(checkpoint_paths, digests, seeds)
            ],
            "reference_sha256": input_hashes["reference"],
            "controls_sha256": input_hashes["controls"],
            "prediction_sha256": _sha256_file(prediction_temp),
            "disagreement_sha256": _sha256_file(disagreement_temp),
            "cells_per_target": cells_per_target,
            "seed": seed,
            "steps": steps,
            "sampling": sampling,
            "effect_strength": effect_strength,
            "allow_unmatched_batch": allow_unmatched_batch,
            "control_sampling": control_sampling,
            "disagreement_is_calibrated": False,
        }
        manifest_temp = temporary / "manifest.json"
        manifest_temp.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        created = []
        try:
            for source, target in zip(
                (prediction_temp, disagreement_temp, manifest_temp), outputs,
            ):
                os.link(source, target)
                created.append(target)
        except OSError:
            for path in created:
                path.unlink()
            raise
