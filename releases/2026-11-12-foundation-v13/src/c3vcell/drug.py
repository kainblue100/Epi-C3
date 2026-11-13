"""SMILES- and dose-conditioned population flow for Tahoe drug responses."""

from __future__ import annotations

import json
import os
import random
import warnings
from pathlib import Path
from typing import Literal

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence

from .data import CONTROL, validate_counts
from .identity import check_cell_id_schemes
from .integrity import file_sha256
from .model import C3VCell, ModelConfig
from .predict import _batches, _contexts, _labels, _resolve_effect_strength, _sample_controls
from .reference import load_reference, reference_metadata
from .tahoe import TAHOE_CONTROL, _smiles
from .training import (
    _batch, _control_groups, fit_representation, representation_diagnostics,
    warn_on_representation_collapse,
)


def validate_tahoe(data: ad.AnnData, *, allow_control_only: bool = False) -> None:
    validate_counts(data)
    columns = (
        "target_gene", "compound", "drug", "canonical_smiles", "dose_um", "context", "batch",
    )
    missing = set(columns) - set(data.obs.columns)
    if missing:
        raise ValueError(f"Tahoe data is missing columns: {sorted(missing)}")
    if data.obs[list(columns)].isna().any().any():
        raise ValueError("Tahoe treatment metadata cannot be missing")
    obs = data.obs
    labels = obs["target_gene"].astype(str)
    compounds = obs["compound"].astype(str)
    controls = labels == CONTROL
    if not np.array_equal(controls.to_numpy(), (compounds == CONTROL).to_numpy()):
        raise ValueError("Tahoe control and compound labels disagree")
    doses = pd.to_numeric(obs["dose_um"], errors="raise").to_numpy(dtype=float)
    if not np.isfinite(doses).all() or np.any(doses[controls] != 0) or np.any(doses[~controls] <= 0):
        raise ValueError("Tahoe controls need zero dose and treatments need a positive dose")
    if (obs.loc[controls, "drug"].astype(str) != TAHOE_CONTROL).any():
        raise ValueError("Tahoe controls must be DMSO_TF")
    if (obs.loc[controls, "canonical_smiles"].astype(str) != "").any():
        raise ValueError("Tahoe controls cannot have a drug SMILES")
    treated = obs.loc[~controls]
    if not len(treated) and not allow_control_only:
        raise ValueError("Tahoe data needs at least one drug treatment")
    if (treated["drug"].astype(str) == TAHOE_CONTROL).any():
        raise ValueError("DMSO_TF cannot be a drug treatment")
    if (treated["compound"].astype(str) != treated["canonical_smiles"].astype(str)).any():
        raise ValueError("Compound groups must match the canonical SMILES")
    for value in treated["canonical_smiles"].astype(str).unique():
        _smiles(value, "treatment")
    grouped = treated.groupby("target_gene", observed=True)[
        ["compound", "drug", "dose_um"]
    ].nunique(dropna=False)
    if (grouped > 1).any().any():
        raise ValueError("Each treatment label must specify one compound, drug, and dose")


class DrugConditioner(nn.Module):
    """Encode the supplied canonical SMILES text and micromolar concentration."""

    def __init__(self, latent_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.char_embedding = nn.Embedding(128, 32, padding_idx=0)
        self.sequence = nn.GRU(32, hidden_dim, batch_first=True)
        self.output = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, smiles: list[str], doses_um: Tensor) -> Tensor:
        if not smiles or doses_um.ndim != 1 or len(smiles) != len(doses_um):
            raise ValueError("Drug SMILES and doses must have matching nonempty batch sizes")
        if not torch.isfinite(doses_um).all() or torch.any(doses_um < 0):
            raise ValueError("Drug doses must be finite and nonnegative")
        sequences = [_smiles(value, "treatment") for value in smiles]
        unique_sequences = list(dict.fromkeys(sequences))
        sequence_index = {value: index for index, value in enumerate(unique_sequences)}
        inverse = torch.tensor(
            [sequence_index[value] for value in sequences],
            dtype=torch.long, device=doses_um.device,
        )
        lengths = torch.tensor([len(value) for value in unique_sequences], dtype=torch.long)
        tokens = torch.zeros(
            (len(unique_sequences), int(lengths.max())), dtype=torch.long, device=doses_um.device,
        )
        for index, value in enumerate(unique_sequences):
            tokens[index, :len(value)] = torch.tensor(
                [ord(char) + 1 for char in value], dtype=torch.long, device=tokens.device,
            )
        packed = pack_padded_sequence(
            self.char_embedding(tokens), lengths, batch_first=True, enforce_sorted=False,
        )
        _, state = self.sequence(packed)
        sequence_state = state[-1].index_select(0, inverse)
        embedding = self.output(torch.cat((sequence_state, torch.log1p(doses_um)[:, None]), dim=-1))
        scale = (doses_um / (1 + doses_um))[:, None]
        return F.normalize(embedding, dim=-1) * scale


def train_drug(
    train_path: Path,
    outdir: Path,
    *,
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
    reference_path: Path | None = None,
    reference_fraction: float = 0.5,
    reference_max_cells: int = 10_000,
    reference_seed: int | None = None,
    reference_min_coverage: float = 0.2,
    expected_reference_identity: str | None = None,
) -> Path:
    if min(pretrain_steps, flow_steps, batch_size) < 1 or learning_rate <= 0:
        raise ValueError("Training steps, batch size, and learning rate must be positive")
    if not np.isfinite(variance_weight) or variance_weight < 0:
        raise ValueError("variance_weight must be finite and nonnegative")
    if not np.isfinite(covariance_weight) or covariance_weight < 0:
        raise ValueError("covariance_weight must be finite and nonnegative")
    if regularization_view not in {"masked", "unmasked"}:
        raise ValueError("regularization_view must be masked or unmasked")
    if covariance_weight > 0 and batch_size < 2:
        raise ValueError("Covariance regularization requires at least two cells per batch")
    if outdir.exists() and any(outdir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty directory: {outdir}")
    if reference_path is not None and not 0 < reference_fraction < 1:
        raise ValueError("Reference fraction must be strictly between zero and one")
    if expected_reference_identity is not None and reference_path is None:
        raise ValueError("An expected reference identity requires reference data")
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
    validate_tahoe(data)
    labels = _labels(data.obs)
    contexts = _contexts(data.obs)
    batches = _batches(data.obs)
    smiles = data.obs["canonical_smiles"].astype(str).to_numpy()
    doses = pd.to_numeric(data.obs["dose_um"]).to_numpy(dtype=np.float32)
    target_rows = {
        target: np.flatnonzero(labels == target)
        for target in sorted(set(labels) - {CONTROL})
    }
    exact, _ = _control_groups(data.obs, CONTROL)
    missing = sorted({
        (contexts[row], batches[row])
        for rows in target_rows.values() for row in rows
        if (contexts[row], batches[row]) not in exact
    })
    if missing:
        raise ValueError(f"Drug training requires matched cell-line/plate DMSO controls: {missing}")
    matrix = sp.csr_matrix(data.X, dtype=np.float32)
    reference = None
    ref_seed = seed if reference_seed is None else reference_seed
    if reference_path is not None:
        reference = load_reference(
            reference_path, data.var_names.astype(str).tolist(),
            max_cells=reference_max_cells, seed=ref_seed, min_coverage=reference_min_coverage,
        )
        check_cell_id_schemes(data.obs_names, reference.cell_ids)
        if expected_reference_identity is not None and reference.source_identity != expected_reference_identity:
            raise ValueError("Reference data changed since the final checkpoint was trained")
    model = C3VCell(ModelConfig(
        data.n_vars, latent_dim, hidden_dim, variance_weight=variance_weight,
        covariance_weight=covariance_weight, regularization_view=regularization_view,
    )).to(device)
    conditioner = DrugConditioner(latent_dim, hidden_dim).to(device)
    outdir.mkdir(parents=True, exist_ok=True)
    with (outdir / "training.jsonl").open("w", encoding="utf-8") as log:
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
                "drug-conditioned predictions may be unreliable.",
                RuntimeWarning, stacklevel=2,
            )
        model.encoder.requires_grad_(False)
        model.decoder.requires_grad_(False)
        model.predictor.requires_grad_(False)
        optimizer = torch.optim.AdamW(
            list(model.velocity.parameters()) + list(conditioner.parameters()), lr=learning_rate,
        )
        targets = np.asarray(sorted(target_rows), dtype=object)
        for step in range(1, flow_steps + 1):
            chosen = rng.choice(targets, size=batch_size)
            rows = np.asarray([rng.choice(target_rows[name]) for name in chosen])
            controls = np.asarray([
                rng.choice(exact[(contexts[row], batches[row])]) for row in rows
            ])
            control_counts = _batch(matrix, controls, device)
            treatment_counts = _batch(matrix, rows, device)
            dose_tensor = torch.as_tensor(doses[rows], dtype=torch.float32, device=device)
            optimizer.zero_grad(set_to_none=True)
            embedding = conditioner(smiles[rows].tolist(), dose_tensor)
            loss, metrics = model.flow_loss_condition(control_counts, treatment_counts, embedding)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite drug flow loss at step {step}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(model.velocity.parameters()) + list(conditioner.parameters()), 5.0,
            )
            optimizer.step()
            if step == 1 or step % 25 == 0 or step == flow_steps:
                record = {"phase": "drug-flow", "step": step, **metrics}
                log.write(json.dumps(record) + "\n")
                log.flush()
                print(json.dumps(record), flush=True)
    if file_sha256(train_path) != training_source_sha256:
        raise RuntimeError("Training data changed while the drug model was being fitted")
    checkpoint = {
        "format_version": 1,
        "kind": "tahoe-drug-smiles-dose",
        "model_config": model.configuration(),
        "model_state": model.state_dict(),
        "conditioner_state": conditioner.state_dict(),
        "genes": data.var_names.astype(str).tolist(),
        "training_cell_ids": data.obs_names.astype(str).tolist(),
        "training_source_sha256": training_source_sha256,
        "representation_diagnostics": diagnostics,
        "training_compounds": sorted(set(data.obs["compound"].astype(str)) - {CONTROL}),
        "training_targets": sorted(target_rows),
        "training_parameters": {
            "pretrain_steps": pretrain_steps, "flow_steps": flow_steps,
            "batch_size": batch_size, "learning_rate": learning_rate,
            "variance_weight": variance_weight, "covariance_weight": covariance_weight,
            "regularization_view": regularization_view, "seed": seed,
        },
        "reference": reference_metadata(
            reference, reference_path, fraction=reference_fraction, max_cells=reference_max_cells,
            seed=ref_seed, min_coverage=reference_min_coverage, model_genes=data.n_vars,
        ),
        "control": CONTROL,
        "concentration_unit": "uM",
    }
    temporary = outdir / "model.pt.tmp"
    torch.save(checkpoint, temporary)
    final = outdir / "model.pt"
    os.replace(temporary, final)
    return final


def load_drug_model(path: Path, device_name: str = "cpu") -> tuple[C3VCell, DrugConditioner, dict]:
    device = torch.device(device_name)
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint.get("format_version") != 1 or checkpoint.get("kind") != "tahoe-drug-smiles-dose":
        raise ValueError("Unsupported drug-response checkpoint")
    warn_on_representation_collapse(checkpoint, path)
    config = ModelConfig(**checkpoint["model_config"])
    model = C3VCell(config).to(device)
    conditioner = DrugConditioner(config.latent_dim, config.hidden_dim).to(device)
    model.load_state_dict(checkpoint["model_state"])
    conditioner.load_state_dict(checkpoint["conditioner_state"])
    model.eval()
    conditioner.eval()
    return model, conditioner, checkpoint


@torch.no_grad()
def predict_drug(
    checkpoint_path: Path,
    reference_path: Path,
    control_path: Path,
    output_path: Path,
    *,
    cells_per_target: int = 400,
    steps: int = 16,
    effect_strength: float | None = None,
    calibration_path: Path | None = None,
    sampling: Literal["negative-binomial", "residual-rounding"] = "residual-rounding",
    seed: int = 0,
    device_name: str = "cpu",
) -> None:
    if cells_per_target < 1:
        raise ValueError("cells_per_target must be positive")
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite {output_path}")
    model, conditioner, checkpoint = load_drug_model(checkpoint_path, device_name)
    strength = _resolve_effect_strength(
        checkpoint, checkpoint_path, sampling, effect_strength, calibration_path,
        calibration_kind="drug", prediction_steps=steps, cells_per_target=cells_per_target,
    )
    reference = ad.read_h5ad(reference_path)
    controls = ad.read_h5ad(control_path)
    validate_tahoe(reference)
    validate_tahoe(controls, allow_control_only=True)
    if reference.var_names.tolist() != checkpoint["genes"] or controls.var_names.tolist() != checkpoint["genes"]:
        raise ValueError("Drug reference, controls, and checkpoint gene axes must match")
    training_ids = set(checkpoint["training_cell_ids"])
    reference_ids = set((checkpoint.get("reference") or {}).get("cell_ids", []))
    check_cell_id_schemes(training_ids, reference_ids, reference.obs_names, controls.obs_names)
    training_ids.update(reference_ids)
    if set(reference.obs_names).intersection(controls.obs_names):
        raise ValueError("Drug inference controls overlap the held-out reference")
    if set(_labels(controls.obs)) != {CONTROL}:
        raise ValueError("Drug inference controls must all be DMSO_TF")
    if training_ids.intersection(reference.obs_names) or training_ids.intersection(controls.obs_names):
        raise ValueError("Training cells overlap drug inference or reference cells")
    test_compounds = set(reference.obs["compound"].astype(str)) - {CONTROL}
    seen = test_compounds.intersection(checkpoint["training_compounds"])
    if seen:
        raise ValueError("Held-out canonical SMILES appeared in drug training")
    labels = _labels(reference.obs)
    contexts = _contexts(reference.obs)
    batches = _batches(reference.obs)
    control_contexts = _contexts(controls.obs)
    control_batches = _batches(controls.obs)
    groups = (
        reference.obs.loc[labels != CONTROL, [
            "context", "target_gene", "drug", "canonical_smiles", "dose_um",
        ]]
        .drop_duplicates()
        .sort_values(["context", "target_gene"])
    )
    if groups.empty:
        raise ValueError("No held-out drug conditions to predict")
    matrix = sp.csr_matrix(controls.X)
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    device = torch.device(device_name)
    chunks, obs_rows = [], []
    for row in groups.itertuples(index=False):
        context, target = str(row.context), str(row.target_gene)
        sampled, matched_batches = _sample_controls(
            contexts, batches, labels, control_contexts, control_batches, context, target,
            cells_per_target=cells_per_target, rng=rng, allow_unmatched_batch=False,
        )
        counts = torch.as_tensor(matrix[sampled].toarray(), dtype=torch.float32, device=device)
        dose = torch.tensor([float(row.dose_um)], dtype=torch.float32, device=device)
        embedding = conditioner([str(row.canonical_smiles)], dose).expand(len(sampled), -1)
        generated = model.predict_counts_condition(
            counts, embedding, steps=steps, sampling=sampling, effect_strength=strength,
        )
        chunks.append(sp.csr_matrix(generated.cpu().numpy()))
        start = len(obs_rows)
        obs_rows.extend({
            "cell_id": f"drug-pred-{start + i:09d}",
            "target_gene": target,
            "drug": str(row.drug),
            "compound": str(row.canonical_smiles),
            "canonical_smiles": str(row.canonical_smiles),
            "dose_um": float(row.dose_um),
            "context": context,
            "batch": str(matched_batches[i]),
        } for i in range(len(sampled)))
    output = ad.AnnData(
        sp.vstack(chunks, format="csr"), obs=pd.DataFrame(obs_rows).set_index("cell_id"),
        var=controls.var.copy(),
    )
    output.uns["c3vcell_prediction"] = {
        "model": "SMILES and dose conditioned population flow",
        "concentration_unit": "uM", "sampling": sampling,
        "effect_strength": strength, "seed": seed,
        "effect_calibration": str(calibration_path) if calibration_path else "",
        "checkpoint": str(checkpoint_path),
    }
    validate_counts(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.write_h5ad(output_path, compression="gzip")
