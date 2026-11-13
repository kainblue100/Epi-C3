"""Joint RNA, accessibility and observed-contact encoder with strict cell identities.

Identity consistency checks do not prove that the source experiment paired the
modalities; the pairing map must be verified against source evidence separately.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import re
from typing import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from epi_c3_shared_foundation_v1 import SharedFoundationEncoder


@dataclass(frozen=True)
class CellKey:
    dataset: str
    experiment: str
    sample: str
    barcode: str
    cell_line: str
    species: str
    assembly: str


@dataclass(frozen=True)
class ModalityIdentity:
    modality: str
    key: CellKey
    pairing_map_sha256: str
    source_accession: str


@dataclass(frozen=True)
class PairedCell:
    rna: ModalityIdentity
    atac: ModalityIdentity
    contacts: ModalityIdentity
    split: str


def validate_paired_cells(
    cells: Sequence[PairedCell], *, species: str, assembly: str, minimum_training_lines: int = 3,
) -> dict:
    if not cells or not species or not assembly or minimum_training_lines < 0:
        raise ValueError("A nonempty paired collection and an explicit genome axis are required")
    seen = set()
    line_counts: Counter = Counter()
    splits: Counter = Counter()
    for cell in cells:
        if cell.split not in ("train", "validation", "test"):
            raise ValueError("Every paired cell needs an explicit split")
        modalities = (cell.rna, cell.atac, cell.contacts)
        if [item.modality for item in modalities] != ["rna", "atac", "contacts"]:
            raise ValueError("Each cell needs exactly one RNA, ATAC and contact observation")
        key = cell.rna.key
        fields = (key.dataset, key.experiment, key.sample, key.barcode, key.cell_line, key.species, key.assembly)
        if any(not isinstance(value, str) or not value.strip() for value in fields):
            raise ValueError("Paired-cell identity fields must be complete")
        if key.species != species or key.assembly != assembly:
            raise ValueError("A cell is incompatible with the selected species and genome assembly")
        if any(item.key != key for item in modalities):
            raise ValueError("Modalities do not identify the same cell, cell line and genome")
        digests = {item.pairing_map_sha256 for item in modalities}
        if (
            len(digests) != 1 or re.fullmatch(r"[0-9a-f]{64}", next(iter(digests))) is None
            or any(not item.source_accession.strip() for item in modalities)
        ):
            raise ValueError("A shared source-backed pairing map and source accessions are required")
        identity = (key.dataset, key.experiment, key.sample, key.barcode)
        if identity in seen:
            raise ValueError("A cell is duplicated within or across splits")
        seen.add(identity)
        splits[cell.split] += 1
        if cell.split == "train":
            line_counts[key.cell_line] += 1
    if len(line_counts) < minimum_training_lines:
        raise ValueError(f"Training needs at least {minimum_training_lines} distinct cell lines")
    return {
        "cells": len(cells), "splits": dict(sorted(splits.items())),
        "training_lines": dict(sorted(line_counts.items())), "species": species, "assembly": assembly,
    }


class MaskedExpressionEncoder(nn.Module):
    def __init__(self, feature_count: int, hidden_dim: int, latent_dim: int) -> None:
        super().__init__()
        if min(feature_count, hidden_dim, latent_dim) < 1:
            raise ValueError("Encoder dimensions must be positive")
        self.feature_count = feature_count
        self.network = nn.Sequential(
            nn.Linear(feature_count * 2, hidden_dim), nn.GELU(),
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, values: Tensor, measured: Tensor) -> tuple[Tensor, Tensor]:
        if (
            values.ndim != 2 or values.shape[1] != self.feature_count
            or measured.shape != values.shape or measured.dtype != torch.bool
            or not torch.is_floating_point(values) or not torch.isfinite(values).all()
            or torch.any(values < 0) or torch.any(values.masked_select(~measured) != 0)
        ):
            raise ValueError("Expression values and feature measurement masks are inconsistent")
        present = measured.any(dim=1)
        library = values.sum(dim=1, keepdim=True)
        scaled = torch.log1p(values * (10000.0 / library.clamp_min(1.0)))
        return self.network(torch.cat((scaled, measured.to(values.dtype)), dim=1)), present


class ObservedContactEncoder(nn.Module):
    def __init__(self, feature_count: int, hidden_dim: int, latent_dim: int) -> None:
        super().__init__()
        if min(feature_count, hidden_dim, latent_dim) < 1:
            raise ValueError("Contact encoder dimensions must be positive")
        self.feature_count = feature_count
        self.edge_embedding = nn.Parameter(torch.empty(feature_count, hidden_dim))
        nn.init.normal_(self.edge_embedding, std=hidden_dim**-0.5)
        self.network = nn.Sequential(
            nn.Linear(hidden_dim + 2, hidden_dim), nn.GELU(),
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, contacts: Tensor, present: Tensor) -> Tensor:
        if (
            contacts.layout != torch.sparse_coo or contacts.ndim != 2
            or contacts.shape[1] != self.feature_count or present.shape != (contacts.shape[0],)
            or present.dtype != torch.bool or not torch.is_floating_point(contacts)
            or contacts.device != present.device
        ):
            raise ValueError("Contact data must be sparse COO with an explicit assay mask")
        raw_values = contacts._values()
        if not torch.isfinite(raw_values).all() or torch.any(raw_values <= 0):
            raise ValueError("Only positive observed contacts may be stored")
        contacts = contacts.coalesce()
        indices, values = contacts.indices(), contacts.values()
        if indices.numel() and not torch.all(present[indices[0]]):
            raise ValueError("A missing contact assay cannot contain observed contacts")
        transformed = torch.log1p(values)
        sparse = torch.sparse_coo_tensor(
            indices, transformed, contacts.shape, device=contacts.device, dtype=contacts.dtype,
        ).coalesce()
        total = torch.zeros(contacts.shape[0], device=contacts.device, dtype=contacts.dtype)
        total = total.index_add(0, indices[0], transformed)
        pooled = torch.sparse.mm(sparse, self.edge_embedding) / total.clamp_min(1.0)[:, None]
        depth = torch.zeros_like(total).index_add(0, indices[0], values)
        features = torch.cat((pooled, torch.log1p(depth)[:, None], present.to(contacts.dtype)[:, None]), dim=1)
        return self.network(features)


class TriModalJointEncoder(nn.Module):
    """Encode all three observed modalities or a documented RNA-only pretraining input."""

    def __init__(
        self, rna_features: int, atac_features: int, contact_features: int,
        *, hidden_dim: int = 128, latent_dim: int = 64,
        rna_encoder: SharedFoundationEncoder | None = None,
        allow_untrained_rna: bool = False,
        allow_exploratory_foundation: bool = False,
    ) -> None:
        super().__init__()
        if rna_encoder is None:
            if not allow_untrained_rna:
                raise ValueError("A completed, verified RNA foundation encoder is required")
            self.rna_encoder = MaskedExpressionEncoder(rna_features, hidden_dim, latent_dim)
        else:
            if (
                not isinstance(rna_encoder, SharedFoundationEncoder)
                or rna_encoder.feature_count != rna_features or rna_encoder.latent_dim != latent_dim
            ):
                raise ValueError("The RNA foundation dimensions do not match the joint model")
            if rna_encoder.checkpoint_scope != "full" and not (
                allow_exploratory_foundation and rna_encoder.checkpoint_scope == "exploratory"
            ):
                raise ValueError("A full foundation checkpoint is required outside explicit exploration")
            self.rna_encoder = rna_encoder
        self.atac_encoder = MaskedExpressionEncoder(atac_features, hidden_dim, latent_dim)
        self.contact_encoder = ObservedContactEncoder(contact_features, hidden_dim, latent_dim)
        self.shared_heads = nn.ModuleList(nn.Linear(latent_dim, latent_dim) for _ in range(3))
        self.specific_heads = nn.ModuleList(nn.Linear(latent_dim, latent_dim) for _ in range(3))
        self.gates = nn.ModuleList(nn.Linear(latent_dim, 1) for _ in range(3))
        self.rna_decoder = nn.Linear(latent_dim, rna_features)

    def forward(
        self, rna: Tensor, rna_measured: Tensor, atac: Tensor, atac_measured: Tensor,
        contacts: Tensor, contact_present: Tensor,
    ) -> dict[str, Tensor]:
        if (
            rna.ndim != 2 or atac.ndim != 2 or rna.shape[0] != atac.shape[0]
            or contacts.shape[0] != rna.shape[0] or contact_present.shape != (rna.shape[0],)
            or rna.device != atac.device or rna.device != contacts.device
            or rna.device != rna_measured.device or atac.device != atac_measured.device
            or rna.dtype != atac.dtype or rna.dtype != contacts.dtype
        ):
            raise ValueError("All modalities must have a common batch, device and floating-point dtype")
        rna_z, rna_present = self.rna_encoder(rna, rna_measured)
        atac_z, atac_present = self.atac_encoder(atac, atac_measured)
        contact_z = self.contact_encoder(contacts, contact_present)
        present = torch.stack((rna_present, atac_present, contact_present), dim=1)
        if not torch.all(present.any(dim=1)):
            raise ValueError("Every cell must have at least one measured modality")
        raw = (rna_z, atac_z, contact_z)
        shared = torch.stack([head(value) for head, value in zip(self.shared_heads, raw, strict=True)], dim=1)
        specific = torch.stack([head(value) for head, value in zip(self.specific_heads, raw, strict=True)], dim=1)
        logits = torch.cat([gate(value) for gate, value in zip(self.gates, raw, strict=True)], dim=1)
        weights = torch.softmax(logits.masked_fill(~present, -torch.inf), dim=1)
        fused = (shared * weights[:, :, None]).sum(dim=1)
        return {
            "fused": fused, "shared": shared, "specific": specific,
            "modality_weights": weights, "modality_present": present,
            "rna_reconstruction": self.rna_decoder(fused),
        }


def masked_rna_loss(prediction: Tensor, target: Tensor, measured: Tensor, heldout: Tensor) -> Tensor:
    if (
        prediction.ndim != 2 or target.shape != prediction.shape or measured.shape != prediction.shape
        or heldout.shape != prediction.shape or measured.dtype != torch.bool or heldout.dtype != torch.bool
        or not torch.isfinite(prediction).all() or not torch.isfinite(target).all()
        or torch.any(heldout & ~measured) or not torch.any(heldout)
    ):
        raise ValueError("RNA reconstruction targets must be measured and explicitly held out")
    return F.mse_loss(prediction[heldout], target[heldout])


def same_sample_contrastive_loss(
    left: Tensor, right: Tensor, cells: Sequence[PairedCell], *, temperature: float = 0.1,
) -> Tensor:
    if (
        left.ndim != 2 or right.shape != left.shape or len(cells) != left.shape[0]
        or not torch.isfinite(left).all() or not torch.isfinite(right).all()
        or not 0 < temperature <= 1
    ):
        raise ValueError("Invalid paired representations or contrastive temperature")
    if not cells:
        raise ValueError("Contrastive training requires paired cells")
    first = cells[0].rna.key
    validate_paired_cells(cells, species=first.species, assembly=first.assembly, minimum_training_lines=0)
    if any(cell.split != "train" for cell in cells):
        raise ValueError("Contrastive optimization may only use training-role cells")
    groups = [(c.rna.key.dataset, c.rna.key.experiment, c.rna.key.sample, c.rna.key.cell_line) for c in cells]
    same = torch.tensor([[a == b for b in groups] for a in groups], dtype=torch.bool, device=left.device)
    usable = same.sum(dim=1) > 1
    if not torch.any(usable):
        raise ValueError("A contrastive group needs at least two distinct same-sample cells")
    logits = F.normalize(left, dim=1) @ F.normalize(right, dim=1).T / temperature
    logits = logits.masked_fill(~same, -torch.inf)
    labels = torch.arange(len(cells), device=left.device)
    return (F.cross_entropy(logits[usable], labels[usable])
            + F.cross_entropy(logits.T[usable], labels[usable])) / 2
