"""Use one verified RNA foundation encoder on explicitly aligned gene axes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from c3vcell.model import C3VCell, ModelConfig


FORMAT = "epi-c3-rna-foundation-v1"


def gene_axis_digest(genes: Sequence[str]) -> str:
    axis = list(genes)
    if not axis or any(not isinstance(gene, str) or not gene.strip() for gene in axis):
        raise ValueError("A gene axis must contain nonempty string identifiers")
    if len(set(axis)) != len(axis):
        raise ValueError("A gene axis cannot contain duplicate identifiers")
    data = json.dumps(axis, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class SharedFoundationEncoder(nn.Module):
    """Project a measured local gene panel into a single foundation gene axis.

    Genes not present in the local panel are unmeasured, not observed zeros.
    """

    def __init__(
        self,
        foundation: C3VCell,
        foundation_genes: Sequence[str],
        local_genes: Sequence[str],
        checkpoint_sha256: str,
        *,
        freeze: bool = True,
    ) -> None:
        super().__init__()
        foundation_axis = list(foundation_genes)
        local_axis = list(local_genes)
        self.foundation_axis_sha256 = gene_axis_digest(foundation_axis)
        self.local_axis_sha256 = gene_axis_digest(local_axis)
        if len(foundation_axis) != foundation.config.n_genes:
            raise ValueError("Foundation model dimensions do not match its saved gene axis")
        if len(checkpoint_sha256) != 64 or any(c not in "0123456789abcdef" for c in checkpoint_sha256):
            raise ValueError("A verified foundation checkpoint SHA-256 is required")
        positions = {gene: index for index, gene in enumerate(foundation_axis)}
        missing = [gene for gene in local_axis if gene not in positions]
        if missing:
            raise ValueError(f"{len(missing)} local genes are absent from the foundation axis: {missing[:5]}")
        self.register_buffer(
            "gene_positions", torch.tensor([positions[gene] for gene in local_axis], dtype=torch.long),
        )
        self.encoder = foundation.encoder
        self.latent_dim = foundation.config.latent_dim
        self.feature_count = len(local_axis)
        self.checkpoint_sha256 = checkpoint_sha256
        self.checkpoint_scope = "unverified"
        self.encoder.requires_grad_(not freeze)

    def gene_embeddings(self) -> Tensor:
        return self.encoder.value.weight.index_select(1, self.gene_positions).transpose(0, 1)

    def forward(self, counts: Tensor, measured: Tensor) -> tuple[Tensor, Tensor]:
        if (
            counts.ndim != 2 or counts.shape[1] != self.feature_count
            or measured.shape != counts.shape or measured.dtype != torch.bool
            or not torch.is_floating_point(counts) or counts.device != measured.device
            or counts.device != self.gene_positions.device
            or counts.dtype != self.encoder.value.weight.dtype
            or not torch.isfinite(counts).all() or torch.any(counts < 0)
            or torch.any(counts.masked_select(~measured) != 0)
        ):
            raise ValueError("Counts and measured-gene masks are incompatible with the foundation axis")
        present = measured.any(dim=1)
        rows = torch.nonzero(present, as_tuple=False).squeeze(1)
        output = counts.new_zeros((counts.shape[0], self.latent_dim))
        if not len(rows):
            return output, present
        observed_counts = counts.index_select(0, rows)
        visible = measured.index_select(0, rows).to(counts.dtype)
        library = observed_counts.sum(dim=1, keepdim=True)
        if torch.any(library <= 0):
            raise ValueError("Every present RNA assay must contain at least one measured count")
        normalized = torch.log1p(observed_counts / library * 10_000.0)
        value_weights = self.encoder.value.weight.index_select(1, self.gene_positions)
        visible_weights = self.encoder.visible.weight.index_select(1, self.gene_positions)
        encoded = (
            F.linear(normalized * visible, value_weights, self.encoder.value.bias)
            + F.linear(visible, visible_weights)
        )
        return output.index_copy(0, rows, self.encoder.body(encoded)), present


def load_foundation(
    path: Path,
    expected_sha256: str,
    local_genes: Sequence[str],
    *,
    freeze: bool = True,
    allow_exploratory: bool = False,
) -> SharedFoundationEncoder:
    if file_digest(path) != expected_sha256:
        raise ValueError("Foundation checkpoint does not match the recorded SHA-256")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or checkpoint.get("format") != FORMAT:
        raise ValueError("The checkpoint is not an Epi-C3 RNA foundation checkpoint")
    if checkpoint.get("status") != "complete":
        raise ValueError("The foundation pretraining run did not complete")
    if checkpoint.get("scope") != "full" and not allow_exploratory:
        raise ValueError("An exploratory foundation checkpoint cannot initialize a final model")
    genes = checkpoint.get("genes")
    if not isinstance(genes, list) or gene_axis_digest(genes) != checkpoint.get("gene_axis_sha256"):
        raise ValueError("The foundation checkpoint gene axis is missing or corrupted")
    config = checkpoint.get("model_config")
    if not isinstance(config, dict):
        raise ValueError("The foundation checkpoint has no model configuration")
    model = C3VCell(ModelConfig(**config))
    model.load_state_dict(checkpoint["model_state"], strict=True)
    if any(not torch.isfinite(value).all() for value in model.state_dict().values()):
        raise ValueError("The foundation checkpoint contains nonfinite parameters")
    encoder = SharedFoundationEncoder(model, genes, local_genes, expected_sha256, freeze=freeze)
    encoder.checkpoint_scope = checkpoint["scope"]
    return encoder
