"""Gene-level perturbation response with unsigned, evidence-specific graph messages."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class GeneLevelResponse(nn.Module):
    """Predict per-gene log-expression shifts from a frozen RNA representation."""

    def __init__(
        self, gene_embeddings: Tensor, node_values: Tensor, node_masks: Tensor,
        relation_adjacency: Tensor, *, use_nodes: bool, use_graph: bool, hidden_dim: int = 32,
    ) -> None:
        super().__init__()
        if (
            gene_embeddings.ndim != 2 or gene_embeddings.shape[0] < 2 or gene_embeddings.shape[1] < 1
            or node_values.ndim != 2 or node_values.shape[0] != gene_embeddings.shape[0]
            or node_values.shape[1] < 1 or node_masks.shape != node_values.shape
            or relation_adjacency.ndim != 3
            or relation_adjacency.shape[0] < 1
            or tuple(relation_adjacency.shape[1:]) != (gene_embeddings.shape[0],) * 2
            or hidden_dim < 1 or (use_graph and not use_nodes)
        ):
            raise ValueError("Invalid gene-level response dimensions or ablation")
        if (
            not torch.is_floating_point(gene_embeddings) or not torch.is_floating_point(node_values)
            or not torch.is_floating_point(node_masks) or relation_adjacency.dtype != torch.bool
            or not torch.isfinite(gene_embeddings).all() or not torch.isfinite(node_values).all()
            or not torch.isfinite(node_masks).all()
            or torch.any((node_values < 0) | (node_values > 1))
            or torch.any((node_masks != 0) & (node_masks != 1))
            or torch.any(node_values[node_masks == 0] != 0)
            or torch.any(torch.diagonal(relation_adjacency, dim1=1, dim2=2))
        ):
            raise ValueError("Invalid graph, node features, or missingness masks")

        self.n_genes, self.embedding_dim = gene_embeddings.shape
        self.use_nodes = use_nodes
        self.use_graph = use_graph
        self.hidden_dim = hidden_dim
        self.register_buffer("gene_embeddings", F.normalize(gene_embeddings.detach().clone(), dim=1), persistent=False)
        self.register_buffer("node_values", node_values.detach().clone(), persistent=False)
        self.register_buffer("node_masks", node_masks.detach().clone(), persistent=False)
        self.register_buffer("relations", relation_adjacency.detach().clone(), persistent=False)

        query_dim = self.embedding_dim * 2
        node_dim = node_values.shape[1] * 2
        self.gene_bias = nn.Parameter(torch.zeros(self.n_genes))
        self.base_query = nn.Linear(query_dim, hidden_dim)
        self.base_key = nn.Linear(self.embedding_dim, hidden_dim)
        self.node_query = nn.Linear(query_dim, hidden_dim)
        self.node_key = nn.Linear(node_dim, hidden_dim)
        self.graph_query = nn.Linear(query_dim, hidden_dim)
        self.graph_key = nn.Linear(self.embedding_dim + node_dim, hidden_dim)
        self.relation_coefficients = nn.Parameter(torch.ones(relation_adjacency.shape[0]))
        for layer in (self.base_query, self.node_query, self.graph_query):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def components(self, latent: Tensor, target_indices: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        if (
            latent.ndim != 2 or latent.shape[1] != self.embedding_dim or not latent.shape[0]
            or target_indices.shape != (latent.shape[0],) or target_indices.dtype != torch.long
            or not torch.isfinite(latent).all()
            or torch.any(target_indices < 0) or torch.any(target_indices >= self.n_genes)
        ):
            raise ValueError("Invalid cell representation or perturbation target indices")
        queries = torch.cat((self.gene_embeddings[target_indices], latent), dim=1)
        scale = math.sqrt(self.hidden_dim)
        base = self.gene_bias + (self.base_query(queries) @ self.base_key(self.gene_embeddings).T) / scale
        nodes = torch.zeros_like(base)
        graph = torch.zeros_like(base)
        if self.use_nodes:
            features = torch.cat((self.node_values, self.node_masks), dim=1)
            nodes = (self.node_query(queries) @ self.node_key(features).T) / scale
            if self.use_graph:
                gene_features = torch.cat((self.gene_embeddings, features), dim=1)
                interaction = (self.graph_query(queries) @ self.graph_key(gene_features).T) / scale
                edges = self.relations[:, target_indices, :].to(interaction.dtype)
                support = (edges * self.relation_coefficients[:, None, None]).sum(dim=0)
                graph = interaction * support
        return base, nodes, graph

    def forward(
        self, log_control: Tensor, latent: Tensor, target_indices: Tensor, *, strength: float = 1.0,
    ) -> Tensor:
        if (
            log_control.ndim != 2 or log_control.shape != (latent.shape[0], self.n_genes)
            or not torch.isfinite(log_control).all() or torch.any(log_control < 0)
            or not math.isfinite(strength) or not 0 <= strength <= 1
        ):
            raise ValueError("Invalid control expression or perturbation strength")
        base, nodes, graph = self.components(latent, target_indices)
        return torch.clamp_min(log_control + strength * (base + nodes + graph), 0.0)


def target_excluded_loss(
    predicted: Tensor, observed: Tensor, target_indices: Tensor, *,
    control_measured: Tensor, observed_measured: Tensor,
) -> Tensor:
    """Score only non-target genes measured in both control and treated cells."""
    if (
        predicted.ndim != 2 or not predicted.shape[0] or observed.shape != predicted.shape
        or predicted.shape[1] < 2 or not torch.is_floating_point(predicted)
        or observed.dtype != predicted.dtype or observed.device != predicted.device
        or control_measured.shape != predicted.shape or observed_measured.shape != predicted.shape
        or control_measured.dtype != torch.bool or observed_measured.dtype != torch.bool
        or control_measured.device != predicted.device or observed_measured.device != predicted.device
        or target_indices.shape != (predicted.shape[0],) or target_indices.dtype != torch.long
        or target_indices.device != predicted.device or not torch.isfinite(predicted).all()
        or not torch.isfinite(observed.masked_select(observed_measured)).all()
        or torch.any(target_indices < 0) or torch.any(target_indices >= predicted.shape[1])
    ):
        raise ValueError("Invalid target-excluded supervised loss inputs")
    eligible = (control_measured & observed_measured).scatter(1, target_indices[:, None], False)
    if not torch.all(eligible.any(dim=1)):
        raise ValueError("Every cell needs a non-target gene measured in both conditions")
    return F.mse_loss(predicted.masked_select(eligible), observed.masked_select(eligible))
