"""A graph-response head initialized from the shared RNA foundation."""

from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor, nn

from epi_c3_gene_grn_v1 import GeneLevelResponse
from epi_c3_shared_foundation_v1 import SharedFoundationEncoder, gene_axis_digest


class FoundationGRNResponse(nn.Module):
    def __init__(
        self,
        foundation: SharedFoundationEncoder,
        graph_genes: Sequence[str],
        node_values: Tensor,
        node_masks: Tensor,
        relation_adjacency: Tensor,
        *,
        use_nodes: bool,
        use_graph: bool,
        hidden_dim: int = 32,
        allow_exploratory_foundation: bool = False,
    ) -> None:
        super().__init__()
        if not isinstance(foundation, SharedFoundationEncoder):
            raise TypeError("A verified shared RNA foundation encoder is required")
        if foundation.checkpoint_scope != "full" and not (
            allow_exploratory_foundation and foundation.checkpoint_scope == "exploratory"
        ):
            raise ValueError("A full foundation checkpoint is required outside explicit exploration")
        if gene_axis_digest(graph_genes) != foundation.local_axis_sha256:
            raise ValueError("The graph and foundation input gene axes do not match")
        if any(parameter.requires_grad for parameter in foundation.encoder.parameters()):
            raise ValueError("This response head requires a frozen foundation encoder")
        self.foundation = foundation
        self.response = GeneLevelResponse(
            foundation.gene_embeddings(), node_values, node_masks, relation_adjacency,
            use_nodes=use_nodes, use_graph=use_graph, hidden_dim=hidden_dim,
        )

    def forward(
        self,
        control_counts: Tensor,
        measured: Tensor,
        target_indices: Tensor,
        *,
        baseline_log: Tensor | None = None,
        strength: float = 1.0,
    ) -> Tensor:
        """Return provisional predictions; only measured control genes have an observed baseline."""
        latent, present = self.foundation(control_counts, measured)
        if not torch.all(present):
            raise ValueError("Graph perturbation prediction requires RNA for every input cell")
        if baseline_log is None:
            library = control_counts.sum(dim=1, keepdim=True)
            baseline_log = torch.log1p(control_counts / library * 10_000.0)
        return self.response(baseline_log, latent, target_indices, strength=strength)
