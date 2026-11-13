"""Exploratory gene-response head for strictly paired human tri-modal controls.

Identity consistency does not establish biological pairing; the source pairing
map and the ordered batch-to-record mapping must be verified independently.
The joint encoder and response head are untrained research prototypes.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from epi_c3_shared_foundation_v1 import SharedFoundationEncoder, gene_axis_digest
from epi_c3_trimodal_joint_v1 import PairedCell, TriModalJointEncoder, validate_paired_cells


class TriModalPerturbationResponse(nn.Module):
    """Predict log1p(CPM) shifts for inhibitory (-1) or activating (+1) interventions."""

    def __init__(
        self,
        joint: TriModalJointEncoder,
        genes: Sequence[str],
        *,
        species: str,
        assembly: str,
        hidden_dim: int = 32,
        allow_untrained_joint: bool = False,
    ) -> None:
        super().__init__()
        if not allow_untrained_joint:
            raise ValueError("The tri-modal joint encoder is an untrained prototype")
        if species != "human" or not isinstance(assembly, str) or not assembly.strip():
            raise ValueError("This prototype requires the shared human axis and an explicit assembly")
        if not isinstance(joint, TriModalJointEncoder):
            raise ValueError("A tri-modal joint encoder is required")
        foundation = joint.rna_encoder
        if not isinstance(foundation, SharedFoundationEncoder):
            raise ValueError("The response head requires the shared RNA foundation encoder")
        if any(parameter.requires_grad for parameter in foundation.encoder.parameters()):
            raise ValueError("The foundation encoder must be frozen for the response head")
        if gene_axis_digest(genes) != foundation.local_axis_sha256:
            raise ValueError("The response gene axis does not match the foundation adapter")
        if len(genes) < 2 or hidden_dim < 1:
            raise ValueError("The response head requires at least two genes and a positive hidden size")

        self.joint = joint
        self.species = species
        self.assembly = assembly
        self.n_genes = len(genes)
        self.hidden_dim = hidden_dim
        embeddings = F.normalize(foundation.gene_embeddings().detach().clone(), dim=1)
        self.register_buffer("gene_embeddings", embeddings, persistent=False)
        self.response_query = nn.Linear(foundation.latent_dim * 2 + 1, hidden_dim)
        self.response_key = nn.Linear(foundation.latent_dim, hidden_dim, bias=False)
        self.gene_bias = nn.Parameter(torch.zeros(self.n_genes))
        nn.init.zeros_(self.response_query.weight)
        nn.init.zeros_(self.response_query.bias)

    def forward(
        self,
        rna: Tensor,
        rna_measured: Tensor,
        atac: Tensor,
        atac_measured: Tensor,
        contacts: Tensor,
        contact_present: Tensor,
        cells: Sequence[PairedCell],
        target_indices: Tensor,
        intervention_direction: Tensor,
        *,
        strength: float = 1.0,
    ) -> dict[str, Tensor]:
        if (
            rna.ndim != 2 or not rna.shape[0] or len(cells) != rna.shape[0]
            or target_indices.shape != (rna.shape[0],) or target_indices.dtype != torch.long
            or target_indices.device != rna.device
            or torch.any(target_indices < 0) or torch.any(target_indices >= self.n_genes)
            or intervention_direction.shape != (rna.shape[0],)
            or intervention_direction.device != rna.device or intervention_direction.dtype != rna.dtype
            or not torch.all((intervention_direction == -1) | (intervention_direction == 1))
            or not math.isfinite(strength) or not 0 <= strength <= 1
        ):
            raise ValueError("Invalid paired batch, gene target, intervention direction or strength")
        validate_paired_cells(cells, species=self.species, assembly=self.assembly, minimum_training_lines=0)
        output = self.joint(rna, rna_measured, atac, atac_measured, contacts, contact_present)
        if not torch.all(output["modality_present"]):
            raise ValueError("Each response input must contain all three measured modalities")
        contact_rows = contacts.coalesce().indices()[0]
        if torch.any(torch.bincount(contact_rows, minlength=rna.shape[0]) == 0):
            raise ValueError("Each response input needs at least one observed contact")
        library = rna.sum(dim=1, keepdim=True)
        baseline = torch.log1p(rna / library * 10_000.0)
        target = self.gene_embeddings.index_select(0, target_indices)
        query = self.response_query(
            torch.cat((output["fused"], target, intervention_direction[:, None]), dim=1)
        )
        shift = self.gene_bias + query @ self.response_key(self.gene_embeddings).T / math.sqrt(self.hidden_dim)
        return {
            "log_expression": torch.clamp_min(baseline + strength * shift, 0.0),
            "measured": rna_measured,
            "modality_weights": output["modality_weights"],
        }


def population_target_excluded_loss(
    predicted: Tensor,
    observed: Tensor,
    *,
    control_measured: Tensor,
    observed_measured: Tensor,
    target_index: int,
) -> Tensor:
    """Compare log-expression group means for one verified condition, excluding its target.

    Both populations must use the same measured panel and normalization. The caller
    must independently verify normalization, condition, cell-line and split membership.
    Control and treated cells are not assumed to be the same biological cells.
    """
    if (
        predicted.ndim != 2 or observed.ndim != 2 or not predicted.shape[0] or not observed.shape[0]
        or predicted.shape[1] < 2 or observed.shape[1] != predicted.shape[1]
        or not torch.is_floating_point(predicted) or observed.dtype != predicted.dtype
        or observed.device != predicted.device
        or control_measured.shape != predicted.shape or observed_measured.shape != observed.shape
        or control_measured.dtype != torch.bool or observed_measured.dtype != torch.bool
        or control_measured.device != predicted.device or observed_measured.device != predicted.device
        or not torch.isfinite(predicted.masked_select(control_measured)).all()
        or not torch.isfinite(observed.masked_select(observed_measured)).all()
        or torch.any(predicted.masked_select(control_measured) < 0)
        or torch.any(observed.masked_select(observed_measured) < 0)
        or isinstance(target_index, bool) or not isinstance(target_index, int)
        or not 0 <= target_index < predicted.shape[1]
    ):
        raise ValueError("Invalid population expression, measured masks or target")
    control_non_target = control_measured.clone()
    observed_non_target = observed_measured.clone()
    control_non_target[:, target_index] = False
    observed_non_target[:, target_index] = False
    if not torch.all(control_non_target.any(dim=1)) or not torch.all(observed_non_target.any(dim=1)):
        raise ValueError("Each population cell needs a measured non-target gene")
    panel = control_measured[0]
    if not torch.all(control_measured == panel) or not torch.all(observed_measured == panel):
        raise ValueError("Population comparison requires the same measured gene panel")
    eligible = panel.clone()
    eligible[target_index] = False
    return F.mse_loss(predicted[:, eligible].mean(dim=0), observed[:, eligible].mean(dim=0))
