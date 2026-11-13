from __future__ import annotations

import pytest
import torch

from c3vcell.model import C3VCell, ModelConfig
from epi_c3_foundation_grn_v1 import FoundationGRNResponse
from epi_c3_shared_foundation_v1 import FORMAT, file_digest, gene_axis_digest, load_foundation


def foundation(tmp_path):
    model = C3VCell(ModelConfig(4, latent_dim=3, hidden_dim=5))
    genes = ["a", "b", "c", "d"]
    path = tmp_path / "foundation.pt"
    torch.save({
        "format": FORMAT, "status": "complete", "scope": "exploratory",
        "genes": genes, "gene_axis_sha256": gene_axis_digest(genes),
        "model_config": model.configuration(), "model_state": model.state_dict(),
    }, path)
    return load_foundation(path, file_digest(path), ["c", "a"], allow_exploratory=True)


def graph():
    return torch.zeros(2, 1), torch.ones(2, 1), torch.zeros(1, 2, 2, dtype=torch.bool)


def test_response_requires_same_verified_foundation_axis(tmp_path):
    encoder = foundation(tmp_path)
    values, masks, relations = graph()
    with pytest.raises(ValueError, match="full foundation"):
        FoundationGRNResponse(encoder, ["c", "a"], values, masks, relations, use_nodes=True, use_graph=True)
    with pytest.raises(ValueError, match="gene axes"):
        FoundationGRNResponse(
            encoder, ["a", "c"], values, masks, relations, use_nodes=True, use_graph=True,
            allow_exploratory_foundation=True,
        )
    result = FoundationGRNResponse(
        encoder, ["c", "a"], values, masks, relations, use_nodes=True, use_graph=True,
        allow_exploratory_foundation=True,
    )
    assert result.foundation is encoder


def test_response_is_gene_level_and_leaves_baseline_at_zero_strength(tmp_path):
    encoder = foundation(tmp_path)
    values, masks, relations = graph()
    model = FoundationGRNResponse(
        encoder, ["c", "a"], values, masks, relations, use_nodes=True, use_graph=True,
        allow_exploratory_foundation=True,
    )
    counts = torch.tensor([[1.0, 3.0], [2.0, 2.0]])
    measured = torch.ones_like(counts, dtype=torch.bool)
    baseline = torch.log1p(counts / counts.sum(1, keepdim=True) * 10000)
    predicted = model(counts, measured, torch.tensor([0, 1]), strength=0.0)
    assert torch.allclose(predicted, baseline)
    with pytest.raises(ValueError, match="RNA"):
        model(torch.tensor([[0.0, 0.0]]), torch.zeros((1, 2), dtype=torch.bool), torch.tensor([0]))


def test_response_refuses_stale_gene_features_if_foundation_is_trainable(tmp_path):
    encoder = foundation(tmp_path)
    encoder.encoder.requires_grad_(True)
    values, masks, relations = graph()
    with pytest.raises(ValueError, match="frozen"):
        FoundationGRNResponse(
            encoder, ["c", "a"], values, masks, relations, use_nodes=True, use_graph=True,
            allow_exploratory_foundation=True,
        )
