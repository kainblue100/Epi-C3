from __future__ import annotations

import pytest
import torch

from c3vcell.model import C3VCell, ModelConfig
from epi_c3_shared_foundation_v1 import (
    FORMAT,
    SharedFoundationEncoder,
    file_digest,
    gene_axis_digest,
    load_foundation,
)


def model():
    torch.manual_seed(4)
    return C3VCell(ModelConfig(n_genes=4, latent_dim=3, hidden_dim=5))


def test_local_axis_projection_equals_full_axis_with_unmeasured_genes():
    base = model()
    adapter = SharedFoundationEncoder(base, ["a", "b", "c", "d"], ["c", "a"], "a" * 64)
    counts = torch.tensor([[2.0, 1.0], [3.0, 0.0]])
    measured = torch.tensor([[True, True], [True, False]])
    full_counts = torch.tensor([[1.0, 0.0, 2.0, 0.0], [0.0, 0.0, 3.0, 0.0]])
    full_measured = torch.tensor([[True, False, True, False], [False, False, True, False]])
    expected = base.encode(full_counts, full_measured.float())
    actual, present = adapter(counts, measured)
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)
    assert present.tolist() == [True, True]
    assert torch.equal(adapter.gene_embeddings(), base.encoder.value.weight[:, [2, 0]].T)


def test_missing_assay_is_not_encoded_as_observed_zero():
    adapter = SharedFoundationEncoder(model(), ["a", "b", "c", "d"], ["c", "a"], "b" * 64)
    counts = torch.tensor([[2.0, 0.0], [0.0, 0.0]])
    measured = torch.tensor([[True, False], [False, False]])
    result, present = adapter(counts, measured)
    assert present.tolist() == [True, False]
    assert torch.equal(result[1], torch.zeros(3))
    with pytest.raises(ValueError):
        adapter(counts, torch.tensor([[False, False], [False, False]]))
    with pytest.raises(ValueError):
        adapter(torch.zeros_like(counts), torch.ones_like(measured))


def test_axis_identity_and_digest_are_required():
    base = model()
    with pytest.raises(ValueError):
        SharedFoundationEncoder(base, ["a", "b", "c", "d"], ["a", "a"], "a" * 64)
    with pytest.raises(ValueError):
        SharedFoundationEncoder(base, ["a", "b", "c", "d"], ["not-in-axis"], "a" * 64)
    with pytest.raises(ValueError):
        SharedFoundationEncoder(base, ["a", "b", "c", "d"], ["a"], "invalid")
    assert gene_axis_digest(["a", "b"]) != gene_axis_digest(["b", "a"])


def test_freeze_is_explicit_and_unfrozen_encoder_receives_gradients():
    frozen = SharedFoundationEncoder(model(), ["a", "b", "c", "d"], ["a", "b"], "a" * 64)
    assert not any(parameter.requires_grad for parameter in frozen.encoder.parameters())
    trainable = SharedFoundationEncoder(
        model(), ["a", "b", "c", "d"], ["a", "b"], "b" * 64, freeze=False,
    )
    output, _ = trainable(torch.tensor([[1.0, 2.0]]), torch.ones((1, 2), dtype=torch.bool))
    output.square().sum().backward()
    assert trainable.encoder.value.weight.grad is not None
    assert torch.isfinite(trainable.encoder.value.weight.grad).all()


def test_checkpoint_requires_hash_complete_status_and_full_scope(tmp_path):
    base = model()
    genes = ["a", "b", "c", "d"]
    path = tmp_path / "foundation.pt"
    payload = {
        "format": FORMAT,
        "status": "complete",
        "scope": "exploratory",
        "genes": genes,
        "gene_axis_sha256": gene_axis_digest(genes),
        "model_config": base.configuration(),
        "model_state": base.state_dict(),
    }
    torch.save(payload, path)
    with pytest.raises(ValueError, match="exploratory"):
        load_foundation(path, file_digest(path), ["a"])
    adapter = load_foundation(path, file_digest(path), ["a"], allow_exploratory=True)
    assert adapter.checkpoint_sha256 == file_digest(path)
    with pytest.raises(ValueError, match="SHA-256"):
        load_foundation(path, "0" * 64, ["a"], allow_exploratory=True)
    payload["status"] = "running"
    torch.save(payload, path)
    with pytest.raises(ValueError, match="did not complete"):
        load_foundation(path, file_digest(path), ["a"], allow_exploratory=True)
    payload["status"] = "complete"
    payload["scope"] = "full"
    payload["gene_axis_sha256"] = "1" * 64
    torch.save(payload, path)
    with pytest.raises(ValueError, match="gene axis"):
        load_foundation(path, file_digest(path), ["a"])
