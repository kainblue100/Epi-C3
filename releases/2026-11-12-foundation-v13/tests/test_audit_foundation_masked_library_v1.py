from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch

from c3vcell.model import GeneEncoder, ModelConfig


path = Path(__file__).parent.parent / "audit-foundation-masked-library-v1.py"
spec = importlib.util.spec_from_file_location("masked_library_audit", path)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def test_visible_only_encoding_ignores_hidden_values():
    torch.manual_seed(7)
    encoder = GeneEncoder(ModelConfig(n_genes=4, hidden_dim=8, latent_dim=4))
    original = torch.tensor([[2.0, 3.0, 4.0, 5.0]])
    altered = torch.tensor([[2.0, 300.0, 4.0, 500.0]])
    visible = torch.tensor([[True, False, True, False]])
    strict = audit.visible_only_encoding(encoder, original, visible)
    changed = audit.visible_only_encoding(encoder, altered, visible)
    assert torch.equal(strict, changed)
    assert not torch.allclose(encoder(original, visible.float()), encoder(altered, visible.float()))


def test_zero_visible_library_is_rejected():
    encoder = GeneEncoder(ModelConfig(n_genes=2, hidden_dim=4, latent_dim=2))
    with pytest.raises(ValueError, match="positive visible library"):
        audit.visible_only_encoding(encoder, torch.tensor([[0.0, 10.0]]), torch.tensor([[True, False]]))


def test_inconsistent_mask_is_rejected():
    encoder = GeneEncoder(ModelConfig(n_genes=2, hidden_dim=4, latent_dim=2))
    with pytest.raises(ValueError, match="visibility"):
        audit.visible_only_encoding(encoder, torch.ones((1, 2)), torch.ones((1, 2)))
