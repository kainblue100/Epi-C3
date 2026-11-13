from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp
import torch

from c3vcell.model import C3VCell, ModelConfig
from epi_c3_foundation_training_v1 import InterleavedFoundationTrainer


class FakeSource:
    def __init__(self, rows, batch_size):
        self.rows = np.asarray(rows, dtype=np.float32)
        self.batch_size = batch_size
        self.expected_cells = len(self.rows)
        self.measured = np.ones(self.rows.shape[1], dtype=bool)
        self.position = 0

    def next_batch(self):
        if self.position == len(self.rows):
            return None
        end = min(self.position + self.batch_size, len(self.rows))
        result = sp.csr_matrix(self.rows[self.position:end])
        self.position = end
        return result

    def cursor(self):
        return {"position": self.position}

    def set_cursor(self, cursor):
        self.position = cursor["position"]


def build():
    torch.manual_seed(71)
    model = C3VCell(ModelConfig(3, latent_dim=4, hidden_dim=6, covariance_weight=0.1))
    sources = {
        "a": FakeSource([[2, 1, 0], [1, 2, 0], [1, 0, 2], [1, 1, 1], [0, 2, 1]], 2),
        "b": FakeSource([[0, 1, 2], [2, 0, 1], [2, 1, 0], [1, 2, 0], [1, 0, 2], [1, 1, 1]], 2),
    }
    return InterleavedFoundationTrainer(model, sources, learning_rate=0.001, seed=3)


def test_resume_matches_uninterrupted_training_exactly(tmp_path):
    full = build()
    full_order = []
    while (result := full.step()) is not None:
        full_order.append(result[0])
    full.verify_exhausted()
    expected = {key: value.clone() for key, value in full.model.state_dict().items()}

    stopped = build()
    prefix = [stopped.step()[0] for _ in range(3)]
    path = tmp_path / "recovery.pt"
    torch.save(stopped.snapshot(), path)
    resumed = build()
    resumed.restore(torch.load(path, weights_only=True))
    suffix = []
    while (result := resumed.step()) is not None:
        suffix.append(result[0])
    resumed.verify_exhausted()
    assert prefix + suffix == full_order
    assert resumed.trained == full.trained
    assert resumed.singleton_updates == 1
    for key, value in resumed.model.state_dict().items():
        assert torch.equal(value, expected[key]), key


def test_source_count_mismatch_fails_loudly():
    trainer = build()
    trainer.sources["a"].expected_cells += 1
    with pytest.raises(ValueError, match="ended before"):
        while trainer.step() is not None:
            pass


def test_recovery_rejects_changed_sources(tmp_path):
    trainer = build()
    trainer.step()
    path = tmp_path / "recovery.pt"
    torch.save(trainer.snapshot(), path)
    other = build()
    other.sources.pop("a")
    with pytest.raises(ValueError, match="sources"):
        other.restore(torch.load(path, weights_only=True))
