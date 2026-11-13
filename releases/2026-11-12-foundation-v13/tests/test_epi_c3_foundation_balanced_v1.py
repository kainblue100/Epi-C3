from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp
import torch

from c3vcell.model import C3VCell, ModelConfig
from epi_c3_foundation_balanced_v1 import BalancedContinualTrainer


class FakeSource:
    def __init__(self, rows, batch_size, measured=None):
        self.rows = np.asarray(rows, dtype=np.float32)
        self.batch_size = batch_size
        self.expected_cells = len(self.rows)
        self.measured = np.asarray(measured if measured is not None else [True] * self.rows.shape[1])
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
        position = cursor["position"]
        if type(position) is not int or not 0 <= position <= len(self.rows):
            raise ValueError("Invalid cursor")
        self.position = position


def build():
    torch.manual_seed(71)
    model = C3VCell(ModelConfig(3, latent_dim=4, hidden_dim=6, covariance_weight=0.1))
    fresh = FakeSource([[2, 1, 0], [1, 0, 2]] * 11, 3)
    replay = {
        "a": FakeSource([[2, 1, 0], [1, 2, 0]] * 2, 3, [True, True, False]),
        "b": FakeSource([[0, 1, 2], [0, 2, 1]] * 3, 2, [False, True, True]),
    }
    return BalancedContinualTrainer(
        model, fresh, replay, learning_rate=0.001,
        source_manifest_sha256={"fresh": "a" * 64, "a": "b" * 64, "b": "c" * 64},
        gene_axis_sha256="d" * 64, initial_checkpoint_sha256="e" * 64,
    )


def run(trainer):
    order = []
    while (result := trainer.step()) is not None:
        order.append(result[0])
        assert abs(trainer.fresh_rows - sum(trainer.replay_rows.values())) <= 3
    trainer.verify_exhausted()
    return order


def test_balanced_replay_and_exact_checkpoint_recovery(tmp_path):
    full = build()
    expected_order = run(full)
    expected = {key: value.clone() for key, value in full.model.state_dict().items()}
    assert full.fresh_rows == 22
    assert 22 <= sum(full.replay_rows.values()) <= 24
    assert full.replay_epochs["a"] >= 1 and full.replay_epochs["b"] >= 1
    assert abs(full.replay_rows["a"] / 4 - full.replay_rows["b"] / 6) <= 0.75

    stopped = build()
    prefix = [stopped.step()[0] for _ in range(7)]
    path = tmp_path / "recovery.pt"
    torch.save(stopped.snapshot(), path)
    resumed = build()
    resumed.restore(torch.load(path, weights_only=True))
    suffix = run(resumed)
    assert prefix + suffix == expected_order
    assert resumed.replay_rows == full.replay_rows
    assert resumed.replay_epochs == full.replay_epochs
    assert resumed.source_updates == full.source_updates
    assert sum(full.source_updates.values()) == full.updates
    for key, value in resumed.model.state_dict().items():
        assert torch.equal(value, expected[key]), key


def test_recovery_rejects_changed_mask_or_corrupt_replay_counts():
    trainer = build()
    trainer.step()
    saved = trainer.snapshot()
    changed = build()
    changed.replay["a"].measured[2] = True
    changed = BalancedContinualTrainer(
        changed.model, changed.fresh, changed.replay, learning_rate=0.001,
        source_manifest_sha256={"fresh": "a" * 64, "a": "b" * 64, "b": "c" * 64},
        gene_axis_sha256="d" * 64, initial_checkpoint_sha256="e" * 64,
    )
    with pytest.raises(ValueError, match="frozen continuation sources"):
        changed.restore(saved)
    saved["replay_rows"]["a"] += 1
    with pytest.raises(ValueError, match="invalid replay counts"):
        build().restore(saved)


def test_source_exhaustion_and_unmeasured_counts_are_rejected():
    trainer = build()
    trainer.replay["a"].expected_cells += 1
    with pytest.raises(ValueError, match="ended before"):
        run(trainer)
    trainer = build()
    trainer.replay["a"].rows[0, 2] = 1
    with pytest.raises(ValueError, match="invalid count batch"):
        run(trainer)


def test_fractional_counts_are_not_silently_rounded():
    trainer = build()
    trainer.fresh.rows[0, 0] = 1.5
    with pytest.raises(ValueError, match="invalid count batch"):
        trainer.step()


def test_recovery_rejects_changed_optimizer_or_source_lineage():
    original = build()
    original.step()
    saved = original.snapshot()
    changed = build()
    changed = BalancedContinualTrainer(
        changed.model, changed.fresh, changed.replay, learning_rate=0.002,
        source_manifest_sha256={"fresh": "a" * 64, "a": "b" * 64, "b": "c" * 64},
        gene_axis_sha256="d" * 64, initial_checkpoint_sha256="e" * 64,
    )
    with pytest.raises(ValueError, match="frozen continuation sources"):
        changed.restore(saved)
    changed = build()
    changed = BalancedContinualTrainer(
        changed.model, changed.fresh, changed.replay, learning_rate=0.001,
        source_manifest_sha256={"fresh": "f" * 64, "a": "b" * 64, "b": "c" * 64},
        gene_axis_sha256="d" * 64, initial_checkpoint_sha256="e" * 64,
    )
    with pytest.raises(ValueError, match="frozen continuation sources"):
        changed.restore(saved)
