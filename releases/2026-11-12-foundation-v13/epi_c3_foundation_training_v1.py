"""Reproducible, source-interleaved RNA representation training."""

from __future__ import annotations

from dataclasses import replace
from typing import Protocol

import numpy as np
import scipy.sparse as sp
import torch

from c3vcell.model import C3VCell


class CountSource(Protocol):
    expected_cells: int
    measured: np.ndarray

    def next_batch(self) -> sp.csr_matrix | None: ...

    def cursor(self) -> dict[str, int]: ...

    def set_cursor(self, cursor: dict[str, int]) -> None: ...


class InterleavedFoundationTrainer:
    def __init__(
        self,
        model: C3VCell,
        sources: dict[str, CountSource],
        *,
        learning_rate: float,
        seed: int,
    ) -> None:
        if not sources or learning_rate <= 0 or not np.isfinite(learning_rate):
            raise ValueError("Training requires sources and a positive finite learning rate")
        for name, source in sources.items():
            if (
                not name or source.expected_cells < 2 or source.measured.dtype != bool
                or source.measured.shape != (model.config.n_genes,) or not np.any(source.measured)
            ):
                raise ValueError(f"Invalid source plan or gene mask: {name}")
        self.model = model
        self.sources = dict(sorted(sources.items()))
        self.optimizer = torch.optim.AdamW(
            list(model.encoder.parameters()) + list(model.predictor.parameters())
            + list(model.decoder.parameters()) + [model.log_theta],
            lr=learning_rate,
        )
        self.rng = np.random.default_rng(seed)
        self.trained = {name: 0 for name in self.sources}
        self.updates = 0
        self.singleton_updates = 0

    def step(self) -> tuple[str, dict[str, float]] | None:
        remaining = {name: source.expected_cells - self.trained[name] for name, source in self.sources.items()}
        if any(value < 0 for value in remaining.values()):
            raise ValueError("Source training exceeded its frozen eligible-cell count")
        total = sum(remaining.values())
        if total == 0:
            return None
        draw = int(self.rng.integers(total))
        chosen = ""
        for name, count in remaining.items():
            if draw < count:
                chosen = name
                break
            draw -= count
        source = self.sources[chosen]
        matrix = source.next_batch()
        if matrix is None:
            raise ValueError(f"Source {chosen} ended before its expected eligible-cell count")
        if (
            not sp.isspmatrix_csr(matrix) or matrix.shape[1] != self.model.config.n_genes
            or not 1 <= matrix.shape[0] <= remaining[chosen]
            or not np.isfinite(matrix.data).all() or np.any(matrix.data < 0)
            or np.any(matrix.data != np.floor(matrix.data)) or np.any(matrix.data > 2**24)
            or np.any(np.asarray(matrix.sum(axis=1)).ravel() <= 0)
            or np.any(~source.measured[matrix.indices])
        ):
            raise ValueError(f"Source {chosen} produced an unexpected batch shape")
        counts = torch.as_tensor(matrix.toarray(), dtype=torch.float32)
        measured = torch.from_numpy(source.measured).expand_as(counts).float()
        self.optimizer.zero_grad(set_to_none=True)
        original_config = self.model.config
        if len(counts) == 1 and self.model.config.covariance_weight > 0:
            self.model.config = replace(self.model.config, covariance_weight=0.0)
            self.singleton_updates += 1
        try:
            loss, metrics = self.model.representation_loss(counts, measured)
        finally:
            self.model.config = original_config
        if not torch.isfinite(loss):
            raise ValueError(f"Nonfinite representation loss from source {chosen}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0, error_if_nonfinite=True)
        self.optimizer.step()
        self.model.update_teacher()
        self.trained[chosen] += len(counts)
        self.updates += 1
        return chosen, metrics

    def verify_exhausted(self) -> None:
        for name, source in self.sources.items():
            if self.trained[name] != source.expected_cells or source.next_batch() is not None:
                raise ValueError(f"Source {name} has not been consumed exactly once")

    def snapshot(self) -> dict:
        return {
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "torch_rng_state": torch.get_rng_state(),
            "numpy_rng_state": self.rng.bit_generator.state,
            "trained": self.trained.copy(),
            "updates": self.updates,
            "singleton_updates": self.singleton_updates,
            "source_cursors": {name: source.cursor() for name, source in self.sources.items()},
        }

    def restore(self, snapshot: dict) -> None:
        if set(snapshot["trained"]) != set(self.sources) or set(snapshot["source_cursors"]) != set(self.sources):
            raise ValueError("Recovery sources do not match the frozen training plan")
        if any(not 0 <= snapshot["trained"][name] <= source.expected_cells for name, source in self.sources.items()):
            raise ValueError("Recovery contains invalid source counts")
        self.model.load_state_dict(snapshot["model_state"], strict=True)
        self.optimizer.load_state_dict(snapshot["optimizer_state"])
        self.rng.bit_generator.state = snapshot["numpy_rng_state"]
        torch.set_rng_state(snapshot["torch_rng_state"])
        self.trained = dict(snapshot["trained"])
        self.updates = int(snapshot["updates"])
        self.singleton_updates = int(snapshot["singleton_updates"])
        for name, source in self.sources.items():
            source.set_cursor(snapshot["source_cursors"][name])
