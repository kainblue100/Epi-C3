"""Reproducible, source-balanced continuation for an RNA foundation model."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from fractions import Fraction
import hashlib
import json
import re

import numpy as np
import scipy.sparse as sp
import torch

from c3vcell.model import C3VCell
from epi_c3_foundation_training_v1 import CountSource


FORMAT = "epi-c3-balanced-continuation-recovery-v1"
SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def source_plan(model: C3VCell, fresh: CountSource, replay: dict[str, CountSource]) -> dict:
    if not replay or "fresh" in replay:
        raise ValueError("Continuation requires distinct, named replay sources")
    sources = {"fresh": fresh, **replay}
    if len({id(source) for source in sources.values()}) != len(sources):
        raise ValueError("Continuation sources must be independent streams")
    plan = {}
    for name, source in sorted(sources.items()):
        if (
            not isinstance(name, str) or not name
            or type(source.expected_cells) is not int or source.expected_cells < 2
            or not isinstance(source.measured, np.ndarray) or source.measured.dtype != bool
            or source.measured.shape != (model.config.n_genes,) or not np.any(source.measured)
        ):
            raise ValueError(f"Invalid source plan or gene mask: {name}")
        plan[name] = {
            "expected_cells": source.expected_cells,
            "measured_sha256": hashlib.sha256(source.measured.tobytes()).hexdigest(),
            "initial_cursor": deepcopy(source.cursor()),
        }
    return plan


def plan_digest(plan: dict) -> str:
    payload = json.dumps(plan, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class BalancedContinualTrainer:
    """Consume fresh rows once while replaying approximately as many prior rows.

    Replay exposure is proportional to the prior sources' eligible row counts.
    All input sources must already enforce their frozen training/holdout splits;
    callers must verify the supplied checkpoint, axis and source manifest hashes.
    """

    def __init__(
        self,
        model: C3VCell,
        fresh: CountSource,
        replay: dict[str, CountSource],
        *,
        learning_rate: float,
        source_manifest_sha256: dict[str, str],
        gene_axis_sha256: str,
        initial_checkpoint_sha256: str,
    ) -> None:
        if learning_rate <= 0 or not np.isfinite(learning_rate):
            raise ValueError("Continuation requires a positive finite learning rate")
        if any(parameter.device.type != "cpu" for parameter in model.parameters()):
            raise ValueError("This reproducible continuation trainer supports CPU models only")
        if (
            set(source_manifest_sha256) != set(replay) | {"fresh"}
            or any(not isinstance(value, str) or SHA256.fullmatch(value) is None
                   for value in source_manifest_sha256.values())
            or not isinstance(gene_axis_sha256, str) or SHA256.fullmatch(gene_axis_sha256) is None
            or not isinstance(initial_checkpoint_sha256, str)
            or SHA256.fullmatch(initial_checkpoint_sha256) is None
        ):
            raise ValueError("Continuation requires verified source, gene axis and checkpoint hashes")
        self.model = model
        self.fresh = fresh
        self.replay = dict(sorted(replay.items()))
        self.source_plan = source_plan(model, fresh, self.replay)
        self.plan = {
            "format": "epi-c3-balanced-continuation-plan-v1",
            "source_plan": self.source_plan,
            "source_manifest_sha256": dict(sorted(source_manifest_sha256.items())),
            "gene_axis_sha256": gene_axis_sha256,
            "initial_checkpoint_sha256": initial_checkpoint_sha256,
            "model_config": model.configuration(),
            "learning_rate": learning_rate,
        }
        self.plan_sha256 = plan_digest(self.plan)
        self.optimizer = torch.optim.AdamW(
            list(model.encoder.parameters()) + list(model.predictor.parameters())
            + list(model.decoder.parameters()) + [model.log_theta],
            lr=learning_rate,
        )
        self.fresh_rows = 0
        self.replay_rows = {name: 0 for name in self.replay}
        self.replay_in_epoch = {name: 0 for name in self.replay}
        self.replay_epochs = {name: 0 for name in self.replay}
        self.updates = 0
        self.source_updates = {name: 0 for name in ("fresh", *self.replay)}
        self.singleton_updates = 0

    def _choose(self) -> tuple[str, CountSource, int]:
        replay_total = sum(self.replay_rows.values())
        if self.fresh_rows < self.fresh.expected_cells and self.fresh_rows <= replay_total:
            return "fresh", self.fresh, self.fresh.expected_cells - self.fresh_rows
        name = min(
            self.replay,
            key=lambda key: (Fraction(self.replay_rows[key], self.replay[key].expected_cells), key),
        )
        source = self.replay[name]
        if self.replay_in_epoch[name] == source.expected_cells:
            if source.next_batch() is not None:
                raise ValueError(f"Replay source {name} exceeded its frozen eligible-row count")
            source.set_cursor(deepcopy(self.source_plan[name]["initial_cursor"]))
            self.replay_epochs[name] += 1
            self.replay_in_epoch[name] = 0
        return name, source, source.expected_cells - self.replay_in_epoch[name]

    def step(self) -> tuple[str, dict[str, float]] | None:
        if self.fresh_rows == self.fresh.expected_cells and sum(self.replay_rows.values()) >= self.fresh_rows:
            return None
        name, source, remaining = self._choose()
        matrix = source.next_batch()
        if matrix is None:
            raise ValueError(f"Source {name} ended before its frozen eligible-row count")
        if (
            not sp.isspmatrix_csr(matrix) or matrix.shape[1] != self.model.config.n_genes
            or not 1 <= matrix.shape[0] <= remaining
            or not np.isfinite(matrix.data).all() or np.any(matrix.data < 0)
            or np.any(matrix.data != np.floor(matrix.data)) or np.any(matrix.data > 2**24)
            or np.any(np.asarray(matrix.sum(axis=1)).ravel() <= 0)
            or np.any(~source.measured[matrix.indices])
        ):
            raise ValueError(f"Source {name} produced an invalid count batch")
        counts = torch.as_tensor(matrix.toarray(), dtype=torch.float32)
        measured = torch.from_numpy(source.measured).expand_as(counts).float()
        self.optimizer.zero_grad(set_to_none=True)
        original_config = self.model.config
        if len(counts) == 1 and original_config.covariance_weight > 0:
            self.model.config = replace(original_config, covariance_weight=0.0)
            self.singleton_updates += 1
        try:
            loss, metrics = self.model.representation_loss(counts, measured)
        finally:
            self.model.config = original_config
        if not torch.isfinite(loss):
            raise ValueError(f"Nonfinite representation loss from source {name}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0, error_if_nonfinite=True)
        self.optimizer.step()
        self.model.update_teacher()
        if name == "fresh":
            self.fresh_rows += len(counts)
        else:
            self.replay_rows[name] += len(counts)
            self.replay_in_epoch[name] += len(counts)
        self.updates += 1
        self.source_updates[name] += 1
        return name, metrics

    def verify_exhausted(self) -> None:
        if (
            self.fresh_rows != self.fresh.expected_cells
            or sum(self.replay_rows.values()) < self.fresh_rows
            or self.fresh.next_batch() is not None
        ):
            raise ValueError("Fresh source or balanced replay has not completed")

    def snapshot(self) -> dict:
        return {
            "format": FORMAT,
            "source_plan_sha256": self.plan_sha256,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "torch_rng_state": torch.get_rng_state(),
            "fresh_rows": self.fresh_rows,
            "replay_rows": self.replay_rows.copy(),
            "replay_in_epoch": self.replay_in_epoch.copy(),
            "replay_epochs": self.replay_epochs.copy(),
            "updates": self.updates,
            "source_updates": self.source_updates.copy(),
            "singleton_updates": self.singleton_updates,
            "source_cursors": {
                "fresh": deepcopy(self.fresh.cursor()),
                **{name: deepcopy(source.cursor()) for name, source in self.replay.items()},
            },
        }

    def restore(self, snapshot: dict) -> None:
        if snapshot.get("format") != FORMAT or snapshot.get("source_plan_sha256") != self.plan_sha256:
            raise ValueError("Recovery does not match the frozen continuation sources")
        names = set(self.replay)
        if (
            set(snapshot["replay_rows"]) != names or set(snapshot["replay_in_epoch"]) != names
            or set(snapshot["replay_epochs"]) != names
            or set(snapshot["source_updates"]) != names | {"fresh"}
            or set(snapshot["source_cursors"]) != names | {"fresh"}
        ):
            raise ValueError("Recovery sources do not match the frozen continuation plan")
        fresh_rows = snapshot["fresh_rows"]
        updates, singleton_updates = snapshot["updates"], snapshot["singleton_updates"]
        if (
            type(fresh_rows) is not int or not 0 <= fresh_rows <= self.fresh.expected_cells
            or type(updates) is not int or updates < 0
            or type(singleton_updates) is not int or not 0 <= singleton_updates <= updates
            or any(type(count) is not int or count < 0 for count in snapshot["source_updates"].values())
            or sum(snapshot["source_updates"].values()) != updates
        ):
            raise ValueError("Recovery contains invalid continuation counts")
        for name, source in self.replay.items():
            rows = snapshot["replay_rows"][name]
            in_epoch = snapshot["replay_in_epoch"][name]
            epochs = snapshot["replay_epochs"][name]
            if (
                type(rows) is not int or type(in_epoch) is not int or type(epochs) is not int
                or rows < 0 or epochs < 0 or not 0 <= in_epoch <= source.expected_cells
                or rows != epochs * source.expected_cells + in_epoch
            ):
                raise ValueError(f"Recovery contains invalid replay counts for {name}")
        self.model.load_state_dict(snapshot["model_state"], strict=True)
        self.optimizer.load_state_dict(snapshot["optimizer_state"])
        torch.set_rng_state(snapshot["torch_rng_state"])
        self.fresh_rows = fresh_rows
        self.replay_rows = dict(snapshot["replay_rows"])
        self.replay_in_epoch = dict(snapshot["replay_in_epoch"])
        self.replay_epochs = dict(snapshot["replay_epochs"])
        self.source_updates = dict(snapshot["source_updates"])
        self.updates, self.singleton_updates = updates, singleton_updates
        self.fresh.set_cursor(deepcopy(snapshot["source_cursors"]["fresh"]))
        for name, source in self.replay.items():
            source.set_cursor(deepcopy(snapshot["source_cursors"][name]))
