from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import scipy.sparse as sp
import torch

from c3vcell.model import C3VCell, ModelConfig
from epi_c3_shared_foundation_v1 import FORMAT, gene_axis_digest


test_dir = Path(__file__).parent
script = test_dir / "train-rna-foundation-stage2-v1.py"
if not script.exists():
    script = test_dir.parent / script.name
spec = importlib.util.spec_from_file_location("stage2_driver", script)
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)


def write_json(path, value):
    path.write_text(json.dumps(value, allow_nan=False), encoding="utf-8")


def stage1_fixture(tmp_path):
    directory = tmp_path / "rna-foundation-stage1-v1"
    directory.mkdir()
    genes = ["ENSG1", "ENSG2", "ENSG3"]
    torch.manual_seed(7)
    model = C3VCell(ModelConfig(3, latent_dim=4, hidden_dim=6, covariance_weight=0.1))
    plan = {
        "format": "epi-c3-shared-rna-stage1-plan-v1", "genes": genes,
        "gene_axis_sha256": gene_axis_digest(genes), "model_config": model.configuration(),
        "sources": {"arc": {"expected_training_cells": 4}, "tabula": {"expected_training_cells": 6}},
        "learning_rate": 0.001,
    }
    digest = driver.canonical_digest(plan)
    counts = {"arc": 4, "tabula": 6}
    diagnostics = {
        name: {"masked_detected_mse": 1.0, "latent_dimensions_std_gt_0_1": 2}
        for name in counts
    }
    state = {
        "status": "stage1-complete", "plan_sha256": digest, "expected_cells": counts,
        "trained_cells": counts, "updates": 5, "diagnostics_after": diagnostics,
    }
    checkpoint = {
        "format": FORMAT, "status": "complete", "scope": "stage1-incomplete", "genes": genes,
        "gene_axis_sha256": plan["gene_axis_sha256"], "model_config": plan["model_config"],
        "model_state": model.state_dict(), "training": {"plan_sha256": digest, "cells": counts, "updates": 5},
    }
    write_json(directory / "plan.json", plan)
    write_json(directory / "state.json", state)
    torch.save(checkpoint, directory / "foundation-stage1.pt")
    return directory, plan, state, checkpoint


def test_gate_accepts_only_consistent_completed_stage1(tmp_path):
    directory, plan, state, checkpoint = stage1_fixture(tmp_path)
    loaded_plan, _, _, plan_sha, checkpoint_sha = driver.load_stage1(tmp_path)
    assert loaded_plan == plan
    assert plan_sha == driver.canonical_digest(plan)
    assert checkpoint_sha == driver.file_digest(directory / "foundation-stage1.pt")
    state["status"] = "running"
    write_json(directory / "state.json", state)
    with pytest.raises(ValueError, match="has not completed"):
        driver.load_stage1(tmp_path)
    state["status"] = "stage1-complete"
    state["trained_cells"]["arc"] = 3
    write_json(directory / "state.json", state)
    with pytest.raises(ValueError, match="does not match"):
        driver.load_stage1(tmp_path)


def test_gate_rejects_collapse_and_changed_checkpoint(tmp_path):
    directory, _, state, checkpoint = stage1_fixture(tmp_path)
    state["diagnostics_after"]["arc"]["latent_dimensions_std_gt_0_1"] = 0
    write_json(directory / "state.json", state)
    with pytest.raises(ValueError, match="collapsed"):
        driver.load_stage1(tmp_path)
    state["diagnostics_after"]["arc"]["latent_dimensions_std_gt_0_1"] = 2
    write_json(directory / "state.json", state)
    checkpoint["training"]["updates"] = 6
    torch.save(checkpoint, directory / "foundation-stage1.pt")
    with pytest.raises(ValueError, match="does not match"):
        driver.load_stage1(tmp_path)


class FakeSource:
    def __init__(self, rows, batch_size):
        self.rows = np.asarray(rows, dtype=np.float32)
        self.expected_cells = len(self.rows)
        self.measured = np.ones(3, dtype=bool)
        self.batch_size = batch_size
        self.position = 0

    def next_batch(self):
        if self.position == len(self.rows):
            return None
        end = min(self.position + self.batch_size, len(self.rows))
        matrix = sp.csr_matrix(self.rows[self.position:end])
        self.position = end
        return matrix

    def cursor(self):
        return {"position": self.position}

    def set_cursor(self, cursor):
        self.position = cursor["position"]

    def close(self):
        pass


def test_stage2_pause_resume_matches_uninterrupted_run(tmp_path, monkeypatch):
    directory, plan, state, checkpoint = stage1_fixture(tmp_path)
    stage1_result = (
        plan, state, checkpoint, driver.canonical_digest(plan),
        driver.file_digest(directory / "foundation-stage1.pt"),
    )
    fake_stage1 = SimpleNamespace(
        SEED=23, check_space=lambda path: None, now=lambda: "2026-01-01T00:00:00Z",
        atomic_json=write_json, atomic_torch=lambda path, value: torch.save(value, path),
        diagnostics=lambda model, sources: {name: {"valid": 1} for name in sources},
    )

    def streams(*args):
        replay = {
            "arc": FakeSource([[2, 1, 0], [1, 2, 0]] * 2, 3),
            "tabula": FakeSource([[0, 1, 2], [0, 2, 1]] * 3, 2),
        }
        fresh = FakeSource([[2, 1, 0], [1, 0, 2]] * 6, 3)
        return replay, fresh, "a" * 64

    monkeypatch.setattr(driver, "load_stage1", lambda root: stage1_result)
    monkeypatch.setattr(driver, "load_script", lambda *args: fake_stage1)
    monkeypatch.setattr(driver, "build_streams", streams)
    monkeypatch.setattr(driver.signal, "signal", lambda *args: None)

    def invoke(output, limit):
        monkeypatch.setattr(sys, "argv", [
            "train", "--data-root", str(tmp_path), "--run-root", str(tmp_path),
            "--output", str(output), "--checkpoint-every", "3",
            "--stop-after-updates", str(limit),
        ])
        driver.main()

    stopped = tmp_path / "resumed"
    invoke(stopped, 3)
    assert json.loads((stopped / "state.json").read_text())["status"] == "paused"
    invoke(stopped, 0)
    full = tmp_path / "full"
    invoke(full, 0)
    resumed_checkpoint = torch.load(stopped / "foundation-stage2.pt", weights_only=True)
    full_checkpoint = torch.load(full / "foundation-stage2.pt", weights_only=True)
    assert resumed_checkpoint["scope"] == "stage2-incomplete"
    assert resumed_checkpoint["training"] == full_checkpoint["training"]
    assert resumed_checkpoint["training"]["fresh_rows"] == 12
    assert sum(resumed_checkpoint["training"]["replay_rows"].values()) >= 12
    for key, value in full_checkpoint["model_state"].items():
        assert torch.equal(resumed_checkpoint["model_state"][key], value), key
