"""Compare full-library and visible-only normalization for a saved RNA checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import torch

import c3vcell.model as model_module
from c3vcell.model import C3VCell, ModelConfig
from epi_c3_foundation_stage1_sources_v1 import digest_json
from epi_c3_shared_foundation_v1 import gene_axis_digest


STAGE1_SHA = "84e177cac2131ece79529f1c85ee16eb36bc7582d566c81dc7234e3a26953230"
MODEL_SHA = "d22568a921afcfc43a26a00f39e423d265b12c4f4ace8afce29169cf0d846cda"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def visible_only_encoding(encoder: torch.nn.Module, counts: torch.Tensor, visible: torch.Tensor) -> torch.Tensor:
    if (
        counts.ndim != 2 or visible.shape != counts.shape or visible.dtype != torch.bool
        or not torch.is_floating_point(counts) or not torch.isfinite(counts).all()
        or torch.any(counts < 0)
    ):
        raise ValueError("Counts and visibility mask are incompatible")
    shown = counts * visible.to(counts.dtype)
    if torch.any(shown.sum(dim=1) <= 0):
        raise ValueError("Visible-only normalization requires a positive visible library in every cell")
    return encoder(shown, visible.to(counts.dtype))


@torch.no_grad()
def compare_source(model: C3VCell, source, seed: int, expected: dict) -> dict:
    counts = torch.as_tensor(source.validation.toarray(), dtype=torch.float32)
    measured = torch.from_numpy(source.measured).expand_as(counts)
    generator = torch.Generator().manual_seed(seed)
    hidden = (torch.rand(counts.shape, generator=generator) < 0.2) & measured
    visible = measured & ~hidden
    positive = hidden & (counts > 0)
    if not torch.any(hidden) or not torch.any(positive):
        raise ValueError("Validation mask lacks held-out measured or detected values")
    full_latent = model.encoder(counts, visible.float())
    strict_latent = visible_only_encoding(model.encoder, counts, visible)
    altered = counts + hidden.to(counts.dtype) * 5.0
    strict_altered = visible_only_encoding(model.encoder, altered, visible)
    if not torch.equal(strict_latent, strict_altered):
        raise ValueError("Visible-only encoding changed when hidden values changed")
    full_altered = model.encoder(altered, visible.float())
    full_shift = float((full_latent - full_altered).abs().max())
    if full_shift <= 1e-7:
        raise ValueError("The audit could not demonstrate full-library dependence on hidden values")
    observed = torch.log1p(counts / counts.sum(dim=1, keepdim=True) * 10_000.0)
    results = {}
    for label, latent in (("full_library", full_latent), ("visible_only_library", strict_latent)):
        predicted = torch.log1p(model.probabilities(latent, measured.float()) * 10_000.0)
        results[label] = {
            "masked_all_mse": float((predicted[hidden] - observed[hidden]).square().mean()),
            "masked_detected_mse": float((predicted[positive] - observed[positive]).square().mean()),
        }
    for key in ("masked_all_mse", "masked_detected_mse"):
        if abs(results["full_library"][key] - expected[key]) > 1e-7:
            raise ValueError(f"Original diagnostic could not be reproduced: {key}")
    return {
        "cells": int(counts.shape[0]),
        "masked_measured_entries": int(hidden.sum()),
        "masked_detected_entries": int(positive.sum()),
        "full_library": results["full_library"],
        "visible_only_library": results["visible_only_library"],
        "mean_absolute_latent_change_between_protocols": float((strict_latent - full_latent).abs().mean()),
        "max_full_library_latent_change_after_hidden_input_alteration": full_shift,
        "visible_only_latent_unchanged_after_hidden_input_alteration": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    stage1_path = args.root / "train-rna-foundation-stage1-v1.py"
    if sha256(stage1_path) != STAGE1_SHA or sha256(Path(model_module.__file__)) != MODEL_SHA:
        raise ValueError("Original stage-one or encoder source changed")
    spec = importlib.util.spec_from_file_location("frozen_stage1_masked_audit", stage1_path)
    if spec is None or spec.loader is None:
        raise ImportError("Cannot load original source setup")
    stage1 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(stage1)
    plan = json.loads((args.snapshot / "plan.json").read_text(encoding="utf-8"))
    previous_path = args.snapshot / "diagnostic.json"
    previous = json.loads(previous_path.read_text(encoding="utf-8"))
    checkpoint_path = args.snapshot / "recovery.pt"
    checkpoint_sha = sha256(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if (
        checkpoint_sha != previous["snapshot_sha256"]
        or checkpoint["plan_sha256"] != digest_json(plan)
        or previous["plan_sha256"] != checkpoint["plan_sha256"]
        or previous["updates"] != checkpoint["trainer"]["updates"]
        or previous["trained_cells"] != checkpoint["trainer"]["trained"]
    ):
        raise ValueError("Checkpoint, plan and earlier diagnostic do not match")
    torch.set_num_threads(1)
    alignment, streams, heldout, provenance = stage1.build_sources()
    try:
        if (
            gene_axis_digest(alignment.genes) != plan["gene_axis_sha256"]
            or provenance != plan["provenance"]
            or digest_json(sorted(heldout)) != plan["heldout_targets_sha256"]
            or any(stream.expected_cells != plan["sources"][name]["expected_training_cells"]
                   or stream.notes != plan["sources"][name]["details"] for name, stream in streams.items())
        ):
            raise ValueError("Original source setup differs from the frozen plan")
        model = C3VCell(ModelConfig(**plan["model_config"]))
        model.load_state_dict(checkpoint["trainer"]["model_state"], strict=True)
        model.eval()
        results = {
            name: compare_source(model, stream, stage1.SEED + index, previous["at_snapshot"][name])
            for index, (name, stream) in enumerate(sorted(streams.items()))
        }
        report = {
            "format": "epi-c3-foundation-masked-library-audit-v1",
            "status": "complete",
            "scope": "read-only-small-validation-diagnostic",
            "updates": previous["updates"],
            "snapshot_sha256": checkpoint_sha,
            "plan_sha256": checkpoint["plan_sha256"],
            "previous_diagnostic_sha256": sha256(previous_path),
            "original_stage1_script_sha256": STAGE1_SHA,
            "original_model_code_sha256": MODEL_SHA,
            "audit_script_sha256": sha256(Path(__file__)),
            "sources": results,
            "limitations": [
                "The original training and reported diagnostic normalize using the complete measured library, including the masked genes.",
                "Visible-only normalization is a stricter input ablation applied to a model trained with full-library normalization; it is not a retrained model.",
                "The ground-truth log1p(CPM 10000) targets use the complete measured library for evaluation, not as strict-model input.",
                "Each source contributes 32 fixed validation cells; Arc and Orion are not donor-disjoint.",
                "Results do not establish perturbation prediction, biological generalization or a completed foundation model.",
            ],
        }
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write("\n")
        print(json.dumps({name: {
            "full_library_detected_mse": value["full_library"]["masked_detected_mse"],
            "visible_only_detected_mse": value["visible_only_library"]["masked_detected_mse"],
        } for name, value in results.items()}, allow_nan=False))
    finally:
        for stream in streams.values():
            stream.close()


if __name__ == "__main__":
    main()
