"""Read-only held-out diagnostics of an immutable stage-one recovery snapshot."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import torch

from c3vcell.model import C3VCell, ModelConfig
from epi_c3_foundation_stage1_sources_v1 import digest_json
from epi_c3_shared_foundation_v1 import gene_axis_digest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Refusing to overwrite an existing diagnostic")
    spec = importlib.util.spec_from_file_location("stage1_diagnostic", args.root / "train-rna-foundation-stage1-v1.py")
    if spec is None or spec.loader is None:
        raise ImportError("Cannot load the frozen foundation source setup")
    stage1 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(stage1)
    plan = json.loads((args.run / "plan.json").read_text())
    with (args.run / "recovery.pt").open("rb") as file:
        digest = hashlib.sha256()
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
        file.seek(0)
        snapshot = torch.load(file, map_location="cpu", weights_only=True)
    if snapshot["plan_sha256"] != digest_json(plan):
        raise ValueError("Recovery snapshot does not match the frozen training plan")
    torch.set_num_threads(1)
    alignment, streams, heldout, provenance = stage1.build_sources()
    try:
        if (
            gene_axis_digest(alignment.genes) != plan["gene_axis_sha256"]
            or provenance != plan["provenance"]
            or digest_json(sorted(heldout)) != plan["heldout_targets_sha256"]
            or any(
                stream.expected_cells != plan["sources"][name]["expected_training_cells"]
                or stream.notes != plan["sources"][name]["details"]
                for name, stream in streams.items()
            )
        ):
            raise ValueError("Diagnostic data do not match the training plan")
        model = C3VCell(ModelConfig(**plan["model_config"]))
        model.load_state_dict(snapshot["trainer"]["model_state"], strict=True)
        after = stage1.diagnostics(model, streams)
        result = {
            "status": "complete", "scope": "early-stage1-read-only-diagnostic",
            "snapshot_sha256": digest.hexdigest(), "plan_sha256": snapshot["plan_sha256"],
            "updates": snapshot["trainer"]["updates"], "trained_cells": snapshot["trainer"]["trained"],
            "before_training": snapshot["diagnostics_before"], "at_snapshot": after,
            "limitations": [
                "Only 32 fixed validation cells per source; Arc and Orion are not donor-disjoint.",
                "Masked reconstruction and latent variance do not establish perturbation prediction quality.",
                "Tahoe and other requested datasets have not yet trained this shared foundation.",
            ],
        }
        with args.output.open("x", encoding="utf-8") as file:
            json.dump(result, file, indent=2, allow_nan=False)
            file.write("\n")
        print(json.dumps({
            "updates": result["updates"], "trained_cells": result["trained_cells"],
            "masked_detected_mse": {name: values["masked_detected_mse"] for name, values in after.items()},
            "latent_std_mean": {name: values["latent_std_mean"] for name, values in after.items()},
        }))
    finally:
        for stream in streams.values():
            stream.close()


if __name__ == "__main__":
    main()
