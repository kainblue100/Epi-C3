"""Audit parameter counts from a trusted, locally produced stage-one checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from c3vcell.model import C3VCell, ModelConfig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--recovery", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    torch.set_num_threads(1)
    plan_bytes = args.plan.read_bytes()
    plan = json.loads(plan_bytes)
    canonical_plan = json.dumps(plan, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    plan_sha256 = hashlib.sha256(canonical_plan.encode("utf-8")).hexdigest()
    with args.recovery.open("rb") as stream:
        digest = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
        stream.seek(0)
        checkpoint = torch.load(stream, map_location="cpu", weights_only=True)
    if checkpoint.get("format") != "epi-c3-shared-rna-stage1-recovery-v1":
        raise ValueError("Unexpected checkpoint format")
    if checkpoint.get("plan_sha256") != plan_sha256:
        raise ValueError("Checkpoint does not match the frozen plan")

    model = C3VCell(ModelConfig(**plan["model_config"]))
    model.load_state_dict(checkpoint["trainer"]["model_state"], strict=True)
    if any(not torch.isfinite(value).all() for value in model.state_dict().values()):
        raise ValueError("Checkpoint contains non-finite tensors")

    optimized = (
        list(model.encoder.parameters())
        + list(model.predictor.parameters())
        + list(model.decoder.parameters())
        + [model.log_theta]
    )
    saved_optimizer = checkpoint["trainer"]["optimizer_state"]
    optimizer = torch.optim.AdamW(optimized, lr=plan["learning_rate"])
    optimizer.load_state_dict(saved_optimizer)
    if len(optimizer.param_groups) != 1 or len(optimizer.param_groups[0]["params"]) != len(optimized):
        raise ValueError("Saved optimizer does not match the representation parameter group")
    if len({id(parameter) for parameter in optimized}) != len(optimized):
        raise ValueError("Repeated optimizer parameter")
    if any(not parameter.requires_grad for parameter in optimized):
        raise ValueError("The representation optimizer includes frozen parameters")

    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    frozen = sum(parameter.numel() for parameter in model.parameters() if not parameter.requires_grad)
    optimized_count = sum(parameter.numel() for parameter in optimized)
    report = {
        "format": "epi-c3-stage1-parameter-audit-v1",
        "status": "complete",
        "scope": "incomplete-stage1-not-a-full-foundation",
        "model_config": plan["model_config"],
        "plan_sha256": plan_sha256,
        "recovery_sha256": digest.hexdigest(),
        "recovery_updates": checkpoint["trainer"]["updates"],
        "checkpoint_loaded_strictly": True,
        "model_trainable_at_construction": trainable,
        "model_frozen_at_construction": frozen,
        "representation_optimizer_parameters": optimized_count,
        "trainable_not_in_representation_optimizer": trainable - optimized_count,
        "nonparameter_buffer_elements": sum(buffer.numel() for buffer in model.buffers()),
        "limitations": [
            "Counts are specific to this gene axis and configuration, not a performance comparison.",
            "Trainable construction parameters include the unused flow module.",
            "The representation optimizer does not train a perturbation flow.",
        ],
    }
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
