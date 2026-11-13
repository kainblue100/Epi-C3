#!/usr/bin/env python3
"""Audit parameter counts from the exact checkpoints used in two internal comparisons."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from importlib.metadata import version
from pathlib import Path, PurePosixPath


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def verify_checkpoint_files(directory: Path, state: dict) -> dict:
    files = state.get("checkpoint_files")
    if not isinstance(files, list) or not files:
        raise ValueError("No checkpoint files are bound to the completed run")
    found = {}
    for item in files:
        relative = PurePosixPath(item["file"])
        if relative.is_absolute() or ".." in relative.parts or relative.as_posix() in found:
            raise ValueError("Unsafe or duplicate checkpoint path")
        path = directory.joinpath(*relative.parts)
        if directory.resolve() not in path.resolve().parents:
            raise ValueError("Checkpoint path escapes the result directory")
        if digest(path) != item["sha256"]:
            raise ValueError(f"Checkpoint checksum mismatch: {relative}")
        found[relative.as_posix()] = item["sha256"]
    return found


def parameter_counts(module) -> dict:
    parameters = list(module.named_parameters())
    buffers = list(module.named_buffers())
    return {
        "parameter_tensors": len(parameters),
        "trainable_at_construction": sum(p.numel() for _, p in parameters if p.requires_grad),
        "frozen_at_construction": sum(p.numel() for _, p in parameters if not p.requires_grad),
        "nonparameter_buffer_elements": sum(b.numel() for _, b in buffers),
    }


def verify_state_dict(module, expected: dict) -> None:
    import torch

    actual = module.state_dict()
    if set(actual) != set(expected):
        raise ValueError("Restored model keys do not match the saved state")
    if any(not torch.equal(actual[key].cpu(), expected[key].cpu()) for key in actual):
        raise ValueError("Restored model values do not match the saved state")


def load_runner(path: Path):
    spec = importlib.util.spec_from_file_location("frozen_khp_runner", path)
    if spec is None or spec.loader is None:
        raise ValueError("Cannot load the frozen benchmark runner")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_khp_case(root: Path, method: str):
    base = root / "khp-cd8-transfer-v1"
    result = base / (
        "scvidr-posthoc-v2/results/scvidr/case-01"
        if method == "scvidr"
        else f"results/{method}/case-01"
    )
    state = read_json(result / "state.json")
    plan = read_json(base / "plan.json")
    if state.get("state") != "complete" or state.get("method") != method or state.get("case_index") != 1:
        raise ValueError("Expected a complete KHP case-01 result")
    if state.get("plan_sha256") != plan["plan_sha256"]:
        raise ValueError("KHP result and plan do not match")
    scripts = {
        "c3_rna_only": "run-khp-c3-baselines-v1.py",
        "scgen": "run-khp-scgen-v1.py",
        "scvidr": "run-khp-scvidr-v1.py",
    }
    for filename, key in (
        (scripts[method], "script_sha256"),
        ("khp_transfer_runtime_v1.py", "runtime_sha256"),
        ("khp_transfer_common_v1.py", "common_sha256"),
    ):
        if digest(root / filename) != state.get(key):
            raise ValueError(f"Frozen KHP implementation changed: {filename}")
    files = verify_checkpoint_files(result, state)
    sys.path.insert(0, str(root))
    import khp_transfer_runtime_v1 as runtime

    manifest, counts, genes, groups = runtime.load_training_case(plan, base / "cases/case-01")
    if state.get("train_counts_sha256") != manifest["train_counts"]["sha256"]:
        raise ValueError("Checkpoint is bound to a different training matrix")
    if counts.shape != (2000, 5000) or len(genes) != 5000:
        raise ValueError("Unexpected KHP case shape")
    return result, state, plan, counts, genes, groups, files


def audit_c3(root: Path) -> dict:
    import torch

    result, state, plan, counts, genes, groups, files = verify_khp_case(root, "c3_rna_only")
    del counts, groups
    package = root / "k562-loader-fix-snapshot-v2"
    model_source = package / "c3vcell/model.py"
    if digest(model_source) != plan["models"]["c3_rna_only"]["implementation_sha256"]:
        raise ValueError("Epi-C3 model implementation differs from the KHP training plan")
    sys.path.insert(0, str(package))
    from c3vcell.model import C3VCell, ModelConfig

    checkpoint = torch.load(result / "model.pt", map_location="cpu", weights_only=True)
    if (
        checkpoint["plan_sha256"] != state["plan_sha256"]
        or checkpoint["train_counts_sha256"] != state["train_counts_sha256"]
    ):
        raise ValueError("Epi-C3 checkpoint provenance differs from its result")
    config = ModelConfig(**checkpoint["model_config"])
    model = C3VCell(config)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    verify_state_dict(model, checkpoint["model_state"])
    condition_count = checkpoint["condition"].numel()
    if config.n_genes != len(genes) or condition_count != config.latent_dim:
        raise ValueError("Epi-C3 gene axis or learned condition vector is inconsistent")
    return {
        "method": "Epi-C3 RNA-only",
        "comparison": "KHP case-01",
        "genes": config.n_genes,
        "latent_dim": config.latent_dim,
        "hidden_dim": config.hidden_dim,
        "training_rows": 2000,
        "checkpoint_sha256": files["model.pt"],
        "model_source_sha256": digest(model_source),
        **parameter_counts(model),
        "separate_learned_condition_vector_parameters": condition_count,
        "flow_optimizer_parameters_including_condition": (
            sum(p.numel() for p in model.velocity.parameters()) + condition_count
        ),
        "checkpoint_values_verified": True,
    }


def audit_gears(root: Path, source: Path) -> dict:
    import torch

    directory = root / "replogle-gwps-gears-posthoc-training-v1"
    state = read_json(directory / "state.json")
    protocol_path = directory / "protocol.json"
    protocol = read_json(protocol_path)
    source_state = read_json(source / "state.json")
    checkpoint_path = directory / "model.pt"
    if (
        state.get("state") != "complete" or state.get("smoke") is not False
        or state.get("step") != 12000 or state.get("steps") != 12000
        or digest(checkpoint_path) != state.get("checkpoint_sha256")
        or digest(protocol_path) != state.get("protocol_file_sha256")
        or digest(source / "state.json") != protocol.get("source_state_sha256")
        or source_state.get("commit") != protocol.get("source_commit")
        or digest(source / "gears/model.py") != source_state["files"]["gears/model.py"]["sha256"]
    ):
        raise ValueError("GEARS checkpoint or source binding is invalid")
    sys.path.insert(0, str(source))
    from gears.model import GEARS_Model

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("protocol_sha256") != state.get("protocol_sha256") or checkpoint.get("step") != state["step"]:
        raise ValueError("GEARS checkpoint does not match the completed training protocol")
    config = checkpoint["config"]
    model = GEARS_Model(config)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    verify_state_dict(model, checkpoint["model_state"])
    if config["num_genes"] != 1861 or config["num_perts"] != 1735:
        raise ValueError("Unexpected GEARS gene or perturbation axis")
    return {
        "method": "GEARS",
        "comparison": "K562 GWPS post-hoc",
        "genes": config["num_genes"],
        "perturbation_identities": config["num_perts"],
        "hidden_dim": config["hidden_size"],
        "decoder_hidden_dim": config["decoder_hidden_size"],
        "training_steps": checkpoint["step"],
        "checkpoint_sha256": state["checkpoint_sha256"],
        "source_commit": source_state["commit"],
        **parameter_counts(model),
        "checkpoint_values_verified": True,
    }


def audit_scgen(root: Path) -> dict:
    import scgen
    import scvi
    import torch

    result, state, plan, counts, genes, groups, files = verify_khp_case(root, "scgen")
    if version("scgen") != state["scgen_version"] or scvi.__version__ != state["scvi_tools_version"]:
        raise ValueError("scGen or scvi-tools version differs from the frozen result")
    runner = load_runner(root / "run-khp-scgen-v1.py")
    data = runner.assemble_adata(counts, genes, groups)
    scgen.SCGEN.setup_anndata(data, batch_key="condition", labels_key="cell_type")
    model = scgen.SCGEN.load(str(result / "model"), adata=data, use_gpu=False)
    checkpoint = torch.load(result / "model/model.pt", map_location="cpu")
    verify_state_dict(model.module, checkpoint["model_state_dict"])
    return {
        "method": "scGen",
        "version": state["scgen_version"],
        "comparison": "KHP case-01",
        "genes": len(genes),
        "training_rows": int(counts.shape[0]),
        "checkpoint_sha256": files["model/model.pt"],
        **parameter_counts(model.module),
        "checkpoint_values_verified": True,
    }


def audit_scvidr(root: Path, source: Path) -> dict:
    import scvi
    import torch

    result, state, plan, counts, genes, groups, files = verify_khp_case(root, "scvidr")
    protocol_path = root / "khp-scvidr-posthoc-protocol-v2.json"
    protocol = read_json(protocol_path)
    source_state = read_json(source / "state.json")
    if (
        scvi.__version__ != "0.13.0"
        or digest(protocol_path) != state["posthoc_protocol_sha256"]
        or digest(source / "state.json") != state["source_state_sha256"]
        or source_state.get("commit") != protocol.get("source_commit")
        or digest(source / "vidr/vidr.py") != source_state["files"]["vidr/vidr.py"]["sha256"]
    ):
        raise ValueError("scVIDR source, protocol, or environment differs from the frozen result")
    sys.path.insert(0, str(source / "vidr"))
    import vidr

    if Path(vidr.__file__).resolve() != (source / "vidr/vidr.py").resolve():
        raise ValueError("Imported scVIDR source differs from the frozen source")
    runner = load_runner(root / "run-khp-scvidr-v1.py")
    data = runner.assemble_adata(counts, genes, groups)
    scvi.data.setup_anndata(data, batch_key="condition", labels_key="cell_type")
    model = vidr.VIDR.load(str(result / "model"), adata=data, use_gpu=False)
    saved = torch.load(result / "model/model_params.pt", map_location="cpu")
    verify_state_dict(model.module, saved)
    return {
        "method": "scVIDR",
        "source_commit": source_state["commit"],
        "comparison": "KHP case-01 post-hoc",
        "genes": len(genes),
        "training_rows": int(counts.shape[0]),
        "checkpoint_sha256": files["model/model_params.pt"],
        **parameter_counts(model.module),
        "separate_two_source_regression_excluded": True,
        "checkpoint_values_verified": True,
        "preprocessing_warning": protocol["preprocessing"]["scvi_warning"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--method", choices=("c3", "gears", "scgen", "scvidr"), required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.method in ("gears", "scvidr") and args.source is None:
        parser.error("--source is required for GEARS and scVIDR")
    if args.output.exists() or not args.output.parent.is_dir():
        raise ValueError("Output already exists or its parent directory is missing")
    if args.method == "c3":
        result = audit_c3(args.root)
    elif args.method == "gears":
        result = audit_gears(args.root, args.source)
    elif args.method == "scgen":
        result = audit_scgen(args.root)
    else:
        result = audit_scvidr(args.root, args.source)
    report = {
        "format": "epi-c3-model-parameter-audit-v1",
        "state": "complete",
        "count_definition": (
            "PyTorch named parameters with requires_grad immediately after the trained model is reconstructed; "
            "frozen parameters and buffers are counted separately. Separately learned adapters are identified."
        ),
        "not_a_model_performance_comparison": True,
        "audit_script_sha256": digest(Path(__file__)),
        **result,
    }
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    print(json.dumps({"method": result["method"], "report_sha256": digest(args.output)}))


if __name__ == "__main__":
    main()
