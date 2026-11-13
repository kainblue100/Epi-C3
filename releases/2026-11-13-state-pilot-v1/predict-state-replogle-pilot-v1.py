"""Predict a frozen pilot with verified, distinct control parents."""

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import random

import anndata as ad
import numpy as np
import pandas as pd
import torch
import yaml


CONTROL = "non-targeting"


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def install_parent_mapping(data_module, plan):
    assignments = plan["assignments"]
    seen = set()
    for subset in data_module.test_datasets:
        dataset = subset.dataset
        if dataset.cell_barcodes is None:
            raise ValueError("CellLoad must expose input and parent identities")
        barcodes = np.asarray(dataset.cell_barcodes, dtype=str)
        if len(set(barcodes)) != len(barcodes):
            raise ValueError("CellLoad source identities are not unique")
        if not np.array_equal(dataset.all_indices, np.arange(len(barcodes))):
            raise ValueError("Unexpected CellLoad file-index mapping")
        lookup = {identity: index for index, identity in enumerate(barcodes)}
        strategy = dataset.mapping_strategy
        if strategy.use_consecutive_loading:
            raise ValueError("Consecutive sampling bypasses frozen parent mappings")
        strategy.cache_perturbation_control_pairs = True
        keep = []
        for value in subset.indices:
            index = int(value)
            identity = barcodes[index]
            if identity not in assignments:
                continue
            if identity in seen:
                raise ValueError("A test cell appears more than once")
            parent = assignments[identity]
            if parent not in lookup:
                raise ValueError("A frozen parent is absent from the model input")
            split = dataset._find_split_for_idx(index)
            if split not in strategy.split_control_mapping:
                raise ValueError("A frozen cell lacks a registered split")
            strategy.split_control_mapping[split][index] = [lookup[parent]]
            seen.add(identity)
            keep.append(index)
        subset.indices = keep
    if seen != set(assignments):
        raise ValueError("CellLoad test rows differ from the frozen parent plan")


def check_maps(data_module, run_dir):
    files = (
        ("pert_onehot_map.pt", data_module.pert_onehot_map),
        ("cell_type_onehot_map.torch", data_module.cell_type_onehot_map),
        ("batch_onehot_map.torch", data_module.batch_onehot_map),
    )
    for filename, actual in files:
        saved = torch.load(run_dir / filename, map_location="cpu", weights_only=False)
        if set(saved) != set(actual):
            raise ValueError(f"One-hot keys differ from training: {filename}")
        if any(not torch.equal(torch.as_tensor(saved[key]), torch.as_tensor(actual[key])) for key in saved):
            raise ValueError(f"One-hot values differ from training: {filename}")


def load_model(run_dir, model_name, data_module, expected_targets):
    from state.tx.models.context_mean import ContextMeanPerturbationModel
    from state.tx.models.perturb_mean import PerturbMeanPerturbationModel
    from state.tx.models.state_transition import StateTransitionPerturbationModel

    classes = {
        "state": StateTransitionPerturbationModel,
        "perturb_mean": PerturbMeanPerturbationModel,
        "context_mean": ContextMeanPerturbationModel,
    }
    config = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    if config["model"]["name"] != model_name:
        raise ValueError("The checkpoint's configured model does not match the requested arm")
    if config["training"]["train_seed"] != 42 or config["model"]["kwargs"]["cell_set_len"] != 8:
        raise ValueError("Unexpected pilot seed or set length")
    checkpoint_path = run_dir / "checkpoints" / "final.ckpt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if model_name == "state" and checkpoint["global_step"] != 512:
        raise ValueError("STATE did not complete the frozen 512-step pilot")
    if model_name == "perturb_mean":
        if not set(expected_targets).issubset(checkpoint.get("pert_mean_offsets", {})):
            raise ValueError("PerturbMean checkpoint lacks one or more test-target offsets")
    if model_name == "context_mean" and "RPE1" not in checkpoint.get("celltype_pert_means", {}):
        raise ValueError("ContextMean checkpoint lacks the target context")
    dimensions = data_module.get_var_dims()
    kwargs = config["model"]["kwargs"]
    model_kwargs = {
        "input_dim": dimensions["input_dim"],
        "hidden_dim": kwargs["hidden_dim"],
        "gene_dim": dimensions["gene_dim"],
        "hvg_dim": dimensions["hvg_dim"],
        "output_dim": dimensions["output_dim"],
        "pert_dim": dimensions["pert_dim"],
        **kwargs,
    }
    model = classes[model_name].load_from_checkpoint(
        checkpoint_path, map_location="cpu", weights_only=False, **model_kwargs
    )
    model.eval()
    return model, config, dimensions


def check_batch(batch, plan, input_data, lookup, seen):
    targets = [str(x) for x in batch["pert_name"]]
    contexts = [str(x) for x in batch["cell_type"]]
    sources = [str(x) for x in batch["pert_cell_barcode"]]
    parents = [str(x) for x in batch["ctrl_cell_barcode"]]
    batches = [str(x) for x in batch["batch_name"]]
    n = len(sources)
    if not n or any(len(values) != n for values in (targets, contexts, parents, batches)):
        raise ValueError("Prediction batch metadata lengths differ")
    if len(set(targets)) != 1 or set(contexts) != {"RPE1"} or n > 8:
        raise ValueError("A test set mixes conditions or exceeds the frozen set length")
    if "pert_cell_counts" in batch:
        raise ValueError("Unexpected target-expression tensor in embedding-space inference")
    if len(set(sources)) != n or seen.intersection(sources):
        raise ValueError("A test cell was emitted more than once")
    if any(plan["assignments"].get(source) != parent for source, parent in zip(sources, parents, strict=True)):
        raise ValueError("A sampled parent differs from the frozen assignment")
    expected_obs = input_data.obs.loc[sources]
    if expected_obs["target_gene"].astype(str).tolist() != targets:
        raise ValueError("Prediction labels differ from input metadata")
    if expected_obs["batch_var"].astype(str).tolist() != batches:
        raise ValueError("Prediction batch metadata differ from input metadata")
    source_x = np.asarray(input_data.X[[lookup[source] for source in sources]])
    parent_x = np.asarray(input_data.X[[lookup[parent] for parent in parents]])
    if not np.array_equal(batch["pert_cell_emb"].cpu().numpy(), source_x):
        raise ValueError("Observed target expressions differ from the frozen input")
    if not np.array_equal(batch["ctrl_cell_emb"].cpu().numpy(), parent_x):
        raise ValueError("Observed parent expressions differ from the frozen input")
    pool = plan["predicted_control_parent_pool"] if targets[0] == CONTROL else plan["perturbation_parent_pool"]
    if not set(parents).issubset(pool):
        raise ValueError("Treatment and control batches use the wrong parent pool")
    if targets[0] == CONTROL and sources != parents:
        raise ValueError("Predicted control cells must use their own parent identity")
    seen.update(sources)
    return targets, sources, parents, batches


def predict(run_dir, model_name, inputs, plan_path, output):
    from cell_load.data_modules import PerturbationDataModule

    if output.exists():
        raise FileExistsError(output)
    manifest_path = inputs / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    data_path = inputs / "data" / "replogle.h5ad"
    reference_path = inputs / "independent-reference-controls.h5ad"
    checks = {
        data_path: manifest["dataset_sha256"],
        reference_path: manifest["reference_controls_sha256"],
        manifest_path: plan["input_manifest_sha256"],
    }
    for path, digest in checks.items():
        if sha256(path) != digest:
            raise ValueError(f"Frozen input hash mismatch: {path.name}")
    input_data = ad.read_h5ad(data_path)
    reference = ad.read_h5ad(reference_path)
    if not input_data.var_names.equals(reference.var_names) or input_data.n_vars != 2000:
        raise ValueError("Input and reference feature axes differ")
    if set(input_data.obs_names.astype(str)) & set(reference.obs_names.astype(str)):
        raise ValueError("Reference controls overlap model inputs")
    lookup = {str(identity): i for i, identity in enumerate(input_data.obs_names)}

    data_module = PerturbationDataModule.load_state(str(run_dir / "data_module.torch"))
    if Path(data_module.toml_config_path).resolve() != (inputs / "split.toml").resolve():
        raise ValueError("Checkpoint does not use the frozen pilot split")
    data_module.setup(stage="test")
    check_maps(data_module, run_dir)
    install_parent_mapping(data_module, plan)
    model, config, dimensions = load_model(run_dir, model_name, data_module, plan["test_targets"])
    if list(map(str, dimensions["gene_names"])) != list(map(str, input_data.var_names)):
        raise ValueError("Checkpoint feature identity or order differs from frozen HVGs")
    if dimensions["input_dim"] != 2000 or dimensions["output_dim"] != 2000:
        raise ValueError("Unexpected model input or output dimension")
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    loader = data_module.test_dataloader()
    seen = set()
    results = []
    records = []
    clipped_low = clipped_high = 0
    with torch.no_grad():
        for set_index, batch in enumerate(loader):
            targets, sources, parents, batches = check_batch(batch, plan, input_data, lookup, seen)
            prediction_batch = dict(batch)
            prediction_batch["pert_cell_emb"] = batch["ctrl_cell_emb"]
            values = model.predict_step(prediction_batch, set_index, padded=False)["preds"]
            values = values.detach().cpu().numpy().astype(np.float32)
            if values.shape != (len(sources), input_data.n_vars) or not np.isfinite(values).all():
                raise ValueError("Prediction dimensions or values are invalid")
            clipped_low += int((values < 0).sum())
            clipped_high += int((values > 14).sum())
            np.clip(values, 0, 14, out=values)
            results.append(values)
            records.extend(
                {
                    "target_gene": target,
                    "cell_type": "RPE1",
                    "batch_var": batch_name,
                    "pert_cell_barcode": source,
                    "ctrl_cell_barcode": parent,
                    "set_index": set_index,
                }
                for target, source, parent, batch_name in zip(targets, sources, parents, batches, strict=True)
            )
    if seen != set(plan["assignments"]):
        raise ValueError("Inference did not emit exactly the frozen test and control rows")
    counts = Counter(item["target_gene"] for item in records)
    expected_counts = {target: plan["cells_per_target"] for target in plan["test_targets"]}
    expected_counts[CONTROL] = len(plan["predicted_control_parent_pool"])
    if counts != expected_counts:
        raise ValueError("Predicted condition counts differ from the frozen plan")

    output.mkdir(mode=0o700, parents=True)
    pred_obs = pd.DataFrame(records)
    pred_obs.index = pd.Index(pred_obs["pert_cell_barcode"].tolist(), name="source_cell")
    pred = ad.AnnData(X=np.concatenate(results), obs=pred_obs, var=input_data.var.copy())
    treated = pred_obs.loc[pred_obs["target_gene"] != CONTROL]
    true_ids = treated["pert_cell_barcode"].tolist()
    ref_ids = plan["reference_control_ids"]
    true = input_data[true_ids].copy()
    ref = reference[ref_ids].copy()
    real = ad.concat([true, ref], axis="obs", join="inner", merge="same", index_unique=None)
    if not real.var_names.equals(pred.var_names):
        raise ValueError("Real and predicted feature order differs")
    if not real.obs_names.is_unique or real.n_obs != pred.n_obs:
        raise ValueError("Real and predicted cell counts or identities are invalid")
    real.X = np.clip(np.asarray(real.X), 0, 14)
    pred.uns["log1p"] = {"base": None}
    real.uns["log1p"] = {"base": None}
    pred_path = output / "predicted.h5ad"
    real_path = output / "real.h5ad"
    pred.write_h5ad(pred_path, compression="gzip")
    real.write_h5ad(real_path, compression="gzip")
    report = {
        "format": "epi-c3-state-replogle-pilot-prediction-v1",
        "model": model_name,
        "parent_plan_sha256": sha256(plan_path),
        "input_manifest_sha256": sha256(manifest_path),
        "checkpoint_sha256": sha256(run_dir / "checkpoints" / "final.ckpt"),
        "config_sha256": sha256(run_dir / "config.yaml"),
        "predicted_h5ad_sha256": sha256(pred_path),
        "real_h5ad_sha256": sha256(real_path),
        "conditions": len(plan["test_targets"]),
        "cells_per_condition": plan["cells_per_target"],
        "predicted_control_cells": len(plan["predicted_control_parent_pool"]),
        "independent_real_control_cells": len(ref_ids),
        "features": input_data.n_vars,
        "clip_below_zero": clipped_low,
        "clip_above_fourteen": clipped_high,
        "training_max_steps": config["training"]["max_steps"],
        "caveats": [
            "STATE predictions within a set are coupled by attention and are not independent biological replicates.",
            "Parents can recur across perturbations, but never within the same perturbation.",
            "Cross-batch parent fallbacks are recorded in the frozen parent plan.",
            "The pilot is a single two-context split and is not the full paper benchmark.",
        ],
    }
    with (output / "manifest.json").open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps(report), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--model", choices=("state", "perturb_mean", "context_mean"), required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--parent-plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    predict(args.run_dir, args.model, args.inputs, args.parent_plan, args.output)


if __name__ == "__main__":
    main()
