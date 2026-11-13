"""Verify CellLoad's actual few-shot split for the frozen Replogle pilot."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import anndata as ad
from cell_load.data_modules import PerturbationDataModule


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    manifest = json.loads((args.inputs / "manifest.json").read_text(encoding="utf-8"))
    data_path = args.inputs / "data" / "replogle.h5ad"
    controls_path = args.inputs / "independent-reference-controls.h5ad"
    split_path = args.inputs / "split.toml"
    for path, key in (
        (args.plan, "plan_sha256"), (data_path, "dataset_sha256"),
        (controls_path, "reference_controls_sha256"), (split_path, "split_sha256"),
    ):
        if sha256(path) != manifest[key]:
            raise ValueError(f"{key} does not match the frozen input manifest")
    data = ad.read_h5ad(data_path, backed="r")
    controls = ad.read_h5ad(controls_path, backed="r")
    if not data.var_names.equals(controls.var_names) or data.n_vars != 2000:
        raise ValueError("Reference controls and input data do not share the frozen HVG axis")
    if set(data.obs_names) & set(controls.obs_names):
        raise ValueError("Reference controls overlap any model input")
    if set(controls.obs["role"].astype(str)) != {"reference_control"}:
        raise ValueError("Reference-control file contains unexpected rows")
    dm = PerturbationDataModule(
        toml_config_path=str(split_path), batch_size=4, cell_sentence_len=8,
        num_workers=0, random_seed=42, pert_col="target_gene", batch_col="batch_var",
        cell_type_key="cell_type", control_pert="non-targeting", embed_key="X_hvg",
        output_space="embedding", downsample_cells=None, basal_mapping_strategy="random",
        should_yield_control_cells=True,
    )
    dm.setup(stage="fit")
    summary = {}
    seen_noncontrol = {}
    for split, subsets in (("train", dm.train_datasets), ("val", dm.val_datasets), ("test", dm.test_datasets)):
        indexes = [int(i) for subset in subsets for i in subset.indices]
        if len(set(indexes)) != len(indexes):
            raise ValueError(f"Duplicated source row within {split}")
        frame = data.obs.iloc[indexes]
        noncontrol = frame[frame["target_gene"].astype(str) != "non-targeting"]
        allowed_roles = {"train"} if split == "train" else {split}
        if not set(noncontrol["role"].astype(str)).issubset(allowed_roles):
            raise ValueError(f"Misassigned perturbation cells in {split}")
        if split != "train" and set(noncontrol["cell_type"].astype(str)) != {"RPE1"}:
            raise ValueError(f"Unexpected validation or test context in {split}")
        if split != "train" and set(noncontrol["target_gene"].astype(str)) != set(plan["targets"][split]):
            raise ValueError(f"Wrong few-shot perturbations in {split}")
        counts = Counter(noncontrol["target_gene"].astype(str))
        if split != "train" and set(counts.values()) != {plan["cells_per_target_context"]}:
            raise ValueError(f"Unequal perturbation cell counts in {split}")
        seen_noncontrol[split] = set(noncontrol.index.astype(str))
        summary[split] = {
            "cells_including_shared_model_controls": len(frame),
            "noncontrol_cells": len(noncontrol),
            "contexts": {str(k): int(v) for k, v in noncontrol["cell_type"].value_counts().items()},
            "perturbation_count": len(counts),
        }
    if any(seen_noncontrol[a] & seen_noncontrol[b] for a, b in (("train", "val"), ("train", "test"), ("val", "test"))):
        raise ValueError("Perturbation cells overlap across splits")
    shared = dm.get_shared_perturbations()
    if not set(plan["targets"]["test"]).issubset(shared):
        raise ValueError("A held-out RPE1 perturbation is absent from other-context training")
    result = {
        "format": "epi-c3-state-replogle-pilot-split-audit-v1",
        "status": "complete",
        "plan_sha256": sha256(args.plan),
        "inputs_manifest_sha256": sha256(args.inputs / "manifest.json"),
        "hvg_count": data.n_vars,
        "independent_reference_controls": controls.n_obs,
        "splits": summary,
        "caveats": [
            "CellLoad shares model-input controls across its few-shot splits; separate reference controls were excluded from the input file.",
            "The pilot covers only two cell lines and one 40-perturbation split.",
            "Cross-source overlap with Epi-C3 foundation pretraining has not been audited.",
        ],
    }
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    data.file.close()
    controls.file.close()
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
