"""Freeze and prepare a small two-context Replogle STATE pipeline pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import shutil

import h5py
import numpy as np


CONTROL = "non-targeting"
SEED = 20261113
TARGETS = 40
CELLS_PER_TARGET = 100
HVG_COUNT = 2000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_strings(node: h5py.Dataset | h5py.Group) -> list[str]:
    if isinstance(node, h5py.Dataset):
        return node.asstr()[:].tolist()
    categories = node["categories"].asstr()[:]
    codes = node["codes"][:]
    if np.any(codes < 0):
        raise ValueError(f"{node.name} contains missing categorical values")
    return categories[codes].tolist()


def observed_strings(node: h5py.Dataset | h5py.Group) -> set[str]:
    if isinstance(node, h5py.Dataset):
        return set(node.asstr()[:].tolist())
    categories = node["categories"].asstr()[:]
    codes = np.unique(node["codes"][:])
    if np.any(codes < 0):
        raise ValueError(f"{node.name} contains missing categorical values")
    return set(categories[codes].tolist())


def load_metadata(path: Path, expected_context: str) -> dict:
    with h5py.File(path, "r") as f:
        matrix = f["X"]
        if not isinstance(matrix, h5py.Dataset) or len(matrix.shape) != 2 or not matrix.chunks:
            raise ValueError("This pilot requires a chunked, dense two-dimensional X")
        obs = f["obs"]
        gene = obs["gene"]
        if not isinstance(gene, h5py.Group):
            raise ValueError("Expected categorical perturbation labels")
        labels = gene["categories"].asstr()[:].tolist()
        codes = gene["codes"][:].astype(np.int32)
        if np.any(codes < 0) or CONTROL not in labels:
            raise ValueError("Missing perturbation labels or non-targeting controls")
        batches = np.asarray(obs["batch"][:])
        nperts = np.asarray(obs["nperts"][:])
        totals = np.asarray(obs["ncounts"][:], dtype=np.float64)
        line = observed_strings(obs["cell_line"])
        organism = observed_strings(obs["organism"])
        if line != {expected_context} or organism != {"human"}:
            raise ValueError("A source has unexpected organism or cell-line identity")
        if not (len(codes) == len(batches) == len(nperts) == len(totals) == matrix.shape[0]):
            raise ValueError("Observation metadata lengths do not match X")
        if not np.all(np.isfinite(totals)) or np.any(totals <= 0):
            raise ValueError("Missing or nonpositive cell library sizes")
        symbols = read_strings(f["var/gene_name"])
        ids = read_strings(f["var/ensembl_id"])
        if len(symbols) != matrix.shape[1] or len(set(symbols)) != len(symbols):
            raise ValueError("Source gene symbols must be unique and match X")
        if len(ids) != len(symbols) or len(set(ids)) != len(ids) or not all(ids):
            raise ValueError("Source Ensembl identifiers must be unique and nonempty")
        control_code = labels.index(CONTROL)
        control_mask = (codes == control_code) & (nperts == 0)
        target_mask = (codes != control_code) & (nperts == 1)
        if np.any((codes == control_code) != (nperts == 0)):
            raise ValueError("Control labels and perturbation counts disagree")
        counts = np.bincount(codes[target_mask], minlength=len(labels))
        return {
            "n_rows": int(matrix.shape[0]),
            "n_genes": int(matrix.shape[1]),
            "chunk_rows": int(matrix.chunks[0]),
            "symbols": symbols,
            "ids": ids,
            "labels": labels,
            "codes": codes,
            "batches": batches,
            "nperts": nperts,
            "totals": totals,
            "control_mask": control_mask,
            "target_mask": target_mask,
            "counts": {label: int(counts[i]) for i, label in enumerate(labels)},
        }


def shared_axis(a: dict, b: dict) -> list[tuple[str, str]]:
    left = dict(zip(a["symbols"], a["ids"], strict=True))
    right = dict(zip(b["symbols"], b["ids"], strict=True))
    shared = sorted((name, left[name]) for name in left.keys() & right.keys() if left[name] == right[name])
    if len(shared) < HVG_COUNT:
        raise ValueError("Not enough identically named and identified shared genes")
    return shared


def select_targets(a: dict, b: dict, symbols: set[str], seed: int = SEED) -> dict[str, list[str]]:
    candidates = sorted(
        name for name in a["counts"].keys() & b["counts"].keys() & symbols
        if name != CONTROL and a["counts"][name] >= CELLS_PER_TARGET
        and b["counts"][name] >= CELLS_PER_TARGET
    )
    if len(candidates) < TARGETS:
        raise ValueError("Not enough shared, sufficiently represented target genes")
    selected = random.Random(seed).sample(candidates, TARGETS)
    return {
        "train": sorted(selected[:10]),
        "val": sorted(selected[10:12]),
        "test": sorted(selected[12:]),
        "eligible_count": len(candidates),
    }


def sample_records(metadata: dict, context: str, targets: dict, rng: random.Random) -> list[dict]:
    labels = metadata["labels"]
    codes = metadata["codes"]
    batches = metadata["batches"]
    roles = {target: role for role in ("train", "val", "test") for target in targets[role]}
    selected = []
    for label in sorted(roles):
        code = labels.index(label)
        available = np.flatnonzero((codes == code) & metadata["target_mask"])
        if len(available) < CELLS_PER_TARGET:
            raise ValueError(f"Too few eligible cells for {context}/{label}")
        for row in sorted(rng.sample(available.tolist(), CELLS_PER_TARGET)):
            selected.append({
                "row": row, "context": context, "target": label,
                "batch": str(batches[row]), "library_size": float(metadata["totals"][row]),
                "role": "train" if context == "K562" else roles[label],
            })
    relevant_batches = sorted({batches[item["row"]] for item in selected})
    for batch in relevant_batches:
        candidates = np.flatnonzero(metadata["control_mask"] & (batches == batch)).tolist()
        required = 20 if context == "RPE1" else 10
        if len(candidates) < required:
            raise ValueError(f"Too few matched controls in {context} batch {batch}")
        sampled = rng.sample(candidates, required)
        for position, row in enumerate(sampled):
            role = "reference_control" if context == "RPE1" and position >= 10 else "train_control"
            selected.append({
                "row": row, "context": context, "target": CONTROL, "batch": str(batch),
                "library_size": float(metadata["totals"][row]), "role": role,
            })
    selected.sort(key=lambda item: item["row"])
    if len({item["row"] for item in selected}) != len(selected):
        raise ValueError("A source cell was selected more than once")
    return selected


def create_plan(k562: Path, rpe1: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError(output)
    if shutil.disk_usage(output.parent).free < 201 * 1024**3:
        raise OSError("Insufficient space above the 200 GiB reserve")
    metadata = {name: load_metadata(path, name) for name, path in (("K562", k562), ("RPE1", rpe1))}
    axis = shared_axis(metadata["K562"], metadata["RPE1"])
    targets = select_targets(metadata["K562"], metadata["RPE1"], {symbol for symbol, _ in axis})
    rng = random.Random(SEED + 1)
    records = {name: sample_records(metadata[name], name, targets, rng) for name in ("K562", "RPE1")}
    sources = {}
    for name, path in (("K562", k562), ("RPE1", rpe1)):
        before = path.stat()
        digest = sha256(path)
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError(f"{name} source changed while hashing")
        sources[name] = {
            "filename": path.name, "sha256": digest, "size": after.st_size,
            "mtime_ns": after.st_mtime_ns, "rows": metadata[name]["n_rows"],
        }
    plan = {
        "format": "epi-c3-state-replogle-pilot-plan-v1",
        "scope": "Two-context feasibility pilot; not a paper Figure 2 reproduction",
        "seed": SEED, "targets": targets, "cells_per_target_context": CELLS_PER_TARGET,
        "features": axis, "sources": sources, "records": records,
        "limitations": [
            "Only two cell lines and a single 40-target split are included.",
            "The feature and sample subset differs from the paper's complete benchmark.",
            "This is RNA-only STATE feasibility, not either new Epi-C3 model.",
            "Cross-source overlap with Epi-C3 foundation pretraining has not been audited.",
        ],
    }
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(plan, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({
        "plan_sha256": sha256(output), "shared_genes": len(axis),
        "eligible_shared_targets": targets["eligible_count"],
        "selected_targets": {role: len(targets[role]) for role in ("train", "val", "test")},
        "selected_rows": {name: len(rows) for name, rows in records.items()},
    }), flush=True)


def read_selected_counts(path: Path, records: list[dict], features: list[list[str]]) -> np.ndarray:
    if not records or len({row["row"] for row in records}) != len(records):
        raise ValueError("Expected unique selected source rows")
    ordered = np.asarray([row["row"] for row in records], dtype=np.int64)
    if not np.all(ordered[1:] > ordered[:-1]):
        raise ValueError("Source rows must be in strictly ascending order")
    with h5py.File(path, "r") as f:
        x = f["X"]
        symbols = read_strings(f["var/gene_name"])
        ids = read_strings(f["var/ensembl_id"])
        positions = {key: i for i, key in enumerate(zip(symbols, ids, strict=True))}
        columns = np.asarray([positions[tuple(feature)] for feature in features], dtype=np.int64)
        out = np.empty((len(records), len(features)), dtype=np.float32)
        chunks = ordered // x.chunks[0]
        unique = np.unique(chunks)
        for number, chunk in enumerate(unique):
            selected = np.flatnonzero(chunks == chunk)
            start = int(chunk * x.chunks[0])
            end = min(start + x.chunks[0], x.shape[0])
            block = x[start:end, :]
            out[selected] = block[ordered[selected] - start][:, columns]
            if (number + 1) % 100 == 0 or number + 1 == len(unique):
                print(f"{path.name}: read {number + 1}/{len(unique)} relevant row chunks", flush=True)
    if not np.isfinite(out).all() or np.any(out < 0) or np.any(out != np.floor(out)):
        raise ValueError("Selected expression data are not finite nonnegative integer counts")
    totals = np.asarray([item["library_size"] for item in records], dtype=np.float64)
    if np.any(out.sum(axis=1, dtype=np.float64) > totals + 0.01):
        raise ValueError("Shared-feature counts exceed original per-cell library size")
    out *= (10_000.0 / totals).astype(np.float32)[:, None]
    np.log1p(out, out=out)
    return out


def prepare(plan_path: Path, k562: Path, rpe1: Path, output: Path) -> None:
    import anndata as ad
    import pandas as pd
    import scanpy as sc

    if output.exists():
        raise FileExistsError(output)
    if shutil.disk_usage(output.parent).free < 201 * 1024**3:
        raise OSError("Insufficient space above the 200 GiB reserve")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan.get("format") != "epi-c3-state-replogle-pilot-plan-v1":
        raise ValueError("Unexpected frozen plan format")
    for name, path in (("K562", k562), ("RPE1", rpe1)):
        expected = plan["sources"][name]
        stat = path.stat()
        if (path.name, stat.st_size, stat.st_mtime_ns) != (
            expected["filename"], expected["size"], expected["mtime_ns"]
        ):
            raise ValueError(f"{name} source changed after plan freezing")
    output.mkdir(mode=0o700)
    matrices, records = [], []
    for name, path in (("K562", k562), ("RPE1", rpe1)):
        sampled = plan["records"][name]
        matrices.append(read_selected_counts(path, sampled, plan["features"]))
        records.extend(sampled)
    matrix = np.concatenate(matrices, axis=0)
    del matrices
    obs = pd.DataFrame(records)
    obs.index = [f"{context}:{row}" for context, row in zip(obs["context"], obs["row"], strict=True)]
    if not obs.index.is_unique:
        raise ValueError("Selected source identities are not unique")
    train_mask = obs["role"].isin(["train", "train_control"]).to_numpy()
    features = [name for name, _ in plan["features"]]
    train = ad.AnnData(X=matrix[train_mask], var=pd.DataFrame(index=features))
    train.uns["log1p"] = {"base": None}
    sc.pp.highly_variable_genes(train, n_top_genes=HVG_COUNT, flavor="seurat")
    selected = np.flatnonzero(train.var["highly_variable"].to_numpy())
    if len(selected) != HVG_COUNT:
        raise ValueError(f"Expected exactly {HVG_COUNT} training-only HVGs; got {len(selected)}")
    names = [features[i] for i in selected]
    del train
    matrix = matrix[:, selected]
    obs["target_gene"] = obs["target"]
    obs["cell_type"] = obs["context"]
    obs["batch_var"] = obs["context"] + ":" + obs["batch"]
    dataset_mask = obs["role"] != "reference_control"
    data = ad.AnnData(X=matrix[dataset_mask], obs=obs.loc[dataset_mask].copy(), var=pd.DataFrame(index=names))
    data.obsm["X_hvg"] = data.X.copy()
    data.uns["log1p"] = {"base": None}
    reference = ad.AnnData(X=matrix[~dataset_mask], obs=obs.loc[~dataset_mask].copy(), var=pd.DataFrame(index=names))
    reference.uns["log1p"] = {"base": None}
    data_dir = output / "data"
    data_dir.mkdir()
    data.write_h5ad(data_dir / "replogle.h5ad", compression="gzip")
    reference.write_h5ad(output / "independent-reference-controls.h5ad", compression="gzip")
    split = (
        f'[datasets]\nreplogle = "{data_dir}"\n\n[training]\nreplogle = "train"\n\n'
        '[fewshot."replogle.RPE1"]\n'
        f'val = {json.dumps(plan["targets"]["val"])}\n'
        f'test = {json.dumps(plan["targets"]["test"])}\n'
    )
    (output / "split.toml").write_text(split, encoding="utf-8", newline="\n")
    manifest = {
        "format": "epi-c3-state-replogle-pilot-inputs-v1",
        "status": "complete",
        "plan_sha256": sha256(plan_path),
        "dataset_sha256": sha256(data_dir / "replogle.h5ad"),
        "reference_controls_sha256": sha256(output / "independent-reference-controls.h5ad"),
        "split_sha256": sha256(output / "split.toml"),
        "cells": int(data.n_obs),
        "independent_reference_controls": int(reference.n_obs),
        "feature_count": len(names),
        "role_counts": {str(k): int(v) for k, v in obs["role"].value_counts().items()},
        "normalization": "log1p(raw counts / original measured-library total * 10000)",
        "hvg_selection": "scanpy seurat flavor on train and train-control rows only",
        "limitations": plan["limitations"],
    }
    with (output / "manifest.json").open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(manifest, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(manifest), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("plan", "prepare"))
    parser.add_argument("--k562", required=True, type=Path)
    parser.add_argument("--rpe1", required=True, type=Path)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.phase == "plan":
        if args.output is not None:
            parser.error("--output is only valid for prepare")
        create_plan(args.k562, args.rpe1, args.plan)
    else:
        if args.output is None:
            parser.error("--output is required for prepare")
        prepare(args.plan, args.k562, args.rpe1, args.output)


if __name__ == "__main__":
    main()
