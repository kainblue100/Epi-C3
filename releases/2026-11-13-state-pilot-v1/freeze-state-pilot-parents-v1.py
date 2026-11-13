"""Freeze distinct input-control parents for a two-context STATE pilot."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import random

import anndata as ad


CONTROL = "non-targeting"
SEED = 20261113


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def shuffled(values, label):
    result = sorted(values)
    random.Random(f"{SEED}:{label}").shuffle(result)
    return result


def build_parent_plan(data_obs, reference_obs, expected_targets, cells_per_target):
    if not data_obs.index.is_unique or not reference_obs.index.is_unique:
        raise ValueError("Cell identities must be unique within each file")
    if set(data_obs.index.astype(str)) & set(reference_obs.index.astype(str)):
        raise ValueError("Reference controls overlap model inputs")
    required = {"cell_type", "target_gene", "batch_var", "role"}
    if not required.issubset(data_obs.columns) or not required.issubset(reference_obs.columns):
        raise ValueError("Missing pilot metadata")
    if set(reference_obs["cell_type"].astype(str)) != {"RPE1"}:
        raise ValueError("Unexpected reference-control context")
    if set(reference_obs["target_gene"].astype(str)) != {CONTROL}:
        raise ValueError("Reference file contains noncontrol cells")
    if set(reference_obs["role"].astype(str)) != {"reference_control"}:
        raise ValueError("Unexpected reference-control role")

    data = data_obs.copy()
    ref = reference_obs.copy()
    data.index = data.index.astype(str)
    ref.index = ref.index.astype(str)
    for frame in (data, ref):
        for key in required:
            frame[key] = frame[key].astype(str)

    controls = data.loc[
        (data["cell_type"] == "RPE1")
        & (data["target_gene"] == CONTROL)
        & (data["role"] == "train_control")
    ]
    tests = data.loc[data["role"] == "test"]
    if set(tests["cell_type"]) != {"RPE1"}:
        raise ValueError("Unexpected test context")
    if set(tests["target_gene"]) != set(expected_targets):
        raise ValueError("Test labels do not match frozen targets")
    counts = Counter(tests["target_gene"])
    if not counts or set(counts.values()) != {cells_per_target}:
        raise ValueError("Test conditions do not have the frozen cell count")
    if controls.empty:
        raise ValueError("No model-input controls are available")

    pool_a = []
    pool_b = []
    chosen_reference = []
    for batch in sorted(controls["batch_var"].unique()):
        ids = shuffled(controls.index[controls["batch_var"] == batch].tolist(), f"pool:{batch}")
        if len(ids) < 2 or len(ids) % 2:
            raise ValueError(f"Expected an even control count of at least two in {batch}")
        midpoint = len(ids) // 2
        pool_a.extend(ids[:midpoint])
        pool_b.extend(ids[midpoint:])
        refs = shuffled(ref.index[ref["batch_var"] == batch].tolist(), f"reference:{batch}")
        if len(refs) < len(ids) - midpoint:
            raise ValueError(f"Too few independent reference controls in {batch}")
        chosen_reference.extend(refs[: len(ids) - midpoint])
    if set(ref["batch_var"]) != set(controls["batch_var"]):
        raise ValueError("Reference and model-control batches differ")
    if len(pool_a) < cells_per_target:
        raise ValueError("Not enough unique perturbation parents per condition")

    parents_by_batch = defaultdict(list)
    for parent in pool_a:
        parents_by_batch[data.at[parent, "batch_var"]].append(parent)

    assignments = {}
    fallback_counts = {}
    for target in sorted(expected_targets):
        rows = tests.loc[tests["target_gene"] == target]
        available = set(pool_a)
        fallback = []
        for batch in sorted(rows["batch_var"].unique()):
            target_rows = shuffled(rows.index[rows["batch_var"] == batch].tolist(), f"rows:{target}:{batch}")
            candidates = shuffled(
                (parent for parent in parents_by_batch[batch] if parent in available),
                f"parents:{target}:{batch}",
            )
            matching = min(len(target_rows), len(candidates))
            for row, parent in zip(target_rows[:matching], candidates[:matching], strict=True):
                assignments[row] = parent
                available.remove(parent)
            fallback.extend(target_rows[matching:])
        alternatives = shuffled(available, f"fallback:{target}")
        if len(alternatives) < len(fallback):
            raise ValueError(f"Insufficient unique parents for {target}")
        for row, parent in zip(fallback, alternatives[: len(fallback)], strict=True):
            assignments[row] = parent
        fallback_counts[target] = len(fallback)
        if len({assignments[row] for row in rows.index}) != len(rows):
            raise ValueError(f"Repeated perturbation parent within {target}")

    for control in pool_b:
        assignments[control] = control
    if set(pool_a) & set(pool_b):
        raise ValueError("Treatment and predicted-control parent pools overlap")
    if len(chosen_reference) != len(pool_b):
        raise ValueError("Predicted and reference control counts differ")

    return {
        "format": "epi-c3-state-replogle-pilot-parent-plan-v1",
        "seed": SEED,
        "assignments": assignments,
        "perturbation_parent_pool": sorted(pool_a),
        "predicted_control_parent_pool": sorted(pool_b),
        "reference_control_ids": sorted(chosen_reference),
        "test_targets": sorted(expected_targets),
        "cells_per_target": cells_per_target,
        "summary": {
            "test_perturbations": len(expected_targets),
            "test_cells": len(tests),
            "perturbation_parent_pool": len(pool_a),
            "predicted_controls": len(pool_b),
            "independent_reference_controls": len(chosen_reference),
            "cross_batch_parent_assignments": sum(fallback_counts.values()),
            "cross_batch_by_target": fallback_counts,
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--split-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    manifest_path = args.inputs / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    audit = json.loads(args.split_audit.read_text(encoding="utf-8"))
    frozen = json.loads(args.plan.read_text(encoding="utf-8"))
    data_path = args.inputs / "data" / "replogle.h5ad"
    reference_path = args.inputs / "independent-reference-controls.h5ad"
    checks = {
        data_path: manifest["dataset_sha256"],
        reference_path: manifest["reference_controls_sha256"],
        args.plan: manifest["plan_sha256"],
        manifest_path: audit["inputs_manifest_sha256"],
    }
    if audit["status"] != "complete":
        raise ValueError("The actual CellLoad split has not passed audit")
    for path, digest in checks.items():
        if sha256(path) != digest:
            raise ValueError(f"Input hash mismatch: {path.name}")
    data = ad.read_h5ad(data_path, backed="r")
    reference = ad.read_h5ad(reference_path, backed="r")
    if not data.var_names.equals(reference.var_names) or data.n_vars != 2000:
        raise ValueError("Reference and model-input feature axes differ")
    plan = build_parent_plan(
        data.obs, reference.obs, frozen["targets"]["test"], frozen["cells_per_target_context"]
    )
    data.file.close()
    reference.file.close()
    plan["frozen_plan_sha256"] = sha256(args.plan)
    plan["input_manifest_sha256"] = sha256(manifest_path)
    plan["split_audit_sha256"] = sha256(args.split_audit)
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(plan, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({"parent_plan_sha256": sha256(args.output), **plan["summary"]}), flush=True)


if __name__ == "__main__":
    main()
