"""Evaluate HCT116 priors on new perturbation targets and disjoint Orion samples."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import re
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

from c3vcell.calibration import _select_effect_strength, calibrate_effect
from c3vcell.data import CONTROL, write_sample_disjoint_split
from c3vcell.epi_selection import priority_panel
from c3vcell.epigenetics import (
    build_epigenetic_bundle, feature_names, load_epigenetic_bundle, shuffle_epigenetic_bundle,
)
from c3vcell.features import gene_token_mapping
from c3vcell.integrity import file_sha256, snapshot_files
from c3vcell.orion import prepare_orion
from c3vcell.predict import _check_disk_reserve, predict_control_baseline, predict_holdout
from c3vcell.research_review import _coverage, _check_scores, compare_scores
from c3vcell.scoring import score_single_context_observable
from c3vcell.training import train


def _write_new(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def _samples(names: list[str], train_count: int, test_count: int) -> tuple[list[str], list[str]]:
    ordered = sorted(set(names))
    if min(train_count, test_count) < 1 or len(ordered) < train_count + test_count:
        raise ValueError("Not enough distinct samples for the requested disjoint split")
    return ordered[:train_count], ordered[-test_count:]


def _selection_metadata(selection_data: Path, train_samples: list[str]) -> tuple[list[str], set[str]]:
    data = ad.read_h5ad(selection_data, backed="r")
    try:
        for column in ("context", "batch"):
            if column not in data.obs or data.obs[column].isna().any():
                raise ValueError("Priority-selection data need context and batch metadata")
        if not data.obs_names.is_unique:
            raise ValueError("Selection data require unique cell identifiers")
        contexts = set(data.obs["context"].astype(str))
        samples = set(data.obs["batch"].astype(str))
        cells = set(data.obs_names.astype(str))
    finally:
        data.file.close()
    if contexts != {"HCT116"} or not samples or not samples <= set(train_samples):
        raise ValueError("Priority-selection data must use only selected HCT116 training samples")
    return sorted(samples), cells


def _selection_samples(selection_data: Path, train_samples: list[str]) -> list[str]:
    return _selection_metadata(selection_data, train_samples)[0]


def _development_record(evidence_path: Path, train_samples: list[str]) -> dict:
    evidence_sha = file_sha256(evidence_path)
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    plan_path = evidence_path.with_name("experiment-plan.json")
    plan_sha = file_sha256(plan_path)
    if evidence.get("experiment_plan_sha256") != plan_sha:
        raise ValueError("Development evidence does not match its experiment plan")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if (
        plan.get("objective") != "expr_mse_unbiased_capped"
        or "population-shift" not in plan.get("flow_targets", [])
        or plan.get("folds") != 2
        or not isinstance(plan.get("inputs_sha256"), dict)
    ):
        raise ValueError("Development evidence lacks the required training-fold protocol")
    inputs = plan["inputs_sha256"]
    sample_files = [Path(path) for path in inputs if Path(path).suffix == ".h5ad"]
    models = [Path(path) for path in inputs if Path(path).suffix == ".pt"]
    if not sample_files or not models:
        raise ValueError("Development evidence needs verifiable training-fold data and models")
    samples: set[str] = set()
    cells: set[str] = set()
    for path, digest in inputs.items():
        if file_sha256(Path(path)) != digest:
            raise ValueError(f"Development evidence input changed: {path}")
    for path in sample_files:
        file_samples, file_cells = _selection_metadata(path, train_samples)
        samples.update(file_samples)
        cells.update(file_cells)
    for path in models:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        model_cells = checkpoint.get("training_cell_ids")
        if (
            not isinstance(model_cells, list) or not model_cells
            or not all(isinstance(cell, str) for cell in model_cells)
            or not set(model_cells) <= cells or checkpoint.get("reference") is not None
        ):
            raise ValueError(f"Development checkpoint uses unverified training cells: {path}")
    for arm in ("no-prior", "hct116-epigenome"):
        result = evidence.get("results", {}).get(f"{arm}-population-shift", {})
        folds = result.get("folds", [])
        if len(folds) != plan["folds"]:
            raise ValueError(f"Development evidence lacks two folds for {arm}")
        selected, means = _select_effect_strength(
            plan["strengths"], [fold["values"] for fold in folds], 0.02, True,
        )
        recorded_means = np.asarray(result.get("mean_objective_by_strength", []), dtype=np.float64)
        if (
            selected <= 0
            or selected != result.get("selected_strength")
            or recorded_means.shape != (len(means),)
            or not np.allclose(means, recorded_means, rtol=0, atol=1e-12)
        ):
            raise ValueError(f"Development evidence does not support population-shift for {arm}")
    if file_sha256(plan_path) != plan_sha or file_sha256(evidence_path) != evidence_sha:
        raise RuntimeError("Development evidence changed while it was being read")
    return {
        "result_sha256": evidence_sha,
        "plan": str(plan_path.resolve()),
        "plan_sha256": plan_sha,
        "inputs_sha256": inputs,
        "samples": sorted(samples),
    }


def _training_targets(
    totals: pd.DataFrame,
    genes: set[str],
    excluded: set[str],
    required: tuple[str, ...],
    count: int,
    minimum_cells: int,
) -> list[str]:
    if count < len(required) or len(set(required)) != len(required):
        raise ValueError("Training panel is too small for its required targets")
    counts = dict(zip(totals["gene_target"].astype(str), totals["n_cells"]))
    if any(gene not in genes or gene in excluded or counts.get(gene, 0) < 10 for gene in required):
        raise ValueError("A required prior-linked training target has insufficient cells")
    eligible = [
        str(row.gene_target) for row in totals.itertuples(index=False)
        if row.gene_target in genes and row.gene_target not in excluded
        and row.gene_target not in required and row.n_cells >= minimum_cells
    ]
    if len(eligible) < count - len(required):
        raise ValueError("Not enough eligible training targets")
    return sorted([*required, *eligible[:count - len(required)]])


def _general_targets(
    totals: pd.DataFrame,
    genes: set[str],
    excluded: set[str],
    count: int,
    minimum_cells: int,
    seed: int,
) -> list[str]:
    eligible = sorted(
        str(row.gene_target) for row in totals.itertuples(index=False)
        if row.gene_target in genes and row.gene_target not in excluded
        and row.n_cells >= minimum_cells
    )
    if count < 2 or len(eligible) < count:
        raise ValueError("Not enough fresh targets for the general heldout panel")
    return sorted(np.random.default_rng(seed).choice(eligible, count, replace=False).tolist())


def _history(root: Path) -> tuple[set[str], list[dict]]:
    targets: set[str] = set()
    records = []
    for path in sorted(root.rglob("split.json")):
        content = json.loads(path.read_text(encoding="utf-8"))
        for field in ("train_targets", "test_targets"):
            values = content.get(field, [])
            if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
                raise ValueError(f"Invalid target history in {path}")
            targets.update(values)
        records.append({"path": str(path.resolve()), "sha256": file_sha256(path)})
    for name in ("orion-targets.txt", "orion-targets-new-panel.txt"):
        path = root / name
        if path.is_file():
            targets.update(path.read_text(encoding="utf-8").split())
            records.append({"path": str(path.resolve()), "sha256": file_sha256(path)})
    return targets, records


def _totals(groups: pd.DataFrame, samples: list[str]) -> pd.DataFrame:
    counts = (
        groups[
            groups["sample"].isin(samples) & (groups["gene_target"] != "Non-Targeting")
        ].groupby("gene_target", as_index=False)["n_cells"].sum()
    )
    return counts.sort_values(["n_cells", "gene_target"], ascending=[False, True])


def _score_panels(
    prediction_path: Path,
    reference_path: Path,
    outdir: Path,
    panels: dict[str, list[str]],
) -> dict:
    prediction = ad.read_h5ad(prediction_path)
    reference = ad.read_h5ad(reference_path)
    pred_labels = prediction.obs["target_gene"].astype(str).to_numpy()
    ref_labels = reference.obs["target_gene"].astype(str).to_numpy()
    output = {}
    for label, targets in panels.items():
        if set(targets) == set(pred_labels):
            pred_file, ref_file = prediction_path, reference_path
        else:
            directory = outdir / f"{label}-inputs"
            directory.mkdir(parents=True)
            pred = prediction[np.isin(pred_labels, targets)].copy()
            ref = reference[np.isin(ref_labels, [CONTROL, *targets])].copy()
            if set(pred.obs["target_gene"].astype(str)) != set(targets):
                raise ValueError(f"Missing prediction targets in panel {label}")
            pred_file, ref_file = directory / "prediction.h5ad", directory / "reference.h5ad"
            pred.write_h5ad(pred_file, compression="gzip")
            ref.write_h5ad(ref_file, compression="gzip")
        score = score_single_context_observable(
            pred_file, ref_file, outdir / label, context="HCT116", threads=2,
        )
        _check_scores(score["raw_metrics"], score["undefined_metrics"])
        output[label] = score
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--orion-root", type=Path, required=True)
    parser.add_argument("--groups", type=Path, required=True)
    parser.add_argument("--history-root", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--priority-selection-data", type=Path, required=True)
    parser.add_argument("--priority-selection-bundle", type=Path, required=True)
    parser.add_argument("--development-evidence", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--training-samples", type=int, default=80)
    parser.add_argument("--heldout-samples", type=int, default=29)
    parser.add_argument("--training-targets", type=int, default=96)
    parser.add_argument("--general-holdouts", type=int, default=16)
    parser.add_argument("--methylation-holdouts", type=int, default=2)
    parser.add_argument("--graph-holdouts", type=int, default=6)
    parser.add_argument("--training-min-cells", type=int, default=80)
    parser.add_argument("--general-min-cells", type=int, default=80)
    parser.add_argument("--priority-min-cells", type=int, default=40)
    parser.add_argument("--max-per-training-target", type=int, default=320)
    parser.add_argument("--max-per-heldout-target", type=int, default=160)
    parser.add_argument("--training-controls", type=int, default=5000)
    parser.add_argument("--heldout-controls", type=int, default=2000)
    parser.add_argument("--genes", type=int, default=4096)
    parser.add_argument("--pretrain-steps", type=int, default=400)
    parser.add_argument("--flow-steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20261112)
    args = parser.parse_args()
    if args.outdir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.outdir}")
    if (
        min(
            args.training_samples, args.heldout_samples, args.training_targets,
            args.general_holdouts, args.training_min_cells, args.general_min_cells,
            args.priority_min_cells, args.max_per_training_target, args.max_per_heldout_target,
            args.training_controls, args.heldout_controls, args.genes,
            args.pretrain_steps, args.flow_steps,
        ) < 1
        or min(args.methylation_holdouts, args.graph_holdouts, args.seed) < 0
        or args.methylation_holdouts + args.graph_holdouts < 2
    ):
        raise ValueError("Invalid sample-holdout experiment sizes")
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("MKL_NUM_THREADS", "2")
    torch.set_num_threads(2)
    if importlib.metadata.version("cell-eval2") != "0.22.0":
        raise ValueError("This experiment requires cell-eval2 0.22.0")
    _check_disk_reserve(args.outdir.parent)
    groups = pd.read_csv(
        args.groups, dtype={"context": str, "gene_target": str, "sample": str},
    )
    if (
        not {"context", "gene_target", "sample", "n_cells"}.issubset(groups.columns)
        or groups[["context", "gene_target", "sample", "n_cells"]].isna().any().any()
        or set(groups["context"]) != {"HCT116"}
        or groups.duplicated(["context", "gene_target", "sample"]).any()
    ):
        raise ValueError("Expected unique HCT116 target/sample passing-cell counts")
    values = pd.to_numeric(groups["n_cells"], errors="raise").to_numpy(dtype=np.float64)
    if not np.isfinite(values).all() or np.any(values < 0) or not np.equal(values, np.floor(values)).all():
        raise ValueError("Passing-cell counts must be finite nonnegative integers")
    groups["n_cells"] = values.astype(np.int64)
    train_samples, test_samples = _samples(
        groups["sample"].astype(str).tolist(), args.training_samples, args.heldout_samples,
    )
    if not all(re.fullmatch(r"HCT116_Batch[1-9][0-9]*", sample) for sample in train_samples + test_samples):
        raise ValueError("Selected Orion sample labels do not match complete HCT116 shard names")
    for names in (train_samples, test_samples):
        controls = groups[(groups["sample"].isin(names)) & (groups["gene_target"] == "Non-Targeting")]
        if set(controls["sample"]) != set(names) or (controls["n_cells"] < 3).any():
            raise ValueError("Every selected sample needs at least three passing control cells")
    metadata = args.orion_root / "metadata/gene_metadata.parquet"
    symbols, _, _ = gene_token_mapping(
        pd.read_parquet(metadata), token_column="gene_token_id", symbol_column="gene_name",
    )
    genes = set(symbols)
    historical, history_records = _history(args.history_root)
    required_training = ("SP1", "CTCF")
    train_totals, test_totals = _totals(groups, train_samples), _totals(groups, test_samples)
    development_record = _development_record(args.development_evidence, train_samples)
    selection_samples = _selection_samples(args.priority_selection_data, train_samples)
    priority_totals = test_totals[
        test_totals["gene_target"].isin(genes)
        & ~test_totals["gene_target"].isin(historical | set(required_training))
        & (test_totals["n_cells"] >= args.priority_min_cells)
    ]
    priority, neighbors, priority_record = priority_panel(
        priority_totals, args.priority_selection_data, args.priority_selection_bundle, args.spec,
        args.methylation_holdouts, additional_graph_count=args.graph_holdouts,
    )
    if not set(neighbors) <= genes:
        raise ValueError("A prespecified network neighbor is absent from the source gene axis")
    general = _general_targets(
        test_totals, genes, historical | set(priority) | set(required_training),
        args.general_holdouts, args.general_min_cells, args.seed,
    )
    heldout = sorted([*priority, *general])
    training = _training_targets(
        train_totals, genes, set(heldout), required_training,
        args.training_targets, args.training_min_cells,
    )
    panels = {"all": heldout, "general": general, "prior-enriched": sorted(priority)}
    train_shards = [args.orion_root / "data" / f"{sample}.parquet" for sample in train_samples]
    test_shards = [args.orion_root / "data" / f"{sample}.parquet" for sample in test_samples]
    shards = train_shards + test_shards
    if not all(path.is_file() and path.suffix == ".parquet" for path in shards):
        raise ValueError("Every selected sample requires a complete local parquet shard")
    shard_records = [
        {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": file_sha256(path)}
        for path in shards
    ]
    args.outdir.mkdir(parents=True)
    project = Path(__file__).resolve().parents[1]
    code = sorted((project / "src/c3vcell").rglob("*.py"))
    code.extend([project / "pyproject.toml", Path(__file__).resolve()])
    code_hashes = {str(path.relative_to(project)): file_sha256(path) for path in code}
    code_snapshot_sha = snapshot_files(project, code_hashes, args.outdir / "code-snapshot.zip")
    fixed_inputs = {
        str(path.resolve()): file_sha256(path)
        for path in (args.groups, metadata, args.spec)
    }
    fixed_inputs[str(args.development_evidence.resolve())] = development_record["result_sha256"]
    fixed_inputs[development_record["plan"]] = development_record["plan_sha256"]
    fixed_inputs.update(development_record["inputs_sha256"])
    fixed_inputs.update({entry["path"]: entry["sha256"] for entry in history_records})
    plan = {
        "selection": (
            "Disjoint context/sample labels; fresh test targets selected solely from prior "
            "metadata and passing-cell counts. Historical targets may be used for training."
        ),
        "train_samples": train_samples,
        "test_samples": test_samples,
        "train_targets": training,
        "test_targets": heldout,
        "panels": panels,
        "priority_selection": priority_record,
        "priority_selection_samples": selection_samples,
        "development_evidence": development_record,
        "previous_target_count": len(historical),
        "history_manifests": history_records,
        "shards": shard_records,
        "fixed_input_sha256": fixed_inputs,
        "source_limits": {
            "max_per_training_target": args.max_per_training_target,
            "max_per_heldout_target": args.max_per_heldout_target,
            "training_controls": args.training_controls,
            "heldout_controls": args.heldout_controls,
            "max_genes": args.genes,
        },
        "training": {
            "latent_dim": 64, "hidden_dim": 128, "batch_size": 48,
            "pretrain_steps": args.pretrain_steps, "flow_steps": args.flow_steps,
            "flow_target": "population-shift", "min_group_cells": 5,
            "variance_weight": 1.0, "covariance_weight": 0.2,
            "regularization_view": "unmasked", "seed": args.seed,
        },
        "ablation": ["no-prior", "hct116-epigenome", "graph-only-shuffle"],
        "calibration": {
            "folds": 2, "strengths": [0.0, 0.1, 0.25, 0.5, 1.0],
            "objective": "expr_mse_unbiased_capped",
            "objective_note": "Unnormalized, unscored training-fold diagnostic.",
            "require_every_fold_improvement": True, "min_relative_improvement": 0.02,
            "cells_per_target": 400, "prediction_steps": 8, "sampling": "residual-rounding",
            "control_sampling": "input-controls",
        },
        "runtime": {
            "python": sys.version, "torch": torch.__version__,
            "cell_eval2": importlib.metadata.version("cell-eval2"),
        },
        "code_sha256": code_hashes,
        "code_snapshot_sha256": code_snapshot_sha,
        "limitations": [
            "Separate Orion sample labels are not a separate dataset or proof of biological replication.",
            "The prior-enriched panel is deliberately nonrepresentative.",
            "The reduced gene axis is selected from outer training counts and is reused in inner folds.",
            "Bulk epigenomic priors are unpaired, and graph edges are candidate associations.",
            "The graph shuffle is constrained and not uniform over all possible networks.",
            "Raw local metrics and diagnostics are not official leaderboard scores.",
        ],
    }
    plan_path = args.outdir / "experiment-plan.json"
    _write_new(plan_path, plan)
    plan_sha = file_sha256(plan_path)
    dev_subset = args.outdir / "training-subset.h5ad"
    test_subset = args.outdir / "heldout-subset.h5ad"
    prepare_orion(
        train_shards, metadata, dev_subset, targets=training, control_label="Non-Targeting",
        max_per_target=args.max_per_training_target, max_controls=args.training_controls,
        seed=args.seed,
    )
    _check_disk_reserve(args.outdir)
    prepare_orion(
        test_shards, metadata, test_subset, targets=heldout, control_label="Non-Targeting",
        max_per_target=args.max_per_heldout_target, max_controls=args.heldout_controls,
        seed=args.seed + 1,
    )
    for entry, path in zip(shard_records, shards):
        if path.stat().st_size != entry["bytes"] or file_sha256(path) != entry["sha256"]:
            raise RuntimeError(f"An Orion source shard changed during preparation: {path}")
    dev = ad.read_h5ad(dev_subset)
    test = ad.read_h5ad(test_subset)
    if not set(dev.obs["batch"].astype(str)) <= set(train_samples) or not set(
        test.obs["batch"].astype(str)
    ) <= set(test_samples):
        raise ValueError("Shard sample metadata disagrees with the prespecified sample roles")
    split = args.outdir / "split"
    write_sample_disjoint_split(
        dev, test, split, seed=args.seed, max_genes=args.genes, required_genes=neighbors or None,
    )
    del dev, test
    source, reference, controls = (
        split / "train.h5ad", split / "reference.h5ad", split / "input_controls.h5ad",
    )
    split_hashes = {
        path.name: file_sha256(path) for path in (source, reference, controls, split / "split.json")
    }
    manifest = json.loads((split / "split.json").read_text(encoding="utf-8"))
    if (
        set(manifest["train_targets"]) != set(training)
        or set(manifest["test_targets"]) != set(heldout)
        or set(manifest["test_targets"]) & historical
        or {row[1] for row in manifest["train_samples"]} & {row[1] for row in manifest["test_samples"]}
    ):
        raise RuntimeError("The materialized split does not match the prespecified plan")
    real, randomized = args.outdir / "epigenome", args.outdir / "graph-control"
    build_epigenetic_bundle(source, args.spec, real)
    shuffle_epigenetic_bundle(real, source, randomized, seed=args.seed, shuffle_features=False)
    train_data = ad.read_h5ad(source, backed="r")
    try:
        coverage = _coverage(real, heldout, train_data)
        _, real_masks, real_metadata = load_epigenetic_bundle(real, train_data)
        _, random_masks, random_metadata = load_epigenetic_bundle(randomized, train_data)
        gene_indices = {gene: index for index, gene in enumerate(train_data.var_names.astype(str))}
    finally:
        train_data.file.close()
    if (
        real_metadata["files"]["features.csv"] != random_metadata["files"]["features.csv"]
        or not np.array_equal(real_masks, random_masks)
        or random_metadata.get("control", {}).get("kind") != "graph-target-shuffle-only"
        or random_metadata["control"].get("reference_manifest_sha256") != real_metadata["manifest_sha256"]
    ):
        raise RuntimeError("The topology-only control does not preserve prior features")
    graph = pd.read_csv(real / "graph.csv", keep_default_na=False)
    random_graph = pd.read_csv(randomized / "graph.csv", keep_default_na=False)
    if (
        graph.groupby("source").size().to_dict() != random_graph.groupby("source").size().to_dict()
        or sorted(graph.groupby("target").size()) != sorted(random_graph.groupby("target").size())
        or set(zip(graph["source"], graph["target"]))
        == set(zip(random_graph["source"], random_graph["target"]))
    ):
        raise RuntimeError("The graph-only control failed its topology checks")
    for target, expected in priority_record["prespecified_outgoing_neighbors"].items():
        actual = set(graph.loc[graph["source"] == target, "target"].astype(str))
        if not set(expected) <= actual:
            raise RuntimeError(f"Prioritized target {target} lost prespecified graph neighbors")
    methylation_columns = [
        feature_names(real_metadata).index(name)
        for name in ("promoter_methylation", "linked_element_methylation")
    ]
    for target in priority_record["methylation_prioritized_holdouts"]:
        if not np.any(real_masks[gene_indices[target], methylation_columns] == 1):
            raise RuntimeError(f"Prioritized target {target} lost its measured methylation")
    _write_new(args.outdir / "network-coverage.json", coverage)
    _check_disk_reserve(args.outdir)
    baseline = args.outdir / "control-baseline.h5ad"
    predict_control_baseline(
        reference, controls, baseline, cells_per_target=400, seed=args.seed,
        control_sampling="input-controls",
    )
    results = {
        "control-baseline": _score_panels(baseline, reference, args.outdir / "baseline-scores", panels),
    }
    for name, prior in (
        ("no-prior", None), ("hct116-epigenome", real), ("graph-only-shuffle", randomized),
    ):
        directory = args.outdir / name
        checkpoint = train(
            source, directory / "model", latent_dim=64, hidden_dim=128, batch_size=48,
            pretrain_steps=args.pretrain_steps, flow_steps=args.flow_steps,
            variance_weight=1.0, covariance_weight=0.2, regularization_view="unmasked",
            flow_target="population-shift", min_group_cells=5,
            seed=args.seed, epigenetic_bundle=prior,
        )
        calibration = calibrate_effect(
            source, checkpoint, directory / "calibration", folds=2,
            strengths=(0, 0.1, 0.25, 0.5, 1.0), min_relative_improvement=0.02,
            cells_per_target=400, prediction_steps=8, seed=args.seed, threads=2,
            control_sampling="input-controls", objective="expr_mse_unbiased_capped",
            require_every_fold_improvement=True,
        )
        selected = json.loads(calibration.read_text(encoding="utf-8"))
        result = {"calibration": selected}
        for label in ("calibrated", "diagnostic-strength-1"):
            prediction = directory / f"{label}.h5ad"
            options = {"calibration_path": calibration} if label == "calibrated" else {
                "effect_strength": 1.0
            }
            predict_holdout(
                checkpoint, reference, controls, prediction, cells_per_target=400,
                steps=8, seed=args.seed, sampling="residual-rounding",
                control_sampling="input-controls", **options,
            )
            if label == "calibrated" and selected["selected_effect_strength"] == 0:
                predicted = ad.read_h5ad(prediction)
                control_prediction = ad.read_h5ad(baseline)
                if (
                    predicted.obs["source_control_id"].tolist()
                    != control_prediction.obs["source_control_id"].tolist()
                    or (sp.csr_matrix(predicted.X) != sp.csr_matrix(control_prediction.X)).nnz
                ):
                    raise RuntimeError("A zero-strength prediction differs from its control baseline")
            result[label] = _score_panels(prediction, reference, directory / f"{label}-scores", panels)
            _check_disk_reserve(args.outdir)
        results[name] = result
        _write_new(directory / "result.json", result)
    if (
        file_sha256(plan_path) != plan_sha
        or file_sha256(args.outdir / "code-snapshot.zip") != code_snapshot_sha
        or any(file_sha256(project / path) != digest for path, digest in code_hashes.items())
        or any(file_sha256(Path(path)) != digest for path, digest in fixed_inputs.items())
        or {path.name: file_sha256(path) for path in (source, reference, controls, split / "split.json")}
        != split_hashes
        or file_sha256(Path(priority_record["selection_data"])) != priority_record["selection_data_sha256"]
        or file_sha256(Path(priority_record["bundle"]) / "manifest.json")
        != priority_record["bundle_manifest_sha256"]
        or any(
            file_sha256(Path(priority_record["bundle"]) / name) != expected
            for name, expected in priority_record["bundle_file_sha256"].items()
        )
    ):
        raise RuntimeError("Experiment code or an input changed during fitting")
    comparisons = {}
    for panel in panels:
        comparisons[panel] = compare_scores(
            results["hct116-epigenome"]["calibrated"][panel]["raw_metrics"],
            {
                "control": results["control-baseline"][panel]["raw_metrics"],
                "no_prior": results["no-prior"]["calibrated"][panel]["raw_metrics"],
                "graph_shuffle": results["graph-only-shuffle"]["calibrated"][panel]["raw_metrics"],
            },
        )
    results["provenance"] = {
        "experiment_plan_sha256": plan_sha,
        "split_sha256": split_hashes,
        "sample_and_target_disjoint": True,
        "external_dataset_validation": False,
        "official_leaderboard_score": False,
        "limitations": plan["limitations"],
    }
    results["comparisons"] = comparisons
    _write_new(args.outdir / "result.json", results)


if __name__ == "__main__":
    main()
