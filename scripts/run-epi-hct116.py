"""Prespecified, bounded HCT116 epigenomic ablation on completed Orion shards."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from c3vcell.calibration import CALIBRATION_OBJECTIVES, OBJECTIVE, calibrate_effect
from c3vcell.data import CONTROL, write_split
from c3vcell.epi_selection import priority_panel
from c3vcell.epigenetics import (
    build_epigenetic_bundle, load_epigenetic_bundle, shuffle_epigenetic_bundle,
)
from c3vcell.integrity import file_sha256, snapshot_files
from c3vcell.orion import prepare_orion
from c3vcell.predict import predict_control_baseline, predict_holdout
from c3vcell.research_review import _coverage
from c3vcell.scoring import score_local, score_single_context_observable
from c3vcell.training import train


def _write_new(path: Path, record: dict) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
        stream.write("\n")


def _priority_panel(
    totals: pd.DataFrame, selection_data: Path, bundle: Path, spec: Path, count: int,
) -> tuple[list[str], list[str], dict]:
    return priority_panel(totals, selection_data, bundle, spec, count, loader=load_epigenetic_bundle)


def _score_observable(prediction: Path, reference: Path, outdir: Path) -> dict:
    return score_single_context_observable(
        prediction, reference, outdir, context="HCT116", threads=2, scorer=score_local,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--orion-root", type=Path, required=True)
    parser.add_argument("--groups", type=Path, required=True)
    parser.add_argument("--history-root", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--targets", type=int, default=32)
    parser.add_argument("--holdouts", type=int, default=8)
    parser.add_argument("--max-per-target", type=int, default=80)
    parser.add_argument("--max-controls", type=int, default=1600)
    parser.add_argument("--genes", type=int, default=4096)
    parser.add_argument("--pretrain-steps", type=int, default=100)
    parser.add_argument("--flow-steps", type=int, default=150)
    parser.add_argument("--prediction-cells", type=int, default=400)
    parser.add_argument("--calibration-cells", type=int, default=400)
    parser.add_argument("--variance-weight", type=float, default=0.1)
    parser.add_argument("--covariance-weight", type=float, default=0.0)
    parser.add_argument("--regularization-view", choices=["masked", "unmasked"], default="masked")
    parser.add_argument("--calibration-objective", choices=CALIBRATION_OBJECTIVES, default=OBJECTIVE)
    parser.add_argument("--require-every-fold-improvement", action="store_true")
    parser.add_argument("--priority-selection-data", type=Path)
    parser.add_argument("--priority-selection-bundle", type=Path)
    parser.add_argument("--prioritized-holdouts", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    if args.outdir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.outdir}")
    if (
        min(
            args.samples, args.targets, args.holdouts, args.max_per_target, args.max_controls,
            args.genes, args.pretrain_steps, args.flow_steps, args.prediction_cells,
            args.calibration_cells,
        ) < 1
        or args.holdouts < 2 or args.targets - args.holdouts < 4
        or args.seed < 0
        or not math.isfinite(args.variance_weight) or args.variance_weight < 0
        or not math.isfinite(args.covariance_weight) or args.covariance_weight < 0
        or not 0 <= args.prioritized_holdouts <= args.holdouts
        or bool(args.prioritized_holdouts) != (
            args.priority_selection_data is not None and args.priority_selection_bundle is not None
        )
        or (args.prioritized_holdouts == 0 and (
            args.priority_selection_data is not None or args.priority_selection_bundle is not None
        ))
    ):
        raise ValueError("Invalid experiment sizes or seed")
    if args.calibration_cells != args.prediction_cells:
        raise ValueError("Calibration and prediction cells per target must match")
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("MKL_NUM_THREADS", "2")
    import torch

    torch.set_num_threads(2)
    scorer_version = importlib.metadata.version("cell-eval2")
    if scorer_version != "0.22.0":
        raise ValueError(f"Expected cell-eval2 0.22.0, found {scorer_version}")
    groups = pd.read_csv(
        args.groups, dtype={"context": str, "gene_target": str, "sample": str},
    )
    if (
        not {"context", "gene_target", "sample", "n_cells"}.issubset(groups.columns)
        or groups[["context", "gene_target", "sample", "n_cells"]].isna().any().any()
        or set(groups["context"]) != {"HCT116"} or (groups["n_cells"] < 0).any()
    ):
        raise ValueError("The target-count table must contain valid HCT116 group counts")
    samples = sorted(groups["sample"].unique())[:args.samples]
    if len(samples) != args.samples:
        raise ValueError("Not enough completed HCT116 samples")
    selected_groups = groups[groups["sample"].isin(samples)].copy()
    controls = selected_groups[selected_groups["gene_target"] == "Non-Targeting"]
    if set(controls["sample"]) != set(samples) or (controls["n_cells"] < 3).any():
        raise ValueError("Each selected sample needs at least three passing control cells")
    shards = [args.orion_root / "data" / f"{sample}.parquet" for sample in samples]
    if not all(path.is_file() for path in shards):
        raise ValueError("Every selected shard must be a complete local parquet file")
    metadata = args.orion_root / "metadata" / "gene_metadata.parquet"
    gene_metadata = pd.read_parquet(metadata, columns=["gene_name"])
    genes = set(gene_metadata["gene_name"].dropna().astype(str))
    historical = set()
    history_records = []
    for path in sorted(args.history_root.rglob("split.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        for field in ("train_targets", "test_targets"):
            historical.update(record.get(field, []))
        history_records.append({"path": str(path), "sha256": file_sha256(path)})
    for name in ("orion-targets.txt", "orion-targets-new-panel.txt"):
        path = args.history_root / name
        if path.is_file():
            historical.update(path.read_text(encoding="utf-8").split())
            history_records.append({"path": str(path), "sha256": file_sha256(path)})
    totals = (
        selected_groups[selected_groups["gene_target"] != "Non-Targeting"]
        .groupby("gene_target", as_index=False)["n_cells"].sum()
    )
    totals = totals.loc[
        totals["gene_target"].isin(genes)
        & ~totals["gene_target"].isin(historical | {"SP1"})
        & (totals["n_cells"] >= 20)
    ].sort_values(["n_cells", "gene_target"], ascending=[False, True])
    if len(totals) < args.targets:
        raise ValueError("Not enough previously unlisted targets with at least 20 passing cells")
    priority = []
    required_genes = []
    priority_record = None
    if args.prioritized_holdouts:
        priority, required_genes, priority_record = _priority_panel(
            totals, args.priority_selection_data, args.priority_selection_bundle,
            args.spec, args.prioritized_holdouts,
        )
        if not set(required_genes) <= genes:
            raise ValueError("A prespecified network neighbor is missing from Orion metadata")
    candidates = priority + [
        gene for gene in totals["gene_target"].tolist() if gene not in set(priority)
    ][:args.targets - len(priority)]
    sp1_count = int(selected_groups.loc[selected_groups["gene_target"] == "SP1", "n_cells"].sum())
    if "SP1" not in genes or sp1_count < 10:
        raise ValueError("The prespecified SP1 training target has insufficient passing cells")
    remaining = [gene for gene in candidates if gene not in set(priority)]
    random_holdouts = np.random.default_rng(args.seed).choice(
        remaining, args.holdouts - len(priority), replace=False,
    ).tolist()
    holdouts = sorted(priority + random_holdouts)
    targets = sorted(candidates + ["SP1"])
    shard_records = [
        {"path": str(path), "bytes": path.stat().st_size, "sha256": file_sha256(path)}
        for path in shards
    ]
    plan = {
        "selection": (
            "HCT116 passing-cell counts and gene availability only; exclude targets in recorded "
            "historical split manifests; SP1 is included deliberately in training. "
            "Optional priority holdouts additionally use only prior methylation masks and graph edges."
        ),
        "priority_selection": priority_record,
        "samples": samples,
        "targets": targets,
        "heldout_targets": holdouts,
        "training_targets": sorted(set(targets) - set(holdouts)),
        "previous_target_count": len(historical),
        "history_manifests": history_records,
        "source_shards": shard_records,
        "groups_sha256": file_sha256(args.groups),
        "gene_metadata_sha256": file_sha256(metadata),
        "epigenetic_spec_sha256": file_sha256(args.spec),
        "max_per_target": args.max_per_target,
        "max_controls": args.max_controls,
        "max_genes": args.genes,
        "training": {
            "latent_dim": 64, "hidden_dim": 128, "batch_size": 48,
            "pretrain_steps": args.pretrain_steps, "flow_steps": args.flow_steps,
            "seed": args.seed, "flow_target": "paired",
            "variance_weight": args.variance_weight, "covariance_weight": args.covariance_weight,
            "regularization_view": args.regularization_view,
        },
        "ablation": ["no-prior", "hct116-epigenome", "shuffled-epigenome"],
        "calibration": {
            "folds": 2, "strengths": [0.0, 0.25, 1.0], "objective": args.calibration_objective,
            "selection": "target-disjoint training folds only", "control_sampling": "input-controls",
            "require_every_fold_improvement": args.require_every_fold_improvement,
            "cells_per_target": args.calibration_cells, "prediction_steps": 8,
            "sampling": "residual-rounding", "min_relative_improvement": 0.02,
        },
        "prediction_cells_per_target": args.prediction_cells,
        "diagnostic_strength": 1.0,
        "runtime": {
            "python": sys.version, "torch": torch.__version__, "cell_eval2": scorer_version,
        },
        "note": (
            "Raw VCC 2026 metrics are local research scores, not the official leaderboard. "
            "Aggregate HCT116 scoring may be confounded by sample/batch composition. "
            "Previously unlisted test targets do not constitute an external dataset."
        ),
    }
    args.outdir.mkdir(parents=True)
    project = Path(__file__).resolve().parents[1]
    code = sorted((project / "src" / "c3vcell").rglob("*.py"))
    code.extend([project / "pyproject.toml", Path(__file__).resolve()])
    code_hashes = {str(path.relative_to(project)): file_sha256(path) for path in code}
    plan["code_sha256"] = code_hashes
    plan["code_snapshot_sha256"] = snapshot_files(
        project, code_hashes, args.outdir / "code-snapshot.zip",
    )
    _write_new(args.outdir / "experiment-plan.json", plan)
    plan_digest = file_sha256(args.outdir / "experiment-plan.json")
    subset = args.outdir / "subset.h5ad"
    prepare_orion(
        shards, metadata, subset, targets=targets, control_label="Non-Targeting",
        max_per_target=args.max_per_target, max_controls=args.max_controls, seed=args.seed,
    )
    for entry, path in zip(shard_records, shards):
        if file_sha256(path) != entry["sha256"]:
            raise RuntimeError(f"An Orion source shard changed during preparation: {path}")
    data = ad.read_h5ad(subset)
    split = args.outdir / "split"
    write_split(
        data, split, control=CONTROL, holdout_fraction=args.holdouts / len(targets),
        holdout_targets=holdouts, max_genes=args.genes, seed=args.seed,
        required_genes=required_genes or None,
    )
    source, reference, controls_path = (
        split / "train.h5ad", split / "reference.h5ad", split / "input_controls.h5ad",
    )
    bundle, shuffled = args.outdir / "epigenome", args.outdir / "shuffled"
    build_epigenetic_bundle(source, args.spec, bundle)
    shuffle_epigenetic_bundle(bundle, source, shuffled, seed=args.seed)
    network_data = ad.read_h5ad(source, backed="r")
    try:
        coverage = _coverage(bundle, holdouts, network_data)
    finally:
        network_data.file.close()
    if priority_record is not None:
        frame = pd.read_csv(bundle / "features.csv").set_index("gene")
        edges = pd.read_csv(bundle / "graph.csv", keep_default_na=False)
        for target in priority:
            observed = (
                frame.loc[target, "promoter_methylation_mask"] == 1
                or frame.loc[target, "linked_element_methylation_mask"] == 1
            )
            actual_neighbors = set(edges.loc[edges["source"] == target, "target"])
            if not observed or not set(
                priority_record["prespecified_outgoing_neighbors"][target]
            ) <= actual_neighbors:
                raise RuntimeError(f"Prioritized target {target} lost prespecified network coverage")
    _write_new(args.outdir / "network-coverage.json", coverage)
    split_hashes = {
        path.name: file_sha256(path)
        for path in (source, reference, controls_path, split / "split.json")
    }
    baseline = args.outdir / "control-baseline.h5ad"
    predict_control_baseline(
        reference, controls_path, baseline, cells_per_target=args.prediction_cells,
        seed=args.seed, control_sampling="input-controls",
    )
    baseline_score = _score_observable(baseline, reference, args.outdir / "control-baseline-score")
    results = {
        "control-baseline": baseline_score["raw_metrics"],
        "control-baseline-undefined": baseline_score["undefined_metrics"],
        "control-baseline-expression-status": baseline_score["normalized_expression_status"],
    }
    for name, prior in (
        ("no-prior", None), ("hct116-epigenome", bundle), ("shuffled-epigenome", shuffled),
    ):
        directory = args.outdir / name
        checkpoint = train(
            source, directory / "model", latent_dim=64, hidden_dim=128, batch_size=48,
            pretrain_steps=args.pretrain_steps, flow_steps=args.flow_steps, seed=args.seed,
            epigenetic_bundle=prior, variance_weight=args.variance_weight,
            covariance_weight=args.covariance_weight, regularization_view=args.regularization_view,
        )
        calibration = calibrate_effect(
            source, checkpoint, directory / "calibration", folds=2, strengths=(0, 0.25, 1),
            min_relative_improvement=0.02, cells_per_target=args.calibration_cells,
            prediction_steps=8,
            seed=args.seed, threads=2, control_sampling="input-controls",
            objective=args.calibration_objective,
            require_every_fold_improvement=args.require_every_fold_improvement,
        )
        result = {"calibration": json.loads(calibration.read_text(encoding="utf-8"))}
        for label in ("calibrated", "diagnostic-strength-1"):
            prediction = directory / f"{label}.h5ad"
            options = (
                {"calibration_path": calibration}
                if label == "calibrated" else {"effect_strength": 1.0}
            )
            predict_holdout(
                checkpoint, reference, controls_path, prediction,
                cells_per_target=args.prediction_cells, steps=8, seed=args.seed,
                sampling="residual-rounding", control_sampling="input-controls", **options,
            )
            score = _score_observable(prediction, reference, directory / f"{label}-score")
            result[label] = score["raw_metrics"]
            result[f"{label}-undefined"] = score["undefined_metrics"]
            result[f"{label}-expression-status"] = score["normalized_expression_status"]
        results[name] = result
        _write_new(directory / "result.json", result)
    if (
        file_sha256(args.outdir / "experiment-plan.json") != plan_digest
        or file_sha256(args.outdir / "code-snapshot.zip") != plan["code_snapshot_sha256"]
        or {str(path.relative_to(project)): file_sha256(path) for path in code} != code_hashes
        or {path.name: file_sha256(path) for path in (source, reference, controls_path, split / "split.json")}
        != split_hashes
        or file_sha256(args.spec) != plan["epigenetic_spec_sha256"]
        or file_sha256(args.groups) != plan["groups_sha256"]
        or file_sha256(metadata) != plan["gene_metadata_sha256"]
    ):
        raise RuntimeError("Experiment code or an input changed during fitting")
    if priority_record is not None:
        current_bundle = Path(priority_record["bundle"])
        if (
            file_sha256(Path(priority_record["selection_data"])) != priority_record["selection_data_sha256"]
            or file_sha256(current_bundle / "manifest.json") != priority_record["bundle_manifest_sha256"]
            or any(
                file_sha256(current_bundle / name) != expected
                for name, expected in priority_record["bundle_file_sha256"].items()
            )
        ):
            raise RuntimeError("Priority-selection inputs changed during fitting")
    limitations = [
        "A target-disjoint panel from the same Orion dataset is not external validation.",
        "The shuffled prior changes features as well as graph structure.",
    ]
    if priority:
        limitations.append("Methylation-prioritized holdouts are enriched, not representative of all targets.")
    results["provenance"] = {
        "experiment_plan_sha256": plan_digest,
        "split_sha256": split_hashes,
        "target_history_exclusion": "Recorded historical train/test targets were excluded.",
        "external_dataset_validation": False,
        "official_leaderboard_score": False,
        "limitations": limitations,
    }
    _write_new(args.outdir / "result.json", results)


if __name__ == "__main__":
    main()
