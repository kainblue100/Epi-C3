"""Reproduce a prespecified, target-disjoint Arc nested-calibration experiment."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path

import anndata as ad
import numpy as np
import scipy.sparse as sp

from c3vcell.integrity import file_sha256, snapshot_files


def _write_new(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def _code_fingerprints() -> dict[str, str]:
    root = Path(__file__).resolve().parents[1]
    paths = sorted((root / "src" / "c3vcell").rglob("*.py"))
    paths.extend([root / "pyproject.toml", Path(__file__).resolve()])
    return {str(path.relative_to(root)): file_sha256(path) for path in paths}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--exclude-manifests", type=Path, nargs="+", required=True)
    parser.add_argument("--exclude-training-targets-manifests", type=Path, nargs="*", default=[])
    parser.add_argument("--exclude-test-training-targets-manifests", type=Path, nargs="*", default=[])
    parser.add_argument("--selection-evidence", type=Path, nargs="*", default=[])
    parser.add_argument("--selection-note", default="")
    parser.add_argument("--seed", type=int, default=20261110)
    parser.add_argument("--train-targets", type=int, default=36)
    parser.add_argument("--test-targets", type=int, default=12)
    parser.add_argument("--max-per-target", type=int, default=250)
    parser.add_argument("--max-controls", type=int, default=8000)
    parser.add_argument("--cells-per-target", type=int, default=400)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--include-null-control", action="store_true")
    parser.add_argument(
        "--control-sampling", choices=["reference-batch", "input-controls"],
        default="reference-batch",
    )
    parser.add_argument("--pretrain-steps", type=int, default=300)
    parser.add_argument("--flow-steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--latent-dim", type=int, default=96)
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--variance-weight", type=float, default=0.1)
    parser.add_argument("--covariance-weight", type=float, default=0.0)
    parser.add_argument("--regularization-view", choices=["masked", "unmasked"], default="masked")
    parser.add_argument("--prediction-steps", type=int, default=8)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.outdir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.outdir}")
    if (
        args.folds < 2 or args.test_targets < 2 or args.train_targets < 2 * args.folds
        or min(
            args.max_per_target, args.max_controls, args.cells_per_target, args.pretrain_steps,
            args.flow_steps, args.batch_size, args.latent_dim, args.hidden_dim,
            args.prediction_steps, args.threads,
        ) < 1
        or not np.isfinite(args.variance_weight) or args.variance_weight < 0
        or not np.isfinite(args.covariance_weight) or args.covariance_weight < 0
        or (args.covariance_weight > 0 and args.batch_size < 2)
    ):
        raise ValueError("Invalid target counts, fold count, or positive run parameters")
    source_digest = file_sha256(args.source)
    if source_digest != args.source_sha256:
        raise ValueError("Arc source SHA-256 does not match the supplied manifest digest")
    source_stat = args.source.stat()
    excluded_evaluation = set()
    excluded_training = set()
    excluded_test_training = set()
    exclusion_sources = []
    for path in args.exclude_manifests:
        digest = file_sha256(path)
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if file_sha256(path) != digest:
            raise RuntimeError(f"Exclusion manifest changed while being read: {path}")
        targets = manifest.get("test_targets", manifest.get("calibration_targets"))
        if not isinstance(targets, list) or not targets or not all(isinstance(t, str) for t in targets):
            raise ValueError(f"No valid held-out target list in {path}")
        excluded_evaluation.update(targets)
        exclusion_sources.append({
            "path": str(path.resolve()), "sha256": digest, "targets": sorted(targets),
            "scope": "prior-evaluation-targets",
        })
    for path in args.exclude_training_targets_manifests:
        digest = file_sha256(path)
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if file_sha256(path) != digest:
            raise RuntimeError(f"Exclusion manifest changed while being read: {path}")
        targets = manifest.get("train_targets")
        if not isinstance(targets, list) or not targets or not all(isinstance(t, str) for t in targets):
            raise ValueError(f"No valid training target list in {path}")
        excluded_training.update(targets)
        exclusion_sources.append({
            "path": str(path.resolve()), "sha256": digest, "targets": sorted(targets),
            "scope": "prior-training-targets",
        })
    for path in args.exclude_test_training_targets_manifests:
        digest = file_sha256(path)
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if file_sha256(path) != digest:
            raise RuntimeError(f"Exclusion manifest changed while being read: {path}")
        targets = manifest.get("training_targets", manifest.get("train_targets"))
        if not isinstance(targets, list) or not targets or not all(isinstance(t, str) for t in targets):
            raise ValueError(f"No valid training target list in {path}")
        excluded_test_training.update(targets)
        exclusion_sources.append({
            "path": str(path.resolve()), "sha256": digest, "targets": sorted(targets),
            "scope": "prior-training-targets-excluded-from-test-only",
        })
    excluded = excluded_evaluation | excluded_training
    selection_evidence = [
        {"path": str(path.resolve()), "sha256": file_sha256(path)}
        for path in args.selection_evidence
    ]
    source = ad.read_h5ad(args.source, backed="r")
    try:
        if "target_gene" not in source.obs or source.obs["target_gene"].isna().any():
            raise ValueError("Arc source needs non-null target_gene labels")
        counts = source.obs["target_gene"].astype(str).value_counts()
        genes = set(source.var_names.astype(str))
        eligible = sorted(
            target for target, count in counts.items()
            if target != "non-targeting" and target in genes
            and count >= args.max_per_target and target not in excluded
        )
        if int(counts.get("non-targeting", 0)) < args.max_controls:
            raise ValueError("Arc source has fewer controls than requested")
        if len(eligible) < args.train_targets + args.test_targets:
            raise ValueError("Not enough eligible targets remain after exclusions")
        shuffled = np.random.default_rng(args.seed).permutation(eligible).tolist()
        test_eligible = [target for target in shuffled if target not in excluded_test_training]
        if len(test_eligible) < args.test_targets:
            raise ValueError("Not enough test-eligible targets remain after prior-training exclusions")
        test = sorted(test_eligible[:args.test_targets])
        training = sorted(
            [target for target in shuffled if target not in set(test)][:args.train_targets]
        )
        gene_count = source.n_vars
    finally:
        source.file.close()
    out = args.outdir.resolve()
    out.mkdir(parents=True)
    fingerprints = _code_fingerprints()
    snapshot_sha256 = snapshot_files(
        Path(__file__).resolve().parents[1], fingerprints, out / "code-snapshot.zip",
    )
    logs = out / "logs"
    logs.mkdir()
    split = out / "split"
    model = out / "model" / "model.pt"
    profile = out / "profile.h5ad"
    calibration = out / "calibration" / "shared-residual-calibration.json"
    python = sys.executable
    common = [python, "-m", "c3vcell"]
    infer_seed = str(args.seed + 100_000)
    commands = [
        ("prepare", [
            *common, "prepare", "--input", str(args.source.resolve()), "--outdir", str(split),
            "--include-targets", *sorted(training + test), "--holdout-targets", *test,
            "--max-per-target", str(args.max_per_target), "--max-controls", str(args.max_controls),
            "--max-genes", "0", "--holdout-fraction", "0.25", "--seed", str(args.seed),
        ]),
        ("train", [
            *common, "train", "--train", str(split / "train.h5ad"), "--outdir", str(out / "model"),
            "--latent-dim", str(args.latent_dim), "--hidden-dim", str(args.hidden_dim),
            "--pretrain-steps", str(args.pretrain_steps), "--flow-steps", str(args.flow_steps),
            "--batch-size", str(args.batch_size), "--flow-target", "population-shift",
            "--variance-weight", str(args.variance_weight),
            "--covariance-weight", str(args.covariance_weight),
            "--regularization-view", args.regularization_view,
            "--min-group-cells", "3", "--seed", str(args.seed),
        ]),
        ("profile", [
            *common, "fit-mean-response", "--train", str(split / "train.h5ad"),
            "--output", str(profile), "--min-group-cells", "3",
        ]),
        ("calibration", [
            *common, "calibrate-shared-residual", "--source", str(split / "train.h5ad"),
            "--checkpoint", str(model), "--profile", str(profile),
            "--outdir", str(out / "calibration"), "--folds", str(args.folds),
            "--strengths", "0", "0.5", "1", "--min-relative-improvement", "0.02",
            "--cells-per-target", str(args.cells_per_target),
            "--prediction-steps", str(args.prediction_steps), "--threads", str(args.threads),
            "--seed", str(args.seed),
        ]),
        ("baseline", [
            *common, "control-baseline", "--reference", str(split / "reference.h5ad"),
            "--controls", str(split / "input_controls.h5ad"), "--output", str(out / "baseline.h5ad"),
            "--cells-per-target", str(args.cells_per_target), "--seed", infer_seed,
        ]),
        ("mean", [
            *common, "predict-mean-response", "--profile", str(profile),
            "--reference", str(split / "reference.h5ad"), "--controls", str(split / "input_controls.h5ad"),
            "--output", str(out / "mean.h5ad"), "--cells-per-target", str(args.cells_per_target),
            "--seed", infer_seed,
        ]),
        ("uncalibrated", [
            *common, "predict-shared-residual", "--profile", str(profile), "--checkpoint", str(model),
            "--reference", str(split / "reference.h5ad"), "--controls", str(split / "input_controls.h5ad"),
            "--output", str(out / "uncalibrated.h5ad"), "--conditional-strength", "1",
            "--cells-per-target", str(args.cells_per_target), "--steps", str(args.prediction_steps),
            "--seed", infer_seed,
        ]),
        ("calibrated", [
            *common, "predict-shared-residual", "--profile", str(profile), "--checkpoint", str(model),
            "--reference", str(split / "reference.h5ad"), "--controls", str(split / "input_controls.h5ad"),
            "--output", str(out / "calibrated.h5ad"), "--conditional-calibration", str(calibration),
            "--cells-per-target", str(args.cells_per_target), "--steps", str(args.prediction_steps),
            "--seed", infer_seed,
        ]),
    ]
    if args.include_null_control:
        next(command for name, command in commands if name == "calibration").append(
            "--include-null-control"
        )
    for name, command in commands:
        if name in {"calibration", "baseline", "mean", "uncalibrated", "calibrated"}:
            command.extend(["--control-sampling", args.control_sampling])
    for name in ("baseline", "mean", "uncalibrated", "calibrated"):
        commands.append((f"score-{name}", [
            *common, "score", "--prediction", str(out / f"{name}.h5ad"),
            "--reference", str(split / "reference.h5ad"), "--outdir", str(out / f"score-{name}"),
            "--threads", str(args.threads),
        ]))
    plan = {
        "kind": "arc-prespecified-target-disjoint-nested-shared-residual-v1",
        "source": str(args.source.resolve()),
        "source_sha256": source_digest,
        "checksum_note": "The expected source digest and downloaded file may share a mirror origin.",
        "source_gene_count": gene_count,
        "source_controls_available": int(counts["non-targeting"]),
        "selection_uses": "target labels, source cell counts, and the fixed gene axis, not expression",
        "eligible_targets": len(eligible),
        "eligible_test_targets": len(test_eligible),
        "excluded_prior_evaluation_targets": sorted(excluded_evaluation),
        "excluded_prior_training_targets": sorted(excluded_training),
        "excluded_from_test_prior_training_targets": sorted(excluded_test_training),
        "exclusion_manifests": exclusion_sources,
        "hyperparameter_selection_evidence": selection_evidence,
        "hyperparameter_selection_note": args.selection_note,
        "uncalibrated_prediction_note": (
            "The fixed-strength prediction is a prespecified diagnostic and is not used "
            "to select the final calibrated model."
        ),
        "training_targets": training,
        "test_targets": test,
        "seed": args.seed,
        "prediction_seed": int(infer_seed),
        "include_null_control": args.include_null_control,
        "control_sampling": args.control_sampling,
        "prior_exposure_note": (
            "These target labels may have appeared in earlier models' training data. "
            "This is target-disjoint within the new experiment, not a fully untouched research history."
        ),
        "code_sha256": fingerprints,
        "code_snapshot_sha256": snapshot_sha256,
        "runtime": {
            "python": sys.version,
            "anndata": version("anndata"),
            "numpy": version("numpy"),
            "scipy": version("scipy"),
            "torch": version("torch"),
            "cell-eval2": version("cell-eval2"),
            "pdex": version("pdex"),
        },
        "commands": [{"stage": name, "argv": command} for name, command in commands],
    }
    _write_new(out / "protocol.json", plan)
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(args.threads)
    env["MKL_NUM_THREADS"] = str(args.threads)
    env["OPENBLAS_NUM_THREADS"] = str(args.threads)
    for name, command in commands:
        print(f"Starting {name}; log: {logs / (name + '.log')}", flush=True)
        try:
            with (logs / f"{name}.log").open("x", encoding="utf-8") as log:
                subprocess.run(command, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, check=True)
        except subprocess.CalledProcessError as exc:
            _write_new(out / "failure.json", {
                "stage": name, "returncode": exc.returncode, "log": str(logs / f"{name}.log"),
            })
            raise
        if name == "prepare":
            current = args.source.stat()
            if (current.st_ino, current.st_size, current.st_mtime_ns) != (
                source_stat.st_ino, source_stat.st_size, source_stat.st_mtime_ns,
            ):
                raise RuntimeError("Arc source changed during preparation")
    calibrated = json.loads(calibration.read_text(encoding="utf-8"))
    if _code_fingerprints() != plan["code_sha256"]:
        raise RuntimeError("Experiment source code changed while the run was active")
    if file_sha256(out / "code-snapshot.zip") != snapshot_sha256:
        raise RuntimeError("Source snapshot changed while the run was active")
    for evidence in exclusion_sources + selection_evidence:
        if file_sha256(Path(evidence["path"])) != evidence["sha256"]:
            raise RuntimeError(f"Experiment evidence changed while the run was active: {evidence['path']}")
    split_manifest = json.loads((split / "split.json").read_text(encoding="utf-8"))
    if split_manifest["train_targets"] != training or split_manifest["test_targets"] != test:
        raise RuntimeError("Prepared targets differ from the prespecified protocol")
    train_ids = set(split_manifest["train_cell_ids"])
    reference_ids = set(split_manifest["test_cell_ids"])
    control_ids = set(split_manifest["input_control_cell_ids"])
    if train_ids & reference_ids or train_ids & control_ids or reference_ids & control_ids:
        raise RuntimeError("Prepared training, reference, and input control cells overlap")
    inputs = ad.read_h5ad(split / "input_controls.h5ad")
    predictions = {
        name: ad.read_h5ad(out / f"{name}.h5ad")
        for name in ("baseline", "mean", "uncalibrated", "calibrated")
    }
    columns = ["target_gene", "context", "batch", "source_control_id"]
    for name, prediction in predictions.items():
        sizes = prediction.obs.groupby("target_gene", observed=True).size()
        if (
            prediction.shape != (len(test) * args.cells_per_target, gene_count)
            or set(sizes.index.astype(str)) != set(test)
            or not (sizes == args.cells_per_target).all()
            or prediction.X.dtype.kind not in "iu" or np.any(prediction.X.data < 0)
            or not prediction.obs_names.is_unique
            or np.any(np.asarray(prediction.X.sum(axis=1)).ravel() > 1_000_000)
            or not set(prediction.obs["source_control_id"]).issubset(inputs.obs_names)
        ):
            raise RuntimeError(f"Invalid prediction shape, counts, or source controls for {name}")
        if not prediction.obs[columns].astype(str).reset_index(drop=True).equals(
            predictions["baseline"].obs[columns].astype(str).reset_index(drop=True)
        ):
            raise RuntimeError(f"Prediction control selection differs for {name}")
    selected_shared = calibrated.get("selected_shared_strength", calibrated["shared_strength"])
    if calibrated["selected_conditional_strength"] == 0 and selected_shared in (0, 1):
        comparator = "baseline" if selected_shared == 0 else "mean"
        if (
            sp.csr_matrix(predictions[comparator].X) != sp.csr_matrix(predictions["calibrated"].X)
        ).nnz:
            raise RuntimeError(f"Zero conditional strength did not reproduce the {comparator} prediction")
    metrics = {}
    for name in predictions:
        provenance = json.loads((out / f"score-{name}" / "scoring-provenance.json").read_text())
        metrics[name] = {
            "raw_metrics": provenance["raw_metrics"],
            "undefined_metrics": provenance["undefined_metrics"],
        }
    result = {
        "official_leaderboard_score": False,
        "note": "Local raw VCC preset metrics; private challenge calibration anchors are unavailable.",
        "protocol_sha256": file_sha256(out / "protocol.json"),
        "checkpoint_sha256": file_sha256(model),
        "profile_sha256": file_sha256(profile),
        "calibration_sha256": file_sha256(calibration),
        "selected_shared_strength": selected_shared,
        "selected_conditional_strength": calibrated["selected_conditional_strength"],
        "mean_objective_by_strength": calibrated["mean_objective_by_strength"],
        "strengths": calibrated["strengths"],
        "test_targets": test,
        "training_cells": split_manifest["train_cells"],
        "reference_cells": split_manifest["test_cells"],
        "input_controls": split_manifest["input_controls"],
        "genes": gene_count,
        "cells_per_target": args.cells_per_target,
        "metrics": metrics,
    }
    _write_new(out / "result.json", result)
    print(f"Verified result: {out / 'result.json'}", flush=True)


if __name__ == "__main__":
    main()
