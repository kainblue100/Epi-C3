"""Explore a new multimodal prior on an existing, fixed HCT116 holdout."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd
import torch

from c3vcell.epigenetics import build_epigenetic_bundle, shuffle_epigenetic_bundle
from c3vcell.integrity import file_sha256
from c3vcell.predict import predict_holdout
from c3vcell.research_review import review_network_diagnostic
from c3vcell.scoring import score_local
from c3vcell.training import train


def _write_new(path: Path, payload: dict) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
        stream.write("\n")


def _bundle_summary(bundle: Path) -> dict:
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    frame = pd.read_csv(bundle / "features.csv")
    edges = pd.read_csv(bundle / "graph.csv", keep_default_na=False)
    features = {}
    for name in manifest["features"]:
        measured = frame.loc[frame[f"{name}_mask"] == 1, name]
        features[name] = {
            "measured_genes": int(len(measured)),
            "nonzero_genes": int((measured != 0).sum()),
            "mean_when_measured": float(measured.mean()) if len(measured) else None,
        }
    return {
        "gene_mapping": manifest["mapping"],
        "contact_count": manifest["loop_count"],
        "edge_count": len(edges),
        "edges_by_evidence": edges["evidence"].value_counts().to_dict(),
        "features": features,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-run", type=Path, required=True)
    parser.add_argument("--parent-diagnostic", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--source-archive", type=Path)
    args = parser.parse_args()
    if args.outdir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.outdir}")
    parent_plan_path = args.parent_run / "experiment-plan.json"
    parent_plan = json.loads(parent_plan_path.read_text(encoding="utf-8"))
    previous_plan_path = args.parent_diagnostic / "diagnostic-plan.json"
    previous_plan = json.loads(previous_plan_path.read_text(encoding="utf-8"))
    previous_result_path = args.parent_diagnostic / "result.json"
    previous_result = json.loads(previous_result_path.read_text(encoding="utf-8"))
    train_path = args.parent_run / "split/train.h5ad"
    reference = args.parent_run / "split/reference.h5ad"
    controls = args.parent_run / "split/input_controls.h5ad"
    baseline = args.parent_run / "no-prior/model/model.pt"
    baseline_prediction = args.parent_diagnostic / "no-prior/diagnostic-strength-1.h5ad"
    settings = parent_plan["training"]
    checkpoint = torch.load(baseline, map_location="cpu", weights_only=True)
    if (
        previous_plan["parent_plan_sha256"] != file_sha256(parent_plan_path)
        or previous_plan["source_sha256"] != file_sha256(train_path)
        or previous_plan["reference_sha256"] != file_sha256(reference)
        or previous_plan["controls_sha256"] != file_sha256(controls)
        or previous_result["no-prior"]["model_sha256"] != file_sha256(baseline)
        or previous_result["no-prior"]["prediction_sha256"] != file_sha256(baseline_prediction)
        or checkpoint["training_source_sha256"] != file_sha256(train_path)
        or checkpoint["training_targets"] != parent_plan["training_targets"]
        or checkpoint["seed"] != settings["seed"]
        or checkpoint.get("epigenetics") is not None
        or checkpoint.get("graph") is not None
        or parent_plan["diagnostic_strength"] != 1.0
        or previous_plan["diagnostic_strength"] != 1.0
        or "nonpositive" not in previous_plan["calibration_status"]
    ):
        raise ValueError("The fixed baseline, split, or undefined calibration has changed")
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    if spec.get("format_version") != 2 or spec.get("context") != "HCT116":
        raise ValueError("Expected a version-2 HCT116 network specification")
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("MKL_NUM_THREADS", "2")
    torch.set_num_threads(2)
    args.outdir.mkdir(parents=True)
    spec_digest = file_sha256(args.spec)
    plan = {
        "parent_plan_sha256": file_sha256(parent_plan_path),
        "previous_diagnostic_plan_sha256": file_sha256(previous_plan_path),
        "previous_results_sha256": file_sha256(previous_result_path),
        "network_spec_sha256": spec_digest,
        "source_sha256": file_sha256(train_path),
        "reference_sha256": file_sha256(reference),
        "controls_sha256": file_sha256(controls),
        "program_sha256": file_sha256(Path(__file__)),
        "source_archive_sha256": file_sha256(args.source_archive) if args.source_archive else None,
        "training": settings,
        "prediction_cells_per_target": parent_plan["prediction_cells_per_target"],
        "effect_strength": 1.0,
        "shuffle_seed": settings["seed"],
        "arms": ["real-network", "shuffled-network"],
        "note": (
            "Exploratory follow-on after the outer holdout was previously scored; fixed training "
            "and prediction settings are reused, but this is not an independent evaluation. "
            "The original inner calibration failed; strength 1 is uncalibrated. Local raw "
            "VCC metrics are not official leaderboard scores."
        ),
    }
    _write_new(args.outdir / "diagnostic-plan.json", plan)
    real, shuffled = args.outdir / "network", args.outdir / "shuffled-network"
    build_epigenetic_bundle(train_path, args.spec, real)
    shuffle_epigenetic_bundle(real, train_path, shuffled, seed=settings["seed"])
    if file_sha256(args.spec) != spec_digest:
        raise RuntimeError("Network specification changed during preparation")
    _write_new(args.outdir / "network-summary.json", _bundle_summary(real))
    result = {
        "previous_diagnostic": {
            "result_sha256": file_sha256(previous_result_path),
            "control_baseline": previous_result["control-baseline"],
            "no_prior": previous_result["no-prior"],
            "v1_prior": previous_result["hct116-epigenome"],
        }
    }
    for label, bundle in (("real-network", real), ("shuffled-network", shuffled)):
        directory = args.outdir / label
        checkpoint_path = train(
            train_path, directory / "model", latent_dim=settings["latent_dim"],
            hidden_dim=settings["hidden_dim"], batch_size=settings["batch_size"],
            pretrain_steps=settings["pretrain_steps"], flow_steps=settings["flow_steps"],
            seed=settings["seed"], flow_target=settings["flow_target"],
            epigenetic_bundle=bundle,
        )
        prediction = directory / "diagnostic-strength-1.h5ad"
        predict_holdout(
            checkpoint_path, reference, controls, prediction,
            cells_per_target=parent_plan["prediction_cells_per_target"], steps=8,
            seed=settings["seed"], sampling="residual-rounding",
            control_sampling="input-controls", effect_strength=1.0,
        )
        scores = score_local(prediction, reference, directory / "score", threads=2)
        result[label] = {
            "model_sha256": file_sha256(checkpoint_path),
            "prediction_sha256": file_sha256(prediction),
            "raw_metrics": scores["raw_metrics"],
            "undefined_metrics": scores["undefined_metrics"],
        }
        _write_new(directory / "result.json", result[label])
    _write_new(args.outdir / "result.json", result)
    review = review_network_diagnostic(
        args.outdir, args.parent_run, args.parent_diagnostic, args.spec,
        source_archive=args.source_archive, program=Path(__file__),
    )
    _write_new(args.outdir / "research-review.json", review)


if __name__ == "__main__":
    main()
