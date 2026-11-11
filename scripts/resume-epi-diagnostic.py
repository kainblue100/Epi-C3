"""Run the prespecified fixed-strength ablation after an undefined calibration."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from c3vcell.integrity import file_sha256
from c3vcell.predict import predict_holdout
from c3vcell.scoring import score_local
from c3vcell.training import train


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()
    if args.outdir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.outdir}")
    plan_path = args.run / "experiment-plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    log_path = args.run.with_suffix(".log")
    failure = log_path.read_text(encoding="utf-8")
    if (
        "the sum of expr_distance_unbiased" not in failure
        or "is not positive" not in failure
        or (args.run / "no-prior/calibration/effect-calibration.json").exists()
    ):
        raise ValueError("This recovery is only for an explicitly undefined reference calibration")
    source = args.run / "split/train.h5ad"
    reference = args.run / "split/reference.h5ad"
    controls = args.run / "split/input_controls.h5ad"
    baseline_checkpoint = args.run / "no-prior/model/model.pt"
    checkpoint = torch.load(baseline_checkpoint, map_location="cpu", weights_only=True)
    if (
        checkpoint["training_source_sha256"] != file_sha256(source)
        or checkpoint["training_targets"] != plan["training_targets"]
        or checkpoint["seed"] != plan["training"]["seed"]
        or checkpoint.get("epigenetics") is not None
        or checkpoint.get("graph") is not None
    ):
        raise ValueError("Saved baseline checkpoint does not match the prespecified experiment")
    if plan["diagnostic_strength"] != 1.0 or plan["calibration"]["folds"] != 2:
        raise ValueError("Unsupported original diagnostic plan")
    real_bundle = args.run / "epigenome"
    shuffled_bundle = args.run / "shuffled"
    real_manifest = real_bundle / "manifest.json"
    shuffled = json.loads((shuffled_bundle / "manifest.json").read_text(encoding="utf-8"))
    if (
        shuffled.get("control", {}).get("reference_manifest_sha256") != file_sha256(real_manifest)
        or shuffled["control"].get("seed") != plan["training"]["seed"]
    ):
        raise ValueError("Shuffled prior is not bound to the prespecified reference and seed")
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("MKL_NUM_THREADS", "2")
    torch.set_num_threads(2)
    args.outdir.mkdir(parents=True)
    provenance = {
        "parent_plan_sha256": file_sha256(plan_path),
        "calibration_status": "failed: the inner-fold reference expression denominator is nonpositive",
        "calibration_error_log": str(log_path),
        "calibration_error_log_sha256": file_sha256(log_path),
        "diagnostic_strength": 1.0,
        "selection_note": (
            "Fixed strength was prespecified before held-out scoring; it is not calibrated "
            "and these results cannot be interpreted as an official leaderboard score."
        ),
        "source_sha256": file_sha256(source),
        "reference_sha256": file_sha256(reference),
        "controls_sha256": file_sha256(controls),
        "real_manifest_sha256": file_sha256(real_manifest),
        "shuffled_manifest_sha256": file_sha256(shuffled_bundle / "manifest.json"),
    }
    with (args.outdir / "diagnostic-plan.json").open("x", encoding="utf-8") as stream:
        json.dump(provenance, stream, indent=2)
        stream.write("\n")
    results = {
        "control-baseline": json.loads(
            (args.run / "control-baseline-score/scoring-provenance.json").read_text(encoding="utf-8")
        )["raw_metrics"]
    }
    settings = plan["training"]
    for name, prior in (
        ("no-prior", None), ("hct116-epigenome", real_bundle),
        ("shuffled-epigenome", shuffled_bundle),
    ):
        directory = args.outdir / name
        if prior is None:
            model_path = baseline_checkpoint
        else:
            model_path = train(
                source, directory / "model", latent_dim=settings["latent_dim"],
                hidden_dim=settings["hidden_dim"], batch_size=settings["batch_size"],
                pretrain_steps=settings["pretrain_steps"], flow_steps=settings["flow_steps"],
                seed=settings["seed"], flow_target=settings["flow_target"],
                epigenetic_bundle=prior,
            )
        prediction = directory / "diagnostic-strength-1.h5ad"
        predict_holdout(
            model_path, reference, controls, prediction,
            cells_per_target=plan["prediction_cells_per_target"], steps=8,
            seed=settings["seed"], sampling="residual-rounding",
            control_sampling="input-controls", effect_strength=1.0,
        )
        result = score_local(prediction, reference, directory / "score", threads=2)
        results[name] = {
            "model_sha256": file_sha256(model_path),
            "prediction_sha256": file_sha256(prediction),
            "raw_metrics": result["raw_metrics"],
            "undefined_metrics": result["undefined_metrics"],
        }
        with (directory / "result.json").open("x", encoding="utf-8") as stream:
            json.dump(results[name], stream, indent=2, allow_nan=False)
            stream.write("\n")
    with (args.outdir / "result.json").open("x", encoding="utf-8") as stream:
        json.dump(results, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
