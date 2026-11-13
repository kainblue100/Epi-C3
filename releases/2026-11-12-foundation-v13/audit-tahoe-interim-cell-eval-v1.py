"""Independently recompute mean-expression cell-eval metrics and parent identity checks."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mean_expression(adata: ad.AnnData, label: str) -> np.ndarray:
    subset = adata[adata.obs["target_gene"].astype(str) == label]
    if not subset.n_obs:
        raise ValueError(f"No cells for {label!r}")
    return np.asarray(subset.X.mean(axis=0)).ravel().astype(np.float64)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Refusing to overwrite an existing independent audit")
    manifest_path = args.inputs / "manifest.json"
    inputs = json.loads(manifest_path.read_text())
    evaluation = json.loads((args.results / "summary.json").read_text())
    if (
        inputs.get("status") != "complete" or evaluation.get("status") != "complete"
        or evaluation.get("input_manifest_sha256") != sha256(manifest_path)
    ):
        raise ValueError("The completed results must match the prepared input manifest")
    table_path = args.results / "per-perturbation.csv"
    table = pd.read_csv(table_path)
    maximum = {metric: 0.0 for metric in ("mse", "mae", "mse_delta", "mae_delta", "pearson_delta")}
    comparisons = 0
    for plate, entry in sorted(inputs["plates"].items()):
        paths = {name: args.inputs / plate / f"{name}.h5ad" for name in ("reference", "full", "noeffect")}
        if any(sha256(path) != entry["prepared_sha256"][name] for name, path in paths.items()):
            raise ValueError(f"Prepared files changed for {plate}")
        real = ad.read_h5ad(paths["reference"])
        real_ctrl = mean_expression(real, "non-targeting")
        for arm in ("full", "noeffect"):
            predicted = ad.read_h5ad(paths[arm])
            if not predicted.var_names.equals(real.var_names):
                raise ValueError("Gene axes do not match")
            pred_control_ids = set(predicted.obs_names[predicted.obs["target_gene"].astype(str) == "non-targeting"])
            parent_rows = predicted.obs[predicted.obs["target_gene"].astype(str) != "non-targeting"]
            parent_ids = set(parent_rows["source_control_id"].astype(str))
            if pred_control_ids & parent_ids or pred_control_ids & set(real.obs_names) or parent_ids & set(real.obs_names):
                raise ValueError("Evaluation inputs do not have disjoint parent, predicted-control and reference cells")
            if parent_rows.duplicated(["target_gene", "source_control_id"]).any():
                raise ValueError("A parent was counted more than once within a treatment")
            pred_ctrl = mean_expression(predicted, "non-targeting")
            for treatment in entry["audit"]["treatments"]:
                real_mean = mean_expression(real, treatment)
                pred_mean = mean_expression(predicted, treatment)
                real_delta, pred_delta = real_mean - real_ctrl, pred_mean - pred_ctrl
                computed = {
                    "mse": float(np.mean((real_mean - pred_mean) ** 2)),
                    "mae": float(np.mean(np.abs(real_mean - pred_mean))),
                    "mse_delta": float(np.mean((real_delta - pred_delta) ** 2)),
                    "mae_delta": float(np.mean(np.abs(real_delta - pred_delta))),
                    "pearson_delta": float(np.corrcoef(real_delta, pred_delta)[0, 1]),
                }
                row = table[(table["plate"] == plate) & (table["arm"] == arm)
                            & (table["perturbation"] == treatment)]
                if len(row) != 1:
                    raise ValueError(f"Missing or duplicate official score: {plate}, {arm}, {treatment}")
                for metric, value in computed.items():
                    recorded = float(row.iloc[0][metric])
                    error = abs(recorded - value)
                    tolerance = 1e-5 if metric == "pearson_delta" else 2e-6
                    if not math.isfinite(error) or error > tolerance:
                        raise ValueError(f"Independent {metric} mismatch for {plate}, {arm}, {treatment}: {error}")
                    maximum[metric] = max(maximum[metric], error)
                    comparisons += 1
    audit = {
        "format": "epi-c3-tahoe-cell-eval-audit-v1",
        "status": "complete",
        "input_manifest_sha256": sha256(manifest_path),
        "evaluation_summary_sha256": sha256(args.results / "summary.json"),
        "official_per_perturbation_sha256": sha256(table_path),
        "script_sha256": sha256(Path(__file__)),
        "independent_metric_comparisons": comparisons,
        "maximum_absolute_error": maximum,
        "control_and_prediction_parents_verified_disjoint": True,
        "same_treatment_duplicate_parents_forbidden": True,
        "limits": [
            "This recomputes five expression metrics, not the package's differential-expression statistics.",
            "The Pearson audit uses float64, whereas the official package evaluates float32 arrays; its tolerance is 1e-5.",
            "The original experiment was post-hoc and is not an evaluation of the new foundation model.",
        ],
    }
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(audit, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({"comparisons": comparisons, "maximum_absolute_error": maximum}), flush=True)


if __name__ == "__main__":
    main()
