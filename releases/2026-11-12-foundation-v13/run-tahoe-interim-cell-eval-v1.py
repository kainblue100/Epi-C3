"""Run official cell-eval on the earlier Tahoe experiment, plate by plate."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from cell_eval import MetricsEvaluator


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if importlib.metadata.version("cell-eval") != "0.8.2":
        raise ValueError("This run requires the pinned official cell-eval 0.8.2 package")
    staging = args.output.with_name(args.output.name + ".partial")
    if args.output.exists() or staging.exists():
        raise FileExistsError("Refusing to overwrite an existing cell-eval run")
    if shutil.disk_usage(args.output.parent).free < 201 * 1024**3:
        raise OSError("Insufficient space above the 200 GiB reserve")
    source_manifest = args.inputs / "manifest.json"
    manifest = json.loads(source_manifest.read_text())
    if manifest.get("format") != "epi-c3-tahoe-cell-eval-inputs-v1" or manifest.get("status") != "complete":
        raise ValueError("A completed, verified evaluator input manifest is required")
    for plate, entry in manifest["plates"].items():
        for name, expected in entry["prepared_sha256"].items():
            if sha256(args.inputs / plate / f"{name}.h5ad") != expected:
                raise ValueError(f"Prepared {plate}/{name} input changed after the frozen manifest")
    staging.mkdir()
    all_results = []
    evaluation = {
        "format": "epi-c3-tahoe-cell-eval-v1",
        "status": "running",
        "scope": manifest["scope"],
        "official_leaderboard_result": False,
        "cell_eval_version": importlib.metadata.version("cell-eval"),
        "pdex_version": importlib.metadata.version("pdex"),
        "input_manifest_sha256": sha256(source_manifest),
        "script_sha256": sha256(Path(__file__)),
        "profile": "full with explicitly skipped metrics where mathematically inapplicable",
        "normalization": "the input manifest specifies log1p CPM 10,000 for both real and predicted cells",
        "limitations": manifest["limitations"] + [
            "Predicted groups contain only 5-10 independent parents, so differential-expression estimates have low power.",
            "A NaN is an undefined metric, not zero or evidence of model success or failure.",
            "Zero significant genes can make official DEG-overlap values uninformative.",
            "Default cell-eval/pdex settings may produce infinite log fold changes for genes with zero expression.",
        ],
        "plates": {},
    }
    for plate, entry in sorted(manifest["plates"].items()):
        treatments = entry["audit"]["treatments"]
        skips = ["clustering_agreement"]
        if len(treatments) < 3:
            skips.extend(["pearson_edistance", "de_spearman_sig"])
        if len(treatments) < 2:
            skips.extend([
                "discrimination_score_l1", "discrimination_score_l2", "discrimination_score_cosine",
            ])
        evaluation["plates"][plate] = {
            "batch": entry["batch"], "treatments": treatments, "skipped_metrics": skips,
            "independent_prediction_parents": entry["audit"]["unique_prediction_parents_per_treatment"],
        }
        real_de_path = None
        for arm in ("full", "noeffect"):
            out = staging / plate / arm
            evaluator = MetricsEvaluator(
                adata_real=str(args.inputs / plate / "reference.h5ad"),
                adata_pred=str(args.inputs / plate / f"{arm}.h5ad"),
                de_real=real_de_path,
                control_pert="non-targeting", pert_col="target_gene", num_threads=1,
                outdir=str(out),
            )
            result, _ = evaluator.compute(profile="full", skip_metrics=skips, break_on_error=True)
            frame = result.to_pandas()
            if set(frame["perturbation"].astype(str)) != set(treatments):
                raise ValueError(f"Evaluator output is missing or adds perturbations for {plate}/{arm}")
            frame.insert(0, "arm", arm)
            frame.insert(0, "plate", plate)
            all_results.append(frame)
            if real_de_path is None:
                real_de_path = str(out / "real_de.csv")
                if not Path(real_de_path).is_file():
                    raise ValueError("cell-eval did not save its reference DE table")
            print(json.dumps({"plate": plate, "arm": arm, "treatments": len(treatments)}), flush=True)
            del evaluator, result, frame
            gc.collect()
    combined = pd.concat(all_results, ignore_index=True, sort=False)
    if combined.duplicated(["arm", "perturbation"]).any():
        raise ValueError("A treatment appears in more than one plate for the same arm")
    combined.to_csv(staging / "per-perturbation.csv", index=False)
    metrics = [column for column in combined if column not in ("plate", "arm", "perturbation")]
    summaries = {}
    for arm in ("full", "noeffect"):
        frame = combined[combined["arm"] == arm]
        summaries[arm] = {}
        for metric in metrics:
            values = pd.to_numeric(frame[metric], errors="raise").to_numpy(dtype=float)
            finite = values[np.isfinite(values)]
            summaries[arm][metric] = {
                "mean_of_finite": float(finite.mean()) if len(finite) else None,
                "finite_conditions": len(finite),
                "undefined_conditions": len(values) - len(finite),
            }
    evaluation["summary"] = summaries
    evaluation["status"] = "complete"
    with (staging / "summary.json").open("x", encoding="utf-8") as stream:
        json.dump(evaluation, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.rename(staging, args.output)
    print(f"Completed exploratory cell-eval for {len(manifest['plates'])} plates.", flush=True)


if __name__ == "__main__":
    main()
