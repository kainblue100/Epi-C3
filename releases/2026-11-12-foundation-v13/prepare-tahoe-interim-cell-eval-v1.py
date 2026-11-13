"""Prepare independent per-plate Tahoe inputs for an exploratory cell-eval run."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil

import anndata as ad

import epi_c3_cell_eval_inputs_v1 as bridge


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matched-dir", type=Path, required=True)
    parser.add_argument("--input-controls", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261115)
    args = parser.parse_args()
    staging = args.output.with_name(args.output.name + ".partial")
    if args.output.exists() or staging.exists():
        raise FileExistsError("Refusing to overwrite an existing evaluator input directory")
    if shutil.disk_usage(args.output.parent).free < 201 * 1024**3:
        raise OSError("Insufficient space above the 200 GiB reserve")
    plates = sorted(path for path in args.matched_dir.glob("plate-*") if path.is_dir())
    if not plates:
        raise ValueError("No per-plate predictions found")
    controls = ad.read_h5ad(args.input_controls)
    staging.mkdir(parents=False)
    manifest = {
        "format": "epi-c3-tahoe-cell-eval-inputs-v1",
        "status": "preparing",
        "scope": "post-hoc exploration of an earlier partially trained Tahoe model",
        "official_leaderboard_result": False,
        "script_sha256": sha256(Path(__file__)),
        "bridge_sha256": sha256(Path(bridge.__file__)),
        "input_controls_sha256": sha256(args.input_controls),
        "source_state_sha256": sha256(args.matched_dir / "state.json"),
        "plates": {},
        "limitations": [
            "The original models and split were examined before this analysis.",
            "This is not a result for the new RNA foundation or either requested final model.",
            "Prediction parents are distinct within each treatment but can be reused across treatments.",
            "Control splitting reduces the effective number of predicted cells and differential-expression power.",
            "Metrics that require multiple perturbations in a plate must not be interpreted when only one is available.",
        ],
    }
    for plate in plates:
        source_paths = {name: plate / f"{name}.h5ad" for name in ("reference", "full", "noeffect")}
        reference = ad.read_h5ad(source_paths["reference"])
        batch_values = reference.obs["batch"].astype(str).unique()
        if len(batch_values) != 1:
            raise ValueError(f"{plate.name} does not contain exactly one plate")
        batch = batch_values[0]
        selected_controls = controls[controls.obs["batch"].astype(str) == batch].copy()
        predictions = {name: ad.read_h5ad(source_paths[name]) for name in ("full", "noeffect")}
        real, prepared, audit = bridge.prepare_cell_eval_inputs(
            reference, selected_controls, predictions,
            perturbation_column="target_gene", control_label="non-targeting",
            seed=args.seed, minimum_parents=3,
        )
        out = staging / plate.name
        out.mkdir()
        outputs = {"reference": real, **prepared}
        output_hashes = {}
        for name, value in outputs.items():
            path = out / f"{name}.h5ad"
            value.write_h5ad(path, compression="gzip")
            output_hashes[name] = sha256(path)
        manifest["plates"][plate.name] = {
            "batch": batch,
            "source_sha256": {name: sha256(path) for name, path in source_paths.items()},
            "prepared_sha256": output_hashes,
            "audit": audit,
        }
        print(json.dumps({"plate": plate.name, **audit["unique_prediction_parents_per_treatment"]}), flush=True)
    manifest["status"] = "complete"
    with (staging / "manifest.json").open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.rename(staging, args.output)
    print(f"Prepared {len(plates)} independent-control plate inputs.", flush=True)


if __name__ == "__main__":
    main()
