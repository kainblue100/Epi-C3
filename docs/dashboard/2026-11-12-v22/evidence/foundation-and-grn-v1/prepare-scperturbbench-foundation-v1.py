"""Freeze audited, control-only benchmark streams without modifying ongoing training."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil

import h5py
import numpy as np
import pandas as pd

import epi_c3_foundation_sources_v1 as foundation
import epi_c3_scperturbbench_foundation_v1 as bench
from epi_c3_shared_foundation_v1 import file_digest, gene_axis_digest
from khp_transfer_common_v1 import read_plan


FILES = {
    "KHP": "Perturb_KHP_sub10.h5ad",
    "CMO": "Perturb_cmo_V1_sub10.h5ad",
}


def verify_prior_stat(path: Path, provenance: dict, expected_sha256: str) -> None:
    stat = path.stat()
    if (
        stat.st_size != provenance["source_size_bytes"]
        or stat.st_mtime_ns != provenance["source_mtime_ns"]
        or provenance["source_sha256"] != expected_sha256
    ):
        raise ValueError(f"Previously verified gene annotation input changed: {path.name}")


def reference_annotations(root: Path, run: Path) -> tuple[dict, list[str], dict[str, set[str]]]:
    path = run / "rna-foundation-stage1-v1/plan.json"
    plan = json.loads(path.read_text(encoding="utf-8"))
    if plan.get("format") != "epi-c3-shared-rna-stage1-plan-v1":
        raise ValueError("The shared foundation plan has an unexpected format")
    provenance = plan["provenance"]
    if file_digest(Path(foundation.__file__)) != provenance["source_code_sha256"]["epi_c3_foundation_sources_v1.py"]:
        raise ValueError("The foundation gene alignment implementation has changed")
    axis = plan["genes"]
    if gene_axis_digest(axis) != plan["gene_axis_sha256"]:
        raise ValueError("The stored shared gene axis has changed")
    arc_path = root / "arc-institute-virtual-cell-dataset/adata_Training.h5ad"
    tabula_path = root / "Tabula-Sapiens-v2-CELLxGENE/Tabula_Sapiens_v2_All_Cells.h5ad"
    for data_path, prior_path, digest_key in (
        (arc_path, run / "arc-stream-full-v1/plan.json", "arc_source_sha256_from_prior_verified_scan"),
        (tabula_path, run / "tabula-stream-full-hct-axis-v1/plan.json", "tabula_source_sha256_from_prior_verified_scan"),
    ):
        prior = json.loads(prior_path.read_text(encoding="utf-8"))
        verify_prior_stat(data_path, prior["provenance"], provenance[digest_key])
    frames = {}
    for name, folder, digest_key, meta_path in (
        ("tahoe", "Tahoe-100M", "tahoe_metadata_plan_sha256_for_gene_axis_only", run / "tahoe-full-metadata-v2/metadata-plan.json"),
        ("orion", "X-Atlas-Orion", "orion_metadata_plan_sha256", run / "orion-full-metadata-v1/metadata-plan.json"),
    ):
        if file_digest(meta_path) != provenance[digest_key]:
            raise ValueError(f"The frozen {name} metadata plan changed")
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        path = root / folder / "metadata/gene_metadata.parquet"
        if metadata["state"] != "complete" or file_digest(path) != metadata["provenance"]["gene_metadata_sha256"]:
            raise ValueError(f"The verified {name} gene metadata changed")
        frames[name] = pd.read_parquet(path)
    with h5py.File(tabula_path, "r") as file:
        tabula_ids = foundation.h5_text(file["raw/var/_index"])
        tabula_symbols = foundation.h5_text(file["raw/var/feature_name"])
    with h5py.File(arc_path, "r") as file:
        arc_symbols = foundation.h5_text(file["var/_index"])
    alignment = foundation.align_human_genes(
        tabula_ids, tabula_symbols, frames["tahoe"], frames["orion"], arc_symbols,
    )
    if list(alignment.genes) != axis:
        raise ValueError("Reconstructed source annotations do not reproduce the frozen foundation axis")
    symbols: dict[str, set[str]] = defaultdict(set)
    for symbol, identifier in zip(tabula_symbols, tabula_ids, strict=True):
        if symbol:
            symbols[symbol].add(identifier)
    for frame, column in ((frames["tahoe"], "gene_symbol"), (frames["orion"], "gene_name")):
        for symbol, identifier in zip(frame[column], frame["ensembl_id"], strict=True):
            if isinstance(symbol, str) and symbol:
                symbols[symbol].add(str(identifier))
    return plan, axis, dict(symbols)


def protected_khp_rows(plan: dict) -> set[int]:
    selection = plan["selection"]
    groups = [selection["target_input_control"], selection["target_reference_control"]]
    groups.extend(selection["target_reference_treatments"].values())
    values = [value for group in groups for value in group]
    if not values or any(type(value) is not int for value in values) or len(values) != len(set(values)):
        raise ValueError("The frozen KHP target evaluation rows are malformed or duplicated")
    return set(values)


def prepare(root: Path, run: Path, output: Path, validation_size: int, seed: int) -> None:
    if output.exists():
        raise FileExistsError(output)
    if shutil.disk_usage(output.parent).free < 201 * 1024**3:
        raise OSError("Insufficient free space above the 200 GiB reserve")
    foundation_plan, axis, symbol_ids = reference_annotations(root, run)
    dataset_root = root / "benchmarks/scPerturbBench-zenodo-20823660"
    state_path = dataset_root / "state.json"
    download = json.loads(state_path.read_text(encoding="utf-8"))
    if download.get("state") != "complete":
        raise ValueError("A completed, verified scPerturbBench download is required")
    khp_path = run / "khp-cd8-transfer-v1/plan.json"
    khp = read_plan(khp_path)
    protected = protected_khp_rows(khp)
    private = {"format": bench.PRIVATE_FORMAT, "gene_axis_sha256": foundation_plan["gene_axis_sha256"], "sources": {}}
    sources = {}
    for source, filename in FILES.items():
        path = dataset_root / filename
        before = path.stat()
        digest = file_digest(path)
        expected = download["verified_files"][filename]["sha256"]
        if digest != expected or source == "KHP" and digest != khp["source"]["sha256"]:
            raise ValueError(f"The {source} source differs from its verified download or evaluation plan")
        with h5py.File(path, "r") as file:
            node = file["layers/counts"]
            shape = tuple(map(int, node.attrs["shape"]))
            genes = foundation.h5_text(file["var/gene_name"])
            direct = foundation.h5_text(file["var/gene_id"]) if source == "CMO" else None
            mapping, mapping_report = bench.resolve_mapping(genes, axis, symbol_ids, direct)
            numeric, positive = bench.audit_sparse_counts(node, mapping)
            obs = file["obs"]
            conditions = bench.categorical(obs, "condition", shape[0])
            cell_types = bench.categorical(obs, "cell_type", shape[0])
            donors = bench.categorical(obs, "donor_id", shape[0]) if source == "CMO" else None
            rows, split = bench.partition_controls(
                source, conditions, cell_types, positive > 0, protected if source == "KHP" else set(),
                donors=donors, validation_size=validation_size, seed=seed,
            )
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError(f"The {source} file changed during the audit")
        private["sources"][source] = {
            "source_sha256": digest, "shape": list(shape), "encoding": numeric["encoding"],
            "mapping": mapping.tolist(), **rows,
        }
        sources[source] = {
            "file": filename, "source_sha256": digest,
            "count_layer": "layers/counts", "mapping": mapping_report,
            "numeric_audit": numeric, "split": split,
        }
        print(f"{source}: {split['training_controls']} training and {split['validation_controls']} validation controls", flush=True)
    output.mkdir(mode=0o700)
    private_path = output / "sources.private.json"
    descriptor = os.open(private_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as file:
        json.dump(private, file, ensure_ascii=False, indent=2, allow_nan=False)
        file.write("\n")
        file.flush()
        os.fsync(file.fileno())
    report = {
        "format": bench.PUBLIC_FORMAT,
        "state": "complete",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "foundation_plan_sha256": file_digest(run / "rna-foundation-stage1-v1/plan.json"),
        "gene_axis_sha256": foundation_plan["gene_axis_sha256"],
        "download_state_sha256": file_digest(state_path),
        "khp_evaluation_plan_sha256": file_digest(khp_path),
        "preparation_script_sha256": file_digest(Path(__file__)),
        "source_module_sha256": file_digest(Path(bench.__file__)),
        "private_sha256": file_digest(private_path),
        "validation_size": validation_size,
        "selection_seed": seed,
        "sources": sources,
        "training_eligibility": "candidate-only; cross-source identity and chemistry isolation remain unresolved",
        "limitations": [
            "Only explicit controls are candidates; every treated row is excluded pending a cross-dataset chemical-identity and benchmark isolation audit.",
            "KHP excludes the entire CD8T group, including all frozen target reference and input-control rows; source-cell controls may overlap older source training.",
            "One CMO donor is excluded, but donor or cell identity overlap with other datasets has not been established.",
            "The 5000-feature author-selected panels may use information from later held-out cells; feature selection was not redone.",
            "Unique gene-symbol mappings use the frozen foundation annotations; unmapped features are discarded rather than represented as observed zeros.",
            "Count-layer numerical integrity does not independently establish original raw-count provenance.",
            "No new foundation training was performed by this audit.",
        ],
    }
    with (output / "audit.json").open("x", encoding="utf-8", newline="\n") as file:
        json.dump(report, file, ensure_ascii=False, indent=2, allow_nan=False)
        file.write("\n")
        file.flush()
        os.fsync(file.fileno())
    print(json.dumps({"state": report["state"], "output": str(output), "private_sha256": report["private_sha256"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validation-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20261117)
    arguments = parser.parse_args()
    prepare(arguments.root, arguments.run_root, arguments.output, arguments.validation_size, arguments.seed)
