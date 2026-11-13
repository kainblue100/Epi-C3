"""Freeze a conservative K562 ChIP source list using local eligibility metadata."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys


HERE = Path(__file__).resolve().parent
PREFLIGHT_DIR = HERE.parent / "2026-11-13-grn-preflight-v1"
if PREFLIGHT_DIR.is_dir():
    sys.path.insert(0, str(PREFLIGHT_DIR))

import audit_grn_pilot as common
import audit_graph_eligibility as eligible
import freeze_encode_catalogs as freeze


PREFLIGHT_SHA = "ed9ad36c80a35c0a195e40ea64063645f0d4af5b3db99beb1a81c1d68e36a47a"
ELIGIBILITY_CODE_SHA = "5eccf4f598048e7e65e135d9e279a4b447a9cc102949a14686412e828bd69854"
FREEZE_CODE_SHA = "c19089a3d5dfb647ed342ac9804142052e52198fac226dc15f904aeeae051e99"
EXPERIMENT = re.compile(r"ENCSR[0-9]{3}[A-Z]{3}")
FILE = re.compile(r"ENCFF[0-9]{3}[A-Z]{3}")
MD5 = re.compile(r"[0-9a-f]{32}")


def unmodified_replicates(record: dict, file_replicates: list) -> str | None:
    replicates = record.get("replicates")
    if not isinstance(replicates, list) or not replicates:
        return "missing_replicate_metadata"
    seen = set()
    for replicate in replicates:
        if not isinstance(replicate, dict):
            return "invalid_replicate_metadata"
        number = replicate.get("biological_replicate_number")
        library = replicate.get("library")
        if type(number) is not int or number < 1 or not isinstance(library, dict):
            return "invalid_replicate_metadata"
        biosample = library.get("biosample")
        if not isinstance(biosample, dict):
            return "missing_biosample_metadata"
        if (
            library.get("treatments") != []
            or biosample.get("treatments") != []
            or biosample.get("applied_modifications") != []
        ):
            return "treated_modified_or_unverified_biosample"
        seen.add(number)
    if (
        not isinstance(file_replicates, list) or len(file_replicates) < 2
        or any(type(number) is not int or number < 1 for number in file_replicates)
        or len(set(file_replicates)) != len(file_replicates)
        or not set(file_replicates).issubset(seen)
    ):
        return "unverified_file_replicates"
    return None


def unmodified_experiment(record: dict, symbol: str, ensembl: str, file_replicates: list) -> str | None:
    target = record.get("target", {})
    if (
        record.get("status") != "released" or record.get("assay_title") != "TF ChIP-seq"
        or record.get("biosample_ontology", {}).get("term_name") != "K562"
        or record.get("biosample_summary") != "Homo sapiens K562"
        or record.get("perturbed") is not False or target.get("label") != symbol
    ):
        return "experiment_context_or_target_mismatch"
    genes = target.get("genes")
    if not isinstance(genes, list) or len(genes) != 1 or genes[0].get("symbol") != symbol:
        return "ambiguous_target_gene"
    xrefs = genes[0].get("dbxrefs")
    if not isinstance(xrefs, list):
        return "missing_target_ensembl_id"
    ids = {x[8:] for x in xrefs if isinstance(x, str) and x.startswith("ENSEMBL:ENSG")}
    if ids != {ensembl}:
        return "target_ensembl_id_mismatch"
    if not isinstance(record.get("lab", {}).get("title"), str) or not record["lab"]["title"]:
        return "missing_producing_lab"
    return unmodified_replicates(record, file_replicates)


def select_files(
    files: list[dict], experiments: list[dict], candidates: set[str], features: dict[str, str],
    foundation_genes: set[str], pilot_targets: dict,
) -> tuple[list[dict], dict]:
    by_experiment = {item["accession"]: item for item in experiments}
    if len(by_experiment) != len(experiments):
        raise ValueError("Duplicate ENCODE experiment accessions")
    grouped = defaultdict(list)
    rejected = Counter()
    catalog_labels = set()
    for item in files:
        label = item.get("target", {}).get("label")
        if label not in candidates:
            continue
        catalog_labels.add(label)
        accession = item.get("accession")
        dataset = item.get("dataset")
        match = re.fullmatch(r"/experiments/(ENCSR[0-9]{3}[A-Z]{3})/", dataset or "")
        replicates = item.get("biological_replicates")
        if (
            not isinstance(accession, str) or not FILE.fullmatch(accession) or not match
            or item.get("status") != "released" or item.get("assay_title") != "TF ChIP-seq"
            or item.get("biosample_ontology", {}).get("term_name") != "K562"
            or item.get("assembly") != "GRCh38" or item.get("file_format") != "bed"
            or item.get("file_format_type") != "narrowPeak"
            or item.get("output_type") not in ("conservative IDR thresholded peaks", "optimal IDR thresholded peaks")
            or type(item.get("file_size")) is not int or item["file_size"] < 1
            or not isinstance(item.get("md5sum"), str) or not MD5.fullmatch(item["md5sum"])
            or item.get("href") != f"/files/{accession}/@@download/{accession}.bed.gz"
        ):
            rejected["invalid_or_unverified_file_metadata"] += 1
            continue
        if features[label] not in foundation_genes:
            rejected["source_missing_from_foundation_axis"] += 1
            continue
        experiment_id = match.group(1)
        experiment = by_experiment.get(experiment_id)
        if experiment is None:
            rejected["experiment_not_in_untreated_catalog"] += 1
            continue
        reason = unmodified_experiment(experiment, label, features[label], replicates)
        if reason is not None:
            rejected[reason] += 1
            continue
        grouped[experiment_id].append({
            "experiment_accession": experiment_id,
            "file_accession": accession,
            "source_symbol": label,
            "source_ensembl_id": features[label],
            "producing_lab": experiment["lab"]["title"],
            "assembly": "GRCh38",
            "output_type": item["output_type"],
            "biological_replicates": sorted(replicates),
            "size": item["file_size"],
            "official_md5": item["md5sum"],
            "href": item["href"],
        })
    selected = []
    for experiment_id, options in sorted(grouped.items()):
        if len({item["source_symbol"] for item in options}) != 1:
            raise ValueError(f"Experiment {experiment_id} has conflicting ChIP target labels")
        options.sort(key=lambda item: (
            item["output_type"] != "conservative IDR thresholded peaks",
            -len(item["biological_replicates"]), item["file_accession"],
        ))
        selected.append(options[0])
    if not selected:
        raise ValueError("No verifiable, untreated ChIP peak files match the eligible source axis")
    selected_labels = {item["source_symbol"] for item in selected}
    return selected, {
        "eligible_shared_targets": len(candidates),
        "eligible_targets_in_full_released_peak_catalog": len(catalog_labels),
        "eligible_targets_with_selected_untreated_peak_experiment": len(selected_labels),
        "selected_experiments": len(selected),
        "selected_peak_files": len(selected),
        "selected_peak_bytes": sum(item["size"] for item in selected),
        "selected_output_types": dict(sorted(Counter(item["output_type"] for item in selected).items())),
        "selected_sources_by_pilot_target_split": {
            role: len(selected_labels & set(pilot_targets[role])) for role in ("train", "val", "test")
        },
        "rejected_candidate_file_counts": dict(sorted(rejected.items())),
    }


def write_json(path: Path, value: dict) -> str:
    data = (json.dumps(value, indent=2, allow_nan=False) + "\n").encode("utf-8")
    if shutil.disk_usage(path.parent).free < common.RESERVE_WITH_HEADROOM:
        raise OSError("Insufficient disk space above the 200 GiB reserve and 1 GiB headroom")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--k562", type=Path, required=True)
    parser.add_argument("--rpe1", type=Path, required=True)
    parser.add_argument("--preparation", type=Path, required=True)
    parser.add_argument("--catalog-dir", type=Path, required=True)
    parser.add_argument("--snapshot-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    for path, digest in (
        (Path(common.__file__), PREFLIGHT_SHA), (Path(eligible.__file__), ELIGIBILITY_CODE_SHA),
        (Path(freeze.__file__), FREEZE_CODE_SHA),
    ):
        if common.sha256(path) != digest:
            raise ValueError(f"Dependency code hash mismatch: {path.name}")
    snapshot = common.read_json(args.catalog_dir / "snapshot.json", args.snapshot_sha256)
    if snapshot.get("format") != "epi-c3-encode-metadata-snapshot-v1" or snapshot["script_sha256"] != FREEZE_CODE_SHA:
        raise ValueError("Unexpected ENCODE snapshot provenance")
    catalogs = {}
    for kind in ("files", "experiments"):
        record = snapshot["catalogs"][kind]
        if record["url"] != freeze.catalog_url(kind):
            raise ValueError("ENCODE catalog query differs from the frozen protocol")
        data = common.read_json(args.catalog_dir / f"{kind}.json", record["sha256"])
        if freeze.validate_catalog(data, kind) != record["records"]:
            raise ValueError("ENCODE catalog count differs from the snapshot")
        catalogs[kind] = data["@graph"]

    pilot = common.read_json(args.root / "state-replogle-pilot-v1" / "frozen-plan.json", common.PILOT_PLAN_SHA)
    foundation = common.read_json(args.root / "rna-foundation-stage1-v1" / "plan.json", common.FOUNDATION_PLAN_SHA)
    if (
        pilot.get("format") != "epi-c3-state-replogle-pilot-plan-v1"
        or foundation.get("format") != "epi-c3-shared-rna-stage1-plan-v1"
    ):
        raise ValueError("Unexpected frozen pilot or foundation plan")
    features = common.pilot_features(pilot)
    foundation_genes = common.unique_strings(foundation["genes"], "Foundation gene axis")
    if common.axis_sha256(foundation_genes) != foundation["gene_axis_sha256"]:
        raise ValueError("Foundation gene-axis hash mismatch")
    source_stats = {
        "K562": eligible.verify_source(args.k562, pilot["sources"]["K562"]),
        "RPE1": eligible.verify_source(args.rpe1, pilot["sources"]["RPE1"]),
    }
    helper = eligible.load_preparation(args.preparation)
    if pilot["cells_per_target_context"] != helper.CELLS_PER_TARGET:
        raise ValueError("The frozen target threshold differs from its preparation code")
    left = helper.load_metadata(args.k562, "K562")
    right = helper.load_metadata(args.rpe1, "RPE1")
    axis = helper.shared_axis(left, right)
    if [list(pair) for pair in axis] != pilot["features"]:
        raise ValueError("Source annotations differ from the frozen shared gene axis")
    if helper.select_targets(left, right, set(features)) != pilot["targets"]:
        raise ValueError("Source metadata do not reconstruct the frozen pilot target selection")
    for name, path in (("K562", args.k562), ("RPE1", args.rpe1)):
        stat = path.stat()
        if (stat.st_size, stat.st_mtime_ns) != source_stats[name]:
            raise ValueError("A Replogle source changed during metadata inspection")
    candidates = eligible.eligible_candidates(left, right, pilot)
    selected, summary = select_files(
        catalogs["files"], catalogs["experiments"], candidates, features, set(foundation_genes), pilot["targets"]
    )
    args.output_dir.mkdir(mode=0o700)
    inputs = {
        "catalog_snapshot": args.snapshot_sha256,
        "pilot_plan": common.PILOT_PLAN_SHA,
        "foundation_stage1_plan": common.FOUNDATION_PLAN_SHA,
        "k562_source": pilot["sources"]["K562"]["sha256"],
        "rpe1_source": pilot["sources"]["RPE1"]["sha256"],
        "preparation_script": eligible.PREPARATION_SHA,
        "eligibility_script": ELIGIBILITY_CODE_SHA,
        "selector_script": common.sha256(Path(__file__)),
    }
    manifest = {
        "format": "epi-c3-encode-k562-chip-selection-v1",
        "input_sha256": inputs,
        "selection_policy": (
            "Exact eligible target symbol and Ensembl ID; released, untreated and unmodified K562 "
            "experiments; replicated GRCh38 narrowPeak IDR files; one file per experiment, "
            "preferring conservative IDR, then more biological replicates, then accession."
        ),
        "files": selected,
    }
    manifest_sha = write_json(args.output_dir / "selected-files.json", manifest)
    report = {
        "format": "epi-c3-encode-k562-chip-selection-summary-v1",
        "status": "source-selection-only-not-a-graph-or-model",
        "input_sha256": {**inputs, "selected_file_manifest": manifest_sha},
        "catalog_counts": {kind: snapshot["catalogs"][kind]["records"] for kind in ("files", "experiments")},
        "selection": summary,
        "limitations": [
            "Experiment metadata describe untreated and unmodified K562 samples; independence of laboratory cultures from benchmark cells is not established.",
            "Exact target labels and Ensembl IDs were required; labels include cofactors and are not all verified transcription factors.",
            "Only two-context eligible single-gene conditions were considered; selection is post-hoc, not a new held-out split.",
            "K562 evidence is cross-cell-line transfer for RPE1; ChIP binding is not causal regulation.",
            "This is metadata selection, not a downloaded peak validation, graph, model, or cell-eval result.",
        ],
    }
    report_sha = write_json(args.output_dir / "selection-summary.json", report)
    print(json.dumps({"state": "complete", "manifest_sha256": manifest_sha,
                      "summary_sha256": report_sha, "selection": summary}), flush=True)


if __name__ == "__main__":
    main()
