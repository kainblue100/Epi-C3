"""Freeze previously evaluated Tahoe identities and recover their source components."""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import anndata as ad
import pandas as pd
import pyarrow.parquet as pq

from c3vcell.tahoe import TAHOE_CONTROL
import epi_c3_tahoe_isolation_v1 as isolation


def read_obs(path: Path) -> pd.DataFrame:
    data = ad.read_h5ad(path, backed="r")
    try:
        return data.obs.copy()
    finally:
        data.file.close()


def component(value) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("An evaluation artifact has missing Tahoe identity metadata")
    return value


def read_legacy(path: Path, pairs: dict, legacy_sources: dict, entries: dict) -> set[str]:
    obs = read_obs(path)
    if not {"source_cell_id", "sample", "source_file"}.issubset(obs.columns):
        raise ValueError("A legacy Tahoe split lacks source identity columns")
    observed = set()
    for record in obs.itertuples(index=False):
        raw = component(record.source_cell_id)
        parts = raw.split("|")
        if len(parts) != 2 or parts[0] != component(record.sample):
            raise ValueError("A legacy source identifier disagrees with its sample")
        sample, barcode = map(component, parts)
        source_file = component(record.source_file)
        if Path(source_file).name != source_file:
            raise ValueError("Legacy source file is not a shard basename")
        relative = f"data/{source_file}"
        if relative not in entries:
            raise ValueError("A legacy source shard is absent from the frozen Tahoe plan")
        cell_id = isolation.identity(sample, barcode)
        isolation.add_pair(pairs, cell_id, sample, barcode)
        legacy_sources[cell_id].add(relative)
        observed.add(cell_id)
    if len(observed) != len(obs):
        raise ValueError("A legacy evaluation split contains duplicate source identities")
    return observed


def validate_modern_id(cell_id: str) -> None:
    prefix = "tahoe-v1-"
    if not cell_id.startswith(prefix) or len(cell_id) != len(prefix) + 64:
        raise ValueError("Modern evaluation has a malformed Tahoe source identity")
    if any(char not in "0123456789abcdef" for char in cell_id[len(prefix):]):
        raise ValueError("Modern evaluation has a malformed Tahoe source identity")


def modern_expectations(obs, kind: str, conditions: dict, control_samples: dict) -> dict[str, set[str] | None]:
    if not {"target_gene", "context", "batch"}.issubset(obs.columns):
        raise ValueError("Modern evaluation split lacks condition metadata")
    expected = {}
    for cell_id, row in obs.iterrows():
        cell_id = component(cell_id)
        validate_modern_id(cell_id)
        if cell_id in expected:
            raise ValueError("A modern evaluation split contains duplicate identities")
        target, context, plate = map(component, (row["target_gene"], row["context"], row["batch"]))
        if kind == "smoke":
            expected[cell_id] = None
        elif target == "non-targeting":
            possible = control_samples.get((context, plate), set())
            if not possible:
                raise ValueError("An interim control has no matching source sample")
            expected[cell_id] = set(possible)
        else:
            condition = conditions.get(target)
            if condition is None or (context, plate) != (condition["cell_line_id"], condition["plate"]):
                raise ValueError("An interim treatment does not match its frozen condition")
            expected[cell_id] = {condition["sample"]}
    return expected


def verify_derived_references(run_root: Path, interim_ids: set[str]) -> int:
    checked = 0
    for dirname in ("tahoe-interim-cell-eval-inputs-v1", "tahoe-interim-matched-score-v1"):
        files = sorted((run_root / dirname).glob("plate-*/reference.h5ad"))
        if not files:
            raise ValueError(f"Missing derived evaluation references: {dirname}")
        for path in files:
            obs = read_obs(path)
            if not set(map(str, obs.index)).issubset(interim_ids):
                raise ValueError("A derived evaluation reference is not covered by its frozen source")
            checked += len(obs)
    return checked


def freeze(root: Path, run_root: Path, legacy_root: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError(output)
    metadata_path = run_root / "tahoe-full-metadata-v2/metadata-plan.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("state") != "complete":
        raise ValueError("A completed Tahoe metadata scan is required")
    entries = {item["path"]: item for item in metadata["provenance"]["files"]}
    if len(entries) != len(metadata["provenance"]["files"]):
        raise ValueError("The Tahoe source plan contains duplicate shards")

    pairs, legacy_sources, artifacts = {}, defaultdict(set), []
    for name in ("tahoe-demo-multidose", "tahoe-demo-plate4"):
        for role in ("reference", "input_controls"):
            path = legacy_root / name / "split" / f"{role}.h5ad"
            ids = read_legacy(path, pairs, legacy_sources, entries)
            artifacts.append({"label": f"{name}/{role}", "sha256": isolation.sha256(path), "identities": len(ids)})

    protocol_path = run_root / "tahoe-interim-evaluation-protocol-v1.json"
    state_path = run_root / "tahoe-interim-eval-v1/state.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if (
        state.get("state") != "complete"
        or state.get("protocol_sha256") != isolation.sha256(protocol_path)
        or protocol.get("metadata_plan_sha256") != isolation.sha256(metadata_path)
    ):
        raise ValueError("The interim evaluation does not match its frozen protocol")
    interim_files = {item["path"] for item in protocol["source_files"]}
    if len(interim_files) != len(protocol["source_files"]):
        raise ValueError("Interim evaluation source shards are duplicated")
    if any(entries.get(item["path"]) != item for item in protocol["source_files"]):
        raise ValueError("Interim evaluation source shards differ from the metadata plan")
    conditions = {item["target_gene"]: item for item in state["conditions"]}
    if len(conditions) != len(state["conditions"]):
        raise ValueError("Interim condition labels are duplicated")
    for condition in conditions.values():
        group = metadata["groups"].get(condition["key"])
        if group is None or any(
            group[field] != condition[field] for field in ("sample", "cell_line_id", "plate", "drug")
        ):
            raise ValueError("An interim condition does not match the full source plan")
    control_samples = defaultdict(set)
    for group in metadata["groups"].values():
        if group["drug"] == TAHOE_CONTROL:
            control_samples[(group["cell_line_id"], group["plate"])].add(group["sample"])

    smoke_metadata_path = run_root / "tahoe-smoke-metadata-v1/metadata-plan.json"
    smoke_metadata = json.loads(smoke_metadata_path.read_text(encoding="utf-8"))
    smoke_entries = smoke_metadata["provenance"]["files"]
    smoke_files = {item["path"] for item in smoke_entries}
    if smoke_metadata.get("state") != "complete" or any(entries.get(item["path"]) != item for item in smoke_entries):
        raise ValueError("Smoke source shards differ from the full metadata plan")

    expectations, origins, interim_reference = {}, {}, set()
    for kind, directory in (
        ("interim", run_root / "tahoe-interim-eval-v1"),
        ("smoke", run_root / "tahoe-real-evaluation-smoke-v1"),
    ):
        for role in ("reference", "input_controls"):
            path = directory / f"{role}.h5ad"
            digest = isolation.sha256(path)
            if kind == "interim" and state[f"{'reference' if role == 'reference' else 'input_control'}_sha256"] != digest:
                raise ValueError("An interim evaluation artifact changed")
            if kind == "smoke":
                mirror = run_root / "tahoe-parent-clean-real-smoke-v1" / path.name
                if isolation.sha256(mirror) != digest:
                    raise ValueError("The two smoke evaluation splits are not identical")
            obs = read_obs(path)
            if role == "input_controls" and not all(obs["target_gene"] == "non-targeting"):
                raise ValueError("An evaluation input-control split contains treated cells")
            if kind == "interim" and len(obs) != state[
                "reference_cells" if role == "reference" else "input_control_cells"
            ]:
                raise ValueError("An interim evaluation count differs from its frozen state")
            expected = modern_expectations(obs, kind, conditions, control_samples)
            label = f"{kind}/{role}"
            expectations[label] = expected
            origins[label] = set(expected)
            if kind == "interim" and role == "reference":
                interim_reference = set(expected)
            artifacts.append({"label": label, "sha256": digest, "identities": len(expected)})
    derived_rows = verify_derived_references(run_root, interim_reference)

    possible = set()
    modern_ids = set()
    for expected in expectations.values():
        modern_ids.update(expected)
        for samples in expected.values():
            if samples is not None:
                possible.update(samples)
    ordered_samples = sorted(possible)
    if not ordered_samples or any(not item.isascii() for item in ordered_samples):
        raise ValueError("Interim candidate samples must be nonempty ASCII strings")
    seen = {label: set() for label in expectations}
    groups_scanned = groups_pruned = 0
    for relative in sorted(interim_files | smoke_files):
        entry = entries[relative]
        parquet = pq.ParquetFile(isolation.check_shard(root, entry))
        for row_group in range(parquet.metadata.num_row_groups):
            smoke_candidate = relative in smoke_files
            interim_candidate = relative in interim_files and isolation.row_group_may_contain(
                parquet, row_group, ordered_samples,
            )
            if not smoke_candidate and not interim_candidate:
                groups_pruned += 1
                continue
            table = parquet.read_row_group(row_group, columns=["sample", "BARCODE_SUB_LIB_ID"])
            groups_scanned += 1
            for sample, barcode in zip(
                table.column("sample").to_pylist(), table.column("BARCODE_SUB_LIB_ID").to_pylist(), strict=True,
            ):
                if not smoke_candidate and sample not in possible:
                    continue
                cell_id = isolation.identity(sample, barcode)
                if cell_id not in modern_ids:
                    continue
                isolation.add_pair(pairs, cell_id, sample, barcode)
                for label, expected in expectations.items():
                    if cell_id not in expected:
                        continue
                    allowed_files = interim_files if label.startswith("interim/") else smoke_files
                    allowed_samples = expected[cell_id]
                    if relative not in allowed_files or (allowed_samples is not None and sample not in allowed_samples):
                        continue
                    seen[label].add(cell_id)
    missing = {label: len(ids - seen[label]) for label, ids in origins.items()}
    if any(missing.values()):
        raise ValueError(f"Could not recover every protected modern identity from its declared sources: {missing}")

    output.mkdir(mode=0o700)
    records = [
        {"cell_id": cell_id, "sample": pair[0], "barcode": pair[1],
         "legacy_source_files": sorted(legacy_sources.get(cell_id, ()))}
        for cell_id, pair in sorted(pairs.items())
    ]
    private_path = output / "identities.private.json"
    with private_path.open("x", encoding="utf-8") as stream:
        os.chmod(private_path, 0o600)
        json.dump({"format": "tahoe-protected-identities-private-v1", "records": records}, stream, allow_nan=False)
        stream.write("\n")
    manifest = {
        "format": "tahoe-protected-identities-v1",
        "state": "complete",
        "metadata_plan_sha256": isolation.sha256(metadata_path),
        "interim_protocol_sha256": isolation.sha256(protocol_path),
        "interim_state_sha256": isolation.sha256(state_path),
        "smoke_metadata_sha256": isolation.sha256(smoke_metadata_path),
        "identity_helper_sha256": isolation.sha256(Path(isolation.__file__)),
        "script_sha256": isolation.sha256(Path(__file__)),
        "private_file": private_path.name,
        "private_sha256": isolation.sha256(private_path),
        "protected_unique_identities": len(records),
        "legacy_identities_with_source_provenance": len(legacy_sources),
        "modern_unique_identities": len(modern_ids),
        "evaluation_artifacts": artifacts,
        "derived_reference_rows_covered": derived_rows,
        "modern_source_shards_checked": len(interim_files | smoke_files),
        "row_groups_scanned": groups_scanned,
        "row_groups_pruned_by_sample_statistics": groups_pruned,
        "source_integrity": "Frozen size and mtime checked; LFS digests were verified during download.",
        "notes": [
            "Private sample and barcode components are retained only for source-isolation auditing.",
            "Derived reference subsets are checked against the protected interim reference.",
            "A later full-shard scan must confirm all protected and validation overlaps before training.",
        ],
    }
    with (output / "manifest.json").open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({key: manifest[key] for key in (
        "protected_unique_identities", "modern_unique_identities", "modern_source_shards_checked",
        "row_groups_scanned", "row_groups_pruned_by_sample_statistics",
    )}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--legacy-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    freeze(args.root, args.run_root, args.legacy_root, args.output)
