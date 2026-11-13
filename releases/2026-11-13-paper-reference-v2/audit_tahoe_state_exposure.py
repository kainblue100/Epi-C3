"""Audit prospective stage-two overlap with the official STATE Tahoe test split."""

from __future__ import annotations

import argparse
import hashlib
import json
import tomllib
from collections import defaultdict
from pathlib import Path


EXPECTED_CONTEXTS = ("C32", "HOP62", "HepG2/C3A", "Hs 766T", "PANC-1")
EXPECTED_SPLIT_SHA256 = "eac16fb06052dec9d6842dfae04d5ca693b847e3432f719a975adce205707783"


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def audit(metadata: dict, isolation: dict, split: dict, cell_lines: list[dict], samples: list[dict]) -> dict:
    if metadata.get("state") != "complete" or isolation.get("state") != "complete":
        raise ValueError("Both input audits must be complete")
    total = int(metadata["rows"])
    eligible = int(isolation["eligible_training_rows_after_exclusions"])
    groups = metadata["groups"]
    if (
        total != int(isolation["source_rows_checked"])
        or sum(int(group["cells"]) for group in groups.values()) != total
        or not 0 <= eligible <= total
    ):
        raise ValueError("Source row universe or eligible count is inconsistent")

    names = defaultdict(set)
    for row in cell_lines:
        names[row["cell_name"]].add(row["Cell_ID_Cellosaur"])
    sample_conditions = {}
    for row in samples:
        sample = row["sample"]
        condition = row["drugname_drugconc"]
        if sample in sample_conditions or not isinstance(condition, str):
            raise ValueError("Sample metadata must have unique identifiers and condition labels")
        sample_conditions[sample] = condition
    if any(group["sample"] not in sample_conditions for group in groups.values()):
        raise ValueError("A source group is absent from sample metadata")

    definitions = split["fewshot"]
    if set(definitions) != {f"tahoe_holdout.{name}" for name in EXPECTED_CONTEXTS}:
        raise ValueError("Unexpected official Tahoe test contexts")
    context_ids, test_labels = {}, {}
    for name in EXPECTED_CONTEXTS:
        ids = names[name]
        if len(ids) != 1 or None in ids:
            raise ValueError(f"Context does not map to exactly one cell-line ID: {name}")
        context_ids[name] = next(iter(ids))
        labels = definitions[f"tahoe_holdout.{name}"]["test"]
        if not isinstance(labels, list) or not labels or not all(isinstance(item, str) for item in labels):
            raise ValueError(f"Unexpected test conditions: {name}")
        test_labels[name] = set(labels)
    if len(set(context_ids.values())) != len(EXPECTED_CONTEXTS):
        raise ValueError("Distinct official contexts map to the same cell-line ID")

    id_to_name = {cell_id: name for name, cell_id in context_ids.items()}
    matched_rows = {name: 0 for name in EXPECTED_CONTEXTS}
    matched_conditions = {name: set() for name in EXPECTED_CONTEXTS}
    for group in groups.values():
        name = id_to_name.get(group["cell_line_id"])
        if name is None:
            continue
        condition = sample_conditions[group["sample"]]
        if condition in test_labels[name]:
            matched_rows[name] += int(group["cells"])
            matched_conditions[name].add(condition)
    contexts = [
        {
            "context": name,
            "test_condition_entries": len(definitions[f"tahoe_holdout.{name}"]["test"]),
            "unique_test_conditions": len(test_labels[name]),
            "matched_unique_test_conditions": len(matched_conditions[name]),
            "matched_source_rows": matched_rows[name],
        }
        for name in EXPECTED_CONTEXTS
    ]
    matched = sum(matched_rows.values())
    excluded_or_ineligible = total - eligible
    return {
        "format": "epi-c3-tahoe-state-exposure-audit-v1",
        "status": "complete",
        "scope": "Prospective exposure under the existing planned stage-two Tahoe stream",
        "stage2_has_been_run_by_this_audit": False,
        "matching": "Exact sample condition string and unambiguous cell-line metadata; aliases are not expanded.",
        "source_rows": total,
        "planned_eligible_stage2_rows": eligible,
        "source_rows_not_eligible_for_stage2": excluded_or_ineligible,
        "matched_official_test_rows": matched,
        "minimum_matched_rows_in_planned_eligible_stream": max(0, matched - excluded_or_ineligible),
        "contexts": contexts,
        "interpretation": (
            "The minimum follows from set cardinalities, not a row-level intersection. "
            "It applies if the planned stage-two stream consumes all eligible rows; "
            "these are row occurrences, not deduplicated biological cells."
        ),
        "limitations": [
            "This is a prospective data-exposure audit, not proof that stage two has started or finished.",
            "Exact matching can miss equivalent condition strings or aliases and is therefore conservative.",
            "The lower bound does not identify which matched rows are eligible or establish their uniqueness.",
            "The historical isolation audit protects listed legacy evaluation identities, not this official split.",
            "Independent cross-source duplicates and prior stage-one exposure are not audited here.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-plan", required=True, type=Path)
    parser.add_argument("--isolation-audit", required=True, type=Path)
    parser.add_argument("--split-toml", required=True, type=Path)
    parser.add_argument("--cell-line-metadata", required=True, type=Path)
    parser.add_argument("--sample-metadata", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    hashes = {
        "metadata_plan": digest(args.metadata_plan),
        "isolation_audit": digest(args.isolation_audit),
        "official_split": digest(args.split_toml),
        "cell_line_metadata": digest(args.cell_line_metadata),
        "sample_metadata": digest(args.sample_metadata),
    }
    if hashes["official_split"] != EXPECTED_SPLIT_SHA256:
        raise ValueError("Unexpected official STATE split version")
    metadata = json.loads(args.metadata_plan.read_text(encoding="utf-8"))
    isolation = json.loads(args.isolation_audit.read_text(encoding="utf-8"))
    if isolation.get("metadata_plan_sha256") != hashes["metadata_plan"]:
        raise ValueError("The isolation audit is not bound to this metadata plan")
    if metadata["provenance"]["sample_metadata_sha256"] != hashes["sample_metadata"]:
        raise ValueError("The metadata plan is not bound to this sample metadata")
    import pyarrow.parquet as pq

    result = audit(
        metadata, isolation,
        tomllib.loads(args.split_toml.read_text(encoding="utf-8")),
        pq.read_table(args.cell_line_metadata, columns=["cell_name", "Cell_ID_Cellosaur"]).to_pylist(),
        pq.read_table(args.sample_metadata, columns=["sample", "drugname_drugconc"]).to_pylist(),
    )
    result["source_sha256"] = hashes
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({
        "matched_official_test_rows": result["matched_official_test_rows"],
        "minimum_matched_rows_in_planned_eligible_stream": result["minimum_matched_rows_in_planned_eligible_stream"],
    }))


if __name__ == "__main__":
    main()
