"""Prepare a metadata-only Tahoe holdout candidate; do not run training."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tomllib
from collections import Counter, defaultdict
from pathlib import Path


CONTEXTS = ("C32", "HOP62", "HepG2/C3A", "Hs 766T", "PANC-1")
CONTROL = "DMSO_TF"
CONTROL_ROLES = ("training", "reference", "input_controls")
EXPOSURE_FORMAT = "epi-c3-tahoe-state-exposure-tight-audit-v1"
EXPECTED_EXPOSURE_SHA256 = "16438bf4867c0691630fbbb7e0808d88f9505f2df81b83679c3ca082d042fd22"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checked_count(value) -> int:
    if type(value) is not int or value < 0:
        raise ValueError("Counts must be nonnegative integers")
    return value


def checked_text(value, what: str) -> str:
    if not isinstance(value, str) or not value or "\t" in value:
        raise ValueError(f"{what} must be a nonempty string without tabs")
    return value


def structure(value):
    if value is None or isinstance(value, float) and math.isnan(value):
        return None
    if not isinstance(value, str):
        raise ValueError("A canonical structure must be text or missing")
    value = value.strip()
    if not value:
        return None
    if len(value) > 512 or not value.isascii() or any(
        char.isspace() or not char.isprintable() for char in value
    ):
        raise ValueError("An invalid canonical structure was supplied")
    return value


def build_plan(metadata, isolation, exposure, split, cell_lines, samples, drugs):
    if metadata.get("state") != "complete" or isolation.get("state") != "complete":
        raise ValueError("The source and historical isolation audits must be complete")
    if exposure.get("format") != EXPOSURE_FORMAT or exposure.get("status") != "complete":
        raise ValueError("The earlier STATE exposure audit must be complete")
    groups = metadata["groups"]
    if not isinstance(groups, dict) or not groups:
        raise ValueError("The frozen source groups are missing")
    total = checked_count(metadata["rows"])
    if isolation["source_rows_checked"] != total or exposure["source_rows"] != total:
        raise ValueError("The audits do not cover the same source universe")

    definitions = split["fewshot"]
    if set(definitions) != {f"tahoe_holdout.{name}" for name in CONTEXTS}:
        raise ValueError("Unexpected official STATE test contexts")
    labels = {}
    for name in CONTEXTS:
        values = definitions[f"tahoe_holdout.{name}"]["test"]
        if not isinstance(values, list) or not values:
            raise ValueError("A STATE context lacks a test-condition list")
        labels[name] = {checked_text(value, "A STATE condition") for value in values}

    names = defaultdict(set)
    for item in cell_lines:
        if item.get("cell_name") in CONTEXTS:
            names[item["cell_name"]].add(checked_text(item.get("Cell_ID_Cellosaur"), "A cell-line ID"))
    context_to_line = {}
    for name in CONTEXTS:
        if len(names[name]) != 1:
            raise ValueError(f"The STATE context {name!r} does not map to exactly one cell line")
        context_to_line[name] = next(iter(names[name]))
    if len(set(context_to_line.values())) != len(CONTEXTS):
        raise ValueError("Distinct STATE contexts map to the same cell line")
    line_to_context = {line: name for name, line in context_to_line.items()}

    sample_conditions = {}
    for item in samples:
        sample = checked_text(item.get("sample"), "A sample")
        condition = checked_text(item.get("drugname_drugconc"), "A sample condition")
        if sample in sample_conditions:
            raise ValueError("Sample metadata contains duplicate samples")
        sample_conditions[sample] = condition

    structures = {}
    for item in drugs:
        drug = checked_text(item.get("drug"), "A drug")
        if drug in structures:
            raise ValueError("Drug metadata contains duplicate names")
        structures[drug] = structure(item.get("canonical_smiles"))

    source_controls = Counter()
    group_rows = 0
    seen_lines = set()
    for key, group in groups.items():
        sample = checked_text(group.get("sample"), "A source sample")
        line = checked_text(group.get("cell_line_id"), "A source cell-line ID")
        plate = checked_text(group.get("plate"), "A source plate")
        drug = checked_text(group.get("drug"), "A source drug")
        if key != f"{sample}\t{line}" or sample not in sample_conditions:
            raise ValueError("A frozen source group is inconsistent with sample metadata")
        cells = checked_count(group.get("cells"))
        if cells == 0:
            raise ValueError("A frozen source group cannot be empty")
        group_rows += cells
        seen_lines.add(line)
        if drug == CONTROL:
            source_controls[f"{line}\t{plate}"] += cells
            continue
        canonical = structure(group.get("canonical_smiles"))
        if canonical is not None:
            if canonical != group["canonical_smiles"]:
                raise ValueError("A frozen group structure is not normalized")
            previous = structures.get(drug)
            if previous is not None and previous != canonical:
                raise ValueError("A drug has inconsistent source structures")
            structures[drug] = canonical
    if group_rows != total or not set(context_to_line.values()).issubset(seen_lines):
        raise ValueError("Source-group totals or STATE cell-line coverage are incomplete")

    heldout = {structure(value) for value in metadata["heldout_compounds"]}
    if None in heldout:
        raise ValueError("A previously held-out structure is missing")
    aliases = {drug for drug, value in structures.items() if value in heldout}
    if not heldout.issubset({structures[drug] for drug in aliases}):
        raise ValueError("A previously held-out structure has no identifiable drug")

    control_groups = metadata["control_groups"]
    if set(source_controls) != set(control_groups):
        raise ValueError("Control-group universe changed")
    control_totals, blocked_control_training = Counter(), Counter()
    for key, roles in control_groups.items():
        line, separator, plate = key.partition("\t")
        if not separator or not line or not plate or "\t" in plate:
            raise ValueError("Invalid source control-group key")
        counts = {role: checked_count(roles[role]) for role in CONTROL_ROLES}
        if sum(counts.values()) != source_controls[key]:
            raise ValueError("A control-group source count does not match its partition")
        control_totals.update(counts)
        if line in line_to_context:
            blocked_control_training[line_to_context[line]] += counts["training"]
    if dict(control_totals) != {
        role: checked_count(metadata["control_counts"][role]) for role in CONTROL_ROLES
    }:
        raise ValueError("Control partition totals changed")

    source_by_context, treated_by_context, matched_by_context = Counter(), Counter(), Counter()
    all_eligible_treated = 0
    for group in groups.values():
        line, drug, sample = group["cell_line_id"], group["drug"], group["sample"]
        cells = group["cells"]
        context = line_to_context.get(line)
        if context is not None:
            source_by_context[context] += cells
            if sample_conditions[sample] in labels[context]:
                matched_by_context[context] += cells
        if drug != CONTROL and structures.get(drug) is not None and drug not in aliases:
            all_eligible_treated += cells
            if context is not None:
                treated_by_context[context] += cells

    prior_contexts = {item["context"]: item for item in exposure["contexts"]}
    if len(prior_contexts) != len(exposure["contexts"]) or set(prior_contexts) != set(CONTEXTS):
        raise ValueError("The earlier exposure audit has different contexts")
    for name in CONTEXTS:
        if matched_by_context[name] != checked_count(prior_contexts[name]["matched_source_rows"]):
            raise ValueError("The earlier exact-condition exposure does not match source groups")
    matched_rows = sum(matched_by_context.values())
    if matched_rows != exposure["matched_official_test_rows"]:
        raise ValueError("The earlier total test exposure does not match source groups")

    eligible_before = all_eligible_treated + control_totals["training"]
    protected = checked_count(isolation["protected_eligible_rows"])
    validation = checked_count(isolation["validation_eligible_rows"])
    private_exclusions = protected + validation
    eligible_after = checked_count(isolation["eligible_training_rows_after_exclusions"])
    if (
        eligible_before != isolation["eligible_rows_before_exclusions"]
        or eligible_before != exposure["eligible_before_identity_exclusions"]
        or eligible_after != eligible_before - private_exclusions
        or eligible_after != exposure["eligible_after_identity_exclusions"]
    ):
        raise ValueError("The historical eligibility or identity-exclusion totals disagree")

    blocked_source = sum(source_by_context.values())
    blocked_treated = sum(treated_by_context.values())
    blocked_controls = sum(blocked_control_training.values())
    blocked_eligible = blocked_treated + blocked_controls
    remaining_before = eligible_before - blocked_eligible
    if (
        blocked_source < matched_rows or not 0 <= blocked_eligible <= eligible_before
        or blocked_eligible < exposure["matched_eligible_lower_before_identity_exclusions"]
        or not 0 <= private_exclusions <= eligible_before
    ):
        raise ValueError("The candidate holdout is inconsistent with the prior exposure audit")
    lower = max(0, remaining_before - private_exclusions)
    upper = min(remaining_before, eligible_after)
    if lower > upper:
        raise ValueError("The candidate remaining-row bounds are invalid")

    private = {
        "format": "epi-c3-tahoe-state-context-holdout-private-v1",
        "status": "candidate-not-enforced",
        "blocked_cell_line_ids": sorted(context_to_line.values()),
        "policy": "Exclude every Tahoe row in the five STATE test contexts from foundation training and validation.",
    }
    summary = {
        "format": "epi-c3-tahoe-state-context-holdout-summary-v1",
        "status": "candidate-not-enforced",
        "source_rows": total,
        "original_eligible_rows_before_identity_exclusions": eligible_before,
        "original_eligible_rows_after_identity_exclusions": eligible_after,
        "official_test_rows_with_exact_condition_match": matched_rows,
        "all_source_rows_in_heldout_contexts": blocked_source,
        "eligible_treated_rows_in_heldout_contexts_before_identity_exclusions": blocked_treated,
        "eligible_training_control_rows_in_heldout_contexts_before_identity_exclusions": blocked_controls,
        "eligible_rows_in_heldout_contexts_before_identity_exclusions": blocked_eligible,
        "remaining_eligible_rows_before_identity_exclusions": remaining_before,
        "remaining_eligible_rows_after_original_identity_exclusions_lower": lower,
        "remaining_eligible_rows_after_original_identity_exclusions_upper": upper,
        "contexts": [
            {
                "context": name,
                "source_rows_in_context": source_by_context[name],
                "matched_official_test_rows": matched_by_context[name],
                "eligible_treated_rows_before_identity_exclusions": treated_by_context[name],
                "eligible_training_control_rows_before_identity_exclusions": blocked_control_training[name],
            }
            for name in CONTEXTS
        ],
        "limitations": [
            "This metadata plan is not enforced by the existing scheduled Tahoe continuation.",
            "All Tahoe rows from five contexts are held out, including every drug, dose, and control.",
            "The count interval reflects unknown overlap with historical protected and validation identities.",
            "A new stream must preserve historical identity exclusions and select validation after this holdout.",
            "The stage-one checkpoint and replay sources have not been proven independent of the test data.",
            "Cell-line annotation, cross-source identity, source integrity, and downstream split correctness require independent verification.",
            "These are source row occurrences, not deduplicated biological cells.",
        ],
    }
    return private, summary


def write_private_json(path: Path, value) -> None:
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "metadata-plan", "isolation-audit", "official-split", "cell-line-metadata",
        "sample-metadata", "drug-metadata", "exposure-audit", "output",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    if sha256(args.exposure_audit) != EXPECTED_EXPOSURE_SHA256:
        raise ValueError("The earlier STATE exposure audit has an unexpected digest")
    exposure = json.loads(args.exposure_audit.read_text(encoding="utf-8"))
    inputs = {
        "metadata_plan": args.metadata_plan, "isolation_audit": args.isolation_audit,
        "official_split": args.official_split, "cell_line_metadata": args.cell_line_metadata,
        "sample_metadata": args.sample_metadata, "drug_metadata": args.drug_metadata,
    }
    hashes = {name: sha256(path) for name, path in inputs.items()}
    if any(exposure["source_sha256"].get(name) != value for name, value in hashes.items()):
        raise ValueError("The candidate inputs differ from the earlier frozen exposure audit")
    metadata = json.loads(args.metadata_plan.read_text(encoding="utf-8"))
    isolation = json.loads(args.isolation_audit.read_text(encoding="utf-8"))
    if (
        isolation.get("metadata_plan_sha256") != hashes["metadata_plan"]
        or metadata["provenance"]["sample_metadata_sha256"] != hashes["sample_metadata"]
        or metadata["provenance"]["drug_metadata_sha256"] != hashes["drug_metadata"]
    ):
        raise ValueError("The candidate source metadata is not provenance-bound")
    import pyarrow.parquet as pq

    private, summary = build_plan(
        metadata, isolation, exposure, tomllib.loads(args.official_split.read_text(encoding="utf-8")),
        pq.read_table(args.cell_line_metadata, columns=["cell_name", "Cell_ID_Cellosaur"]).to_pylist(),
        pq.read_table(args.sample_metadata, columns=["sample", "drugname_drugconc"]).to_pylist(),
        pq.read_table(args.drug_metadata, columns=["drug", "canonical_smiles"]).to_pylist(),
    )
    private["source_sha256"] = hashes
    private["exposure_audit_sha256"] = EXPECTED_EXPOSURE_SHA256
    private["planner_sha256"] = sha256(Path(__file__))
    args.output.mkdir(mode=0o700)
    private_path = args.output / "context-holdout.private.json"
    write_private_json(private_path, private)
    summary["source_sha256"] = hashes
    summary["exposure_audit_sha256"] = EXPECTED_EXPOSURE_SHA256
    summary["planner_sha256"] = private["planner_sha256"]
    summary["private_plan_sha256"] = sha256(private_path)
    write_private_json(args.output / "summary.json", summary)
    print(json.dumps({
        key: summary[key] for key in (
            "status", "all_source_rows_in_heldout_contexts",
            "eligible_rows_in_heldout_contexts_before_identity_exclusions",
            "remaining_eligible_rows_after_original_identity_exclusions_lower",
            "remaining_eligible_rows_after_original_identity_exclusions_upper",
        )
    }), flush=True)


if __name__ == "__main__":
    main()
