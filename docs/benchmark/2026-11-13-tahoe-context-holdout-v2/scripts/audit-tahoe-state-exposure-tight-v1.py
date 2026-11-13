"""Bound Tahoe STATE-test exposure using the frozen training eligibility rules."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import tomllib
from collections import Counter, defaultdict
from pathlib import Path


CONTEXTS = ("C32", "HOP62", "HepG2/C3A", "Hs 766T", "PANC-1")
CONTROL = "DMSO_TF"
CONTROL_ROLES = ("training", "reference", "input_controls")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_smiles(value):
    if value is None or isinstance(value, float) and math.isnan(value):
        return None
    if not isinstance(value, str):
        raise ValueError("A canonical SMILES value must be text or missing")
    value = value.strip()
    if not value:
        return None
    if len(value) > 512 or not value.isascii() or any(
        char.isspace() or not char.isprintable() for char in value
    ):
        raise ValueError("A canonical SMILES value is invalid")
    return value


def count(value):
    if type(value) is not int or value < 0:
        raise ValueError("Counts must be nonnegative integers")
    return value


def audit(metadata, isolation, split, cell_lines, samples, drugs):
    if metadata.get("state") != "complete" or isolation.get("state") != "complete":
        raise ValueError("Metadata and identity audits must be complete")
    groups = metadata["groups"]
    total = count(metadata["rows"])
    if sum(count(group["cells"]) for group in groups.values()) != total:
        raise ValueError("Group counts do not account for the source universe")
    if isolation["source_rows_checked"] != total:
        raise ValueError("Identity audit has a different source universe")

    structures = {}
    for row in drugs:
        drug = row["drug"]
        if not isinstance(drug, str) or not drug or drug in structures:
            raise ValueError("Drug metadata must have unique nonempty names")
        structures[drug] = normalized_smiles(row["canonical_smiles"])
    for key, group in groups.items():
        if key != f"{group['sample']}\t{group['cell_line_id']}":
            raise ValueError("Unexpected group identity")
        drug = group["drug"]
        structure = group["canonical_smiles"]
        if drug == CONTROL or structure is None:
            continue
        if normalized_smiles(structure) != structure:
            raise ValueError("A source group has an unnormalized structure")
        previous = structures.get(drug)
        if previous is not None and previous != structure:
            raise ValueError("A drug has inconsistent structures")
        structures[drug] = structure
    heldout = set(metadata["heldout_compounds"])
    aliases = {drug for drug, structure in structures.items() if structure in heldout}
    if not heldout.issubset({structures[drug] for drug in aliases}):
        raise ValueError("A held-out structure has no identifiable drug")

    names = defaultdict(set)
    for row in cell_lines:
        names[row["cell_name"]].add(row["Cell_ID_Cellosaur"])
    line_to_context = {}
    for name in CONTEXTS:
        ids = names[name]
        if len(ids) != 1 or None in ids:
            raise ValueError("An official context does not map to one cell line")
        line = next(iter(ids))
        if line in line_to_context:
            raise ValueError("Official contexts map to the same cell line")
        line_to_context[line] = name
    definitions = split["fewshot"]
    if set(definitions) != {f"tahoe_holdout.{name}" for name in CONTEXTS}:
        raise ValueError("Unexpected official test contexts")
    labels = {}
    for name in CONTEXTS:
        values = definitions[f"tahoe_holdout.{name}"]["test"]
        if not isinstance(values, list) or not values or not all(
            isinstance(value, str) and value for value in values
        ):
            raise ValueError("Invalid official test-condition labels")
        labels[name] = set(values)
    sample_conditions = {}
    for row in samples:
        sample, condition = row["sample"], row["drugname_drugconc"]
        if (
            not isinstance(sample, str) or not sample or sample in sample_conditions
            or not isinstance(condition, str) or not condition
        ):
            raise ValueError("Sample metadata must have unique complete labels")
        sample_conditions[sample] = condition

    results = {name: Counter() for name in CONTEXTS}
    source_controls, matching_controls = Counter(), Counter()
    eligible_treated = 0
    for group in groups.values():
        sample, line, plate, drug = (group[k] for k in ("sample", "cell_line_id", "plate", "drug"))
        if sample not in sample_conditions:
            raise ValueError("A source sample is absent from sample metadata")
        n = count(group["cells"])
        context = line_to_context.get(line)
        matched = context is not None and sample_conditions[sample] in labels[context]
        if matched:
            results[context]["matched_source_rows"] += n
        if drug == CONTROL:
            key = f"{line}\t{plate}"
            source_controls[key] += n
            if matched:
                matching_controls[key] += n
                results[context]["matched_control_rows"] += n
        elif structures.get(drug) is None:
            if matched:
                results[context]["matched_missing_structure_rows"] += n
        elif drug in aliases:
            if matched:
                results[context]["matched_heldout_compound_rows"] += n
        else:
            eligible_treated += n
            if matched:
                results[context]["matched_eligible_treated_rows"] += n

    control_totals = Counter()
    if set(source_controls) != set(metadata["control_groups"]):
        raise ValueError("Control-group universe differs from scanned source groups")
    for key, roles in metadata["control_groups"].items():
        counts = {role: count(roles[role]) for role in CONTROL_ROLES}
        control_totals.update(counts)
        group_total = sum(counts.values())
        if group_total != source_controls[key]:
            raise ValueError("Control-group source and role counts differ")
        line, separator, _ = key.partition("\t")
        if not separator:
            raise ValueError("Control group lacks a cell-line and plate")
        context = line_to_context.get(line)
        if context is not None:
            matched = matching_controls[key]
            training = counts["training"]
            results[context]["matched_control_training_lower"] += max(
                0, training - (group_total - matched)
            )
            results[context]["matched_control_training_upper"] += min(training, matched)
    if dict(control_totals) != {
        role: count(metadata["control_counts"][role]) for role in CONTROL_ROLES
    }:
        raise ValueError("Control partition totals disagree")

    before = eligible_treated + control_totals["training"]
    protected = count(isolation["protected_eligible_rows"])
    validation = count(isolation["validation_eligible_rows"])
    after = count(isolation["eligible_training_rows_after_exclusions"])
    if before != isolation["eligible_rows_before_exclusions"] or after != before - protected - validation:
        raise ValueError("Frozen training eligibility totals disagree")
    private_exclusions = protected + validation
    by_context = []
    lower_before = upper_before = matched_source = 0
    for context in CONTEXTS:
        values = results[context]
        expected = (
            values["matched_eligible_treated_rows"] + values["matched_heldout_compound_rows"]
            + values["matched_missing_structure_rows"] + values["matched_control_rows"]
        )
        if expected != values["matched_source_rows"]:
            raise ValueError("Matched rows do not partition by eligibility")
        lower = values["matched_eligible_treated_rows"] + values["matched_control_training_lower"]
        upper = values["matched_eligible_treated_rows"] + values["matched_control_training_upper"]
        lower_before += lower
        upper_before += upper
        matched_source += values["matched_source_rows"]
        by_context.append({
            "context": context,
            "matched_source_rows": values["matched_source_rows"],
            "matched_eligible_treated_rows": values["matched_eligible_treated_rows"],
            "matched_heldout_compound_rows": values["matched_heldout_compound_rows"],
            "matched_missing_structure_rows": values["matched_missing_structure_rows"],
            "matched_control_rows": values["matched_control_rows"],
            "matched_control_training_lower": values["matched_control_training_lower"],
            "matched_control_training_upper": values["matched_control_training_upper"],
            "matched_eligible_lower_before_identity_exclusions": lower,
            "matched_eligible_upper_before_identity_exclusions": upper,
        })
    return {
        "format": "epi-c3-tahoe-state-exposure-tight-audit-v1",
        "status": "complete",
        "scope": "Prospective exposure under the existing planned Tahoe continuation",
        "source_rows": total,
        "eligible_before_identity_exclusions": before,
        "eligible_after_identity_exclusions": after,
        "matched_official_test_rows": matched_source,
        "matched_eligible_lower_before_identity_exclusions": lower_before,
        "matched_eligible_upper_before_identity_exclusions": upper_before,
        "private_identity_exclusions_across_all_conditions": private_exclusions,
        "minimum_matched_rows_after_identity_exclusions": max(0, lower_before - private_exclusions),
        "maximum_matched_rows_after_identity_exclusions": min(upper_before, after),
        "contexts": by_context,
        "limitations": [
            "Prospective bounds apply only if the existing scheduled Tahoe stream is fully consumed.",
            "Exact condition labels can miss aliases or equivalent doses; rows are not unique cells.",
            "When only part of a plate's control group matches, control eligibility is bounded, not assumed.",
            "The identity exclusions are subtracted in the worst case, not joined to test conditions.",
            "This does not establish stage-one or cross-source isolation or prove continuation has started.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("metadata-plan", "isolation-audit", "official-split", "cell-line-metadata",
                 "sample-metadata", "drug-metadata", "previous-audit", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    inputs = {
        "metadata_plan": args.metadata_plan, "isolation_audit": args.isolation_audit,
        "official_split": args.official_split, "cell_line_metadata": args.cell_line_metadata,
        "sample_metadata": args.sample_metadata,
    }
    previous = json.loads(args.previous_audit.read_text(encoding="utf-8"))
    hashes = {name: sha256(path) for name, path in inputs.items()}
    if previous.get("source_sha256") != hashes:
        raise ValueError("Inputs do not match the previous published exposure audit")
    metadata = json.loads(args.metadata_plan.read_text(encoding="utf-8"))
    isolation = json.loads(args.isolation_audit.read_text(encoding="utf-8"))
    drug_hash = sha256(args.drug_metadata)
    if (
        isolation.get("metadata_plan_sha256") != hashes["metadata_plan"]
        or metadata["provenance"]["sample_metadata_sha256"] != hashes["sample_metadata"]
        or metadata["provenance"]["drug_metadata_sha256"] != drug_hash
    ):
        raise ValueError("Source metadata is not bound to the frozen plan")
    import pyarrow.parquet as pq

    result = audit(
        metadata, isolation, tomllib.loads(args.official_split.read_text(encoding="utf-8")),
        pq.read_table(args.cell_line_metadata, columns=["cell_name", "Cell_ID_Cellosaur"]).to_pylist(),
        pq.read_table(args.sample_metadata, columns=["sample", "drugname_drugconc"]).to_pylist(),
        pq.read_table(args.drug_metadata, columns=["drug", "canonical_smiles"]).to_pylist(),
    )
    if result["matched_official_test_rows"] != previous["matched_official_test_rows"]:
        raise ValueError("Recomputed matched rows disagree with the published audit")
    if result["minimum_matched_rows_after_identity_exclusions"] < previous[
        "minimum_matched_rows_in_planned_eligible_stream"
    ]:
        raise ValueError("The refined bound is weaker than the published set bound")
    result["source_sha256"] = {
        **hashes, "drug_metadata": drug_hash, "previous_audit": sha256(args.previous_audit),
        "audit_script": sha256(Path(__file__)),
    }
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({
        key: result[key] for key in (
            "matched_official_test_rows", "minimum_matched_rows_after_identity_exclusions",
            "maximum_matched_rows_after_identity_exclusions",
        )
    }))


if __name__ == "__main__":
    main()
