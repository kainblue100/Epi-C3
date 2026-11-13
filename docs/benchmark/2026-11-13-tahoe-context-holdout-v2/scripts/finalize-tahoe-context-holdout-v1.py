"""Combine the candidate Tahoe context plan with its private-overlap audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


CONTEXTS = {"C32", "HOP62", "HepG2/C3A", "Hs 766T", "PANC-1"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def count(value) -> int:
    if type(value) is not int or value < 0:
        raise ValueError("Audit counts must be nonnegative integers")
    return value


def combine(summary, old_audit, intersection):
    if (
        summary.get("format") != "epi-c3-tahoe-state-context-holdout-summary-v1"
        or summary.get("status") != "candidate-not-enforced"
        or old_audit.get("state") != "complete"
        or intersection.get("format") != "epi-c3-tahoe-context-identity-intersection-v1"
        or intersection.get("status") != "complete"
    ):
        raise ValueError("The candidate plan or identity intersection is incomplete")
    source = summary["source_sha256"]
    for field in ("metadata_plan", "cell_line_metadata"):
        if intersection[f"{field}_sha256"] != source[field]:
            raise ValueError("The identity intersection used different source metadata")
    if (
        intersection["isolation_audit_sha256"] != source["isolation_audit"]
        or old_audit["metadata_plan_sha256"] != source["metadata_plan"]
    ):
        raise ValueError("The identity intersection used a different historical audit")
    contexts = summary["contexts"]
    if not isinstance(contexts, list) or len(contexts) != len(CONTEXTS) or {
        item["context"] for item in contexts
    } != CONTEXTS:
        raise ValueError("The candidate context summaries are incomplete or duplicated")
    source_rows = sum(count(item["source_rows_in_context"]) for item in contexts)
    test_rows = sum(count(item["matched_official_test_rows"]) for item in contexts)
    context_eligible = sum(
        count(item["eligible_treated_rows_before_identity_exclusions"])
        + count(item["eligible_training_control_rows_before_identity_exclusions"])
        for item in contexts
    )
    if (
        source_rows != count(summary["all_source_rows_in_heldout_contexts"])
        or test_rows != count(summary["official_test_rows_with_exact_condition_match"])
        or context_eligible != count(summary["eligible_rows_in_heldout_contexts_before_identity_exclusions"])
        or test_rows > source_rows or source_rows > count(summary["source_rows"])
        or context_eligible > source_rows
        or count(summary["remaining_eligible_rows_before_identity_exclusions"])
        != count(summary["original_eligible_rows_before_identity_exclusions"]) - context_eligible
    ):
        raise ValueError("The candidate context row totals disagree")
    protected = count(old_audit["protected_eligible_rows"])
    validation = count(old_audit["validation_eligible_rows"])
    if (
        count(intersection["protected_eligible"]) != protected
        or count(intersection["validation_eligible"]) != validation
        or count(intersection["protected_with_verified_shard"]) != protected
        or count(intersection["validation_with_verified_shard"]) != validation
        or count(intersection["protected_unresolved"]) != 0
        or count(intersection["validation_unresolved"]) != 0
        or count(intersection["source_files_scanned"]) != count(intersection["target_shards"])
        or count(intersection["target_shards"]) == 0
    ):
        raise ValueError("The historical identity intersection is incomplete")
    heldout = count(intersection["protected_heldout"]) + count(intersection["validation_heldout"])
    retained = count(intersection["protected_retained"]) + count(intersection["validation_retained"])
    if (
        count(intersection["protected_heldout"]) + count(intersection["protected_retained"]) != protected
        or count(intersection["validation_heldout"]) + count(intersection["validation_retained"]) != validation
        or protected + validation != summary["original_eligible_rows_before_identity_exclusions"]
        - summary["original_eligible_rows_after_identity_exclusions"]
    ):
        raise ValueError("The historical identity occurrence counts disagree")
    remaining = count(summary["remaining_eligible_rows_before_identity_exclusions"]) - retained
    lower = count(summary["remaining_eligible_rows_after_original_identity_exclusions_lower"])
    upper = count(summary["remaining_eligible_rows_after_original_identity_exclusions_upper"])
    if not lower <= remaining <= upper:
        raise ValueError("The exact remaining count falls outside its original bounds")
    output = {
        "format": "epi-c3-tahoe-state-context-holdout-exact-summary-v1",
        "status": "candidate-not-enforced",
        "source_rows": count(summary["source_rows"]),
        "official_test_rows_with_exact_condition_match": count(
            summary["official_test_rows_with_exact_condition_match"]
        ),
        "all_source_rows_in_heldout_contexts": count(summary["all_source_rows_in_heldout_contexts"]),
        "eligible_rows_in_heldout_contexts_before_identity_exclusions": count(
            summary["eligible_rows_in_heldout_contexts_before_identity_exclusions"]
        ),
        "remaining_eligible_rows_before_identity_exclusions": count(
            summary["remaining_eligible_rows_before_identity_exclusions"]
        ),
        "previous_identity_exclusions_already_in_heldout_contexts": heldout,
        "previous_identity_exclusions_outside_heldout_contexts": retained,
        "remaining_eligible_rows_after_original_identity_exclusions": remaining,
        "new_validation_selection_still_required": True,
        "contexts": contexts,
        "source_sha256": source,
        "limitations": [
            "The existing Tahoe continuation does not enforce this candidate.",
            "Every Tahoe row in the five official STATE test contexts is excluded by the candidate, including all controls and doses.",
            "The count is before any additional validation identities are selected for a new protected stream.",
            "Stage-one and replay source overlap with test cells or conditions has not been ruled out.",
            "Source-shard validation uses frozen size and mtime plus previously verified download hashes.",
            "Historical identity counts rely on the earlier audit and the present metadata-only shard check.",
            "Rows are source occurrences, not deduplicated biological cells.",
        ],
    }
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("summary", "old-audit", "intersection", "intersection-script", "guard", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    old_audit = json.loads(args.old_audit.read_text(encoding="utf-8"))
    intersection = json.loads(args.intersection.read_text(encoding="utf-8"))
    if (
        sha256(args.old_audit) != summary["source_sha256"]["isolation_audit"]
        or sha256(args.intersection_script) != intersection["script_sha256"]
    ):
        raise ValueError("The historical audit or intersection script changed")
    result = combine(summary, old_audit, intersection)
    result["candidate_summary_sha256"] = sha256(args.summary)
    result["identity_intersection_sha256"] = sha256(args.intersection)
    result["identity_intersection_script_sha256"] = intersection["script_sha256"]
    result["guard_sha256"] = sha256(args.guard)
    descriptor = os.open(args.output, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({
        "status": result["status"],
        "remaining_eligible_rows_after_original_identity_exclusions":
            result["remaining_eligible_rows_after_original_identity_exclusions"],
        "new_validation_selection_still_required": result["new_validation_selection_still_required"],
    }), flush=True)


if __name__ == "__main__":
    main()
