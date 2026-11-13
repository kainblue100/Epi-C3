import copy
import importlib.util
import unittest
from pathlib import Path


spec = importlib.util.spec_from_file_location(
    "finalize_holdout", Path(__file__).with_name("finalize-tahoe-context-holdout-v1.py")
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def fixture():
    source = {"metadata_plan": "metadata", "isolation_audit": "audit", "cell_line_metadata": "lines"}
    summary = {
        "format": "epi-c3-tahoe-state-context-holdout-summary-v1", "status": "candidate-not-enforced",
        "source_sha256": source, "source_rows": 110,
        "original_eligible_rows_before_identity_exclusions": 104,
        "original_eligible_rows_after_identity_exclusions": 101,
        "official_test_rows_with_exact_condition_match": 50,
        "all_source_rows_in_heldout_contexts": 85,
        "eligible_rows_in_heldout_contexts_before_identity_exclusions": 80,
        "remaining_eligible_rows_before_identity_exclusions": 24,
        "remaining_eligible_rows_after_original_identity_exclusions_lower": 21,
        "remaining_eligible_rows_after_original_identity_exclusions_upper": 24,
        "contexts": [{
            "context": name, "source_rows_in_context": 17, "matched_official_test_rows": 10,
            "eligible_treated_rows_before_identity_exclusions": 12,
            "eligible_training_control_rows_before_identity_exclusions": 4,
        } for name in sorted(module.CONTEXTS)],
    }
    audit = {
        "state": "complete", "metadata_plan_sha256": "metadata",
        "protected_eligible_rows": 2, "validation_eligible_rows": 1,
    }
    intersection = {
        "format": "epi-c3-tahoe-context-identity-intersection-v1", "status": "complete",
        "metadata_plan_sha256": "metadata", "isolation_audit_sha256": "audit",
        "cell_line_metadata_sha256": "lines", "protected_eligible": 2, "validation_eligible": 1,
        "protected_with_verified_shard": 2, "validation_with_verified_shard": 1,
        "protected_unresolved": 0, "validation_unresolved": 0, "source_files_scanned": 2,
        "target_shards": 2, "protected_heldout": 1, "protected_retained": 1,
        "validation_heldout": 0, "validation_retained": 1,
    }
    return summary, audit, intersection


class FinalizeTests(unittest.TestCase):
    def test_exact_count_still_requires_a_new_validation_selection(self):
        result = module.combine(*fixture())
        self.assertEqual(result["status"], "candidate-not-enforced")
        self.assertEqual(result["previous_identity_exclusions_already_in_heldout_contexts"], 1)
        self.assertEqual(result["previous_identity_exclusions_outside_heldout_contexts"], 2)
        self.assertEqual(result["remaining_eligible_rows_after_original_identity_exclusions"], 22)
        self.assertTrue(result["new_validation_selection_still_required"])

    def test_rejects_mismatched_metadata_or_incomplete_scans(self):
        args = fixture()
        args[2]["metadata_plan_sha256"] = "different"
        with self.assertRaisesRegex(ValueError, "different source"):
            module.combine(*args)
        args = fixture()
        args[2]["source_files_scanned"] = 1
        with self.assertRaisesRegex(ValueError, "incomplete"):
            module.combine(*args)
        args = fixture()
        args[2]["protected_retained"] = 2
        with self.assertRaisesRegex(ValueError, "occurrence counts"):
            module.combine(*args)

    def test_rejects_changed_context_counts_or_out_of_range_result(self):
        args = fixture()
        args[0]["contexts"][0]["source_rows_in_context"] += 1
        with self.assertRaisesRegex(ValueError, "row totals"):
            module.combine(*args)
        args = fixture()
        args[0]["remaining_eligible_rows_after_original_identity_exclusions_lower"] = 23
        with self.assertRaisesRegex(ValueError, "outside"):
            module.combine(*args)

    def test_does_not_mutate_source_reports(self):
        args = fixture()
        before = copy.deepcopy(args)
        module.combine(*args)
        self.assertEqual(args, before)


if __name__ == "__main__":
    unittest.main()
