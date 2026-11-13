import copy
import importlib.util
import unittest
from pathlib import Path


SPEC = importlib.util.spec_from_file_location(
    "tahoe_exposure", Path(__file__).with_name("audit_tahoe_state_exposure.py")
)
exposure = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(exposure)


def fixture():
    groups, lines, samples, splits = {}, [], [], {}
    for index, name in enumerate(exposure.EXPECTED_CONTEXTS):
        cell_id, sample = f"line-{index}", f"sample-{index}"
        lines.append({"cell_name": name, "Cell_ID_Cellosaur": cell_id})
        samples.append({"sample": sample, "drugname_drugconc": "drug-a"})
        groups[sample] = {"sample": sample, "cell_line_id": cell_id, "cells": 10}
        splits[f"tahoe_holdout.{name}"] = {"test": ["drug-a", "drug-a", "drug-b"]}
    groups["unmatched"] = {"sample": "unmatched", "cell_line_id": "other", "cells": 50}
    samples.append({"sample": "unmatched", "drugname_drugconc": "drug-a"})
    return (
        {"state": "complete", "rows": 100, "groups": groups},
        {"state": "complete", "source_rows_checked": 100, "eligible_training_rows_after_exclusions": 80},
        {"fewshot": splits}, lines, samples,
    )


class TahoeExposureTests(unittest.TestCase):
    def test_lower_bound_is_conservative_and_counts_rows_not_cells(self):
        result = exposure.audit(*fixture())
        self.assertEqual(result["matched_official_test_rows"], 50)
        self.assertEqual(result["minimum_matched_rows_in_planned_eligible_stream"], 30)
        self.assertFalse(result["stage2_has_been_run_by_this_audit"])
        self.assertEqual(result["contexts"][0]["test_condition_entries"], 3)
        self.assertEqual(result["contexts"][0]["unique_test_conditions"], 2)
        self.assertEqual(result["contexts"][0]["matched_unique_test_conditions"], 1)

    def test_zero_lower_bound_when_all_matched_rows_could_be_excluded(self):
        metadata, isolation, split, lines, samples = fixture()
        isolation["eligible_training_rows_after_exclusions"] = 20
        result = exposure.audit(metadata, isolation, split, lines, samples)
        self.assertEqual(result["minimum_matched_rows_in_planned_eligible_stream"], 0)

    def test_rejects_inconsistent_row_universe(self):
        metadata, isolation, split, lines, samples = fixture()
        isolation["source_rows_checked"] = 99
        with self.assertRaisesRegex(ValueError, "row universe"):
            exposure.audit(metadata, isolation, split, lines, samples)

    def test_rejects_ambiguous_context_or_missing_sample(self):
        metadata, isolation, split, lines, samples = fixture()
        ambiguous = copy.deepcopy(lines)
        ambiguous.append({"cell_name": exposure.EXPECTED_CONTEXTS[0], "Cell_ID_Cellosaur": "different"})
        with self.assertRaisesRegex(ValueError, "exactly one"):
            exposure.audit(metadata, isolation, split, ambiguous, samples)
        with self.assertRaisesRegex(ValueError, "absent from sample metadata"):
            exposure.audit(metadata, isolation, split, lines, samples[1:])


if __name__ == "__main__":
    unittest.main()
