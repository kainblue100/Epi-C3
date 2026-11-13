import copy
import importlib.util
import unittest
from pathlib import Path


spec = importlib.util.spec_from_file_location(
    "tight_exposure", Path(__file__).with_name("audit-tahoe-state-exposure-tight-v1.py")
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def fixture():
    groups, lines, samples, splits, controls = {}, [], [], {}, {}
    for index, name in enumerate(module.CONTEXTS):
        line, plate = f"line-{index}", f"plate-{index}"
        lines.append({"cell_name": name, "Cell_ID_Cellosaur": line})
        splits[f"tahoe_holdout.{name}"] = {"test": ["test-a", "test-b", "test-unknown", "test-control"]}
        for suffix, drug, structure, cells, label in (
            ("a", "drug-a", "C", 10, "test-a"),
            ("b", "drug-b", "N", 6, "test-b"),
            ("u", "unknown", None, 4, "test-unknown"),
            ("c", module.CONTROL, None, 7, "test-control"),
            ("d", module.CONTROL, None, 3, "other-control"),
        ):
            sample = f"sample-{index}-{suffix}"
            samples.append({"sample": sample, "drugname_drugconc": label})
            groups[f"{sample}\t{line}"] = {
                "sample": sample, "cell_line_id": line, "plate": plate, "drug": drug,
                "canonical_smiles": structure, "cells": cells,
            }
        controls[f"{line}\t{plate}"] = {"training": 8, "reference": 1, "input_controls": 1}
    metadata = {
        "state": "complete", "rows": 150, "groups": groups, "heldout_compounds": ["N"],
        "control_groups": controls, "control_counts": {
            "training": 40, "reference": 5, "input_controls": 5,
        },
    }
    isolation = {
        "state": "complete", "source_rows_checked": 150, "eligible_rows_before_exclusions": 90,
        "protected_eligible_rows": 2, "validation_eligible_rows": 1,
        "eligible_training_rows_after_exclusions": 87,
    }
    drugs = [
        {"drug": "drug-a", "canonical_smiles": "C"},
        {"drug": "drug-b", "canonical_smiles": "N"},
        {"drug": "unknown", "canonical_smiles": None},
    ]
    return metadata, isolation, {"fewshot": splits}, lines, samples, drugs


class ExposureTests(unittest.TestCase):
    def test_partial_control_groups_produce_conservative_bounds(self):
        result = module.audit(*fixture())
        self.assertEqual(result["matched_official_test_rows"], 135)
        self.assertEqual(result["matched_eligible_lower_before_identity_exclusions"], 75)
        self.assertEqual(result["matched_eligible_upper_before_identity_exclusions"], 85)
        self.assertEqual(result["minimum_matched_rows_after_identity_exclusions"], 72)
        self.assertEqual(result["maximum_matched_rows_after_identity_exclusions"], 85)
        self.assertEqual(result["contexts"][0]["matched_heldout_compound_rows"], 6)
        self.assertEqual(result["contexts"][0]["matched_missing_structure_rows"], 4)

    def test_complete_control_group_is_exact_before_identity_exclusions(self):
        args = fixture()
        for section in args[2]["fewshot"].values():
            section["test"].append("other-control")
        result = module.audit(*args)
        self.assertEqual(result["matched_official_test_rows"], 150)
        self.assertEqual(result["matched_eligible_lower_before_identity_exclusions"], 90)
        self.assertEqual(result["matched_eligible_upper_before_identity_exclusions"], 90)
        self.assertEqual(result["minimum_matched_rows_after_identity_exclusions"], 87)
        self.assertEqual(result["maximum_matched_rows_after_identity_exclusions"], 87)

    def test_rejects_mismatched_training_or_control_totals(self):
        args = fixture()
        args[1]["eligible_rows_before_exclusions"] += 1
        with self.assertRaisesRegex(ValueError, "eligibility"):
            module.audit(*args)
        args = fixture()
        args[0]["control_groups"]["line-0\tplate-0"]["training"] -= 1
        with self.assertRaisesRegex(ValueError, "Control-group"):
            module.audit(*args)

    def test_rejects_ambiguous_line_and_inconsistent_structures(self):
        args = fixture()
        args[3].append({"cell_name": module.CONTEXTS[0], "Cell_ID_Cellosaur": "other"})
        with self.assertRaisesRegex(ValueError, "one cell line"):
            module.audit(*args)
        args = fixture()
        altered = copy.deepcopy(args[5])
        altered[0]["canonical_smiles"] = "CC"
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            module.audit(*args[:5], altered)


if __name__ == "__main__":
    unittest.main()
