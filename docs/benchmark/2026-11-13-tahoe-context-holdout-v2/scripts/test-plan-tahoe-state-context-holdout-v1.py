import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


spec = importlib.util.spec_from_file_location(
    "context_holdout", Path(__file__).with_name("plan-tahoe-state-context-holdout-v1.py")
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def fixture():
    groups, controls, samples, lines, definitions, contexts = {}, {}, [], [], {}, []
    for index, name in enumerate(module.CONTEXTS):
        line, plate = f"line-{index}", f"plate-{index}"
        lines.append({"cell_name": name, "Cell_ID_Cellosaur": line})
        definitions[f"tahoe_holdout.{name}"] = {"test": ["test-a"]}
        contexts.append({"context": name, "matched_source_rows": 10})
        for suffix, drug, canonical, count, label in (
            ("a", "drug-a", "C", 10, "test-a"),
            ("b", "drug-b", "N", 2, "training-b"),
            ("c", module.CONTROL, "", 5, "vehicle"),
        ):
            sample = f"sample-{index}-{suffix}"
            samples.append({"sample": sample, "drugname_drugconc": label})
            groups[f"{sample}\t{line}"] = {
                "sample": sample, "cell_line_id": line, "plate": plate, "drug": drug,
                "canonical_smiles": canonical, "cells": count,
            }
        controls[f"{line}\t{plate}"] = {"training": 4, "reference": 1, "input_controls": 0}
    for suffix, drug, canonical, count, label in (
        ("a", "drug-a", "C", 20, "test-a"),
        ("c", module.CONTROL, "", 5, "vehicle"),
    ):
        sample = f"outside-{suffix}"
        samples.append({"sample": sample, "drugname_drugconc": label})
        groups[f"{sample}\tline-other"] = {
            "sample": sample, "cell_line_id": "line-other", "plate": "plate-other",
            "drug": drug, "canonical_smiles": canonical, "cells": count,
        }
    controls["line-other\tplate-other"] = {"training": 4, "reference": 1, "input_controls": 0}
    metadata = {
        "state": "complete", "rows": 110, "groups": groups, "heldout_compounds": [],
        "control_groups": controls,
        "control_counts": {"training": 24, "reference": 6, "input_controls": 0},
    }
    isolation = {
        "state": "complete", "source_rows_checked": 110, "eligible_rows_before_exclusions": 104,
        "protected_eligible_rows": 2, "validation_eligible_rows": 1,
        "eligible_training_rows_after_exclusions": 101,
    }
    exposure = {
        "format": module.EXPOSURE_FORMAT, "status": "complete", "source_rows": 110,
        "eligible_before_identity_exclusions": 104, "eligible_after_identity_exclusions": 101,
        "matched_official_test_rows": 50, "matched_eligible_lower_before_identity_exclusions": 50,
        "contexts": contexts,
    }
    drugs = [
        {"drug": "drug-a", "canonical_smiles": "C"},
        {"drug": "drug-b", "canonical_smiles": "N"},
    ]
    return metadata, isolation, exposure, {"fewshot": definitions}, lines, samples, drugs


class ContextHoldoutTests(unittest.TestCase):
    def test_holds_out_complete_contexts_including_controls_and_non_test_drugs(self):
        private, summary = module.build_plan(*fixture())
        self.assertEqual(private["status"], "candidate-not-enforced")
        self.assertEqual(set(private["blocked_cell_line_ids"]), {f"line-{i}" for i in range(5)})
        self.assertNotIn("line-other", private["blocked_cell_line_ids"])
        self.assertEqual(summary["official_test_rows_with_exact_condition_match"], 50)
        self.assertEqual(summary["all_source_rows_in_heldout_contexts"], 85)
        self.assertEqual(summary["eligible_treated_rows_in_heldout_contexts_before_identity_exclusions"], 60)
        self.assertEqual(summary["eligible_training_control_rows_in_heldout_contexts_before_identity_exclusions"], 20)
        self.assertEqual(summary["eligible_rows_in_heldout_contexts_before_identity_exclusions"], 80)
        self.assertEqual(summary["remaining_eligible_rows_before_identity_exclusions"], 24)
        self.assertEqual(summary["remaining_eligible_rows_after_original_identity_exclusions_lower"], 21)
        self.assertEqual(summary["remaining_eligible_rows_after_original_identity_exclusions_upper"], 24)
        public = json.dumps(summary)
        self.assertNotIn("line-0", public)
        self.assertNotIn("sample-0", public)

    def test_bounds_account_for_original_exclusions_inside_and_outside_holdout(self):
        args = fixture()
        args[1].update(
            protected_eligible_rows=88, validation_eligible_rows=2,
            eligible_training_rows_after_exclusions=14,
        )
        args[2]["eligible_after_identity_exclusions"] = 14
        _, summary = module.build_plan(*args)
        self.assertEqual(summary["remaining_eligible_rows_after_original_identity_exclusions_lower"], 0)
        self.assertEqual(summary["remaining_eligible_rows_after_original_identity_exclusions_upper"], 14)

    def test_rejects_ambiguous_context_mappings(self):
        args = fixture()
        args[4].append({"cell_name": module.CONTEXTS[0], "Cell_ID_Cellosaur": "line-other"})
        with self.assertRaisesRegex(ValueError, "exactly one"):
            module.build_plan(*args)
        args = fixture()
        args[4][1]["Cell_ID_Cellosaur"] = args[4][0]["Cell_ID_Cellosaur"]
        with self.assertRaisesRegex(ValueError, "Distinct STATE contexts"):
            module.build_plan(*args)

    def test_rejects_changed_metadata_and_old_exposure(self):
        args = fixture()
        args[5][0]["drugname_drugconc"] = "changed"
        with self.assertRaisesRegex(ValueError, "exposure"):
            module.build_plan(*args)
        args = fixture()
        args[0]["control_groups"]["line-0\tplate-0"]["training"] -= 1
        with self.assertRaisesRegex(ValueError, "control-group"):
            module.build_plan(*args)
        args = fixture()
        args[0]["groups"]["sample-0-a\tline-0"]["cells"] += 1
        with self.assertRaisesRegex(ValueError, "totals"):
            module.build_plan(*args)

    def test_private_json_never_overwrites_existing_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "candidate.json"
            module.write_private_json(path, {"first": True})
            with self.assertRaises(FileExistsError):
                module.write_private_json(path, {"second": True})
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"first": True})

    def test_candidate_is_deterministic_for_group_order(self):
        args = fixture()
        expected = module.build_plan(*args)
        changed = copy.deepcopy(args)
        changed[0]["groups"] = dict(reversed(list(changed[0]["groups"].items())))
        self.assertEqual(module.build_plan(*changed), expected)


if __name__ == "__main__":
    unittest.main()
