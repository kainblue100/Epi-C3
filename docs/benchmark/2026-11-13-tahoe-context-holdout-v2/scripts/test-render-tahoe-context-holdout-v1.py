import copy
import importlib.util
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path


spec = importlib.util.spec_from_file_location(
    "render_holdout", Path(__file__).with_name("render-tahoe-context-holdout-v1.py")
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def fixture():
    return {
        "format": "epi-c3-tahoe-state-context-holdout-exact-summary-v1",
        "status": "candidate-not-enforced", "source_rows": 110,
        "official_test_rows_with_exact_condition_match": 50,
        "all_source_rows_in_heldout_contexts": 85,
        "eligible_rows_in_heldout_contexts_before_identity_exclusions": 80,
        "remaining_eligible_rows_before_identity_exclusions": 24,
        "previous_identity_exclusions_outside_heldout_contexts": 2,
        "remaining_eligible_rows_after_original_identity_exclusions": 22,
        "new_validation_selection_still_required": True,
        "contexts": [{
            "context": name, "source_rows_in_context": 17, "matched_official_test_rows": 10,
            "eligible_treated_rows_before_identity_exclusions": 12,
            "eligible_training_control_rows_before_identity_exclusions": 4,
        } for name in sorted(module.CONTEXTS)],
    }


class RenderTests(unittest.TestCase):
    def test_candidate_status_counts_and_accessible_figure_are_visible(self):
        page = module.render(fixture(), "a" * 64)
        self.assertIn("候选方案，尚未接入训练", page)
        self.assertIn("不是两个 Epi-C3 模型的预测成绩", page)
        self.assertIn("22", page)
        self.assertIn("尚未扣除", page)
        self.assertIn("本次交集核查读取了私有身份元数据", page)
        self.assertIn("未读取原始表达，也不公开单细胞身份", page)
        self.assertIn("<summary>查看聚合摘要（JSON）</summary>", page)
        self.assertNotIn("不读取或展示单细胞表达及身份", page)
        self.assertIn("aria-labelledby=\"chart-title\"", page)
        for context in module.CONTEXTS:
            self.assertIn(context, page)
        self.assertEqual(page.count("<tr>"), 6)
        start, end = page.index("<svg "), page.index("</svg>") + len("</svg>")
        root = ET.fromstring(page[start:end])
        self.assertEqual(root.attrib["viewBox"], "0 0 920 275")
        for rectangle in root.findall(".//rect"):
            self.assertGreaterEqual(float(rectangle.attrib["width"]), 0)
            self.assertLessEqual(float(rectangle.attrib["width"]), 480)

    def test_invalid_or_unverified_counts_do_not_render(self):
        report = fixture()
        report["remaining_eligible_rows_after_original_identity_exclusions"] = 23
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            module.render(report, "a" * 64)
        report = fixture()
        report["new_validation_selection_still_required"] = False
        with self.assertRaisesRegex(ValueError, "incompatible"):
            module.render(report, "a" * 64)
        report = fixture()
        report["contexts"] = copy.deepcopy(report["contexts"])
        report["contexts"][0]["matched_official_test_rows"] = 99
        with self.assertRaisesRegex(ValueError, "totals"):
            module.render(report, "a" * 64)
        with self.assertRaisesRegex(ValueError, "incompatible"):
            module.render(fixture(), "<script>" * 8)


if __name__ == "__main__":
    unittest.main()
