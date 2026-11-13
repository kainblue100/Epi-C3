from __future__ import annotations

import copy
import hashlib
from html.parser import HTMLParser
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import render_grn_preflight as page


class PageParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.tag_names = []

    def handle_starttag(self, tag, attrs):
        self.tag_names.append(tag)


class PageTests(unittest.TestCase):
    def setUp(self):
        self.audit = page.read_pinned(page.HERE / "audit.json", page.AUDIT_SHA)
        self.eligibility = page.read_pinned(page.HERE / "eligibility.json", page.ELIGIBILITY_SHA)

    def test_page_contains_aggregate_results_and_non_model_disclaimer(self):
        html = page.render(self.audit, self.eligibility)
        axes = self.audit["axis_and_graph"]
        counts = self.eligibility["eligibility"]
        targets = axes["pilot_target_partitions"]["test"]
        self.assertIn(f'{counts["eligible_targets_sourcing_real_graph_edges"]} / {counts["eligible_shared_targets"]}', html)
        self.assertIn(f'{targets["targets"] - targets["targets_on_hvg_axis"]} / {targets["targets"]}', html)
        self.assertIn("没有训练新 GRN 模型", html)
        self.assertIn("不是去重细胞数", html)
        parser = PageParser()
        parser.feed(html)
        self.assertIn("main", parser.tag_names)
        self.assertIn("table", parser.tag_names)
        self.assertNotIn("script", parser.tag_names)

    def test_inconsistent_provenance_and_modified_aggregate_fail(self):
        modified = copy.deepcopy(self.eligibility)
        modified["input_sha256"]["pilot_plan"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "provenance disagrees"):
            page.render(self.audit, modified)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "modified.json"
            path.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                page.read_pinned(path, page.AUDIT_SHA)

    def test_cli_writes_deterministically_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "index.html"
            command = [sys.executable, "-B", str(page.HERE / "render_grn_preflight.py"), "--output", str(output)]
            first = subprocess.run(command, capture_output=True, text=True, check=True)
            self.assertEqual(first.stdout.strip(), hashlib.sha256(output.read_bytes()).hexdigest())
            expected = page.render(self.audit, self.eligibility).encode("utf-8")
            self.assertEqual(output.read_bytes(), expected)
            second = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(second.returncode, 0)
            self.assertEqual(output.read_bytes(), expected)


if __name__ == "__main__":
    unittest.main()
