from __future__ import annotations

import copy
import hashlib
from html.parser import HTMLParser
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import render_encode_report as page


class ReportTests(unittest.TestCase):
    def setUp(self):
        self.data = [page.load_pinned(name) for name in page.FILES]

    def test_page_contains_actual_results_and_explicit_limitations(self):
        html = page.render(*self.data)
        result = self.data[1]["graph"]
        self.assertIn(f'{result["candidate_edges"]:,}', html)
        self.assertIn(f'{result["source_genes"]} / 565', html)
        self.assertIn("没有训练扰动模型", html)
        self.assertIn("不能作为盲选图", html)
        self.assertIn("不是去重细胞", html)
        parser = HTMLParser()
        parser.feed(html)
        self.assertNotIn("<script", html)

    def test_conflicting_control_provenance_fails(self):
        altered = copy.deepcopy(self.data)
        altered[2]["input_sha256"]["graph_summary"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "provenance"):
            page.render(*altered)

    def test_degree_mismatch_is_not_presented_as_zero(self):
        altered = copy.deepcopy(self.data)
        altered[3]["additional_control_checks"]["controls"][0]["degree_difference_nodes"] = 1
        with self.assertRaisesRegex(ValueError, "counts disagree"):
            page.render(*altered)

    def test_cli_writes_deterministically_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "index.html"
            command = [sys.executable, "-B", str(page.HERE / "render_encode_report.py"), "--output", str(output)]
            first = subprocess.run(command, capture_output=True, text=True, check=True)
            self.assertEqual(first.stdout.strip(), hashlib.sha256(output.read_bytes()).hexdigest())
            expected = page.render(*self.data).encode("utf-8")
            self.assertEqual(output.read_bytes(), expected)
            second = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(second.returncode, 0)
            self.assertEqual(output.read_bytes(), expected)


if __name__ == "__main__":
    unittest.main()
