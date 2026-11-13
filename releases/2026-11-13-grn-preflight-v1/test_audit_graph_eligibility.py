from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest

import audit_graph_eligibility as audit
from test_audit_grn_pilot import fixture


class EligibilityTests(unittest.TestCase):
    def test_source_verification_uses_content_not_original_path_or_timestamp(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "renamed-source"
            path.write_bytes(b"source")
            expected = {
                "filename": "old-name",
                "size": len(b"source"),
                "mtime_ns": 1,
                "sha256": audit.common.sha256(path),
            }
            os.utime(path, ns=(2_000_000_000, 2_000_000_000))
            audit.verify_source(path, expected)
            path.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "differs"):
                audit.verify_source(path, expected)

    def test_graph_coverage_uses_both_contexts_and_exact_feature_axis(self):
        pilot, foundation, graph, real, _ = fixture()
        pilot["cells_per_target_context"] = 100
        pilot["targets"]["eligible_count"] = 3
        left = {"counts": {"A": 100, "B": 101, "C": 130, "D": 300}}
        right = {"counts": {"A": 110, "B": 101, "C": 200, "D": 99}}
        result = audit.summarize_candidates(left, right, pilot, foundation, graph, real)
        self.assertEqual(result["eligible_shared_targets"], 3)
        self.assertEqual(result["eligible_targets_sourcing_real_graph_edges"], 2)
        self.assertEqual(result["candidate_edges_from_eligible_graph_sources"], 2)
        self.assertEqual(result["eligible_graph_sources_in_legacy_target_splits"],
                         {"train": 1, "validation": 1, "test": 0})

    def test_mismatch_and_unavailable_frozen_targets_fail(self):
        pilot, foundation, graph, real, _ = fixture()
        pilot["cells_per_target_context"] = 100
        left = {"counts": {"A": 100, "B": 100, "C": 100}}
        right = {"counts": {"A": 100, "B": 100, "C": 100}}
        pilot["targets"]["eligible_count"] = 4
        with self.assertRaisesRegex(ValueError, "count differs"):
            audit.summarize_candidates(left, right, pilot, foundation, graph, real)
        pilot["targets"]["eligible_count"] = 2
        right["counts"]["C"] = 99
        with self.assertRaisesRegex(ValueError, "no longer eligible"):
            audit.summarize_candidates(left, right, pilot, foundation, graph, real)


if __name__ == "__main__":
    unittest.main()
