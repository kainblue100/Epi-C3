import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from tahoe_context_holdout_guard_v1 import TahoeContextHoldoutGuard


def save(path, value):
    data = (json.dumps(value, sort_keys=True) + "\n").encode()
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def fixture(root):
    metadata = {
        "state": "complete", "rows": 6,
        "groups": {
            f"sample-{index}\tline-{index}": {
                "sample": f"sample-{index}", "cell_line_id": f"line-{index}", "cells": 1,
            }
            for index in range(6)
        },
    }
    metadata_path, private_path, summary_path = (
        root / "metadata.json", root / "private.json", root / "summary.json",
    )
    metadata_sha = save(metadata_path, metadata)
    private = {
        "format": "epi-c3-tahoe-state-context-holdout-private-v1",
        "status": "candidate-not-enforced", "blocked_cell_line_ids": [f"line-{i}" for i in range(5)],
        "source_sha256": {"metadata_plan": metadata_sha}, "planner_sha256": "a" * 64,
        "exposure_audit_sha256": "b" * 64,
    }
    private_sha = save(private_path, private)
    summary = {
        "format": "epi-c3-tahoe-state-context-holdout-summary-v1",
        "status": "candidate-not-enforced", "private_plan_sha256": private_sha,
        "source_sha256": private["source_sha256"], "planner_sha256": private["planner_sha256"],
        "exposure_audit_sha256": private["exposure_audit_sha256"], "source_rows": 6,
    }
    summary_sha = save(summary_path, summary)
    return metadata_path, private_path, summary_path, summary_sha


class ContextGuardTests(unittest.TestCase):
    def test_filters_selected_rows_for_training_or_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            metadata, private, summary, digest = fixture(Path(directory))
            guard = TahoeContextHoldoutGuard(metadata, private, summary, expected_summary_sha256=digest)
            samples = ["sample-0", "sample-1", "sample-5"]
            lines = ["line-0", "line-1", "line-5"]
            self.assertEqual(guard.filter_selected(samples, lines, [2, 0, 1]), [2])
            self.assertEqual(guard.filter_selected(samples, lines, []), [])

    def test_unknown_or_invalid_rows_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            metadata, private, summary, digest = fixture(Path(directory))
            guard = TahoeContextHoldoutGuard(metadata, private, summary, expected_summary_sha256=digest)
            with self.assertRaisesRegex(ValueError, "outside"):
                guard.filter_selected(["unknown"], ["line-5"], [0])
            with self.assertRaisesRegex(ValueError, "incomplete"):
                guard.filter_selected([None], ["line-5"], [0])
            with self.assertRaisesRegex(ValueError, "different lengths"):
                guard.filter_selected(["sample-5"], [], [])
            for indices in ([0, 0], [-1], [1], [True], [0.0]):
                with self.subTest(indices=indices), self.assertRaises(ValueError):
                    guard.filter_selected(["sample-5"], ["line-5"], indices)

    def test_rejects_tampered_frozen_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metadata, private, summary, digest = fixture(root)
            private.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "changed"):
                TahoeContextHoldoutGuard(metadata, private, summary, expected_summary_sha256=digest)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metadata, private, summary, digest = fixture(root)
            metadata.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "changed"):
                TahoeContextHoldoutGuard(metadata, private, summary, expected_summary_sha256=digest)
        with tempfile.TemporaryDirectory() as directory:
            metadata, private, summary, digest = fixture(Path(directory))
            summary.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "changed"):
                TahoeContextHoldoutGuard(metadata, private, summary, expected_summary_sha256=digest)

    def test_summary_cannot_disagree_with_private_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            metadata, private, summary, _ = fixture(Path(directory))
            value = json.loads(summary.read_text(encoding="utf-8"))
            value["planner_sha256"] = "c" * 64
            digest = save(summary, value)
            with self.assertRaisesRegex(ValueError, "disagree"):
                TahoeContextHoldoutGuard(metadata, private, summary, expected_summary_sha256=digest)


if __name__ == "__main__":
    unittest.main()
