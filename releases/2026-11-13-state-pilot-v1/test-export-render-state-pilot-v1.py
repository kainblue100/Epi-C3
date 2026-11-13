"""Focused checks for public-data whitelisting and figure validation."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET


def load(name, filename):
    path = Path(__file__).with_name(filename)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


export = load("export_pilot", "export-state-pilot-aggregate-v1.py")
render = load("render_pilot", "render-state-pilot-v1.py")


def fixtures():
    summary = {
        "format": "epi-c3-state-replogle-pilot-scores-v1",
        "cell_eval_version": "0.8.2",
        "pdex_version": "0.3.0",
        "conditions": 28,
        "cells_per_condition": 100,
        "hvg_count": 2000,
        "control_counts": {
            "perturbation_parent_pool": 280, "predicted_control": 280,
            "independent_real_control": 280,
        },
        "cross_batch_parent_assignments": 33,
        "aggregate": {},
        "audit": {},
        "private_cell_identity": "DO_NOT_EXPORT",
    }
    audit = {
        "format": "epi-c3-state-pilot-independent-metric-audit-v1",
        "status": "complete", "source_summary_sha256": "scorehash",
        "conditions": 28, "features": 2000,
        "infinite_real_lfc": 3, "bh_corrected_groups_per_model": 56,
        "mann_whitney_spot_checks_per_model": 12,
        "finite_lfc_sensitivity": {
            "method": "shared finite values", "affected_conditions": 7,
            "removed_condition_gene_occurrences": 16, "common_conditions": 22,
            "mean_spearman": {model: 0.1 for model in export.MODELS},
            "private_cell_identity": "DO_NOT_EXPORT",
        },
        "model_audits": {},
    }
    for model in export.MODELS:
        summary["aggregate"][model] = {
            metric: {
                "value": None if metric == export.METRICS[-1] and model == "context_mean" else 0.2,
                "n_conditions": n,
            }
            for metric, n in zip(export.METRICS, export.EXPECTED_SUPPORT, strict=True)
        }
        summary["audit"][model] = {
            "clipped_below_zero": 1666400 if model == "perturb_mean" else 0,
            "clipped_above_fourteen": 0,
            "minimum_unique_prediction_vectors_per_condition": 1,
            "maximum_unique_prediction_vectors_per_condition": 1,
            "conditions_with_one_unique_prediction_vector": 28,
        }
        audit["model_audits"][model] = {
            "infinite_predicted_lfc": 704 if model == "state" else 0,
            "maximum_absolute_metric_differences": {},
            "overlap_cutoff_ties": 0,
        }
    plan = {
        "seed": 20261113, "cells_per_target_context": 100,
        "targets": {"eligible_count": 565, "train": list(range(10)),
                    "val": list(range(2)), "test": list(range(28))},
        "sources": {
            context: {"filename": filename, "sha256": sha}
            for context, (filename, sha) in export.SOURCE_HASHES.items()
        },
        "records": {"private": "DO_NOT_EXPORT"},
    }
    checkpoints = {
        model: {
            "actual_global_step": step, "configured_max_steps": 512,
            "checkpoint_sha256": "x", "config_sha256": "y",
            "private_cell_identity": "DO_NOT_EXPORT",
        }
        for model, step in export.EXPECTED_STEPS.items()
    }
    return summary, audit, plan, checkpoints


class PublicExportTests(unittest.TestCase):
    def test_only_aggregate_data_are_exported(self):
        data = export.build_public_summary(*fixtures(), "scorehash", "audithash")
        self.assertNotIn("DO_NOT_EXPORT", json.dumps(data))
        self.assertEqual(len(data["metrics"]), 6)
        self.assertIsNone(data["metrics"][-1]["scores"]["context_mean"])
        render.validate(data)

    def test_rejects_mismatched_audit_and_support(self):
        summary, audit, plan, checkpoints = fixtures()
        with self.assertRaisesRegex(ValueError, "audit does not match"):
            export.build_public_summary(summary, audit, plan, checkpoints, "wrong", "audit")
        summary["aggregate"]["state"]["pr_auc"]["n_conditions"] = 23
        with self.assertRaisesRegex(ValueError, "common-condition count"):
            export.build_public_summary(summary, audit, plan, checkpoints, "scorehash", "audit")

    def test_renderer_rejects_nonfinite_and_missing_model(self):
        data = export.build_public_summary(*fixtures(), "scorehash", "audithash")
        data["metrics"][0]["scores"]["state"] = float("nan")
        with self.assertRaises(ValueError):
            render.validate(data)
        data["metrics"][0]["scores"].pop("state")
        with self.assertRaises(ValueError):
            render.validate(data)

    def test_undefined_model_is_visible_in_last_panel(self):
        data = export.build_public_summary(*fixtures(), "scorehash", "audithash")
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            render.render(data, output)
            svg = output / "state-pilot-comparison.svg"
            self.assertTrue(all(line == line.rstrip(" \t") for line in svg.read_text(encoding="utf-8").splitlines()))
            root = ET.parse(svg).getroot()
            width = float(root.attrib["viewBox"].split()[2])
            texts = root.findall(".//{http://www.w3.org/2000/svg}text")
            labels = [node for node in texts if "".join(node.itertext()) == "CM"]
            missing = [node for node in texts if "".join(node.itertext()) == "N/A"]
            self.assertEqual(len(labels), 6)
            self.assertEqual(len(missing), 1)
            self.assertLess(float(missing[0].attrib["x"]), width)


if __name__ == "__main__":
    unittest.main()
