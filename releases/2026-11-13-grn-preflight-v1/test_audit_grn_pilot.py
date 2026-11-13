from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

import audit_grn_pilot as audit


def fixture():
    pilot = {
        "features": [["A", "ENS1"], ["B", "ENS2"], ["C", "ENS3"], ["D", "ENS4"]],
        "targets": {"train": ["A"], "val": ["B"], "test": ["C"]},
        "sources": {"K562": {"sha256": "same-source", "rows": 20}},
        "records": {"K562": [
            {"row": 1, "target": "A", "role": "train", "context": "K562"},
            {"row": 2, "target": "C", "role": "train", "context": "K562"},
            {"row": 3, "target": "non-targeting", "role": "train_control", "context": "K562"},
        ]},
    }
    genes = ["ENS1", "ENS2", "ENS3"]
    foundation = {"genes": genes, "gene_axis_sha256": audit.axis_sha256(genes)}
    graph = {
        "source_sha256": "same-source",
        "gene_axis": {"symbols": ["A", "B", "C", "D"]},
        "target_split": {"train": ["A"], "validation": ["B"], "test": ["C"]},
        "conditions": {
            "train": {"A": {"rows": [1, 4]}},
            "validation": {"B": {"treatment_rows": [5]}},
            "test": {"C": {"treatment_rows": [2, 6]}},
        },
    }
    real = {("A", "C"), ("B", "D")}
    shuffled = {("A", "D"), ("B", "C")}
    return pilot, foundation, graph, real, shuffled


class AuditTests(unittest.TestCase):
    def test_axis_and_projected_graph_coverage(self):
        pilot, foundation, graph, real, shuffled = fixture()
        result = audit.summarize_axes(pilot, foundation, graph, ["A", "C", "D"], real, shuffled)
        self.assertEqual(result["hvgs_with_exact_ensembl_id_on_foundation_axis"], 2)
        self.assertEqual(result["pilot_target_partitions"]["val"]["targets_on_hvg_axis"], 0)
        self.assertEqual(result["pilot_target_partitions"]["test"]["targets_directly_sourcing_real_edges"], 0)
        self.assertEqual(result["graph_full_axis"]["nodes_with_different_in_degree"], 0)
        self.assertEqual(result["graph_full_axis"]["nodes_with_different_out_degree"], 0)
        projected = result["graph_projected_to_pilot_hvgs"]
        self.assertEqual(projected["real_edges"], 1)
        self.assertEqual(projected["shuffled_edges"], 1)
        self.assertEqual(projected["nodes_with_different_in_degree"], 2)

    def test_axes_and_partitions_reject_duplicate_and_unverified_mapping(self):
        pilot, foundation, graph, real, shuffled = fixture()
        with self.assertRaisesRegex(ValueError, "HVG"):
            audit.summarize_axes(pilot, foundation, graph, ["A", "A"], real, shuffled)
        pilot["features"][1][1] = "ENS1"
        with self.assertRaisesRegex(ValueError, "identifiers"):
            audit.summarize_axes(pilot, foundation, graph, ["A", "C"], real, shuffled)
        pilot, foundation, graph, real, shuffled = fixture()
        pilot["targets"]["test"] = ["A"]
        with self.assertRaisesRegex(ValueError, "overlap"):
            audit.summarize_axes(pilot, foundation, graph, ["A", "C"], real, shuffled)
        pilot, foundation, graph, real, shuffled = fixture()
        foundation["gene_axis_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            audit.summarize_axes(pilot, foundation, graph, ["A", "C"], real, shuffled)

    def test_same_source_row_overlap_and_label_checks(self):
        pilot, _, graph, _, _ = fixture()
        result = audit.training_row_overlap(pilot, graph)
        self.assertEqual(result["pilot_k562_training_treatment_rows"], 2)
        self.assertEqual(result["legacy_graph_partitions"]["train"]["overlapping_source_rows"], 1)
        self.assertEqual(result["legacy_graph_partitions"]["test"]["overlapping_source_rows"], 1)
        graph["source_sha256"] = "different-source"
        with self.assertRaisesRegex(ValueError, "same source"):
            audit.training_row_overlap(pilot, graph)
        pilot, _, graph, _, _ = fixture()
        graph["conditions"]["test"]["C"]["treatment_rows"] = [1]
        with self.assertRaisesRegex(ValueError, "overlap or repeat"):
            audit.training_row_overlap(pilot, graph)
        pilot, _, graph, _, _ = fixture()
        pilot["records"]["K562"][1]["target"] = "B"
        with self.assertRaisesRegex(ValueError, "different perturbation labels"):
            audit.training_row_overlap(pilot, graph)

    def test_normalization_distinguishes_library_and_naive_log_input(self):
        counts = np.asarray([[1.0, 3.0, 0.0], [2.0, 2.0, 1.0]])
        total_library = np.asarray([[10.0], [20.0]])
        pilot_x = np.log1p(counts / total_library * 10000.0)
        result = audit.normalization_diagnostic(pilot_x)
        expected_local = np.log1p(counts / counts.sum(axis=1, keepdims=True) * 10000.0)
        wrong = np.log1p(pilot_x / pilot_x.sum(axis=1, keepdims=True) * 10000.0)
        self.assertAlmostEqual(result["hvg_fraction_of_original_library"]["mean"], 0.325)
        self.assertAlmostEqual(
            result["local_panel_vs_original_library_mean_absolute_difference"],
            float(np.abs(expected_local - pilot_x).mean()),
        )
        self.assertAlmostEqual(
            result["naive_log_input_vs_local_panel_mean_absolute_difference"],
            float(np.abs(wrong - expected_local).mean()),
        )
        for invalid in (np.zeros((1, 2)), np.asarray([[-1.0, 0.0]]), np.asarray([[np.nan, 1.0]])):
            with self.assertRaises(ValueError):
                audit.normalization_diagnostic(invalid)

    def test_hash_and_edge_semantics_are_enforced(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            edge_path = root / "edges-real.csv"
            edge_path.write_text(
                "source,target,evidence_origin,regulatory_sign\nA,B,unpaired_bulk_candidate,unknown\n",
                encoding="utf-8",
            )
            manifest = {"edge_count": 1, "artifacts": {edge_path.name: audit.sha256(edge_path)}}
            self.assertEqual(audit.read_graph(edge_path, manifest, {"A", "B"}, shuffled=False), {("A", "B")})
            edge_path.write_text(edge_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                audit.read_graph(edge_path, manifest, {"A", "B"}, shuffled=False)
            payload = root / "payload.json"
            payload.write_text(json.dumps({"test": 1}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                audit.read_json(payload, "0" * 64)

    def test_cli_writes_and_reports_a_new_aggregate_without_overwriting(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "audit.json"
            arguments = ["audit_grn_pilot.py", "--root", str(root), "--output", str(output)]
            disk = SimpleNamespace(free=audit.RESERVE_WITH_HEADROOM)
            with mock.patch("sys.argv", arguments), mock.patch.object(
                audit.shutil, "disk_usage", return_value=disk
            ), mock.patch.object(audit, "audit", return_value={"status": "preflight-complete-not-a-model"}):
                with mock.patch("builtins.print") as printed:
                    audit.main()
                self.assertEqual(json.loads(output.read_text(encoding="utf-8"))["status"],
                                 "preflight-complete-not-a-model")
                message = json.loads(printed.call_args.args[0])
                self.assertEqual(message["output_sha256"], audit.sha256(output))
                with self.assertRaises(FileExistsError):
                    audit.main()

    @unittest.skipUnless(importlib.util.find_spec("h5py"), "h5py is not installed")
    def test_hdf5_reads_only_training_controls_and_checks_file_hash(self):
        import h5py

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pilot.h5ad"
            text = h5py.string_dtype("utf-8")
            with h5py.File(path, "w") as file:
                file.create_dataset("X", data=np.log1p(np.asarray([
                    [1000.0, 2000.0], [500.0, 1500.0], [5000.0, 4000.0],
                ])))
                file.create_dataset("var/_index", data=np.asarray(["A", "B"], dtype=object), dtype=text)
                file.create_dataset("obs/role", data=np.asarray(
                    ["train_control", "train_control", "test"], dtype=object), dtype=text)
                file.create_dataset("obs/context", data=np.asarray(
                    ["RPE1", "RPE1", "RPE1"], dtype=object), dtype=text)
                file.create_dataset("obs/target_gene", data=np.asarray(
                    ["non-targeting", "non-targeting", "A"], dtype=object), dtype=text)
            manifest = {
                "dataset_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "cells": 3, "feature_count": 2, "role_counts": {"train_control": 2, "test": 1},
            }
            axis, result = audit.read_pilot_input(path, manifest, sample_size=2)
            self.assertEqual(axis, ["A", "B"])
            self.assertEqual(result["training_control_cells"], 2)
            self.assertAlmostEqual(result["hvg_fraction_of_original_library"]["mean"], 0.25)
            manifest["dataset_sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "dataset hash mismatch"):
                audit.read_pilot_input(path, manifest, sample_size=2)


if __name__ == "__main__":
    unittest.main()
