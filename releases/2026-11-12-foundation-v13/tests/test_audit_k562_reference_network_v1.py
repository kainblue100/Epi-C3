from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


SPEC = importlib.util.spec_from_file_location(
    "k562_reference_audit", Path(__file__).resolve().parents[1] / "audit-k562-reference-network-v1.py"
)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def fixture(root: Path):
    reference = root / "RN119.tsv"
    reference.write_text(
        "Source\tTarget\tRelationship\nA\tB\tActivation\nA\tB\tRepression\n"
        "A\tB\tActivation\nC\tB\tActivation\nC\tZ\tRepression\n",
        encoding="utf-8",
    )
    graph = root / "edges-real.csv"
    graph.write_text(
        "source,target,evidence_origin,regulatory_sign\n"
        "A,B,unpaired_bulk_candidate,unknown\nC,A,unpaired_bulk_candidate,unknown\n",
        encoding="utf-8",
    )
    plan = root / "plan.json"
    plan.write_text(json.dumps({
        "format": "epi-c3-replogle-gwps-graph-plan-v1",
        "gene_axis": {"symbols": ["A", "B", "C", "D"]},
        "target_split": {"train": ["A"], "validation": ["D"], "test": ["C"]},
        "graph_source_targets": {"train": ["A"], "validation": [], "test": ["C"]},
    }), encoding="utf-8")
    manifest = root / "manifest.json"
    manifest.write_text(json.dumps({
        "format": "epi-c3-evidence-grn-v1", "state": "complete",
        "context": "K562", "assembly": "GRCh38", "edge_count": 2,
        "source_plan_sha256": audit.digest(plan),
        "artifacts": {graph.name: audit.digest(graph)},
    }), encoding="utf-8")
    return reference, plan, manifest, graph


def run(root: Path):
    reference, plan, manifest, graph = fixture(root)
    return audit.audit(
        reference, audit.digest(reference), plan, audit.digest(plan),
        manifest, audit.digest(manifest), graph, 5,
    )


class ReferenceAuditTests(unittest.TestCase):
    def test_conflicting_signs_deduplication_and_split_overlap(self):
        with tempfile.TemporaryDirectory() as directory:
            result = run(Path(directory))
        ref = result["reference"]
        self.assertEqual(ref["rows"], 5)
        self.assertEqual(ref["unique_directed_gene_pairs"], 3)
        self.assertEqual(ref["pairs_with_both_signs"], 1)
        self.assertEqual(ref["duplicate_identical_rows"], 1)
        self.assertTrue(ref["catalog_row_count_matches"])
        axis = result["frozen_axis_comparison"]
        self.assertEqual(axis["reference_unique_edges_both_ends_on_axis"], 2)
        self.assertEqual(axis["exact_directed_gene_pair_overlap"], 1)
        self.assertEqual(axis["overlapping_pairs_with_both_signs"], 1)
        self.assertEqual(result["frozen_target_partitions"]["test"]["reference_unique_edges_from_partition"], 2)
        self.assertFalse(result["used_for_training_or_evaluation"])

    def test_changed_reference_and_graph_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference, plan, manifest, graph = fixture(root)
            with self.assertRaisesRegex(ValueError, "Reference network differs"):
                audit.read_reference(reference, "0" * 64)
            graph.write_text(graph.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "artifact digest"):
                audit.audit(reference, audit.digest(reference), plan, audit.digest(plan),
                            manifest, audit.digest(manifest), graph, 5)

    def test_graph_must_reference_the_frozen_plan(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference, plan, manifest, graph = fixture(root)
            metadata = json.loads(manifest.read_text(encoding="utf-8"))
            metadata["source_plan_sha256"] = "0" * 64
            manifest.write_text(json.dumps(metadata), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "not derived"):
                audit.audit(reference, audit.digest(reference), plan, audit.digest(plan),
                            manifest, audit.digest(manifest), graph, 5)

    def test_bad_relationship_and_overlapping_partitions_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference, plan, manifest, graph = fixture(root)
            reference.write_text("Source\tTarget\tRelationship\nA\tB\tUnknown\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "relationship"):
                audit.read_reference(reference, audit.digest(reference))
            reference, plan, manifest, graph = fixture(root)
            data = json.loads(plan.read_text(encoding="utf-8"))
            data["target_split"]["test"].append("A")
            plan.write_text(json.dumps(data), encoding="utf-8")
            graph_metadata = json.loads(manifest.read_text(encoding="utf-8"))
            graph_metadata["source_plan_sha256"] = audit.digest(plan)
            manifest.write_text(json.dumps(graph_metadata), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "partitions overlap"):
                audit.audit(reference, audit.digest(reference), plan, audit.digest(plan),
                            manifest, audit.digest(manifest), graph, 5)


if __name__ == "__main__":
    unittest.main()
