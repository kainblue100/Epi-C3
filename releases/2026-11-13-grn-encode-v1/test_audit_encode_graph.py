from __future__ import annotations

import copy
import unittest

import audit_encode_graph as audit


def edge(source, target, synthetic=False):
    return {
        "source": source, "target": target, "source_ensembl_id": "id-" + source,
        "target_ensembl_id": "id-" + target,
        "evidence_origin": "synthetic_structural_control" if synthetic else "unpaired_encode_k562_chip_atac_promoter",
        "regulatory_sign": "unknown", "chip_experiment_accessions": "" if synthetic else "ENCSR000AAA",
        "chip_file_accessions": "" if synthetic else "ENCFF000AAA",
        "supporting_peak_gene_overlaps": "" if synthetic else "1",
    }


class IndependentAuditTests(unittest.TestCase):
    def test_graph_matching_and_evidence_separation(self):
        features = {value: "id-" + value for value in ("A", "B", "C", "D")}
        foundation = set(features.values())
        sources = {"ENCFF000AAA": {"source_symbol": "A", "experiment_accession": "ENCSR000AAA"},
                   "ENCFF000BBB": {"source_symbol": "C", "experiment_accession": "ENCSR000BBB"}}
        real = [edge("A", "B"), {**edge("C", "D"), "chip_file_accessions": "ENCFF000BBB",
                                 "chip_experiment_accessions": "ENCSR000BBB"}]
        control = [edge("A", "D", True), edge("C", "B", True)]
        result = audit.audit_structure(real, control, features, foundation, sources)
        self.assertEqual(result["changed_fraction"], 1.0)
        self.assertEqual(result["degree_difference_nodes"], 0)
        bad = copy.deepcopy(control)
        bad[0]["chip_file_accessions"] = "ENCFF000AAA"
        with self.assertRaisesRegex(ValueError, "must not retain"):
            audit.audit_structure(real, bad, features, foundation, sources)
        bad = copy.deepcopy(real)
        bad[0]["source_ensembl_id"] = "wrong"
        with self.assertRaisesRegex(ValueError, "unmapped gene"):
            audit.audit_structure(bad, control, features, foundation, sources)

    def test_half_open_overlap_excludes_touching_boundaries(self):
        self.assertFalse(audit.overlap((0, 10), (10, 20)))
        self.assertTrue(audit.overlap((0, 11), (10, 20)))

    def test_extra_controls_require_exactly_two_distinct_seed_records(self):
        with self.assertRaisesRegex(ValueError, "two additional"):
            audit.audit_extra_controls(
                None,
                {"format": "epi-c3-encode-k562-additional-degree-controls-v1",
                 "input_sha256": {"graph_summary": "abc"}, "additional_controls": []},
                "abc", [], {}, set(), {},
            )


if __name__ == "__main__":
    unittest.main()
