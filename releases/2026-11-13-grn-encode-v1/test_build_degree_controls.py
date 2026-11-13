from __future__ import annotations

import unittest

import build_degree_controls as controls


class ControlTests(unittest.TestCase):
    def test_control_rows_keep_gene_identifiers_but_no_biological_evidence(self):
        rows = controls.graph_rows({("A", "B")}, {"A": "ENSG1", "B": "ENSG2"})
        self.assertEqual(rows[0]["source_ensembl_id"], "ENSG1")
        self.assertEqual(rows[0]["target_ensembl_id"], "ENSG2")
        self.assertEqual(rows[0]["evidence_origin"], "synthetic_structural_control")
        self.assertEqual(rows[0]["chip_file_accessions"], "")


if __name__ == "__main__":
    unittest.main()
