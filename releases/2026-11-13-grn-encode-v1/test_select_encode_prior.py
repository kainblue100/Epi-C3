from __future__ import annotations

import copy
import unittest

import select_encode_prior as select


def experiment():
    return {
        "accession": "ENCSR000AAA", "status": "released", "assay_title": "TF ChIP-seq",
        "biosample_ontology": {"term_name": "K562"}, "biosample_summary": "Homo sapiens K562",
        "perturbed": False, "lab": {"title": "Example laboratory"},
        "target": {"label": "A", "genes": [{"symbol": "A", "dbxrefs": ["ENSEMBL:ENSG1"]}]},
        "replicates": [
            {"biological_replicate_number": i, "library": {
                "treatments": [], "biosample": {"treatments": [], "applied_modifications": []}
            }} for i in (1, 2)
        ],
    }


def peak(accession, output_type):
    return {
        "accession": accession, "dataset": "/experiments/ENCSR000AAA/", "target": {"label": "A"},
        "status": "released", "assay_title": "TF ChIP-seq", "biosample_ontology": {"term_name": "K562"},
        "assembly": "GRCh38", "file_format": "bed", "file_format_type": "narrowPeak",
        "output_type": output_type, "file_size": 10, "md5sum": "a" * 32,
        "href": f"/files/{accession}/@@download/{accession}.bed.gz", "biological_replicates": [1, 2],
    }


class SelectionTests(unittest.TestCase):
    def test_prefers_one_conservative_file_per_verified_experiment(self):
        files = [peak("ENCFF000AAA", "optimal IDR thresholded peaks"),
                 peak("ENCFF000BBB", "conservative IDR thresholded peaks")]
        selected, summary = select.select_files(files, [experiment()], {"A"}, {"A": "ENSG1"}, {"ENSG1"},
                                                {"train": [], "val": [], "test": ["A"]})
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]["file_accession"], "ENCFF000BBB")
        self.assertEqual(summary["selected_sources_by_pilot_target_split"]["test"], 1)

    def test_rejects_modified_and_unmatched_ensembl_biosamples(self):
        exp = experiment()
        changed = copy.deepcopy(exp)
        changed["replicates"][0]["library"]["biosample"]["applied_modifications"] = ["unknown"]
        self.assertEqual(select.unmodified_experiment(changed, "A", "ENSG1", [1, 2]),
                         "treated_modified_or_unverified_biosample")
        changed = copy.deepcopy(exp)
        changed["target"]["genes"][0]["dbxrefs"] = ["ENSEMBL:ENSG2"]
        self.assertEqual(select.unmodified_experiment(changed, "A", "ENSG1", [1, 2]),
                         "target_ensembl_id_mismatch")
        self.assertEqual(select.unmodified_experiment(exp, "A", "ENSG1", [1, 3]),
                         "unverified_file_replicates")

    def test_untrusted_or_unavailable_metadata_cannot_yield_a_graph_source(self):
        bad = peak("ENCFF000AAA", "conservative IDR thresholded peaks")
        bad["assembly"] = "GRCh37"
        with self.assertRaisesRegex(ValueError, "No verifiable"):
            select.select_files([bad], [experiment()], {"A"}, {"A": "ENSG1"}, {"ENSG1"},
                                {"train": ["A"], "val": [], "test": []})


if __name__ == "__main__":
    unittest.main()
