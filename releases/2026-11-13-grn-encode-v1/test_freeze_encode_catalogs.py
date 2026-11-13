from __future__ import annotations

import unittest
import urllib.parse

import freeze_encode_catalogs as freeze


class CatalogTests(unittest.TestCase):
    def test_search_urls_request_all_released_public_k562_records(self):
        for kind in ("files", "experiments"):
            query = urllib.parse.parse_qs(urllib.parse.urlparse(freeze.catalog_url(kind)).query)
            self.assertEqual(query["status"], ["released"])
            self.assertEqual(query["biosample_ontology.term_name"], ["K562"])
            self.assertEqual(query["limit"], ["all"])
        self.assertEqual(len(urllib.parse.parse_qs(
            urllib.parse.urlparse(freeze.catalog_url("files")).query
        )["output_type"]), 2)

    def test_rejects_truncated_or_duplicate_search(self):
        self.assertEqual(freeze.validate_catalog({"total": 2, "@graph": [
            {"accession": "A"}, {"accession": "B"}
        ]}, "files"), 2)
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            freeze.validate_catalog({"total": 3, "@graph": [{"accession": "A"}]}, "files")
        with self.assertRaisesRegex(ValueError, "duplicate"):
            freeze.validate_catalog({"total": 2, "@graph": [
                {"accession": "A"}, {"accession": "A"}
            ]}, "files")


if __name__ == "__main__":
    unittest.main()
