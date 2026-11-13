from __future__ import annotations

import copy
import unittest

import test_select_encode_prior as fixture
import verify_atac_prior as atac


class AtacTests(unittest.TestCase):
    def test_requires_frozen_public_atac_metadata_and_clean_replicates(self):
        experiment = fixture.experiment()
        experiment.update(accession=atac.ATAC_EXPERIMENT, assay_title="ATAC-seq")
        metadata = {
            "accession": atac.ATAC_ACCESSION, "dataset": f"/experiments/{atac.ATAC_EXPERIMENT}/",
            "status": "released", "assembly": "GRCh38", "file_format": "bed",
            "file_format_type": "narrowPeak", "output_type": "conservative IDR thresholded peaks",
            "file_size": atac.ATAC_BYTES, "md5sum": atac.ATAC_MD5,
            "href": f"/files/{atac.ATAC_ACCESSION}/@@download/{atac.ATAC_ACCESSION}.bed.gz",
            "biological_replicates": [1, 2],
        }
        source = atac.verify_metadata(metadata, experiment)
        self.assertEqual(source["file_accession"], atac.ATAC_ACCESSION)
        modified = copy.deepcopy(experiment)
        modified["replicates"][0]["library"]["biosample"]["treatments"] = ["drug"]
        with self.assertRaisesRegex(ValueError, "replicate provenance"):
            atac.verify_metadata(metadata, modified)
        metadata["assembly"] = "GRCh37"
        with self.assertRaisesRegex(ValueError, "file metadata"):
            atac.verify_metadata(metadata, experiment)


if __name__ == "__main__":
    unittest.main()
