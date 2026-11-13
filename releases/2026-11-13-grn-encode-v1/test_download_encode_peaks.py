from __future__ import annotations

import gzip
import hashlib
from pathlib import Path
import tempfile
import unittest

import download_encode_peaks as download


class PeakTests(unittest.TestCase):
    def test_checks_complete_gzip_checksum_and_narrowpeak_records(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "peak.gz"
            with gzip.open(path, "wt", encoding="utf-8") as stream:
                stream.write("chr1\t100\t130\tp1\t100\t.\t4.0\t2.0\t1.0\t12\n")
            raw = path.read_bytes()
            result = download.verify_peak_file(path, len(raw), hashlib.md5(raw).hexdigest())
            self.assertEqual(result["rows"], 1)
            self.assertEqual(result["invalid_summit_rows_excluded"], 0)
            self.assertTrue(result["gzip_crc_checked"])
            with self.assertRaisesRegex(ValueError, "checksum"):
                download.verify_peak_file(path, len(raw), "0" * 32)
            with gzip.open(path, "wt", encoding="utf-8") as stream:
                stream.write("chr1\t100\t100\tp1\t100\t.\t4.0\t2.0\t1.0\t0\n")
            raw = path.read_bytes()
            with self.assertRaisesRegex(ValueError, "invalid narrowPeak"):
                download.verify_peak_file(path, len(raw), hashlib.md5(raw).hexdigest())

    def test_out_of_interval_summit_is_counted_and_excluded(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "peak.gz"
            with gzip.open(path, "wt", encoding="utf-8") as stream:
                stream.write("chr1\t100\t130\tp1\t100\t.\t4.0\t2.0\t1.0\t30\n")
                stream.write("chr1\t200\t230\tp2\t100\t.\t4.0\t2.0\t1.0\t0\n")
            raw = path.read_bytes()
            result = download.verify_peak_file(path, len(raw), hashlib.md5(raw).hexdigest())
            self.assertEqual(result["invalid_summit_rows_excluded"], 1)
            self.assertEqual(list(download.usable_intervals(path)), [("chr1", 200, 230)])

    def test_rejects_noncanonical_download_url(self):
        with self.assertRaisesRegex(ValueError, "invalid provenance"):
            download.validate_selected_record({
                "file_accession": "ENCFF000AAA", "href": "https://example.invalid/file",
                "size": 1, "official_md5": "a" * 32, "assembly": "GRCh38",
                "output_type": "conservative IDR thresholded peaks",
            })


if __name__ == "__main__":
    unittest.main()
