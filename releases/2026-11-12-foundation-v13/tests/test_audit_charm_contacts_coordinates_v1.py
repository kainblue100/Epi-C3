from __future__ import annotations

import gzip
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest


MODULE_PATH = Path(__file__).resolve().parents[1] / "audit-charm-contacts-coordinates-v1.py"
SPEC = importlib.util.spec_from_file_location("charm_contact_audit", MODULE_PATH)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)

HEADER = (
    b"## pairs format v1.0\n"
    b"#chromosome: chr1 10\n"
    b"#chromosome: chr2 20\n"
    b"#columns:readID\tchr1\tpos1\tchr2\tpos2\tstrand1\tstrand2\tphase0\tphase1\n"
)
RECORD = b"read\tchr1\t1\tchr2\t20\t+\t-\t.\t1\n"
REFERENCE = {b"chr1": 10, b"chr2": 20}


def archive_fixture(root: Path, members: list[tuple[str, bytes]], matched: list[str]):
    archive_path = root / "contacts.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        for name, content in members:
            compressed = gzip.compress(content)
            member = tarfile.TarInfo(name)
            member.size = len(compressed)
            archive.addfile(member, io.BytesIO(compressed))
    identities = root / "identities.private.json"
    identities.write_text(json.dumps({"rna_positive_qc_atac_contacts": matched}), encoding="utf-8")
    pairing = {
        "state": "complete",
        "genome_assembly": "GRCm38",
        "inputs": {
            "contacts": {
                "name": archive_path.name,
                "bytes": archive_path.stat().st_size,
                "sha256": audit.digest_file(archive_path),
            }
        },
        "contacts": {"cells_with_contact_files": len(members)},
        "overlap": {"rna_positive_qc_atac_and_contacts": len(matched)},
        "private_identity_sha256": audit.digest_file(identities),
    }
    pairing_path = root / "pairing.json"
    pairing_path.write_text(json.dumps(pairing), encoding="utf-8")
    return archive_path, pairing_path, identities


class CoordinateAuditTests(unittest.TestCase):
    def test_all_records_and_matched_subset_are_counted_without_identifiers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive, pairing_path, identities = archive_fixture(
                root,
                [("P1.pairs.gz", HEADER + RECORD * 2), ("P2.pairs.gz", HEADER + RECORD)],
                ["P1"],
            )
            pairing, matched, _ = audit.load_pairing(pairing_path, identities)
            result = audit.audit_archive(archive, pairing, matched, REFERENCE, progress_every=0)
            self.assertEqual(result["mapped_record_check"], "pass")
            self.assertEqual(result["all_contacts"]["records"], 3)
            self.assertEqual(result["matched_contacts"]["records"], 2)
            self.assertEqual(result["matched_contact_files"], 1)
            self.assertNotIn("P1", json.dumps(result))

    def test_invalid_records_are_reported_without_treating_them_as_valid(self):
        rows = [
            b"read\tchr1\t0\tchr2\t21\t+\t-\t.\t1\n",
            b"read\tchrX\t2\tchr1\tx\t+\t.\t.\t9\n",
            b"read\t!\t0\tchr2\t1\t+\t?\t0\t1\n",
            b"too\tfew\n",
            b"\tchr1\t1\tchr2\t20\t+\t-\t.\t1\n",
        ]
        metrics, header = audit.audit_stream(io.BytesIO(HEADER + b"".join(rows)), REFERENCE)
        self.assertEqual(header["header_problems"], 0)
        self.assertEqual(metrics["records"], 5)
        self.assertEqual(metrics["out_of_bounds_endpoints"], 2)
        self.assertEqual(metrics["unknown_chromosome_endpoints"], 1)
        self.assertEqual(metrics["undeclared_chromosome_endpoints"], 1)
        self.assertEqual(metrics["noninteger_position_endpoints"], 1)
        self.assertEqual(metrics["unmapped_endpoints"], 1)
        self.assertEqual(metrics["malformed_records"], 1)
        self.assertEqual(metrics["missing_strand_endpoints"], 1)
        self.assertEqual(metrics["invalid_strand_endpoints"], 1)
        self.assertEqual(metrics["invalid_phase_endpoints"], 1)
        self.assertEqual(metrics["empty_read_ids"], 1)

    def test_mismatched_header_refuses_to_interpret_records(self):
        header = HEADER.replace(b"chr1 10", b"chr1 11")
        metrics, report = audit.audit_stream(io.BytesIO(header + RECORD), REFERENCE)
        self.assertGreater(report["header_problems"], 0)
        self.assertEqual(metrics["records_unchecked_due_to_schema"], 1)

    def test_archive_and_private_mapping_are_bound_to_frozen_audit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive, pairing_path, identities = archive_fixture(
                root, [("P1.pairs.gz", HEADER + RECORD)], ["P1"]
            )
            pairing, matched, _ = audit.load_pairing(pairing_path, identities)
            pairing["inputs"]["contacts"]["sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "archive SHA-256"):
                audit.audit_archive(archive, pairing, matched, REFERENCE, progress_every=0)
            identities.write_text('{"rna_positive_qc_atac_contacts":[]}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "identity map"):
                audit.load_pairing(pairing_path, identities)

    def test_duplicate_and_missing_members_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive, pairing_path, identities = archive_fixture(
                root,
                [("P1.pairs.gz", HEADER + RECORD), ("other/P1.pairs.gz", HEADER + RECORD)],
                ["P1"],
            )
            pairing, matched, _ = audit.load_pairing(pairing_path, identities)
            with self.assertRaisesRegex(ValueError, "duplicate"):
                audit.audit_archive(archive, pairing, matched, REFERENCE, progress_every=0)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive, pairing_path, identities = archive_fixture(
                root, [("P1.pairs.gz", HEADER + RECORD)], ["P2"]
            )
            pairing, matched, _ = audit.load_pairing(pairing_path, identities)
            with self.assertRaisesRegex(ValueError, "matched identities"):
                audit.audit_archive(archive, pairing, matched, REFERENCE, progress_every=0)

    def test_reference_hash_and_lengths_are_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            reference = Path(directory) / "reference.tsv"
            reference.write_bytes(b"chr1\t10\nchr2\t20\n")
            self.assertEqual(audit.load_reference(reference, audit.digest_file(reference)), REFERENCE)
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                audit.load_reference(reference, "0" * 64)
            reference.write_bytes(b"chr1\t10\nchr1\t20\n")
            with self.assertRaisesRegex(ValueError, "duplicate"):
                audit.load_reference(reference, audit.digest_file(reference))


if __name__ == "__main__":
    unittest.main()
