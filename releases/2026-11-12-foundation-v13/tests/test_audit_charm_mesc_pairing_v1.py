from __future__ import annotations

import gzip
import importlib.util
import io
import json
from pathlib import Path
import random
import tarfile

import pytest


test_dir = Path(__file__).parent
script = test_dir / "audit-charm-mesc-pairing-v1.py"
if not script.exists():
    script = test_dir.parent / script.name
spec = importlib.util.spec_from_file_location("charm_audit", script)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def write_gzip(path, text):
    with gzip.open(path, "wt", encoding="utf-8", newline="") as stream:
        stream.write(text)


def make_files(tmp_path, *, duplicate_contact=False):
    rna, qc, atac, contacts = [tmp_path / name for name in
                               ("rna.gz", "qc.gz", "atac.gz", "pairs.tar.gz")]
    write_gzip(rna, "gene\tP1\tP2\tP3\tP4\nA\t1\t0.5\t1\t0\nB\t0\t1\t0\t0\n")
    write_gzip(qc, "cellname\tcelltype\tnUMIs\tnGenes\tcleaned_contacts\tnCount_accessibility"
                   "\thistone_modification_type\n"
                   "P1\tmESC\t1\t1\t2\t2\tH3K27ac\n"
                   "P2\tmESC\t1.5\t2\t2\t1\tH3K27ac\n"
                   "P3\tmESC\t1\t1\t2\t1\tH3K27ac\n"
                   "R1P1\tbrain\t1\t1\t2\t1\tH3K27ac\n")
    write_gzip(atac, "chr1\t0\t10\t.\t60\t+\tP1\nchr1\t10\t20\t.\t60\t-\tP1\n"
                     "chr2\t0\t10\t.\t60\t+\tP3\n")
    with tarfile.open(contacts, "w:gz") as archive:
        names = ["P1", "P2"] + (["P1"] if duplicate_contact else [])
        for name in names:
            data = gzip.compress(b"#pairs format v1.0\nread1\tchr1\t1\tchr1\t2\t+\t-\n")
            info = tarfile.TarInfo(f"pairs/{name}.pairs.gz")
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    return rna, qc, atac, contacts


def test_direct_intersection_and_fractional_rna(tmp_path):
    rna, qc, atac, contacts = make_files(tmp_path)
    report, private = audit.audit_sources(rna, qc, atac, contacts)
    assert report["state"] == "complete"
    assert report["final_training_ready"] is False
    assert report["rna"]["fractional_entries"] == 1
    assert report["overlap"] == {
        "rna_positive_and_qc": 3, "rna_positive_qc_and_atac": 2,
        "rna_positive_qc_atac_and_contacts": 1,
        "atac_cells_in_qc": 2, "atac_fragment_count_equals_qc": 2,
    }
    assert private["rna_positive_qc_atac_contacts"] == ["P1"]
    output = tmp_path / "result"
    audit.write_results(output, report, private)
    public = json.loads((output / "audit.json").read_text())
    assert public["private_identity_sha256"] == audit.describe_input(output / "identities.private.json")["sha256"]
    assert "P1" not in (output / "audit.json").read_text()


def test_missing_contact_is_explicitly_partial(tmp_path):
    rna, qc, atac, _ = make_files(tmp_path)
    report, private = audit.audit_sources(rna, qc, atac, None)
    assert report["state"] == "partial-no-contact-archive"
    assert report["contacts"] is None
    assert report["overlap"]["rna_positive_qc_atac_and_contacts"] is None
    assert private["rna_positive_qc_atac_contacts"] is None


def test_duplicate_contact_identifiers_are_rejected(tmp_path):
    rna, qc, atac, contacts = make_files(tmp_path, duplicate_contact=True)
    with pytest.raises(ValueError, match="duplicated cell identifier"):
        audit.audit_sources(rna, qc, atac, contacts)


def test_malformed_fragment_is_rejected(tmp_path):
    rna, qc, atac, contacts = make_files(tmp_path)
    write_gzip(atac, "chr1\t10\t0\t.\t60\t+\tP1\n")
    with pytest.raises(ValueError, match="invalid genomic coordinates"):
        audit.audit_sources(rna, qc, atac, contacts)


def test_partial_download_is_rejected(tmp_path):
    path = tmp_path / "data.part"
    path.write_bytes(b"unfinished")
    with pytest.raises(ValueError, match="partial download"):
        audit.describe_input(path)


def test_malformed_numeric_suffix_is_not_silently_truncated(tmp_path):
    rna, qc, atac, contacts = make_files(tmp_path)
    write_gzip(rna, "gene\tP1\tP2\nA\t1\t2invalid\n")
    with pytest.raises(ValueError):
        audit.audit_sources(rna, qc, atac, contacts)


def test_contact_archive_can_stream_past_large_inner_member(tmp_path):
    archive_path = tmp_path / "large.tar.gz"
    payload = b"read1\tchr1\t1\tchr1\t2\t+\t-\n" + random.Random(42).randbytes(250_000)
    with tarfile.open(archive_path, "w:gz") as archive:
        for cell in ("P1", "P2"):
            data = gzip.compress(payload)
            info = tarfile.TarInfo(f"pairs/{cell}.pairs.gz")
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    report, cells = audit.contact_summary(archive_path)
    assert report["cells_with_contact_files"] == 2
    assert cells == {"P1", "P2"}
