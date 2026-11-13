from __future__ import annotations

import gzip
import importlib.util
import io
from pathlib import Path
import tarfile

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "audit-charm-contact-header-probe-v1.py"
spec = importlib.util.spec_from_file_location("charm_header_probe", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def make_archive(path: Path) -> dict:
    with tarfile.open(path, "w:gz") as archive:
        for index, has_y in enumerate((False, False, True), 1):
            chrom = b"#chromsize: chr1 100\n"
            if has_y:
                chrom += b"#chromsize: chrY 90\n"
            payload = gzip.compress(b"## pairs format v1.0\n" + chrom + b"#columns: sample\nr1\tchr1\n")
            member = tarfile.TarInfo(f"P{index}.pairs.gz")
            member.size = len(payload)
            archive.addfile(member, io.BytesIO(payload))
    return {
        "status": "complete", "mapped_record_check": "pass", "distinct_declared_chromosome_sets": 2,
        "archive_name": path.name, "archive_bytes": path.stat().st_size, "archive_sha256": module.digest(path),
    }


def test_header_probe_distinguishes_chromosome_sets_without_identities(tmp_path):
    path = tmp_path / "contacts.tar.gz"
    audit = make_archive(path)
    result = module.inspect_headers(path, audit)
    assert result["status"] == "complete"
    assert result["sampled_files"] == 3
    assert result["declared_sets_found"] == 2
    assert result["groups"] == [
        {"chromosomes": ["chr1"], "sampled_files": 2},
        {"chromosomes": ["chr1", "chrY"], "sampled_files": 1},
    ]
    assert "P1" not in str(result) and "P2" not in str(result)


def test_header_probe_rejects_mismatched_audit_and_reports_bounded_scan(tmp_path):
    path = tmp_path / "contacts.tar.gz"
    audit = make_archive(path)
    result = module.inspect_headers(path, audit, limit=1)
    assert result["status"] == "incomplete"
    assert result["sampled_files"] == 1
    with pytest.raises(ValueError, match="size"):
        module.inspect_headers(path, {**audit, "archive_bytes": 1})
    with pytest.raises(ValueError, match="completed"):
        module.inspect_headers(path, {**audit, "mapped_record_check": "findings"})
