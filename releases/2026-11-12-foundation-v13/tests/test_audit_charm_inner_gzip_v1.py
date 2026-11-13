import gzip
import hashlib
import importlib.util
import io
from pathlib import Path
import tarfile

import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "audit-charm-inner-gzip-v1.py"
SPEC = importlib.util.spec_from_file_location("charm_inner_integrity", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def fixture_archive(tmp_path, entries):
    archive_path = tmp_path / "contacts.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        for name, payload in entries:
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
    data = archive_path.read_bytes()
    audit = {
        "state": "complete",
        "inputs": {
            "contacts": {
                "name": archive_path.name,
                "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        },
        "contacts": {"cells_with_contact_files": len(entries)},
    }
    return archive_path, audit


def test_all_nested_streams_are_checked_and_hash_bound(tmp_path):
    first = b"#contact\nrecord one\n"
    second = b"#contact\nrecord two\n"
    path, audit = fixture_archive(
        tmp_path, [("P1.pairs.gz", gzip.compress(first)), ("folder/P2.pairs.gz", gzip.compress(second))]
    )
    result = MODULE.verify_archive(path, audit)
    assert result["status"] == "complete"
    assert result["inner_gzip_streams_verified"] == 2
    assert result["inner_uncompressed_bytes"] == len(first) + len(second)
    assert result["archive_sha256"] == audit["inputs"]["contacts"]["sha256"]
    assert "P1.pairs.gz" not in repr(result)


def test_corrupt_inner_crc_is_not_accepted(tmp_path):
    damaged = bytearray(gzip.compress(b"#contact\nrecord\n"))
    damaged[-8] ^= 1
    path, audit = fixture_archive(tmp_path, [("P1.pairs.gz", bytes(damaged))])
    with pytest.raises(gzip.BadGzipFile, match="CRC"):
        MODULE.verify_archive(path, audit)


def test_archive_hash_and_count_are_not_silently_trusted(tmp_path):
    path, audit = fixture_archive(tmp_path, [("P1.pairs.gz", gzip.compress(b"record\n"))])
    audit["inputs"]["contacts"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="hash differs"):
        MODULE.verify_archive(path, audit)
    audit["inputs"]["contacts"]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    audit["contacts"]["cells_with_contact_files"] = 2
    with pytest.raises(ValueError, match="count differs"):
        MODULE.verify_archive(path, audit)


@pytest.mark.parametrize("names", [
    ["P1.pairs.gz", "nested/P1.pairs.gz"],
    ["../P1.pairs.gz"],
    ["/P1.pairs.gz"],
    ["unknown.gz"],
])
def test_unsafe_or_duplicate_members_are_rejected(tmp_path, names):
    path, audit = fixture_archive(tmp_path, [(name, gzip.compress(b"record\n")) for name in names])
    with pytest.raises(ValueError, match="Unexpected or duplicate"):
        MODULE.verify_archive(path, audit)
