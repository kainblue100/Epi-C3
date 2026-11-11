import hashlib
import json
import os
import zipfile
from pathlib import Path

import pytest

from c3vcell.integrity import file_sha256, snapshot_files, verified_source_record


def test_source_snapshot_preserves_hashes_and_rejects_bad_inputs(tmp_path: Path) -> None:
    root = tmp_path / "source"
    root.mkdir()
    source = root / "model.py"
    source.write_text("value = 1\n", encoding="utf-8")
    fingerprints = {"model.py": file_sha256(source)}
    archive = tmp_path / "snapshot.zip"
    digest = snapshot_files(root, fingerprints, archive)
    assert digest == file_sha256(archive)
    with zipfile.ZipFile(archive) as snapshot:
        assert snapshot.read("model.py") == source.read_bytes()
    with pytest.raises(FileExistsError, match="overwrite"):
        snapshot_files(root, fingerprints, archive)
    with pytest.raises(RuntimeError, match="changed"):
        snapshot_files(root, {"model.py": "incorrect"}, tmp_path / "stale.zip")
    with pytest.raises(ValueError, match="escapes"):
        snapshot_files(root, {"../outside.py": "incorrect"}, tmp_path / "outside.zip")


def test_source_snapshot_accepts_reproducible_archive_timestamps(tmp_path: Path) -> None:
    source = tmp_path / "model.py"
    source.write_text("value = 1\n", encoding="utf-8")
    os.utime(source, (0, 0))
    archive = tmp_path / "snapshot.zip"

    snapshot_files(tmp_path, {"model.py": file_sha256(source)}, archive)

    with zipfile.ZipFile(archive) as snapshot:
        assert snapshot.read("model.py") == source.read_bytes()
        assert snapshot.getinfo("model.py").date_time == (1980, 1, 1, 0, 0, 0)


@pytest.mark.parametrize("has_source_md5", [True, False])
def test_verified_source_records_distinguish_source_checksums(
    tmp_path: Path, has_source_md5: bool,
) -> None:
    source = tmp_path / "input.txt"
    source.write_bytes(b"example")
    md5 = hashlib.md5(b"example").hexdigest() if has_source_md5 else None
    record = {
        "source": "official", "source_md5": md5, "sha256": file_sha256(source),
        "bytes": source.stat().st_size,
    }
    sidecar = source.with_name(source.name + ".json")
    sidecar.write_text(json.dumps(record), encoding="utf-8")
    assert verified_source_record(
        source, source="official", source_md5=md5, expected_sha256=record["sha256"],
    ) == record
    with pytest.raises(ValueError, match="provenance changed"):
        verified_source_record(source, source="wrong", source_md5=md5)
    with pytest.raises(ValueError, match="provenance changed"):
        verified_source_record(source, source="official", source_md5=md5, expected_sha256="bad")
    source.write_bytes(b"changed")
    with pytest.raises(ValueError, match="provenance changed"):
        verified_source_record(source, source="official", source_md5=md5)
