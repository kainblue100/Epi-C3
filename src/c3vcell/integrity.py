"""Content fingerprints for input and model artifacts."""

import hashlib
import json
import zipfile
from pathlib import Path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verified_source_record(
    path: Path, *, source: str, source_md5: str | None, expected_sha256: str | None = None,
) -> dict:
    record = json.loads(path.with_name(path.name + ".json").read_text(encoding="utf-8"))
    if not isinstance(record, dict):
        raise ValueError(f"Source provenance must be an object: {path}")
    sha = hashlib.sha256()
    md5 = hashlib.md5() if source_md5 is not None else None
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(chunk)
            if md5 is not None:
                md5.update(chunk)
    if (
        record.get("source") != source or "source_md5" not in record
        or record["source_md5"] != source_md5
        or (md5 is not None and md5.hexdigest() != source_md5)
        or record.get("sha256") != sha.hexdigest()
        or (expected_sha256 is not None and sha.hexdigest() != expected_sha256)
        or record.get("bytes") != path.stat().st_size
    ):
        raise ValueError(f"Source file or provenance changed: {path}")
    return record


def snapshot_files(root: Path, fingerprints: dict[str, str], output: Path) -> str:
    root = root.resolve()
    if not fingerprints:
        raise ValueError("A source snapshot requires files")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    with zipfile.ZipFile(
        output, "x", compression=zipfile.ZIP_DEFLATED, strict_timestamps=False,
    ) as archive:
        for name, expected in sorted(fingerprints.items()):
            relative = Path(name)
            source = (root / relative).resolve()
            if relative.is_absolute() or not source.is_relative_to(root):
                raise ValueError(f"Snapshot source escapes the project root: {name}")
            if file_sha256(source) != expected:
                raise RuntimeError(f"Source changed before snapshot: {name}")
            archive.write(source, arcname=relative.as_posix())
    with zipfile.ZipFile(output) as archive:
        for name, expected in fingerprints.items():
            if hashlib.sha256(archive.read(Path(name).as_posix())).hexdigest() != expected:
                raise RuntimeError(f"Snapshot content differs from its fingerprint: {name}")
    return file_sha256(output)
