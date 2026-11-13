"""Verify every nested CHARM contact gzip against a frozen source audit."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import tarfile


CHUNK_SIZE = 1024 * 1024
IDENTIFIER = re.compile(r"P[0-9]+\.pairs\.gz\Z")


class HashingReader:
    def __init__(self, stream):
        self.stream = stream
        self.digest = hashlib.sha256()

    def read(self, size=-1):
        data = self.stream.read(size)
        self.digest.update(data)
        return data


def verify_archive(path: Path, expected: dict) -> dict:
    expected_input = expected["inputs"]["contacts"]
    expected_count = expected["contacts"]["cells_with_contact_files"]
    if expected["state"] != "complete" or path.name != expected_input["name"]:
        raise ValueError("The completed pairing audit does not describe this archive")
    if not isinstance(expected_count, int) or expected_count <= 0:
        raise ValueError("The pairing audit has an invalid contact-file count")

    identifiers: set[str] = set()
    uncompressed_bytes = 0
    with path.open("rb") as source:
        before = os.fstat(source.fileno())
        if before.st_size != expected_input["bytes"]:
            raise ValueError("Archive size differs from the frozen pairing audit")
        reader = HashingReader(source)
        with tarfile.open(fileobj=reader, mode="r|gz") as archive:
            for member in archive:
                if member.isdir():
                    continue
                entry = PurePosixPath(member.name)
                if (
                    not member.isfile() or entry.is_absolute() or ".." in entry.parts
                    or IDENTIFIER.fullmatch(entry.name) is None
                    or entry.name in identifiers
                ):
                    raise ValueError("Unexpected or duplicate contact archive member")
                raw = archive.extractfile(member)
                if raw is None:
                    raise ValueError("Unreadable contact archive member")
                member_bytes = 0
                with gzip.GzipFile(fileobj=raw) as stream:
                    for chunk in iter(lambda: stream.read(CHUNK_SIZE), b""):
                        member_bytes += len(chunk)
                if member_bytes == 0:
                    raise ValueError("An inner contact gzip is empty")
                uncompressed_bytes += member_bytes
                identifiers.add(entry.name)
                if len(identifiers) % 25 == 0:
                    print(f"Verified {len(identifiers)} of {expected_count} inner gzip streams", flush=True)
        for _ in iter(lambda: reader.read(CHUNK_SIZE), b""):
            pass
        after = os.fstat(source.fileno())

    current = path.stat()
    original_identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    if original_identity != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
        raise ValueError("The archive changed while it was being verified")
    if original_identity != (current.st_dev, current.st_ino, current.st_size, current.st_mtime_ns):
        raise ValueError("The archive path changed while it was being verified")
    if len(identifiers) != expected_count:
        raise ValueError("The contact-file count differs from the frozen pairing audit")
    archive_sha256 = reader.digest.hexdigest()
    if archive_sha256 != expected_input["sha256"]:
        raise ValueError("The archive hash differs from the frozen pairing audit")
    return {
        "format": "epi-c3-charm-inner-gzip-integrity-v1",
        "status": "complete",
        "archive_name": path.name,
        "archive_bytes": before.st_size,
        "archive_sha256": archive_sha256,
        "inner_gzip_streams_verified": len(identifiers),
        "inner_uncompressed_bytes": uncompressed_bytes,
        "limitations": [
            "All inner gzip streams were read to EOF, validating their gzip CRC and length.",
            "The archive hash is local and is not an independent source-provided digest.",
            "This integrity check does not validate all contact records, coordinates, or biological pairing.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--pairing-audit", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    audit_bytes = args.pairing_audit.read_bytes()
    expected = json.loads(audit_bytes)
    args.output.mkdir(mode=0o700, parents=True, exist_ok=False)
    report = verify_archive(args.archive, expected)
    report["pairing_audit_sha256"] = hashlib.sha256(audit_bytes).hexdigest()
    report["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    destination = args.output / "audit.json"
    descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(report, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
