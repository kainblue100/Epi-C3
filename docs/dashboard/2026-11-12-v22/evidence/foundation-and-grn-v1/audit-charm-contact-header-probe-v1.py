"""Inspect a bounded sample of CHARM chromosome headers without exposing cell IDs."""

from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import tarfile


MEMBER = re.compile(r"P[0-9]+\.pairs\.gz\Z")
MAX_LINE_BYTES = 1024 * 1024


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def inspect_headers(archive: Path, full_audit: dict, limit: int = 100) -> dict:
    if (
        full_audit.get("status") != "complete"
        or full_audit.get("mapped_record_check") != "pass"
        or full_audit.get("distinct_declared_chromosome_sets") != 2
        or full_audit.get("archive_name") != archive.name
        or not isinstance(full_audit.get("archive_bytes"), int)
        or not isinstance(full_audit.get("archive_sha256"), str)
        or re.fullmatch(r"[0-9a-f]{64}", full_audit["archive_sha256"]) is None
        or limit < 1
    ):
        raise ValueError("A completed, matching full-coordinate audit and positive limit are required")
    groups: Counter = Counter()
    seen: set[str] = set()
    with archive.open("rb") as source:
        before = os.fstat(source.fileno())
        if before.st_size != full_audit["archive_bytes"]:
            raise ValueError("Archive size differs from the completed audit")
        with tarfile.open(fileobj=source, mode="r|gz") as members:
            for member in members:
                if member.isdir():
                    continue
                entry = PurePosixPath(member.name)
                if (
                    not member.isfile() or entry.is_absolute() or ".." in entry.parts
                    or MEMBER.fullmatch(entry.name) is None or entry.name in seen
                ):
                    raise ValueError("Unexpected or duplicate archive member")
                seen.add(entry.name)
                raw = members.extractfile(member)
                if raw is None:
                    raise ValueError("Unreadable contact member")
                chromosomes: dict[str, int] = {}
                with gzip.GzipFile(fileobj=raw) as stream:
                    first_line = stream.readline(MAX_LINE_BYTES + 1)
                    if first_line.rstrip(b"\r\n") != b"## pairs format v1.0":
                        raise ValueError("Invalid pairs header")
                    while True:
                        line = stream.readline(MAX_LINE_BYTES + 1)
                        if len(line) > MAX_LINE_BYTES:
                            raise ValueError("A contact header exceeds the maximum line length")
                        if not line.startswith(b"#"):
                            if not line:
                                raise ValueError("A contact file has no records")
                            break
                        key, separator, value = line.partition(b":")
                        if separator and key in (b"#chromosome", b"#chromsize"):
                            fields = value.split()
                            if len(fields) != 2:
                                raise ValueError("Invalid chromosome header")
                            name = fields[0].decode("ascii")
                            length = int(fields[1])
                            if name in chromosomes or length <= 0:
                                raise ValueError("Duplicate or invalid chromosome declaration")
                            chromosomes[name] = length
                if not chromosomes:
                    raise ValueError("A contact header has no chromosome declarations")
                groups[tuple(sorted(chromosomes.items()))] += 1
                if len(groups) == 2 or len(seen) == limit:
                    break
        after = os.fstat(source.fileno())
    current = archive.stat()
    identity = lambda stat: (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
    if identity(before) != identity(after) or identity(before) != identity(current):
        raise ValueError("The archive changed during the header probe")
    return {
        "format": "epi-c3-charm-contact-header-probe-v1",
        "status": "complete" if len(groups) == 2 else "incomplete",
        "sampled_files": len(seen),
        "sample_limit": limit,
        "declared_sets_found": len(groups),
        "groups": [
            {"chromosomes": [name for name, _ in group], "sampled_files": count}
            for group, count in sorted(groups.items())
        ],
        "archive_name": archive.name,
        "archive_bytes": before.st_size,
        "archive_sha256_from_full_audit": full_audit["archive_sha256"],
        "limitations": [
            "Only the first sampled headers were inspected; group counts are not archive-wide frequencies.",
            "The archive hash is inherited from the earlier full audit, not independently recomputed by this probe.",
            "The reason for differences in declared chromosomes is not established.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--full-audit", required=True, type=Path)
    parser.add_argument("--full-audit-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if digest(args.full_audit) != args.full_audit_sha256:
        raise ValueError("Full-coordinate audit hash does not match")
    full_audit = json.loads(args.full_audit.read_text(encoding="utf-8"))
    report = inspect_headers(args.archive, full_audit)
    report["full_audit_sha256"] = args.full_audit_sha256
    report["script_sha256"] = digest(Path(__file__))
    descriptor = os.open(args.output, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
