"""Download and fully validate the frozen public ENCODE narrowPeak files."""

from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import math
import os
from pathlib import Path
import shutil
import time
import urllib.parse
import urllib.request

import select_encode_prior as selection


def iter_narrowpeak(path: Path):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip() or line.startswith(("#", "track ", "browser ")):
                continue
            fields = line.rstrip("\r\n").split("\t")
            if len(fields) != 10:
                raise ValueError(f"{path.name}:{line_number}: expected ten narrowPeak columns")
            try:
                start, end, score, summit = int(fields[1]), int(fields[2]), int(fields[4]), int(fields[9])
                values = [float(fields[index]) for index in (6, 7, 8)]
            except ValueError as exc:
                raise ValueError(f"{path.name}:{line_number}: invalid narrowPeak value") from exc
            if (
                not fields[0].startswith("chr") or start < 0 or end <= start or not 0 <= score <= 1000
                or fields[5] not in ("+", "-", ".") or not all(math.isfinite(value) for value in values)
                or summit < -1
            ):
                raise ValueError(f"{path.name}:{line_number}: invalid narrowPeak interval or measurement")
            yield fields[0], start, end, summit == -1 or summit < end - start


def usable_intervals(path: Path):
    for chrom, start, end, valid_summit in iter_narrowpeak(path):
        if valid_summit:
            yield chrom, start, end


def verify_peak_file(path: Path, expected_size: int, expected_md5: str, expected_sha: str | None = None) -> dict:
    before = path.stat()
    if before.st_size != expected_size:
        raise ValueError(f"Peak file size mismatch: {path.name}")
    md5, sha = hashlib.md5(), hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            md5.update(block)
            sha.update(block)
    if md5.hexdigest() != expected_md5 or (expected_sha is not None and sha.hexdigest() != expected_sha):
        raise ValueError(f"Peak file checksum mismatch: {path.name}")
    chromosomes = Counter()
    rows = dropped = 0
    for chrom, _, _, valid_summit in iter_narrowpeak(path):
        chromosomes[chrom] += 1
        rows += 1
        dropped += not valid_summit
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns) or rows <= dropped:
        raise ValueError(f"Peak file changed during validation or has no usable rows: {path.name}")
    return {
        "sha256": sha.hexdigest(), "official_md5": md5.hexdigest(), "bytes": before.st_size,
        "rows": rows, "usable_rows": rows - dropped, "invalid_summit_rows_excluded": dropped,
        "chromosome_rows": dict(sorted(chromosomes.items())),
        "gzip_crc_checked": True, "all_rows_structurally_checked": True,
    }


def validate_selected_record(record: dict) -> None:
    accession, href, size, md5 = (
        record.get("file_accession"), record.get("href"), record.get("size"), record.get("official_md5")
    )
    if (
        not isinstance(accession, str) or not selection.FILE.fullmatch(accession)
        or href != f"/files/{accession}/@@download/{accession}.bed.gz"
        or type(size) is not int or size < 1
        or not isinstance(md5, str) or not selection.MD5.fullmatch(md5)
        or record.get("assembly") != "GRCh38"
        or record.get("output_type") not in ("conservative IDR thresholded peaks", "optimal IDR thresholded peaks")
    ):
        raise ValueError("Selected peak record has invalid provenance")


def download_one(record: dict, outdir: Path) -> dict:
    validate_selected_record(record)
    if shutil.disk_usage(outdir).free < selection.common.RESERVE_WITH_HEADROOM:
        raise OSError("Insufficient disk space above the 200 GiB reserve and 1 GiB headroom")
    accession = record["file_accession"]
    path = outdir / f"{accession}.bed.gz"
    partial = outdir / f".{accession}.bed.gz.part"
    request = urllib.request.Request(
        selection.freeze.HOST + record["href"],
        headers={"Accept-Encoding": "identity", "User-Agent": "Epi-C3-research/1.0"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        final_url = urllib.parse.urlparse(response.geturl())
        if final_url.scheme != "https":
            raise ValueError("ENCODE file download redirected to a non-HTTPS URL")
        descriptor = os.open(partial, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        written = 0
        with os.fdopen(descriptor, "wb") as stream:
            while True:
                block = response.read(1024 * 1024)
                if not block:
                    break
                written += len(block)
                if written > record["size"]:
                    raise ValueError(f"Peak download exceeds the official size: {accession}")
                if shutil.disk_usage(outdir).free < selection.common.RESERVE_WITH_HEADROOM:
                    raise OSError("Disk reserve reached during download")
                stream.write(block)
            stream.flush()
            os.fsync(stream.fileno())
    verified = verify_peak_file(partial, record["size"], record["official_md5"])
    os.link(partial, path)
    partial.unlink()
    return {**record, **verified}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--selection-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = selection.common.read_json(args.selection, args.selection_sha256)
    if (
        manifest.get("format") != "epi-c3-encode-k562-chip-selection-v1"
        or manifest["input_sha256"]["selector_script"] != selection.common.sha256(Path(selection.__file__))
    ):
        raise ValueError("Selected file manifest does not match the selector")
    records = manifest.get("files")
    if not isinstance(records, list) or not records:
        raise ValueError("The selected file manifest is empty")
    for record in records:
        validate_selected_record(record)
    accessions = [item["file_accession"] for item in records]
    if len(set(accessions)) != len(accessions):
        raise ValueError("Duplicate selected peak accessions")
    if shutil.disk_usage(args.output_dir.parent).free < selection.common.RESERVE_WITH_HEADROOM:
        raise OSError("Insufficient disk space above the 200 GiB reserve and 1 GiB headroom")
    args.output_dir.mkdir(mode=0o700)
    verified = []
    for number, record in enumerate(records, 1):
        verified.append(download_one(record, args.output_dir))
        if number % 10 == 0 or number == len(records):
            print(f"Verified {number}/{len(records)} peak files", flush=True)
        time.sleep(0.3)
    result = {
        "format": "epi-c3-encode-k562-chip-verified-files-v1",
        "input_sha256": {
            "selection": args.selection_sha256,
            "downloader_script": selection.common.sha256(Path(__file__)),
        },
        "files": verified,
    }
    digest = selection.write_json(args.output_dir / "verified-files.json", result)
    print(f"verified_manifest_sha256={digest}", flush=True)
    print(f"invalid_summit_rows_excluded={sum(item['invalid_summit_rows_excluded'] for item in verified)}",
          flush=True)


if __name__ == "__main__":
    main()
