"""Resume one documented public 4DN HFFc6 Hi-C example, not an HCT116 reference."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
import urllib.request
from pathlib import Path

URL = (
    "https://4dn-open-data-public.s3.amazonaws.com/fourfront-webprod/wfoutput/"
    "bd06e6c5-765f-4660-8b1c-23b681bcfff0/4DNFIB59T7NN.mcool"
)
ACCESSION_URL = (
    "https://data.4dnucleome.org/files-processed/4DNFIB59T7NN/"
    "@@download/4DNFIB59T7NN.mcool"
)
REFERENCE_URL = (
    "https://github.com/open2c/cooltools/blob/master/"
    "docs/notebooks_old/07_pileups2.ipynb"
)
SIZE = 22_837_705_555
ETAG = "b2bb36baaba61e620940f758da9b8bc4-2723"
PART_SIZE = 8 * 1024**2
PROGRESS_INTERVAL = 256 * 1024**2


def _write_status(path: Path, record: dict) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", prefix=".4dn-status-", dir=path.parent, delete=False,
    ) as stream:
        temporary = Path(stream.name)
        json.dump(record, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def download(directory: Path, reserve_gib: int) -> None:
    if reserve_gib < 200:
        raise ValueError("At least 200 GiB of free space must be reserved")
    directory.mkdir(parents=True, exist_ok=True)
    final = directory / "4DNFIB59T7NN.mcool"
    partial = directory / "4DNFIB59T7NN.mcool.part"
    status_path = directory / "4DNFIB59T7NN.download.json"
    if final.exists():
        raise FileExistsError(f"Refusing to overwrite {final}")
    if partial.exists() != status_path.exists():
        raise FileExistsError("A partial download without its matching status cannot be resumed")
    if partial.exists():
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if status.get("url") != URL or status.get("expected_etag") != ETAG:
            raise ValueError("Partial download provenance does not match the requested object")
    else:
        with partial.open("xb"):
            pass
        status = {
            "url": URL,
            "accession_url": ACCESSION_URL,
            "source_description_url": REFERENCE_URL,
            "accession": "4DNFIB59T7NN",
            "biosample": "HFFc6",
            "assay": "DpnII Hi-C",
            "genome_assembly": "unknown in file metadata",
            "expected_bytes": SIZE,
            "expected_etag": ETAG,
            "etag_part_bytes": PART_SIZE,
            "state": "downloading",
            "bytes": 0,
            "note": (
                "HFFc6 is not HCT116. Multipart ETag verifies the public object bytes, "
                "not the biological metadata; this file is not paired single-cell data."
            ),
        }
        with status_path.open("x", encoding="utf-8") as stream:
            json.dump(status, stream, indent=2)
            stream.write("\n")

    offset = partial.stat().st_size
    if offset > SIZE:
        raise ValueError("Partial file is larger than the expected source object")
    if shutil.disk_usage(directory).free - (SIZE - offset) < reserve_gib * 1024**3:
        raise OSError("Completing the download would violate the disk-space reserve")
    sha256 = hashlib.sha256()
    part_md5 = hashlib.md5()
    part_digests = []
    current_part = 0

    def update(chunk: bytes) -> None:
        nonlocal part_md5, current_part
        sha256.update(chunk)
        start = 0
        while start < len(chunk):
            length = min(PART_SIZE - current_part, len(chunk) - start)
            part_md5.update(chunk[start:start + length])
            current_part += length
            start += length
            if current_part == PART_SIZE:
                part_digests.append(part_md5.digest())
                part_md5 = hashlib.md5()
                current_part = 0

    with partial.open("rb") as stream:
        while True:
            chunk = stream.read(1024**2)
            if not chunk:
                break
            update(chunk)
    status.update(state="downloading", bytes=offset)
    _write_status(status_path, status)
    try:
        if offset < SIZE:
            headers = {"If-Match": f'"{ETAG}"'}
            if offset:
                headers["Range"] = f"bytes={offset}-"
            request = urllib.request.Request(URL, headers=headers)
            with urllib.request.urlopen(request, timeout=120) as response, partial.open("ab") as out:
                expected_status = 206 if offset else 200
                if response.status != expected_status:
                    raise RuntimeError(f"Unexpected download HTTP status {response.status}")
                if response.headers.get("ETag", "").strip('"') != ETAG:
                    raise RuntimeError("The public S3 object ETag changed")
                if offset and not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                    raise RuntimeError("The server did not honor the requested resume offset")
                next_progress = offset + PROGRESS_INTERVAL
                while True:
                    chunk = response.read(1024**2)
                    if not chunk:
                        break
                    if offset + len(chunk) > SIZE:
                        raise RuntimeError("The server returned more bytes than expected")
                    out.write(chunk)
                    update(chunk)
                    offset += len(chunk)
                    if offset >= next_progress:
                        out.flush()
                        if shutil.disk_usage(directory).free < reserve_gib * 1024**3:
                            raise OSError("Stopping because the disk-space reserve was reached")
                        status["bytes"] = offset
                        _write_status(status_path, status)
                        print(f"4DN HFFc6: {offset / 1024**3:.2f} / {SIZE / 1024**3:.2f} GiB", flush=True)
                        next_progress = offset + PROGRESS_INTERVAL
                out.flush()
                os.fsync(out.fileno())
        if offset != SIZE:
            raise RuntimeError(f"Incomplete 4DN object: {offset} of {SIZE} bytes")
        if current_part:
            part_digests.append(part_md5.digest())
        actual_etag = f"{hashlib.md5(b''.join(part_digests)).hexdigest()}-{len(part_digests)}"
        if actual_etag != ETAG:
            raise RuntimeError(f"4DN multipart ETag mismatch: {actual_etag}")
        os.link(partial, final)
        partial.unlink()
        status.update(state="complete", bytes=offset, sha256=sha256.hexdigest())
        _write_status(status_path, status)
        print(f"Verified and published {final} ({offset} bytes)", flush=True)
    except Exception as exc:
        status.update(state="error", bytes=partial.stat().st_size if partial.exists() else offset)
        status["error"] = f"{type(exc).__name__}: {exc}"
        _write_status(status_path, status)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--reserve-gib", type=int, default=200)
    args = parser.parse_args()
    download(args.outdir, args.reserve_gib)


if __name__ == "__main__":
    main()
