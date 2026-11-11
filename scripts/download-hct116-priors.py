"""Download verified HCT116 RRBS, RAD21 ChIA-PET, and genome-conversion references."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
import urllib.request
from pathlib import Path

ENCODE = "https://www.encodeproject.org"
UCSC = "https://hgdownload.soe.ucsc.edu/goldenPath"
ENTRIES = (
    {
        "id": "ENCFF001TMM", "experiment": "ENCSR000DFS", "assay": "RRBS",
        "assembly": "hg19", "format": "bed", "format_type": "bedMethyl",
        "output_type": "methylation state at CpG", "replicate": 1,
        "bytes": 11_461_707, "md5": "c368ce628590962021fa9b04b8681248",
    },
    {
        "id": "ENCFF001TMN", "experiment": "ENCSR000DFS", "assay": "RRBS",
        "assembly": "hg19", "format": "bed", "format_type": "bedMethyl",
        "output_type": "methylation state at CpG", "replicate": 2,
        "bytes": 15_378_739, "md5": "a08913af52cca626aa4ffa344a7bf711",
    },
    {
        "id": "ENCFF625LBL", "experiment": "ENCSR028AOM", "assay": "ChIA-PET",
        "assembly": "GRCh38", "format": "bedpe", "format_type": None,
        "output_type": "loops", "replicate": 2, "target": "RAD21",
        "bytes": 577_826, "md5": "28c59dd8d79703db0be952a6ba47d972",
    },
    {
        "id": "ENCFF769MZQ", "experiment": "ENCSR309KYA", "assay": "ChIA-PET",
        "assembly": "GRCh38", "format": "bedpe", "format_type": None,
        "output_type": "loops", "replicate": 2, "target": "RAD21",
        "bytes": 711_254, "md5": "ef42546db65683f7e33de918bc764a75",
    },
)
REFERENCES = (
    {
        "name": "hg19ToHg38.over.chain.gz",
        "url": f"{UCSC}/hg19/liftOver/hg19ToHg38.over.chain.gz",
        "md5": "35887f73fe5e2231656504d1f6430900",
    },
    {
        "name": "hg38.2bit", "url": f"{UCSC}/hg38/bigZips/hg38.2bit",
        "md5": "dcc3ea27079aa6dc3f9deccd7275e0f8", "bytes": 835_393_456,
    },
)
PREVIEW_REFERENCES = (
    REFERENCES[0],
    {
        "name": "chr21.fa.gz", "url": f"{UCSC}/hg38/chromosomes/chr21.fa.gz",
        "md5": "184df2bd9b812b6e6b6da16c6021369e",
    },
    {
        "name": "chr22.fa.gz", "url": f"{UCSC}/hg38/chromosomes/chr22.fa.gz",
        "md5": "41b47ce1cc21b558409c19b892e1c0d1",
    },
)


def _json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=60) as response:
        return json.load(response)


def _hashes(path: Path) -> tuple[str, str]:
    md5, sha = hashlib.md5(), hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            md5.update(chunk)
            sha.update(chunk)
    return md5.hexdigest(), sha.hexdigest()


def _download(url: str, final: Path, expected_md5: str, expected_bytes: int | None) -> str:
    final.parent.mkdir(parents=True, exist_ok=True)
    partial = final.with_name(final.name + ".part")
    if final.exists():
        if expected_bytes is not None and final.stat().st_size != expected_bytes:
            raise FileExistsError(f"Existing file has an unexpected size: {final}")
        md5, sha = _hashes(final)
        if md5 != expected_md5:
            raise FileExistsError(f"Existing file has an unexpected checksum: {final}")
        return sha
    with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=60) as response:
        length = int(response.headers["Content-Length"])
    if expected_bytes is not None and length != expected_bytes:
        raise ValueError(f"Source size changed for {url}")
    expected_bytes = length
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > length:
        raise ValueError(f"Partial file exceeds the expected source size: {partial}")
    if shutil.disk_usage(final.parent).free - (length - offset) < 200 * 1024**3:
        raise OSError("Completing this download would violate the 200 GiB reserve")
    if offset < length:
        headers = {"Range": f"bytes={offset}-"} if offset else {}
        request = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(request, timeout=120) as response:
            if response.status != (206 if offset else 200):
                raise RuntimeError(f"Source did not honor the requested offset for {url}")
            if offset and not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                raise RuntimeError(f"Unexpected resumed range for {url}")
            with partial.open("ab") as stream:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    offset += len(chunk)
                    if offset > length:
                        raise RuntimeError(f"Source returned too many bytes for {url}")
                    if shutil.disk_usage(final.parent).free < 200 * 1024**3:
                        raise OSError("Stopped because the 200 GiB reserve was reached")
                    stream.write(chunk)
                stream.flush()
                os.fsync(stream.fileno())
    if offset != length:
        raise RuntimeError(f"Download incomplete: {offset}/{length}: {url}")
    md5, sha = _hashes(partial)
    if md5 != expected_md5:
        raise ValueError(f"Official MD5 mismatch: {url}")
    os.link(partial, final)
    partial.unlink()
    return sha


def _record(path: Path, record: dict) -> None:
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != record:
            raise FileExistsError(f"Existing provenance differs: {path}")
        return
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", prefix=".prior-metadata-", dir=path.parent, delete=False,
    ) as stream:
        temp = Path(stream.name)
        json.dump(record, stream, indent=2, allow_nan=False)
        stream.write("\n")
    try:
        os.link(temp, path)
    finally:
        temp.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--preview-reference", action="store_true")
    args = parser.parse_args()
    for entry in (() if args.preview_reference else ENTRIES):
        experiment = _json(f"{ENCODE}/experiments/{entry['experiment']}/?format=json&frame=object")
        file = _json(f"{ENCODE}/files/{entry['id']}/?format=json&frame=object")
        if (
            experiment.get("status") != "released"
            or experiment.get("assay_title") != entry["assay"]
            or experiment.get("biosample_summary") != "Homo sapiens HCT116"
            or experiment.get("perturbed") is True
            or file.get("status") != "released"
            or file.get("dataset") != f"/experiments/{entry['experiment']}/"
            or file.get("assembly") != entry["assembly"]
            or file.get("file_format") != entry["format"]
            or file.get("file_format_type") != entry["format_type"]
            or file.get("output_type") != entry["output_type"]
            or file.get("biological_replicates") != [entry["replicate"]]
            or file.get("file_size") != entry["bytes"] or file.get("md5sum") != entry["md5"]
            or (entry.get("target") and experiment.get("target") != f"/targets/{entry['target']}-human/")
        ):
            raise ValueError(f"Unexpected ENCODE source metadata: {entry['id']}")
        href = file["href"]
        url = ENCODE + href
        path = args.outdir / ("rrbs-hg19" if entry["assay"] == "RRBS" else "rad21-chia-pet") / Path(href).name
        sha = _download(url, path, entry["md5"], entry["bytes"])
        record = {
            "source": "ENCODE", "source_url": url, "experiment": entry["experiment"],
            "accession": entry["id"], "assay": entry["assay"], "target": entry.get("target"),
            "biosample": "HCT116", "biosample_summary": experiment["biosample_summary"],
            "genome_assembly": entry["assembly"], "biological_replicates": [entry["replicate"]],
            "file_format": entry["format"], "file_format_type": entry["format_type"],
            "output_type": entry["output_type"], "source_md5": entry["md5"],
            "bytes": entry["bytes"], "sha256": sha,
            "replication_type": experiment.get("replication_type"),
            "note": (
                "ENCODE reports no treatment in this experiment summary. "
                "Different experiments are not paired measurements; consult ENCODE audit warnings."
            ),
        }
        _record(path.with_name(path.name + ".json"), record)
        print(f"Verified {entry['id']} -> {path}", flush=True)
    for entry in (PREVIEW_REFERENCES if args.preview_reference else REFERENCES):
        path = args.outdir / "ucsc" / entry["name"]
        sha = _download(entry["url"], path, entry["md5"], entry.get("bytes"))
        _record(path.with_name(path.name + ".json"), {
            "source": "UCSC", "source_url": entry["url"], "source_md5": entry["md5"],
            "bytes": path.stat().st_size, "sha256": sha,
        })
        print(f"Verified {entry['name']} -> {path}", flush=True)


if __name__ == "__main__":
    main()
