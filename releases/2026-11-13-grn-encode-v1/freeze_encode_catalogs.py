"""Freeze compact public ENCODE K562 ChIP-seq metadata for offline selection."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import urllib.parse
import urllib.request


HOST = "https://www.encodeproject.org"
RESERVE_WITH_HEADROOM = 201 * 1024**3
MAX_METADATA_BYTES = 64 * 1024**2
FILE_FIELDS = (
    "accession", "dataset", "target.label", "assembly", "status", "output_type",
    "file_format", "file_format_type", "file_size", "md5sum", "href",
    "biological_replicates", "lab.title", "biosample_ontology.term_name", "assay_title",
)
EXPERIMENT_FIELDS = (
    "accession", "status", "assay_title", "biosample_ontology.term_name",
    "biosample_summary", "perturbed", "lab.title", "target.label",
    "target.genes.symbol", "target.genes.dbxrefs",
    "replicates.biological_replicate_number",
    "replicates.library.biosample.applied_modifications",
    "replicates.library.biosample.treatments", "replicates.library.treatments",
)


def catalog_url(kind: str) -> str:
    if kind == "files":
        params = [
            ("type", "File"), ("assay_title", "TF ChIP-seq"),
            ("biosample_ontology.term_name", "K562"), ("assembly", "GRCh38"),
            ("status", "released"), ("file_format", "bed"),
            ("output_type", "optimal IDR thresholded peaks"),
            ("output_type", "conservative IDR thresholded peaks"),
        ]
        fields = FILE_FIELDS
    elif kind == "experiments":
        params = [
            ("type", "Experiment"), ("assay_title", "TF ChIP-seq"),
            ("biosample_ontology.term_name", "K562"), ("status", "released"),
            ("biosample_summary", "Homo sapiens K562"), ("perturbed", "false"),
        ]
        fields = EXPERIMENT_FIELDS
    else:
        raise ValueError(f"Unknown catalog: {kind}")
    params.extend([("format", "json"), ("limit", "all")])
    params.extend(("field", field) for field in fields)
    return HOST + "/search/?" + urllib.parse.urlencode(params)


def validate_catalog(value: object, kind: str) -> int:
    if not isinstance(value, dict):
        raise ValueError(f"Invalid {kind} search response")
    graph, total = value.get("@graph"), value.get("total")
    if not isinstance(graph, list) or type(total) is not int or total < 1 or len(graph) != total:
        raise ValueError(f"Incomplete or empty {kind} search response")
    accessions = [item.get("accession") if isinstance(item, dict) else None for item in graph]
    if any(not isinstance(item, str) or not item for item in accessions) or len(set(accessions)) != total:
        raise ValueError(f"Missing or duplicate {kind} accessions")
    return total


def fetch_catalog(url: str, kind: str) -> tuple[bytes, int]:
    request = urllib.request.Request(
        url, headers={"Accept": "application/json", "User-Agent": "Epi-C3-research/1.0"}
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        endpoint = urllib.parse.urlparse(response.geturl())
        if endpoint.scheme != "https" or endpoint.hostname != "www.encodeproject.org":
            raise ValueError("ENCODE metadata redirected outside the official HTTPS portal")
        data = response.read(MAX_METADATA_BYTES + 1)
    if len(data) > MAX_METADATA_BYTES:
        raise ValueError("ENCODE metadata exceeds the configured maximum")
    return data, validate_catalog(json.loads(data), kind)


def write_new(path: Path, data: bytes) -> None:
    if shutil.disk_usage(path.parent).free < RESERVE_WITH_HEADROOM:
        raise OSError("Insufficient disk space above the 200 GiB reserve and 1 GiB headroom")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True, help="New private output directory")
    args = parser.parse_args()
    if shutil.disk_usage(args.output_dir.parent).free < RESERVE_WITH_HEADROOM:
        raise OSError("Insufficient disk space above the 200 GiB reserve and 1 GiB headroom")
    args.output_dir.mkdir(mode=0o700)
    records = {}
    for kind in ("files", "experiments"):
        url = catalog_url(kind)
        data, count = fetch_catalog(url, kind)
        name = f"{kind}.json"
        write_new(args.output_dir / name, data)
        records[kind] = {"url": url, "sha256": hashlib.sha256(data).hexdigest(), "records": count}
        time.sleep(0.25)
    manifest = {
        "format": "epi-c3-encode-metadata-snapshot-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "catalogs": records,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    data = (json.dumps(manifest, indent=2, allow_nan=False) + "\n").encode("utf-8")
    write_new(args.output_dir / "snapshot.json", data)
    print(json.dumps({"state": "complete", "snapshot_sha256": hashlib.sha256(data).hexdigest(),
                      "counts": {kind: item["records"] for kind, item in records.items()}}), flush=True)


if __name__ == "__main__":
    main()
