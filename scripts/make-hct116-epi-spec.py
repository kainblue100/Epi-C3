"""Verify previously downloaded ENCODE/GENCODE files and write a source specification."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

TRACKS = (
    ("atac", "ENCFF070LSQ", "ENCSR872WGW", "ATAC-seq", None),
    ("ctcf", "ENCFF850PKJ", "ENCSR000DTO", "TF ChIP-seq", "CTCF"),
    ("sp1", "ENCFF314MDR", "ENCSR000BSF", "TF ChIP-seq", "SP1"),
    ("usf1", "ENCFF028UPG", "ENCSR000BVK", "TF ChIP-seq", "USF1"),
)


def _check_file(path: Path, record: dict) -> str:
    if not path.is_file() or path.stat().st_size != record["bytes"]:
        raise ValueError(f"Missing file or byte-count mismatch: {path}")
    sha256, md5 = hashlib.sha256(), hashlib.md5()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            sha256.update(chunk)
            md5.update(chunk)
    if sha256.hexdigest() != record["sha256"] or md5.hexdigest() != record["source_md5"]:
        raise ValueError(f"Checksum mismatch: {path}")
    return sha256.hexdigest()


def make_spec(encode_dir: Path, annotation_dir: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    annotation = annotation_dir / "gencode.v44.annotation.gtf.gz"
    record = json.loads((annotation_dir / (annotation.name + ".json")).read_text(encoding="utf-8"))
    if (
        record.get("source") != "GENCODE" or record.get("release") != 44
        or record.get("genome_assembly") != "GRCh38"
    ):
        raise ValueError("Unexpected gene-annotation provenance")
    sources = [{
        "role": "annotation", "path": str(annotation.resolve()),
        "source_url": record["source_url"], "source_md5_url": record["source_md5_url"],
        "source_md5": record["source_md5"], "sha256": _check_file(annotation, record),
        "assembly": "GRCh38", "release": 44,
        "note": "The exact Orion 10x 2024-A annotation release has not been established.",
    }]
    for role, accession, experiment, assay, target in TRACKS:
        path = encode_dir / (accession + ".bed.gz")
        record = json.loads((encode_dir / (accession + ".json")).read_text(encoding="utf-8"))
        if (
            record.get("source") != "ENCODE"
            or record.get("accession") != accession or record.get("experiment") != experiment
            or record.get("assay") != assay or record.get("target") != target
            or record.get("biosample") != "HCT116"
            or record.get("biosample_summary") != "Homo sapiens HCT116"
            or record.get("genome_assembly") != "GRCh38"
            or record.get("output_type") != "conservative IDR thresholded peaks"
            or record.get("biological_replicates") != [1, 2]
        ):
            raise ValueError(f"Unexpected ENCODE provenance for {accession}")
        sources.append({
            "role": role, "path": str(path.resolve()), "source_url": record["source_url"],
            "source_md5": record["source_md5"], "sha256": _check_file(path, record),
            "assembly": "GRCh38", "context": "HCT116", "condition": "no-reported-treatment",
            "coordinates": "zero-based-half-open", "accession": accession, "experiment": experiment,
            "assay": assay, "target": target, "output_type": record["output_type"],
            "biological_replicates": record["biological_replicates"],
        })
    spec = {
        "format_version": 1, "assembly": "GRCh38", "context": "HCT116",
        "condition": "no-reported-treatment", "sources": sources,
        "note": (
            "ENCODE reports no treatment in these selected experiment summaries. "
            "Different laboratories and culture conditions may still differ."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(spec, stream, indent=2, allow_nan=False)
        stream.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encode-dir", type=Path, required=True)
    parser.add_argument("--annotation-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    make_spec(args.encode_dir, args.annotation_dir, args.output)


if __name__ == "__main__":
    main()
