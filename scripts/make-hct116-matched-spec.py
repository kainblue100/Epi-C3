"""Prepare a matched HCT116 Hi-C and sequence-validated WGBS network specification."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from c3vcell.epigenetics import CONDITION, COORDINATES, _loops, _source_records
from c3vcell.integrity import file_sha256, verified_source_record
from c3vcell.methylation import FastaReference, prepare_uncovered_cpg_fractions

PREVIEW_LENGTHS = {"chr21": 46_709_983, "chr22": 50_818_468}
REFERENCE_MD5 = {
    "chr21.fa.gz": "184df2bd9b812b6e6b6da16c6021369e",
    "chr22.fa.gz": "41b47ce1cc21b558409c19b892e1c0d1",
    "hg38.2bit": "dcc3ea27079aa6dc3f9deccd7275e0f8",
}
LOOP_MD5 = "d6cb89c8f7515c84e4ef2ba37b4bcd63"
WGBS_SHA256 = "965835cfe0ce6cf70aed1c6fdecb9e2f41154a8d29c2719de3f95c2a4a659ae7"
LOOP_URL = (
    "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE237nnn/GSE237721/suppl/"
    "GSE237721_ENCFF308MMM_loops_GRCh38.bedpe.gz"
)
WGBS_URL = (
    "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM3317nnn/GSM3317488/suppl/"
    "GSM3317488_HCT116_WGBS_CpG.bedGraph.gz"
)


def _validate_records(loop: dict, wgbs: dict) -> None:
    if (
        loop.get("source_url") != LOOP_URL or loop.get("accession") != "ENCFF308MMM"
        or loop.get("experiment") != "ENCSR477GZK" or loop.get("geo_series") != "GSE237721"
        or loop.get("assay") != "intact Hi-C"
        or loop.get("biosample_summary") != "Homo sapiens HCT116"
        or loop.get("perturbed") is not False or loop.get("genome_assembly") != "GRCh38"
        or loop.get("biological_replicates") != [1, 2] or loop.get("output_type") != "loops"
    ):
        raise ValueError("The Hi-C record is not the expected HCT116 called-loop source")
    if (
        wgbs.get("source_url") != WGBS_URL or wgbs.get("accession") != "GSM3317488"
        or wgbs.get("series") != "GSE118030" or wgbs.get("assay") != "WGBS"
        or wgbs.get("sample") != "HCT116_WT_WGBS" or wgbs.get("treatment") != "No treatment"
        or wgbs.get("genome_assembly") != "hg38" or wgbs.get("gzip_integrity_checked") is not True
    ):
        raise ValueError("The WGBS record is not the expected untreated HCT116 source")


def _loop_inventory(path: Path) -> dict:
    widths: Counter[tuple[int, int]] = Counter()
    chromosomes = set()
    for first, second in _loops(path):
        widths[(first[2] - first[1], second[2] - second[1])] += 1
        chromosomes.update((first[0], second[0]))
    expected = {(1000, 1000): 17331, (2000, 2000): 8999, (5000, 5000): 11477,
                (10000, 10000): 6412}
    if dict(widths) != expected:
        raise ValueError("The pinned Hi-C loop inventory has changed")
    return {
        "loops": sum(widths.values()),
        "anchor_widths_bp": sorted({width for pair in widths for width in pair}),
        "anchor_width_pairs": [
            {"widths_bp": list(pair), "count": count} for pair, count in sorted(widths.items())
        ],
        "covered_chromosomes": sorted(chromosomes),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-spec", type=Path, required=True)
    parser.add_argument("--matched-dir", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--preview-chromosomes", nargs="+", choices=sorted(PREVIEW_LENGTHS))
    args = parser.parse_args()
    if args.outdir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.outdir}")
    if args.preview_chromosomes and len(set(args.preview_chromosomes)) != len(args.preview_chromosomes):
        raise ValueError("Preview chromosomes cannot be repeated")
    base_digest = file_sha256(args.base_spec)
    base = json.loads(args.base_spec.read_text(encoding="utf-8"))
    if (
        base.get("format_version") != 1 or base.get("context") != "HCT116"
        or base.get("assembly") != "GRCh38" or base.get("condition") != CONDITION
    ):
        raise ValueError("Expected the original HCT116 GRCh38 base specification")
    base_paths, _ = _source_records(args.base_spec, base)
    base_sources = [
        {**entry, "path": str(base_paths[entry["role"]].resolve())} for entry in base["sources"]
    ]
    loop_path = args.matched_dir / "encode-intact-hic" / "ENCFF308MMM.bedpe.gz"
    wgbs_path = args.matched_dir / "geo-GSM3317488" / "GSM3317488_HCT116_WGBS_CpG.bedGraph.gz"
    loop = verified_source_record(loop_path, source="ENCODE via NCBI GEO", source_md5=LOOP_MD5)
    wgbs = verified_source_record(
        wgbs_path, source="NCBI GEO", source_md5=None, expected_sha256=WGBS_SHA256,
    )
    _validate_records(loop, wgbs)
    inventory = _loop_inventory(loop_path)
    references = (
        [args.reference_dir / f"{chrom}.fa.gz" for chrom in args.preview_chromosomes]
        if args.preview_chromosomes else [args.reference_dir / "hg38.2bit"]
    )
    records = [
        verified_source_record(path, source="UCSC", source_md5=REFERENCE_MD5[path.name])
        for path in references
    ]
    if args.preview_chromosomes:
        reference = FastaReference(references, expected_lengths=PREVIEW_LENGTHS)
        chromosomes = set(args.preview_chromosomes)
    else:
        import py2bit

        reference = py2bit.open(str(references[0]))
        if reference is None:
            raise ValueError("Could not open the complete GRCh38 2bit reference")
        chromosomes = {f"chr{number}" for number in range(1, 23)} | {"chrX", "chrY"}
        if not chromosomes <= set(reference.chroms()):
            reference.close()
            raise ValueError("The GRCh38 reference lacks a primary chromosome")
    args.outdir.mkdir(parents=True)
    methyl_path = args.outdir / "methylation-reference-cpg-fractions.bed.gz"
    try:
        methylation = prepare_uncovered_cpg_fractions(
            wgbs_path, reference, methyl_path, chromosomes=chromosomes,
        )
    finally:
        if not args.preview_chromosomes:
            reference.close()
    for path, record in zip([loop_path, wgbs_path, *references], [loop, wgbs, *records]):
        if file_sha256(path) != record["sha256"]:
            raise RuntimeError(f"Source changed while preparing matched inputs: {path}")
    if file_sha256(args.base_spec) != base_digest:
        raise RuntimeError("Base specification changed while preparing matched inputs")
    _source_records(args.base_spec, base)
    common = {
        "assembly": "GRCh38", "context": "HCT116", "condition": CONDITION,
        "coordinates": COORDINATES,
    }
    spec = {
        **base, "format_version": 2, "base_specification_sha256": base_digest,
        "sources": [
            *base_sources,
            {
                **common, "role": "chromatin_contacts", "path": str(loop_path.resolve()),
                "sha256": loop["sha256"], "source_url": LOOP_URL, "assay": "Hi-C",
                "source_assay": "intact Hi-C", "target": None, "evidence_type": "called-loop",
                "resolution_bp": None, "resolution_kind": "multi-resolution-anchor-interval",
                "anchor_widths_bp": inventory["anchor_widths_bp"],
                "covered_chromosomes": inventory["covered_chromosomes"],
                "inventory": inventory, "input_files": [loop],
                "note": (
                    "Pooled biological replicates are not independent samples. Called loops "
                    "are candidate contacts, not proof of regulatory causation."
                ),
            },
            {
                **common, "role": "methylation", "path": str(methyl_path.resolve()),
                "sha256": methylation["sha256"], "source_url": WGBS_URL, "assay": "WGBS",
                "file_format": "cpg-fraction-bed-no-coverage", "min_coverage": None,
                "site_coverage": "not-available", "aggregation": "equal-strand-then-cpg-mean",
                "reference_sequence_validated": True,
                "covered_chromosomes": methylation["covered_chromosomes"],
                "processing": methylation, "input_files": [wgbs], "reference_files": records,
                "note": (
                    "The original percentage bedGraph lacks read coverage. Available strand "
                    "fractions are averaged equally per reference CpG dyad; positions that do "
                    "not match a reference CpG are excluded. GEO did not publish an independent "
                    "source checksum. Hi-C and WGBS are bulk, separately collected samples."
                ),
            },
        ],
    }
    output = args.outdir / "sources.json"
    with output.open("x", encoding="utf-8") as stream:
        json.dump(spec, stream, indent=2, allow_nan=False)
        stream.write("\n")
    _source_records(output, spec)
    print(json.dumps({"spec": str(output), "contacts": inventory, "methylation": methylation}, indent=2))


if __name__ == "__main__":
    main()
