"""Build a provenance-preserving HCT116 3D/ATAC/CpG network specification."""

from __future__ import annotations

import argparse
import gzip
import io
import json
import sqlite3
import tempfile
from collections import Counter
from pathlib import Path

from pyliftover import LiftOver

from c3vcell.contacts import overlap_supported_contacts
from c3vcell.integrity import file_sha256, verified_source_record
from c3vcell.methylation import FastaReference, lifted_cpgs

CHROMOSOME_LENGTHS = {"chr21": 46_709_983, "chr22": 50_818_468}
PINNED_MD5 = {
    "ENCFF001TMM.bed.gz": "c368ce628590962021fa9b04b8681248",
    "ENCFF001TMN.bed.gz": "a08913af52cca626aa4ffa344a7bf711",
    "ENCFF625LBL.bedpe.gz": "28c59dd8d79703db0be952a6ba47d972",
    "ENCFF769MZQ.bedpe.gz": "ef42546db65683f7e33de918bc764a75",
    "hg19ToHg38.over.chain.gz": "35887f73fe5e2231656504d1f6430900",
    "hg38.2bit": "dcc3ea27079aa6dc3f9deccd7275e0f8",
    "chr21.fa.gz": "184df2bd9b812b6e6b6da16c6021369e",
    "chr22.fa.gz": "41b47ce1cc21b558409c19b892e1c0d1",
}


def _verified(path: Path, source: str) -> dict:
    expected_md5 = PINNED_MD5.get(path.name)
    if expected_md5 is None:
        raise ValueError(f"No pinned source checksum: {path}")
    return verified_source_record(path, source=source, source_md5=expected_md5)


def _pool_methylation(
    inputs: list[Path], records: list[dict], converter, reference, output: Path, *,
    source_chromosomes: set[str] | None, min_coverage: int,
) -> dict:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    if len(inputs) != len(records):
        raise ValueError("Methylation inputs and provenance records differ")
    with tempfile.NamedTemporaryFile(prefix=".cpg-pool-", suffix=".sqlite", dir=output.parent, delete=False) as file:
        temporary = Path(file.name)
    stats = []
    try:
        with sqlite3.connect(temporary) as connection:
            connection.execute(
                "CREATE TABLE cpg (chrom TEXT, start INTEGER, weighted REAL, coverage INTEGER, "
                "PRIMARY KEY(chrom, start)) WITHOUT ROWID"
            )
            for path, record in zip(inputs, records):
                counts: Counter[str] = Counter()
                batch = []
                for chrom, start, fraction, coverage in lifted_cpgs(
                    path, converter, reference, min_coverage=min_coverage, stats=counts,
                    source_chromosomes=source_chromosomes,
                ):
                    batch.append((chrom, start, fraction * coverage, coverage))
                    if len(batch) >= 10_000:
                        connection.executemany(
                            "INSERT INTO cpg VALUES (?,?,?,?) ON CONFLICT(chrom,start) "
                            "DO UPDATE SET weighted=weighted+excluded.weighted, "
                            "coverage=coverage+excluded.coverage",
                            batch,
                        )
                        batch.clear()
                if batch:
                    connection.executemany(
                        "INSERT INTO cpg VALUES (?,?,?,?) ON CONFLICT(chrom,start) "
                        "DO UPDATE SET weighted=weighted+excluded.weighted, "
                        "coverage=coverage+excluded.coverage",
                        batch,
                    )
                connection.commit()
                if file_sha256(path) != record["sha256"]:
                    raise RuntimeError(f"Methylation source changed while processing: {path}")
                stats.append({"accession": record["accession"], **counts})
                print(f"Processed {record['accession']}: {dict(counts)}", flush=True)
            sites = connection.execute("SELECT COUNT(*) FROM cpg").fetchone()[0]
            if not sites:
                raise ValueError("No sequence-validated CpG observations remain")
            with output.open("xb") as raw:
                with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
                    with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as stream:
                        stream.write("#chrom\tstart\tend\tmethyl_fraction\tcoverage\n")
                        for chrom, start, weighted, coverage in connection.execute(
                            "SELECT chrom,start,weighted,coverage FROM cpg ORDER BY chrom,start"
                        ):
                            stream.write(
                                f"{chrom}\t{start}\t{start + 1}\t"
                                f"{weighted / coverage:.8g}\t{coverage}\n"
                            )
    finally:
        temporary.unlink(missing_ok=True)
    return {"unique_cpgs": sites, "source_statistics": stats, "sha256": file_sha256(output)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-spec", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--preview-chromosomes", nargs="+", choices=sorted(CHROMOSOME_LENGTHS))
    parser.add_argument("--min-coverage", type=int, default=5)
    args = parser.parse_args()
    if args.min_coverage < 1:
        raise ValueError("Minimum CpG coverage must be positive")
    if args.preview_chromosomes and len(set(args.preview_chromosomes)) != len(args.preview_chromosomes):
        raise ValueError("Preview chromosomes cannot be repeated")
    if args.outdir.exists() and any(args.outdir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty directory: {args.outdir}")
    base_digest = file_sha256(args.base_spec)
    base = json.loads(args.base_spec.read_text(encoding="utf-8"))
    if (
        base.get("format_version") != 1 or base.get("context") != "HCT116"
        or base.get("assembly") != "GRCh38" or base.get("condition") != "no-reported-treatment"
    ):
        raise ValueError("Unexpected base HCT116 reference specification")
    raw = args.raw_dir
    methyl_paths = [raw / "rrbs-hg19" / f"{name}.bed.gz" for name in ("ENCFF001TMM", "ENCFF001TMN")]
    contact_paths = [
        raw / "rad21-chia-pet" / f"{name}.bedpe.gz" for name in ("ENCFF625LBL", "ENCFF769MZQ")
    ]
    methyl_records = [_verified(path, "ENCODE") for path in methyl_paths]
    contact_records = [_verified(path, "ENCODE") for path in contact_paths]
    if any(
        record.get("biosample") != "HCT116" or record.get("assay") != "RRBS"
        or record.get("genome_assembly") != "hg19" or record.get("experiment") != "ENCSR000DFS"
        or record.get("accession") != accession or record.get("biological_replicates") != [replicate]
        or record.get("file_format_type") != "bedMethyl"
        or record.get("output_type") != "methylation state at CpG"
        for record, accession, replicate in zip(
            methyl_records, ("ENCFF001TMM", "ENCFF001TMN"), (1, 2)
        )
    ) or any(
        record.get("biosample") != "HCT116" or record.get("assay") != "ChIA-PET"
        or record.get("target") != "RAD21" or record.get("genome_assembly") != "GRCh38"
        or record.get("accession") != accession or record.get("experiment") != experiment
        or record.get("output_type") != "loops" or record.get("replication_type") != "unreplicated"
        for record, accession, experiment in zip(
            contact_records, ("ENCFF625LBL", "ENCFF769MZQ"), ("ENCSR028AOM", "ENCSR309KYA")
        )
    ):
        raise ValueError("Epigenomic sources are not the expected HCT116 assays")
    chain = raw / "ucsc" / "hg19ToHg38.over.chain.gz"
    chain_record = _verified(chain, "UCSC")
    sequence_paths = (
        [raw / "ucsc" / f"{chrom}.fa.gz" for chrom in args.preview_chromosomes]
        if args.preview_chromosomes else [raw / "ucsc" / "hg38.2bit"]
    )
    sequence_records = [_verified(path, "UCSC") for path in sequence_paths]
    if args.preview_chromosomes:
        reference = FastaReference(sequence_paths, expected_lengths=CHROMOSOME_LENGTHS)
    else:
        import py2bit

        reference = py2bit.open(str(sequence_paths[0]))
    if reference is None:
        raise ValueError("Could not open GRCh38 reference sequence")
    converter = LiftOver(str(chain))
    args.outdir.mkdir(parents=True, exist_ok=True)
    contact_path = args.outdir / "rad21-overlap-supported.bedpe"
    contacts = overlap_supported_contacts(contact_paths[0], contact_paths[1], contact_path)
    print(f"Prepared {contacts['supported_contacts']} overlap-supported 3D contacts", flush=True)
    methyl_path = args.outdir / "methylation-cpg-fractions.bed.gz"
    try:
        methylation = _pool_methylation(
            methyl_paths, methyl_records, converter, reference, methyl_path, min_coverage=args.min_coverage,
            source_chromosomes=set(args.preview_chromosomes) if args.preview_chromosomes else None,
        )
    finally:
        if not args.preview_chromosomes:
            reference.close()
    for path, record in zip([chain, *sequence_paths], [chain_record, *sequence_records]):
        if file_sha256(path) != record["sha256"]:
            raise RuntimeError(f"Reference changed during processing: {path}")
    if file_sha256(args.base_spec) != base_digest:
        raise RuntimeError("Base specification changed while constructing the network")
    common = {
        "assembly": "GRCh38", "context": "HCT116", "condition": "no-reported-treatment",
        "coordinates": "zero-based-half-open",
    }
    spec = {
        **base, "format_version": 2, "base_specification_sha256": base_digest,
        "sources": [
            *base["sources"],
            {
                **common, "role": "chromatin_contacts", "path": str(contact_path.resolve()),
                "sha256": contacts["sha256"],
                "source_url": "https://www.encodeproject.org/experiments/ENCSR028AOM/",
                "assay": "ChIA-PET", "target": "RAD21", "evidence_type": "called-loop",
                "resolution_bp": None, "resolution_kind": "anchor-interval",
                "covered_chromosomes": contacts["covered_chromosomes"],
                "overlap_support": contacts, "input_files": contact_records,
                "note": (
                    "RAD21-enriched contacts, not Hi-C; each ENCODE experiment is designated "
                    "unreplicated. Overlap is not proof of independent biological replication."
                ),
            },
            {
                **common, "role": "methylation", "path": str(methyl_path.resolve()),
                "sha256": methylation["sha256"],
                "source_url": "https://www.encodeproject.org/experiments/ENCSR000DFS/",
                "assay": "RRBS", "file_format": "cpg-fraction-bed",
                "min_coverage": args.min_coverage, "original_assembly": "hg19",
                "input_files": methyl_records, "chain": chain_record,
                "reference_sequence": sequence_records, "conversion_statistics": methylation,
                "source_chromosomes": args.preview_chromosomes if args.preview_chromosomes else "all",
                "note": (
                    "Only unique, contiguous hg19-to-GRCh38 CpGs matching the target reference "
                    "were retained. Fractions use the coverage-weighted, rounded percentages in "
                    "the source bedMethyl; RRBS is sparse. The two nominal replicates are "
                    "associated with the same biosample in the ENCODE audit and should not be "
                    "treated as independent validation. These are bulk, unpaired measurements."
                ),
            },
        ],
        "note": (
            "HCT116 is shared across bulk assays, but laboratories, cultures and dates differ; "
            "the measurements are not paired and contacts are not causal."
        ),
    }
    path = args.outdir / "sources.json"
    with path.open("x", encoding="utf-8") as stream:
        json.dump(spec, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"Wrote {path} with {methylation['unique_cpgs']} covered CpGs", flush=True)


if __name__ == "__main__":
    main()
