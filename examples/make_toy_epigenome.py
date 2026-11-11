"""Create fictitious epigenomic inputs for the synthetic quick-start dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import anndata as ad


DESCRIPTION = "Artificial data for software demonstrations, not biology."
CONDITION = "no-reported-treatment"
COORDINATES = "zero-based-half-open"


def create_toy_epigenome(source: Path, outdir: Path) -> None:
    data = ad.read_h5ad(source, backed="r")
    try:
        if (
            data.uns.get("description") != DESCRIPTION
            or data.uns.get("c3vcell_genome_assembly") != "GRCh38"
            or set(data.obs["context"].astype(str)) != {"toy"}
            or "ensembl_ids" not in data.var
        ):
            raise ValueError("This example accepts only the generated synthetic toy dataset")
        genes = list(zip(data.var_names.astype(str), data.var["ensembl_ids"].astype(str)))
    finally:
        data.file.close()
    if len(genes) != 64 or not {"G4", "G10", "SP1", "USF1"} <= {gene for gene, _ in genes}:
        raise ValueError("The synthetic gene axis is incomplete")
    outdir.mkdir(parents=True, exist_ok=False)

    def write(name: str, text: str) -> Path:
        path = outdir / name
        with path.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
        return path

    annotation = write(
        "genes.gtf",
        "".join(
            f'chr1\ttoy\tgene\t{10_001 + i * 10_000}\t{10_101 + i * 10_000}\t.\t+\t.\t'
            f'gene_id "{identifier}.1"; gene_name "{gene}";\n'
            for i, (gene, identifier) in enumerate(genes)
        ),
    )
    atac = write("atac.bed", "chr1\t50000\t50020\nchr1\t110000\t110020\n")
    sp1 = write("sp1.bed", "chr1\t110005\t110015\n")
    usf1 = write("usf1.bed", "chr1\t50005\t50015\n")
    contacts = write("loops.bedpe", "chr1\t50000\t50020\tchr1\t110000\t110020\n")
    methylation = write(
        "methylation.bed",
        "#chrom\tstart\tend\tfraction\tcoverage\n"
        "chr1\t50005\t50006\t0.8\t12\nchr1\t110005\t110006\t0.25\t15\n",
    )

    def entry(role: str, path: Path, **metadata: object) -> dict:
        return {
            "role": role,
            "path": path.name,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "source_url": f"https://example.invalid/synthetic/{path.name}",
            "assembly": "GRCh38",
            **metadata,
        }

    common = {"context": "toy", "condition": CONDITION, "coordinates": COORDINATES}
    spec = {
        "format_version": 2,
        "context": "toy",
        "assembly": "GRCh38",
        "condition": CONDITION,
        "sources": [
            entry("annotation", annotation),
            entry("atac", atac, **common, assay="ATAC-seq", target=None, accession="TOY-ATAC"),
            entry("sp1", sp1, **common, assay="TF ChIP-seq", target="SP1", accession="TOY-SP1"),
            entry("usf1", usf1, **common, assay="TF ChIP-seq", target="USF1", accession="TOY-USF1"),
            entry(
                "chromatin_contacts", contacts, **common, assay="Hi-C", target=None,
                resolution_bp=20, covered_chromosomes=["chr1"], evidence_type="called-loop",
            ),
            entry(
                "methylation", methylation, **common, assay="RRBS",
                file_format="cpg-fraction-bed", min_coverage=5,
            ),
        ],
    }
    spec_path = write("sources.json", json.dumps(spec, indent=2) + "\n")
    print(f"Wrote artificial epigenomic sources: {spec_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()
    create_toy_epigenome(args.source, args.outdir)


if __name__ == "__main__":
    main()
