from collections import Counter
import gzip
from pathlib import Path

import pytest

from c3vcell.integrity import file_sha256
from c3vcell.methylation import (
    FastaReference, lifted_cpgs, prepare_uncovered_cpg_fractions, validated_percent_cpgs,
)


class Converter:
    def convert_coordinate(self, chromosome: str, position: int, strand: str):
        if chromosome == "chrMissing":
            return None
        if chromosome == "chrAmbiguous":
            return [("chr1", position, strand, 1), ("chr2", position, strand, 1)]
        if chromosome == "chrReverse":
            return [("chr1", 50 - position, "-" if strand == "+" else "+", 1)]
        return [("chr1", position, strand, 1)]


class Reference:
    def chroms(self):
        return {"chr1": 100}

    def sequence(self, chromosome: str, start: int, end: int):
        assert chromosome == "chr1" and end == start + 2
        return "CG" if start in (10, 39) else "CA"


def test_liftover_validates_both_bases_strand_and_reference_cpg(tmp_path: Path) -> None:
    path = tmp_path / "source.bed"
    path.write_text(
        "track name=test\n"
        "chr1\t10\t11\t.\t0\t+\t10\t11\t0\t10\t25\n"
        "chr1\t11\t12\t.\t0\t-\t11\t12\t0\t20\t75\n"
        "chrReverse\t10\t11\t.\t0\t+\t10\t11\t0\t5\t50\n"
        "chr1\t20\t21\t.\t0\t+\t20\t21\t0\t10\t100\n"
        "chrMissing\t10\t11\t.\t0\t+\t10\t11\t0\t10\t0\n"
        "chrAmbiguous\t10\t11\t.\t0\t+\t10\t11\t0\t10\t0\n"
        "chr1\t10\t11\t.\t0\t+\t10\t11\t0\t2\t0\n",
        encoding="utf-8",
    )
    stats = Counter()
    result = list(lifted_cpgs(path, Converter(), Reference(), min_coverage=5, stats=stats))
    assert result == [("chr1", 10, 0.25, 10), ("chr1", 10, 0.75, 20), ("chr1", 39, 0.5, 5)]
    assert stats == {
        "input_sites": 7, "retained_sites": 3, "not_reference_cpg": 1,
        "unmapped_or_ambiguous": 2, "low_coverage": 1,
    }


def test_preview_only_processes_requested_source_chromosomes(tmp_path: Path) -> None:
    path = tmp_path / "source.bed"
    path.write_text(
        "chr1\t10\t11\t.\t0\t+\t10\t11\t0\t10\t25\n"
        "chrOther\t10\t11\t.\t0\t+\t10\t11\t0\t10\t75\n",
        encoding="utf-8",
    )
    stats = Counter()
    result = list(lifted_cpgs(
        path, Converter(), Reference(), min_coverage=5, stats=stats,
        source_chromosomes={"chr1"},
    ))
    assert result == [("chr1", 10, 0.25, 10)]
    assert stats == {
        "input_sites": 2, "retained_sites": 1, "outside_requested_source_chromosomes": 1,
    }


def test_real_pyliftover_keeps_both_strands_of_an_inverted_cpg(tmp_path: Path) -> None:
    pyliftover = pytest.importorskip("pyliftover")
    chain = tmp_path / "inverted.chain"
    chain.write_text(
        "chain 1000 chrSource 100 + 10 20 chr1 100 - 30 40 1\n10\n",
        encoding="utf-8",
    )
    path = tmp_path / "source.bed"
    path.write_text(
        "chrSource\t10\t11\t.\t0\t+\t10\t11\t0\t10\t25\n"
        "chrSource\t11\t12\t.\t0\t-\t11\t12\t0\t20\t75\n",
        encoding="utf-8",
    )

    class InvertedReference(Reference):
        def sequence(self, chromosome: str, start: int, end: int):
            return "CG" if (chromosome, start, end) == ("chr1", 68, 70) else "CA"

    stats = Counter()
    result = list(lifted_cpgs(
        path, pyliftover.LiftOver(str(chain)), InvertedReference(), min_coverage=5, stats=stats,
    ))
    assert result == [("chr1", 68, 0.25, 10), ("chr1", 68, 0.75, 20)]
    assert stats == {"input_sites": 2, "retained_sites": 2}


@pytest.mark.parametrize("line", [
    "chr1\t10\t11\t.\t0\t+\t10\t11\t0\t3.5\t50\n",
    "chr1\t10\t12\t.\t0\t+\t10\t12\t0\t10\t50\n",
    "chr1\t10\t11\t.\t0\t.\t10\t11\t0\t10\t50\n",
    "chr1\t10\t11\t.\t0\t+\t10\t11\t0\t10\tnan\n",
    "chr1\t10\t11\t.\t0\t+\t10\t11\t0\t10\t101\n",
])
def test_liftover_rejects_malformed_cpg_values(tmp_path: Path, line: str) -> None:
    path = tmp_path / "bad.bed"
    path.write_text(line, encoding="utf-8")
    with pytest.raises(ValueError, match="bedMethyl|invalid CpG"):
        list(lifted_cpgs(path, Converter(), Reference(), min_coverage=5, stats=Counter()))


class StrandReference:
    def __init__(self) -> None:
        sequence = list("A" * 100)
        for start in (10, 20, 30):
            sequence[start:start + 2] = "CG"
        self.bases = "".join(sequence)

    def chroms(self):
        return {"chr1": len(self.bases)}

    def sequence(self, chromosome: str, start: int, end: int):
        assert chromosome == "chr1"
        return self.bases[start:end]


def test_uncovered_wgbs_averages_strands_per_verified_dyad(tmp_path: Path) -> None:
    source = tmp_path / "raw.bedGraph"
    source.write_text(
        "track type=bedGraph\n"
        "chr1\t10\t11\t25\n"
        "chr1\t11\t12\t75\n"
        "chr1\t20\t21\t0\n"
        "chr1\t30\t31\t50\n"
        "chr1\t31\t32\t0\n"
        "chr1\t40\t41\t100\n"
        "chr2\t10\t11\t100\n",
        encoding="utf-8",
    )
    output = tmp_path / "fractions.bed.gz"
    result = prepare_uncovered_cpg_fractions(
        source, StrandReference(), output, chromosomes={"chr1"},
    )
    with gzip.open(output, "rt", encoding="utf-8") as stream:
        assert stream.read() == (
            "#chrom\tstart\tend\tmethyl_fraction\n"
            "chr1\t10\t11\t0.5\n"
            "chr1\t20\t21\t0\n"
            "chr1\t30\t31\t0.25\n"
        )
    assert result == {
        "unique_cpg_dyads": 3,
        "single_strand_dyads": 1,
        "both_strands_dyads": 2,
        "covered_chromosomes": ["chr1"],
        "source_statistics": {
            "input_rows": 7, "outside_requested_chromosomes": 1, "forward_rows": 3,
            "reverse_rows": 2, "retained_rows": 5, "not_reference_cpg": 1,
        },
        "sha256": file_sha256(output),
    }
    with pytest.raises(FileExistsError):
        prepare_uncovered_cpg_fractions(source, StrandReference(), output, chromosomes={"chr1"})


@pytest.mark.parametrize("line", [
    "chr1\t10\t12\t50\n",
    "chr1\t100\t101\t50\n",
    "chr1\t10\t11\tnan\n",
    "chr1\t10\t11\t101\n",
    "chr1\t10\t11\t-1\n",
    "chr1\t10\t11\n",
    "chr1\t10\t11\t25\nchr1\t10\t11\t75\n",
    "chr1\t20\t21\t25\nchr1\t10\t11\t75\n",
])
def test_uncovered_wgbs_rejects_invalid_rows_without_publishing(
    tmp_path: Path, line: str,
) -> None:
    source = tmp_path / "bad.bedGraph"
    source.write_text(line, encoding="utf-8")
    output = tmp_path / "fractions.bed.gz"
    with pytest.raises(ValueError):
        prepare_uncovered_cpg_fractions(source, StrandReference(), output, chromosomes={"chr1"})
    assert not output.exists()
    assert sorted(path.name for path in tmp_path.iterdir()) == ["bad.bedGraph"]


def test_uncovered_wgbs_requires_reference_and_observed_cpg(tmp_path: Path) -> None:
    source = tmp_path / "source.bedGraph"
    source.write_text("chr1\t40\t41\t50\n", encoding="utf-8")
    with pytest.raises(ValueError, match="present in the reference"):
        list(validated_percent_cpgs(source, StrandReference(), stats=Counter(), chromosomes={"chr2"}))
    with pytest.raises(ValueError, match="No valid CpG"):
        prepare_uncovered_cpg_fractions(
            source, StrandReference(), tmp_path / "empty.gz", chromosomes={"chr1"},
        )


def test_fasta_reference_validates_name_length_and_sequence(tmp_path: Path) -> None:
    path = tmp_path / "chr1.fa.gz"
    with gzip.open(path, "wt", encoding="ascii") as stream:
        stream.write(">chr1\nACGTN\n")
    reference = FastaReference([path], expected_lengths={"chr1": 5})
    assert reference.chroms() == {"chr1": 5}
    assert reference.sequence("chr1", 1, 3) == "CG"
    with pytest.raises(ValueError, match="FASTA sequence"):
        FastaReference([path], expected_lengths={"chr1": 4})
