from pathlib import Path

import pytest

from c3vcell.contacts import overlap_supported_contacts


def test_overlap_supported_contacts_accepts_reversed_anchors_and_deduplicates(tmp_path: Path) -> None:
    first, second = tmp_path / "first.bedpe", tmp_path / "second.bedpe"
    first.write_text("chr1\t100\t110\tchr1\t200\t210\n", encoding="utf-8")
    second.write_text(
        "chr1\t202\t212\tchr1\t102\t112\n"
        "chr1\t102\t112\tchr1\t202\t212\n"
        "chr1\t105\t115\tchr1\t250\t260\n",
        encoding="utf-8",
    )
    output = tmp_path / "supported.bedpe"
    result = overlap_supported_contacts(first, second, output)
    assert output.read_text().splitlines()[1] == "chr1\t102\t110\tchr1\t202\t210"
    assert result["first_unique_contacts"] == 1
    assert result["second_unique_contacts"] == 2
    assert result["supported_contacts"] == 1
    assert result["covered_chromosomes"] == ["chr1"]


def test_contact_consensus_requires_reciprocal_overlap_and_nonempty_sources(tmp_path: Path) -> None:
    first, second = tmp_path / "first.bedpe", tmp_path / "second.bedpe"
    first.write_text("chr1\t100\t200\tchr1\t300\t400\n", encoding="utf-8")
    second.write_text("chr1\t150\t250\tchr1\t390\t490\n", encoding="utf-8")
    with pytest.raises(ValueError, match="No called loops"):
        overlap_supported_contacts(first, second, tmp_path / "none.bedpe")
    with pytest.raises(ValueError, match="overlap must"):
        overlap_supported_contacts(first, second, tmp_path / "bad.bedpe", reciprocal_overlap=0)
