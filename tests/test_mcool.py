import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from c3vcell.integrity import file_sha256
from c3vcell.mcool import extract_contact_candidates


@pytest.fixture
def contact_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    path = tmp_path / "source.mcool"
    with h5py.File(path, "w") as file:
        group = file.create_group("resolutions/10000")
        group.attrs["bin-size"] = 10_000
        group.attrs["genome-assembly"] = "hg38"
        chroms = group.create_group("chroms")
        chroms.create_dataset("name", data=np.asarray([b"chr1", b"chr2"]))
        chroms.create_dataset("length", data=np.asarray([40_000, 20_000]))
        bins = group.create_group("bins")
        bins.create_dataset("chrom", data=np.asarray([0, 0, 0, 0, 1, 1]))
        bins.create_dataset("start", data=np.asarray([0, 10_000, 20_000, 30_000, 0, 10_000]))
        bins.create_dataset("end", data=np.asarray([10_000, 20_000, 30_000, 40_000, 10_000, 20_000]))
        bins.create_dataset("weight", data=np.ones(6))
        pixels = group.create_group("pixels")
        pixels.create_dataset("bin1_id", data=np.asarray([0, 0, 0, 1, 1, 2, 4]))
        pixels.create_dataset("bin2_id", data=np.asarray([1, 2, 3, 2, 3, 3, 5]))
        pixels.create_dataset("count", data=np.asarray([1, 10, 1, 1, 1, 1, 1]))
    source = tmp_path / "source.json"
    source.write_text(json.dumps({
        "source_url": "https://example.org/source.mcool", "sha256": file_sha256(path),
        "context": "HCT116", "condition": "no-reported-treatment", "assembly": "GRCh38",
        "assay": "Hi-C",
    }), encoding="utf-8")
    sizes = tmp_path / "hg38.chrom.sizes"
    sizes.write_text("chr1\t40000\nchr2\t20000\n", encoding="utf-8")
    return path, source, sizes


def test_distance_expected_uses_zero_and_nonzero_possible_pairs(
    contact_inputs: tuple[Path, Path, Path], tmp_path: Path,
) -> None:
    mcool, source, sizes = contact_inputs
    metadata = extract_contact_candidates(
        mcool, source, sizes, tmp_path / "contacts", resolution=10_000,
        chromosomes=["chr1"], max_distance=30_000, min_count=2,
        min_observed_expected=1.5, chunk_pixels=2,
    )
    rows = (metadata.parent / "contacts.bedpe").read_text(encoding="utf-8").splitlines()
    assert len(rows) == 2
    fields = rows[1].split("\t")
    assert fields[:6] == ["chr1", "0", "10000", "chr1", "20000", "30000"]
    assert int(fields[6]) == 10
    assert float(fields[7]) == pytest.approx(10 / 5.5)
    record = json.loads(metadata.read_text(encoding="utf-8"))
    assert record["evidence_type"] == "distance-normalized-contact"
    assert record["covered_chromosomes"] == ["chr1"]
    assert record["parameters"]["output_contacts"] == 1
    assert "not FDR-controlled" in record["note"]


def test_missing_pixels_are_included_in_the_expected_denominator(
    contact_inputs: tuple[Path, Path, Path], tmp_path: Path,
) -> None:
    mcool, source, sizes = contact_inputs
    with h5py.File(mcool, "a") as file:
        pixels = file["resolutions/10000/pixels"]
        for field in ("bin1_id", "bin2_id", "count"):
            values = pixels[field][:][[0, 1, 2, 3, 5, 6]]
            del pixels[field]
            pixels.create_dataset(field, data=values)
    record = json.loads(source.read_text(encoding="utf-8"))
    record["sha256"] = file_sha256(mcool)
    source.write_text(json.dumps(record), encoding="utf-8")
    metadata = extract_contact_candidates(
        mcool, source, sizes, tmp_path / "missing", resolution=10_000,
        chromosomes=["chr1"], max_distance=30_000, min_count=2,
        min_observed_expected=1.5,
    )
    fields = (metadata.parent / "contacts.bedpe").read_text().splitlines()[1].split("\t")
    assert float(fields[7]) == pytest.approx(2.0)


def test_pixel_columns_can_be_unordered_within_sorted_bin1_rows(
    contact_inputs: tuple[Path, Path, Path], tmp_path: Path,
) -> None:
    mcool, source, sizes = contact_inputs
    with h5py.File(mcool, "a") as file:
        pixels = file["resolutions/10000/pixels"]
        for field in ("bin1_id", "bin2_id", "count"):
            values = pixels[field][:][[2, 0, 1, 3, 4, 5, 6]]
            del pixels[field]
            pixels.create_dataset(field, data=values)
    record = json.loads(source.read_text(encoding="utf-8"))
    record["sha256"] = file_sha256(mcool)
    source.write_text(json.dumps(record), encoding="utf-8")
    metadata = extract_contact_candidates(
        mcool, source, sizes, tmp_path / "unordered", resolution=10_000,
        chromosomes=["chr1"], max_distance=30_000, min_count=2,
        min_observed_expected=1.5, chunk_pixels=2,
    )
    fields = (metadata.parent / "contacts.bedpe").read_text().splitlines()[1].split("\t")
    assert fields[:6] == ["chr1", "0", "10000", "chr1", "20000", "30000"]
    assert float(fields[7]) == pytest.approx(10 / 5.5)


def test_assembly_mismatch_and_unverified_assembly_fail(
    contact_inputs: tuple[Path, Path, Path], tmp_path: Path,
) -> None:
    mcool, source, sizes = contact_inputs
    sizes.write_text("chr1\t40001\nchr2\t20000\n", encoding="utf-8")
    with pytest.raises(ValueError, match="chromosome size"):
        extract_contact_candidates(
            mcool, source, sizes, tmp_path / "wrong-size", resolution=10_000,
            chromosomes=["chr1"], min_count=2,
        )
    sizes.write_text("chr1\t40000\nchr2\t20000\n", encoding="utf-8")
    with h5py.File(mcool, "a") as file:
        file["resolutions/10000"].attrs["genome-assembly"] = "unknown"
    record = json.loads(source.read_text(encoding="utf-8"))
    record.update(sha256=file_sha256(mcool), assembly="unknown", condition="unknown")
    source.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="explicit size-based inference"):
        extract_contact_candidates(
            mcool, source, sizes, tmp_path / "no-inference", resolution=10_000,
            chromosomes=["chr1"], min_count=2,
        )
    inferred = extract_contact_candidates(
        mcool, source, sizes, tmp_path / "inferred", resolution=10_000,
        chromosomes=["chr1"], max_distance=30_000, min_count=2,
        min_observed_expected=1.5, allow_assembly_inference=True,
    )
    derived = json.loads(inferred.read_text(encoding="utf-8"))
    assert "not definitive" in derived["assembly_evidence"]
    assert derived["condition"] == "unknown"


def test_contact_map_hash_and_pixel_order_are_checked(
    contact_inputs: tuple[Path, Path, Path], tmp_path: Path,
) -> None:
    mcool, source, sizes = contact_inputs
    record = json.loads(source.read_text(encoding="utf-8"))
    record["sha256"] = "0" * 64
    source.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256"):
        extract_contact_candidates(
            mcool, source, sizes, tmp_path / "bad-hash", resolution=10_000,
            chromosomes=["chr1"], min_count=2,
        )
    record["sha256"] = file_sha256(mcool)
    source.write_text(json.dumps(record), encoding="utf-8")
    with h5py.File(mcool, "a") as file:
        file["resolutions/10000/pixels/bin2_id"][1] = 1
    record["sha256"] = file_sha256(mcool)
    source.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="sorted by bin1 and unique"):
        extract_contact_candidates(
            mcool, source, sizes, tmp_path / "bad-order", resolution=10_000,
            chromosomes=["chr1"], min_count=2, chunk_pixels=1,
        )


@pytest.mark.parametrize(("field", "message"), [
    ("count", "raw integer counts"),
    ("bin1_id", "bin IDs"),
])
def test_fractional_contact_data_is_rejected(
    contact_inputs: tuple[Path, Path, Path], tmp_path: Path, field: str, message: str,
) -> None:
    mcool, source, sizes = contact_inputs
    with h5py.File(mcool, "a") as file:
        pixels = file["resolutions/10000/pixels"]
        values = pixels[field][:].astype(float)
        values[0] += 0.5
        del pixels[field]
        pixels.create_dataset(field, data=values)
    record = json.loads(source.read_text(encoding="utf-8"))
    record["sha256"] = file_sha256(mcool)
    source.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        extract_contact_candidates(
            mcool, source, sizes, tmp_path / "fractional", resolution=10_000,
            chromosomes=["chr1"], min_count=2,
        )
