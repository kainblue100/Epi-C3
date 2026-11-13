from __future__ import annotations

import gzip
import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "probe-schicar-brain-v1.py"
spec = importlib.util.spec_from_file_location("schicar_prefix_probe", SCRIPT)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def source_files():
    a, b, c, d = (letter * 18 for letter in "ACGT")
    text = {
        "rna_barcodes": f"{a}\n{b}\n{c}\n{d}\n",
        "metadata": (
            f"RNAbarcode\tDNAbarcode\n{a}\t{b}\n{b}\t{a}\n{c}\t{c}\n{d}\t{d}\n"
        ),
        "atac": f"chr1\t0\t10\t{b}\t60\t+\nchr1\t10\t20\t{b}\t60\t+\nchr1\t20\t30\t{c}\t60\t-\n",
        "contacts": f"{b}\tchr1\t1\tchr1\t2\t+\t-\tUU\n{a}\tchr1\t3\tchr1\t4\t+\t-\tUU\n",
    }
    return {probe.FILES[key]: gzip.compress(value.encode(), mtime=0) for key, value in text.items()}


def fake_fetch(files):
    return lambda url, limit: files[url.rsplit("/", 1)[-1]][:limit]


def test_source_map_detects_wrong_direct_join_without_exposing_barcodes():
    report = probe.audit(fake_fetch(source_files()))
    assert report["source_pairing_map"] == {
        "rna_axis_cells": 4,
        "mapped_rna_cells": 4,
        "mapped_dna_cells": 4,
        "same_barcode_in_source_paired_row": 2,
        "dna_barcodes_also_on_rna_axis": 4,
        "dna_barcodes_equal_to_another_cells_rna_barcode": 2,
    }
    assert report["assay_prefixes"]["atac"]["rows_matching_rna_axis_directly"] == 3
    assert report["assay_prefixes"]["atac"]["direct_rna_matches_conflicting_with_source_map"] == 2
    assert report["mapped_dna_cells_seen_in_both_assay_prefixes"] == 1
    assert report["final_human_cell_line_training_ready"] is False
    assert report["model_result"] is False
    for letter in "ACGT":
        assert letter * 18 not in json.dumps(report)


def test_duplicate_dna_mapping_and_mismatched_rna_axis_are_rejected():
    files = source_files()
    a, b, c, d = (letter * 18 for letter in "ACGT")
    duplicate = f"RNAbarcode\tDNAbarcode\n{a}\t{a}\n{b}\t{a}\n{c}\t{c}\n{d}\t{d}\n"
    files[probe.FILES["metadata"]] = gzip.compress(duplicate.encode(), mtime=0)
    with pytest.raises(ValueError, match="one-to-one"):
        probe.audit(fake_fetch(files))
    files = source_files()
    files[probe.FILES["rna_barcodes"]] = gzip.compress(f"{a}\n{b}\n{c}\n".encode(), mtime=0)
    with pytest.raises(ValueError, match="do not match"):
        probe.audit(fake_fetch(files))


def test_complete_metadata_is_required_and_partial_gzip_is_explicit():
    raw = gzip.compress(b"first\nsecond\nincomplete", mtime=0)
    partial = raw[:-5]
    with pytest.raises(ValueError, match="complete gzip"):
        probe.unpack_prefix(partial, len(partial), require_complete=True)
    lines, description = probe.unpack_prefix(partial, len(partial), require_complete=False)
    assert lines == ["first", "second"]
    assert not description["gzip_stream_complete"]
    assert description["discarded_incomplete_final_line"]
    with pytest.raises(ValueError, match="short"):
        probe.unpack_prefix(partial, len(partial) + 1, require_complete=False)


def test_bad_assay_schema_and_oversized_prefix_are_rejected():
    files = source_files()
    files[probe.FILES["atac"]] = gzip.compress(b"chr1\t1\t2\tbad\n", mtime=0)
    with pytest.raises(ValueError, match="invalid schema"):
        probe.audit(fake_fetch(files))
    with pytest.raises(ValueError, match="8 MiB"):
        probe.audit(fake_fetch(source_files()), prefix_bytes=probe.MAX_PREFIX_BYTES + 1)
