"""Reconcile called chromatin-contact anchors without claiming independent replication."""

from __future__ import annotations

import csv
import math
from pathlib import Path

from .epigenetics import _IntervalIndex, _loops
from .integrity import file_sha256

Anchor = tuple[str, int, int]
Contact = tuple[Anchor, Anchor]


def _canonical(contact: Contact) -> Contact:
    first, second = contact
    return (first, second) if first <= second else (second, first)


def _intersection(left: Anchor, right: Anchor, fraction: float) -> Anchor | None:
    if left[0] != right[0]:
        return None
    start, end = max(left[1], right[1]), min(left[2], right[2])
    overlap = end - start
    if (
        overlap <= 0 or overlap < fraction * (left[2] - left[1])
        or overlap < fraction * (right[2] - right[1])
    ):
        return None
    return left[0], start, end


def overlap_supported_contacts(
    first_path: Path, second_path: Path, output_path: Path, *, reciprocal_overlap: float = 0.5,
) -> dict:
    if not math.isfinite(reciprocal_overlap) or not 0 < reciprocal_overlap <= 1:
        raise ValueError("Reciprocal anchor overlap must be in (0, 1]")
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite {output_path}")
    first_digest, second_digest = file_sha256(first_path), file_sha256(second_path)
    first = sorted({_canonical(loop) for loop in _loops(first_path)})
    second = sorted({_canonical(loop) for loop in _loops(second_path)})
    if not first or not second:
        raise ValueError("Both input contact files must contain called loops")
    index = _IntervalIndex([
        (anchor[0], anchor[1], anchor[2], str(number))
        for number, (anchor, _) in enumerate(second)
    ])
    supported: set[Contact] = set()
    for left, right in first:
        candidates = index.overlaps(*left)
        if len(candidates) > 50_000:
            raise ValueError("An anchor overlaps too many candidate contacts")
        for _, _, number in candidates:
            other_left, other_right = second[int(number)]
            a = _intersection(left, other_left, reciprocal_overlap)
            b = _intersection(right, other_right, reciprocal_overlap)
            if a is not None and b is not None:
                supported.add(_canonical((a, b)))
    if not supported:
        raise ValueError("No called loops have the requested overlap support")
    if file_sha256(first_path) != first_digest or file_sha256(second_path) != second_digest:
        raise RuntimeError("Contact source changed while constructing the supported graph")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
        stream.write("#chrom1\tstart1\tend1\tchrom2\tstart2\tend2\n")
        for left, right in sorted(supported):
            writer.writerow((*left, *right))
    return {
        "first_sha256": first_digest,
        "second_sha256": second_digest,
        "first_unique_contacts": len(first),
        "second_unique_contacts": len(second),
        "supported_contacts": len(supported),
        "reciprocal_overlap": reciprocal_overlap,
        "covered_chromosomes": sorted({anchor[0] for pair in supported for anchor in pair}),
        "sha256": file_sha256(output_path),
        "note": (
            "Coordinate intersection across called-loop files is not independent biological "
            "replication or evidence of causal enhancer-promoter relationships."
        ),
    }
