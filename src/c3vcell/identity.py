"""Stable source-cell identities and compatibility checks."""

from __future__ import annotations

import hashlib
import json
import re
import warnings
from collections.abc import Iterable
from typing import Literal


Source = Literal["orion", "tahoe"]
_LEGACY = {
    source: re.compile(rf"^{source}-[0-9]{{9}}$")
    for source in ("orion", "tahoe")
}
_STABLE = {
    source: re.compile(rf"^{source}-v1-[0-9a-f]{{64}}$")
    for source in ("orion", "tahoe")
}


def source_cell_id(source: Source, *components: str) -> str:
    if source not in _STABLE or not components or any(
        not isinstance(value, str) or not value.strip() for value in components
    ):
        raise ValueError("Source cell identifiers require nonempty string components")
    payload = json.dumps(
        [source, *components], ensure_ascii=False, separators=(",", ":"),
    ).encode("utf-8")
    return f"{source}-v1-{hashlib.sha256(payload).hexdigest()}"


def check_cell_id_schemes(*groups: Iterable[str]) -> None:
    """Reject mixed legacy/stable IDs and flag legacy cross-file comparisons."""
    found = {source: {"legacy": 0, "stable": 0} for source in _LEGACY}
    for identifiers in groups:
        group_found = {source: set() for source in _LEGACY}
        for value in identifiers:
            cell_id = str(value)
            for source in _LEGACY:
                if _LEGACY[source].fullmatch(cell_id):
                    group_found[source].add("legacy")
                elif _STABLE[source].fullmatch(cell_id):
                    group_found[source].add("stable")
        for source, schemes in group_found.items():
            for scheme in schemes:
                found[source][scheme] += 1
    for source, schemes in found.items():
        if schemes["legacy"] and schemes["stable"]:
            raise ValueError(
                f"Legacy and stable {source} cell IDs cannot be compared safely; "
                "reprepare the data and retrain legacy checkpoints"
            )
        if schemes["legacy"] > 1:
            warnings.warn(
                f"Legacy {source} cell IDs were assigned by position and cannot verify "
                "independence between separately prepared datasets. Reprepare and retrain "
                "using stable source-cell IDs for cross-dataset comparisons.",
                RuntimeWarning, stacklevel=2,
            )
