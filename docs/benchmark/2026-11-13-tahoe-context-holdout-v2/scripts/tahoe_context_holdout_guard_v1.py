"""Candidate Tahoe context filter; not integrated into the existing trainer."""

from __future__ import annotations

import hashlib
import json
import numbers
from pathlib import Path


def read_pinned_json(path: Path, expected_sha256: str) -> dict:
    if (
        not isinstance(expected_sha256, str) or len(expected_sha256) != 64
        or any(char not in "0123456789abcdef" for char in expected_sha256)
    ):
        raise ValueError("A lowercase SHA-256 digest is required")
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected_sha256:
        raise ValueError("A frozen context-holdout input changed")
    value = json.loads(data)
    if not isinstance(value, dict):
        raise ValueError("A frozen context-holdout input is not an object")
    return value


def valid_component(value) -> bool:
    return isinstance(value, str) and bool(value) and "\t" not in value


class TahoeContextHoldoutGuard:
    def __init__(
        self, metadata_path: Path, private_plan_path: Path, summary_path: Path,
        *, expected_summary_sha256: str,
    ) -> None:
        summary = read_pinned_json(summary_path, expected_summary_sha256)
        if (
            summary.get("format") != "epi-c3-tahoe-state-context-holdout-summary-v1"
            or summary.get("status") != "candidate-not-enforced"
        ):
            raise ValueError("The frozen context-holdout summary is incompatible")
        private = read_pinned_json(private_plan_path, summary["private_plan_sha256"])
        if (
            private.get("format") != "epi-c3-tahoe-state-context-holdout-private-v1"
            or private.get("status") != "candidate-not-enforced"
            or private.get("source_sha256") != summary.get("source_sha256")
            or private.get("exposure_audit_sha256") != summary.get("exposure_audit_sha256")
            or private.get("planner_sha256") != summary.get("planner_sha256")
        ):
            raise ValueError("The candidate private plan and summary disagree")
        metadata = read_pinned_json(metadata_path, summary["source_sha256"]["metadata_plan"])
        groups = metadata.get("groups")
        blocked = private.get("blocked_cell_line_ids")
        if (
            metadata.get("state") != "complete" or not isinstance(groups, dict) or not groups
            or not isinstance(blocked, list) or len(blocked) != 5
            or any(not valid_component(value) for value in blocked) or len(set(blocked)) != 5
        ):
            raise ValueError("The frozen Tahoe groups or held-out contexts are invalid")
        seen_lines = set()
        rows = 0
        for key, group in groups.items():
            sample, line, count = group.get("sample"), group.get("cell_line_id"), group.get("cells")
            if (
                not valid_component(sample) or not valid_component(line) or key != f"{sample}\t{line}"
                or type(count) is not int or count < 1
            ):
                raise ValueError("A frozen Tahoe group is invalid")
            seen_lines.add(line)
            rows += count
        if rows != metadata.get("rows") or rows != summary.get("source_rows") or not set(blocked) <= seen_lines:
            raise ValueError("The frozen Tahoe group universe is inconsistent")
        self._groups = groups
        self._blocked = frozenset(blocked)

    def filter_selected(self, samples, lines, selected) -> list[int]:
        """Filter already-eligible rows; historical and validation identity filters remain required."""
        if len(samples) != len(lines):
            raise ValueError("Tahoe sample and cell-line columns have different lengths")
        kept, seen = [], set()
        for raw_index in selected:
            if isinstance(raw_index, bool) or not isinstance(raw_index, numbers.Integral):
                raise ValueError("A selected Tahoe row index is invalid")
            index = int(raw_index)
            if index < 0 or index >= len(samples) or index in seen:
                raise ValueError("A selected Tahoe row index is invalid or duplicated")
            seen.add(index)
            sample, line = samples[index], lines[index]
            if not valid_component(sample) or not valid_component(line):
                raise ValueError("Selected Tahoe metadata is incomplete")
            group = self._groups.get(f"{sample}\t{line}")
            if group is None or group["sample"] != sample or group["cell_line_id"] != line:
                raise ValueError("Selected Tahoe metadata is outside the frozen source groups")
            if line not in self._blocked:
                kept.append(index)
        return kept
