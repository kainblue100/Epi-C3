"""Choose an Orion target panel by passing-cell counts, never by expression."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--groups", type=Path, required=True)
    parser.add_argument("--gene-metadata", type=Path, required=True)
    parser.add_argument("--control-label", required=True)
    parser.add_argument("--targets", type=int, default=20)
    parser.add_argument("--min-cells", type=int, default=25)
    parser.add_argument(
        "--exclude-targets", type=Path,
        help="newline-delimited previously inspected targets to exclude",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.targets < 3 or args.min_cells < 1:
        raise ValueError("Select at least three targets and a positive minimum cell count")
    summary_path = args.output.with_suffix(".json")
    if summary_path == args.output:
        raise ValueError("The target list needs a non-JSON extension, such as .txt")
    if args.output.exists() or summary_path.exists():
        raise FileExistsError("Refusing to overwrite an existing target list or summary")

    groups = pd.read_csv(args.groups, dtype={"context": str, "gene_target": str, "sample": str})
    required = {"context", "gene_target", "sample", "n_cells"}
    if not required.issubset(groups.columns) or groups[list(required)].isna().any().any():
        raise ValueError("The Orion count table has missing fields")
    if (groups["n_cells"] < 0).any():
        raise ValueError("Cell counts cannot be negative")
    metadata = pd.read_parquet(args.gene_metadata, columns=["gene_name"])
    genes = set(metadata["gene_name"].dropna().astype(str))
    excluded = set()
    if args.exclude_targets is not None:
        excluded = {
            line.strip() for line in args.exclude_targets.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
    control = groups.loc[groups["gene_target"] == args.control_label]
    if control.empty:
        raise ValueError(f"No passing cells for control label {args.control_label!r}")

    totals = groups.groupby("gene_target", as_index=False)["n_cells"].sum()
    totals = totals.loc[
        (totals["gene_target"] != args.control_label)
        & totals["gene_target"].isin(genes)
        & ~totals["gene_target"].isin(excluded)
        & (totals["n_cells"] >= args.min_cells)
    ].sort_values(["n_cells", "gene_target"], ascending=[False, True])
    if len(totals) < args.targets:
        raise ValueError(
            f"Only {len(totals)} targets have at least {args.min_cells} passing cells; "
            "inspect more complete shards or lower the threshold explicitly"
        )
    chosen = totals.head(args.targets)
    control_keys = set(zip(control["context"], control["sample"]))
    target_rows = groups.loc[groups["gene_target"].isin(chosen["gene_target"])].copy()
    target_rows["has_control"] = [
        (context, sample) in control_keys
        for context, sample in zip(target_rows["context"], target_rows["sample"])
    ]
    target_cells = int(target_rows["n_cells"].sum())
    matched_cells = int(target_rows.loc[target_rows["has_control"], "n_cells"].sum())
    summary = {
        "selection": "passing cell counts only; no expression values",
        "control_label": args.control_label,
        "excluded_targets": sorted(excluded),
        "control_cells": int(control["n_cells"].sum()),
        "control_context_sample_groups": len(control_keys),
        "targets": chosen.to_dict(orient="records"),
        "target_cells": target_cells,
        "target_cells_with_context_sample_controls": matched_cells,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(chosen["gene_target"]) + "\n", encoding="utf-8")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
