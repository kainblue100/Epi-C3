#!/usr/bin/env python3
"""Choose a reproducible Tahoe panel using cell counts, not expression."""

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--groups", type=Path, required=True)
    parser.add_argument("--cell-line", required=True)
    parser.add_argument("--plate", required=True)
    parser.add_argument("--drugs", type=int, default=20)
    parser.add_argument("--min-cells", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary_path = args.output.with_suffix(".json")
    if args.output.exists() or summary_path.exists():
        raise FileExistsError("Refusing to overwrite the drug panel or its summary")
    if min(args.drugs, args.min_cells) < 1:
        raise ValueError("Drug count and minimum cells must be positive")
    frame = pd.read_csv(args.groups)
    required = {"drug", "plate", "cell_line_id", "n_cells"}
    if not required.issubset(frame):
        raise ValueError(f"Group table must contain {sorted(required)}")
    subset = frame[
        (frame["plate"] == args.plate) & (frame["cell_line_id"] == args.cell_line)
    ]
    counts = subset.groupby("drug", dropna=False, observed=True)["n_cells"].sum()
    if counts.get("DMSO_TF", 0) < args.min_cells:
        raise ValueError("Not enough matched DMSO_TF controls")
    eligible = [
        (str(drug), int(n)) for drug, n in counts.items()
        if pd.notna(drug) and drug != "DMSO_TF" and n >= args.min_cells
    ]
    eligible.sort(key=lambda item: (-item[1], item[0]))
    if len(eligible) < args.drugs:
        raise ValueError(f"Only {len(eligible)} drugs meet the minimum cell count")
    chosen = eligible[:args.drugs]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(f"{drug}\n" for drug, _ in chosen), encoding="utf-8")
    summary = {
        "selection": "cell counts only; no expression or response was inspected",
        "cell_line_id": args.cell_line, "plate": args.plate,
        "minimum_cells": args.min_cells, "control_cells": int(counts["DMSO_TF"]),
        "selected": [{"drug": drug, "n_cells": n} for drug, n in chosen],
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Selected {len(chosen)} drugs; summary: {summary_path}")


if __name__ == "__main__":
    main()
