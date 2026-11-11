"""Verify and review a repeated, uncalibrated HCT116 network diagnostic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from c3vcell.research_review import review_network_diagnostic


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--parent-run", type=Path, required=True)
    parser.add_argument("--parent-diagnostic", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--source-archive", type=Path)
    parser.add_argument("--program", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    review = review_network_diagnostic(
        args.run, args.parent_run, args.parent_diagnostic, args.spec,
        source_archive=args.source_archive, program=args.program,
    )
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(review, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
