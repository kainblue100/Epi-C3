"""Compare one target's predictions with independent, replicated bulk TPM."""

from __future__ import annotations

import argparse
from pathlib import Path

from c3vcell.external_bulk import compare_bulk_tpm


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--controls", type=Path, required=True)
    parser.add_argument("--bulk", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--context", required=True)
    parser.add_argument("--control-samples", nargs="+", required=True)
    parser.add_argument("--perturbed-samples", nargs="+", required=True)
    parser.add_argument("--minimum-tpm", type=float, default=1.0)
    parser.add_argument("--pseudocount", type=float, default=1.0)
    args = parser.parse_args()
    compare_bulk_tpm(
        args.prediction, args.controls, args.bulk, args.output, target=args.target,
        context=args.context, control_samples=args.control_samples,
        perturbed_samples=args.perturbed_samples, minimum_tpm=args.minimum_tpm,
        pseudocount=args.pseudocount,
    )


if __name__ == "__main__":
    main()
