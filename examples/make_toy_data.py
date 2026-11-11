"""Create a small, synthetic raw-count dataset for the quick-start tutorial."""

from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp


def create_toy_data(output: Path, *, seed: int = 42) -> None:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    rng = np.random.default_rng(seed)
    genes = [f"G{i}" for i in range(62)] + ["SP1", "USF1"]
    targets = genes[:6]
    labels = ["non-targeting"] * 120 + [gene for gene in targets for _ in range(40)]
    baseline = rng.uniform(1.0, 5.0, len(genes))
    rates = np.broadcast_to(baseline, (len(labels), len(genes))).copy()
    for index, target in enumerate(targets):
        rows = np.array(labels) == target
        rates[rows, index] *= 0.4
        rates[rows, 10 + index] *= 2.0
        rates[rows, 20 + index] *= 0.5
    counts = rng.poisson(rates).astype(np.int32)
    counts[:, -1] += 1
    data = ad.AnnData(
        X=sp.csr_matrix(counts),
        obs=pd.DataFrame(
            {"target_gene": labels, "context": "toy", "batch": "toy-batch"},
            index=[f"toy-cell-{i:04d}" for i in range(len(labels))],
        ),
        var=pd.DataFrame(
            {"ensembl_ids": [f"ENSG{i:011d}" for i in range(1, len(genes) + 1)]},
            index=genes,
        ),
    )
    data.uns["description"] = "Artificial data for software demonstrations, not biology."
    data.uns["c3vcell_genome_assembly"] = "GRCh38"
    output.parent.mkdir(parents=True, exist_ok=True)
    data.write_h5ad(output, compression="gzip")
    print(f"Wrote synthetic raw counts: {output} ({data.n_obs} cells, {data.n_vars} genes)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    create_toy_data(args.output, seed=args.seed)


if __name__ == "__main__":
    main()
