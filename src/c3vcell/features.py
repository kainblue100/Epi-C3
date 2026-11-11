"""Map dataset gene tokens to a unique, count-preserving symbol axis."""

from __future__ import annotations

import numpy as np
import pandas as pd


def gene_token_mapping(
    metadata: pd.DataFrame, *, token_column: str, symbol_column: str,
) -> tuple[list[str], dict[int, int], pd.Series]:
    required = {token_column, symbol_column, "ensembl_id"}
    if not required.issubset(metadata.columns):
        raise ValueError(f"Gene metadata must contain {sorted(required)}")
    frame = metadata.dropna(subset=list(required)).copy()
    if frame.empty:
        raise ValueError("Gene metadata contains no complete gene records")
    if frame[token_column].duplicated().any():
        raise ValueError("A gene token maps to multiple metadata rows")
    numeric = pd.to_numeric(frame[token_column], errors="raise").to_numpy(dtype=np.float64)
    if not np.isfinite(numeric).all() or np.any(numeric < 0) or not np.equal(numeric, np.floor(numeric)).all():
        raise ValueError("Gene tokens must be finite nonnegative integers")
    frame[symbol_column] = frame[symbol_column].astype(str)
    if (frame[symbol_column].str.len() == 0).any():
        raise ValueError("Gene symbols cannot be empty")
    symbols = list(dict.fromkeys(frame[symbol_column]))
    lookup = {name: index for index, name in enumerate(symbols)}
    tokens = {
        int(token): lookup[str(symbol)]
        for token, symbol in zip(frame[token_column], frame[symbol_column])
    }
    if len(tokens) != len(frame):
        raise ValueError("Gene tokens must be uniquely representable as integers")
    ensembls = frame.groupby(symbol_column, sort=False)["ensembl_id"].agg(
        lambda values: "|".join(sorted(set(map(str, values))))
    )
    collisions = int((frame.groupby(symbol_column)["ensembl_id"].nunique() > 1).sum())
    if collisions:
        print(
            f"Aggregating {collisions} duplicated gene symbols; Ensembl IDs remain in var",
            flush=True,
        )
    return symbols, tokens, ensembls
