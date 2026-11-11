"""Select prior-covered genetic targets without inspecting perturbation expression."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from .epigenetics import feature_names, load_epigenetic_bundle
from .integrity import file_sha256


def priority_panel(
    totals: pd.DataFrame,
    selection_data: Path,
    bundle: Path,
    spec: Path,
    methylation_count: int,
    *,
    additional_graph_count: int = 0,
    loader: Callable = load_epigenetic_bundle,
) -> tuple[list[str], list[str], dict]:
    if methylation_count < 0 or additional_graph_count < 0:
        raise ValueError("Priority panel sizes must be nonnegative")
    if (
        not {"gene_target", "n_cells"}.issubset(totals.columns)
        or totals["gene_target"].isna().any()
        or totals["gene_target"].astype(str).duplicated().any()
    ):
        raise ValueError("Target totals need unique, non-null gene labels and cell counts")
    counts = pd.to_numeric(totals["n_cells"], errors="raise").to_numpy(dtype=np.float64)
    if not np.isfinite(counts).all() or np.any(counts < 0) or not np.equal(counts, np.floor(counts)).all():
        raise ValueError("Target totals need finite nonnegative integer cell counts")
    expected_order = totals.sort_values(
        ["n_cells", "gene_target"], ascending=[False, True],
    )["gene_target"].astype(str).tolist()
    if totals["gene_target"].astype(str).tolist() != expected_order:
        raise ValueError("Target totals must be ordered by descending counts and gene name")
    source_digest = file_sha256(selection_data)
    data = ad.read_h5ad(selection_data, backed="r")
    try:
        _, masks, metadata = loader(bundle, data)
        genes = data.var_names.astype(str).to_numpy()
    finally:
        data.file.close()
    if (
        metadata.get("context") != "HCT116"
        or metadata.get("format_version") != 2
        or metadata.get("control") is not None
        or metadata.get("specification_sha256") != file_sha256(spec)
    ):
        raise ValueError("Priority selection needs a real HCT116 network from the same specification")
    names = feature_names(metadata)
    columns = [names.index("promoter_methylation"), names.index("linked_element_methylation")]
    measured = set(genes[np.any(masks[:, columns] == 1, axis=1)])
    edges = pd.read_csv(bundle / "graph.csv", keep_default_na=False)
    if (
        file_sha256(bundle / "graph.csv") != metadata["files"]["graph.csv"]
        or file_sha256(selection_data) != source_digest
    ):
        raise RuntimeError("Priority-selection inputs changed while reading")
    outgoing = set(edges["source"].astype(str))
    eligible = [
        str(gene) for gene in totals["gene_target"]
        if gene in measured and gene in outgoing
    ]
    if len(eligible) < methylation_count:
        raise ValueError(
            f"Only {len(eligible)} fresh eligible targets have measured methylation "
            "and outgoing graph edges"
        )
    methylation = eligible[:methylation_count]
    additional = [
        str(gene) for gene in totals["gene_target"]
        if gene in outgoing and gene not in set(methylation)
    ][:additional_graph_count]
    if len(additional) < additional_graph_count:
        raise ValueError("Not enough additional fresh targets have outgoing graph edges")
    selected = methylation + additional
    neighbors = {
        target: sorted(edges.loc[edges["source"] == target, "target"].astype(str).tolist())
        for target in selected
    }
    required = sorted(set().union(*map(set, neighbors.values()))) if neighbors else []
    rule = (
        "Among fresh targets with enough passing cells, select by descending cell count "
        "and gene name when prior methylation is measured and outgoing edges exist."
    )
    if additional_graph_count:
        rule += " Additional graph-only targets have outgoing edges but may lack methylation."
    return selected, required, {
        "selection_data": str(selection_data.resolve()),
        "selection_data_sha256": source_digest,
        "bundle": str(bundle.resolve()),
        "bundle_manifest_sha256": metadata["manifest_sha256"],
        "bundle_file_sha256": metadata["files"],
        "rule": rule,
        "prioritized_holdouts": selected,
        "methylation_prioritized_holdouts": methylation,
        "graph_only_prioritized_holdouts": additional,
        "prespecified_outgoing_neighbors": neighbors,
        "additional_required_genes": required,
    }
