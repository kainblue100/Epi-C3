"""Count-preserving data preparation and perturbation-disjoint splitting."""

from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from .identity import check_cell_id_schemes


CONTROL = "non-targeting"


def validate_counts(adata: ad.AnnData, *, allow_zero_cells: bool = False) -> None:
    if not adata.var_names.is_unique:
        raise ValueError("Gene names must be unique; resolve duplicates before training")
    if not adata.obs_names.is_unique:
        raise ValueError("Cell identifiers must be unique")
    matrix = sp.csr_matrix(adata.X)
    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("The count matrix is empty")
    values = matrix.data
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Counts must be finite and nonnegative")
    if not np.equal(values, np.floor(values)).all():
        raise ValueError("X must contain raw integer counts, not normalized expression")
    if not allow_zero_cells and np.any(np.asarray(matrix.sum(axis=1)).ravel() <= 0):
        raise ValueError("Cells with zero total counts are not supported")


def _choose(indices: np.ndarray, limit: int, rng: np.random.Generator) -> np.ndarray:
    if limit <= 0:
        raise ValueError("Sampling limits must be positive")
    if len(indices) <= limit:
        return indices
    return rng.choice(indices, limit, replace=False)


def read_h5ad_subset(
    path: Path,
    *,
    label_column: str,
    control: str,
    max_per_target: int,
    max_controls: int,
    max_targets: int,
    seed: int,
    include_targets: list[str] | None = None,
) -> ad.AnnData:
    """Read only selected rows from a completed, raw-count H5AD."""
    rng = np.random.default_rng(seed)
    source = ad.read_h5ad(path, backed="r")
    try:
        if label_column not in source.obs:
            raise ValueError(f"Missing perturbation column: {label_column}")
        if source.obs[label_column].isna().any():
            raise ValueError(f"{label_column} contains missing labels")
        labels = source.obs[label_column].astype(str).to_numpy()
        controls = np.flatnonzero(labels == control)
        if not len(controls):
            raise ValueError(f"No controls match {control!r}")
        available = np.array(sorted(set(labels) - {control}), dtype=object)
        if include_targets is not None:
            if (
                not include_targets or len(set(include_targets)) != len(include_targets)
                or control in include_targets or any(not name for name in include_targets)
            ):
                raise ValueError("Included targets must be unique, nonempty, and exclude the control")
            missing = sorted(set(include_targets) - set(available))
            if missing:
                raise ValueError(f"Included targets are absent from the source: {missing}")
            available = np.asarray(sorted(include_targets), dtype=object)
        elif max_targets > 0 and len(available) > max_targets:
            available = np.sort(rng.choice(available, max_targets, replace=False))
        chosen = [_choose(controls, max_controls, rng)]
        for label in available:
            chosen.append(_choose(np.flatnonzero(labels == label), max_per_target, rng))
        rows = np.sort(np.concatenate(chosen))
        matrix = sp.csr_matrix(source.X[rows, :])
        obs = source.obs.iloc[rows].copy()
        obs["target_gene"] = labels[rows]
        if "context" not in obs:
            obs["context"] = "local"
        result = ad.AnnData(X=matrix, obs=obs, var=source.var.copy())
        if "c3vcell_genome_assembly" in source.uns:
            result.uns["c3vcell_genome_assembly"] = source.uns["c3vcell_genome_assembly"]
    finally:
        source.file.close()
    validate_counts(result)
    return result


def split_by_target(
    adata: ad.AnnData,
    *,
    control: str = CONTROL,
    holdout_fraction: float = 0.25,
    seed: int = 0,
    holdout_targets: list[str] | None = None,
    split_column: str = "target_gene",
    require_targets_on_gene_axis: bool = True,
) -> tuple[ad.AnnData, ad.AnnData, ad.AnnData, dict]:
    """Hold out target groups and disjoint training, input and scoring controls."""
    validate_counts(adata)
    if not 0 < holdout_fraction < 1:
        raise ValueError("holdout_fraction must be between zero and one")
    if split_column not in adata.obs:
        raise ValueError(f"The count matrix needs obs[{split_column!r}]")
    for column in set(("target_gene", "context", "batch", split_column)):
        if column in adata.obs and adata.obs[column].isna().any():
            raise ValueError(f"{column} contains missing values")
    labels = adata.obs[split_column].astype(str).to_numpy()
    gene_set = set(map(str, adata.var_names))
    perturbations = sorted(set(labels) - {control})
    if require_targets_on_gene_axis:
        missing = sorted(set(perturbations) - gene_set)
        if missing:
            raise ValueError(f"Target genes absent from the model gene axis: {missing[:20]}")
    if len(perturbations) < 3:
        raise ValueError("At least three distinct perturbations are needed for a target split")
    rng = np.random.default_rng(seed)
    if holdout_targets is None:
        shuffled = rng.permutation(perturbations)
        n_holdout = min(max(2, round(len(shuffled) * holdout_fraction)), len(shuffled) - 1)
        heldout = set(shuffled[:n_holdout].tolist())
    else:
        heldout = set(holdout_targets)
        if len(heldout) != len(holdout_targets) or len(heldout) < 2:
            raise ValueError("Explicit holdout targets must contain at least two unique labels")
        if not heldout < set(perturbations):
            raise ValueError("Holdout targets must be present and leave at least one training target")
    training = set(perturbations) - heldout

    controls = np.flatnonzero(labels == control)
    if len(controls) < 3:
        raise ValueError("At least three independent control cells are required")
    obs = adata.obs.copy()
    context = obs["context"].astype(str) if "context" in obs else pd.Series("local", index=obs.index)
    batch = obs["batch"].astype(str) if "batch" in obs else pd.Series("", index=obs.index)
    train_contexts = set(context.iloc[np.flatnonzero(np.isin(labels, list(training)))])
    heldout_contexts = set(context.iloc[np.flatnonzero(np.isin(labels, list(heldout)))])
    groups: dict[tuple[str, str], list[int]] = {}
    for i in controls:
        groups.setdefault((context.iloc[i], batch.iloc[i]), []).append(i)
    train_controls: list[int] = []
    input_controls: list[int] = []
    reference_controls: list[int] = []
    for (group_context, _), indices in groups.items():
        shuffled_controls = rng.permutation(indices)
        if group_context not in heldout_contexts or len(indices) < 3:
            train_controls.extend(shuffled_controls.tolist())
            continue
        n_each = min(max(1, round(len(indices) * holdout_fraction / 2)), (len(indices) - 1) // 2)
        input_controls.extend(shuffled_controls[:n_each].tolist())
        reference_controls.extend(shuffled_controls[n_each : 2 * n_each].tolist())
        train_controls.extend(shuffled_controls[2 * n_each :].tolist())
    for role, indices, required_contexts in (
        ("training", train_controls, train_contexts),
        ("input", input_controls, heldout_contexts),
        ("reference", reference_controls, heldout_contexts),
    ):
        missing_contexts = required_contexts - set(context.iloc[indices])
        if missing_contexts:
            raise ValueError(
                f"No {role} controls for contexts {sorted(missing_contexts)}; "
                "each context needs a batch with at least three controls"
            )

    train_mask = np.isin(labels, list(training))
    test_mask = np.isin(labels, list(heldout))
    train_mask[train_controls] = True
    test_mask[reference_controls] = True
    train = adata[train_mask].copy()
    test = adata[test_mask].copy()
    inputs = adata[np.sort(input_controls)].copy()
    manifest = {
        "seed": seed,
        "control": control,
        "split_column": split_column,
        "require_targets_on_gene_axis": require_targets_on_gene_axis,
        "holdout_fraction": holdout_fraction,
        "train_targets": sorted(training),
        "test_targets": sorted(heldout),
        "train_cells": train.n_obs,
        "test_cells": test.n_obs,
        "train_controls": len(train_controls),
        "input_controls": len(input_controls),
        "reference_controls": len(reference_controls),
        "train_cell_ids": train.obs_names.astype(str).tolist(),
        "test_cell_ids": test.obs_names.astype(str).tolist(),
        "input_control_cell_ids": inputs.obs_names.astype(str).tolist(),
    }
    return train, test, inputs, manifest


def _validate_gene_selection(
    adata: ad.AnnData, max_genes: int, required_genes: list[str] | None,
) -> None:
    if max_genes < 0:
        raise ValueError("max_genes must be nonnegative; zero keeps all genes")
    if required_genes is not None:
        if (
            any(not isinstance(gene, str) or not gene.strip() for gene in required_genes)
            or len(set(required_genes)) != len(required_genes)
        ):
            raise ValueError("Prespecified required genes must be unique nonempty strings")
        missing = sorted(set(required_genes) - set(adata.var_names.astype(str)))
        if missing:
            raise ValueError(f"Prespecified required genes are absent from the source: {missing[:20]}")


def _select_gene_axis(
    train: ad.AnnData,
    test: ad.AnnData,
    inputs: ad.AnnData,
    manifest: dict,
    *,
    max_genes: int,
    required_genes: list[str] | None,
    require_targets_on_gene_axis: bool,
) -> tuple[ad.AnnData, ad.AnnData, ad.AnnData]:
    if max_genes and train.n_vars > max_genes:
        required = (
            set(manifest["train_targets"]) | set(manifest["test_targets"])
            if require_targets_on_gene_axis else set()
        )
        required.update(required_genes or [])
        if len(required) > max_genes:
            raise ValueError("max_genes is smaller than the number of required genes")
        genes = train.var_names.astype(str).to_numpy()
        required_indices = np.flatnonzero(np.isin(genes, list(required)))
        totals = np.asarray(sp.csr_matrix(train.X).sum(axis=0)).ravel().astype(np.float64)
        order = np.lexsort((np.arange(len(genes)), -totals))
        candidate_indices = order[~np.isin(order, required_indices)]
        selected = np.sort(np.concatenate(
            (required_indices, candidate_indices[: max_genes - len(required_indices)])
        ))
        train = train[:, selected].copy()
        test = test[:, selected].copy()
        inputs = inputs[:, selected].copy()
        for subset in (train, test, inputs):
            validate_counts(subset)
        manifest["gene_selection"] = {
            "method": (
                "training-count totals plus the public target panel"
                if require_targets_on_gene_axis else "training-count totals"
            ),
            "n_genes": len(selected),
            "source_genes": len(genes),
        }
        if required_genes:
            manifest["gene_selection"]["method"] += " plus prespecified required genes"
            manifest["gene_selection"]["required_genes"] = sorted(required_genes)
    else:
        manifest["gene_selection"] = {
            "method": "all genes",
            "n_genes": train.n_vars,
            "source_genes": train.n_vars,
        }
        if required_genes:
            manifest["gene_selection"]["required_genes"] = sorted(required_genes)
    return train, test, inputs


def _write_split_files(
    train: ad.AnnData, test: ad.AnnData, inputs: ad.AnnData, manifest: dict, outdir: Path,
) -> None:
    train.write_h5ad(outdir / "train.h5ad", compression="gzip")
    test.write_h5ad(outdir / "reference.h5ad", compression="gzip")
    inputs.write_h5ad(outdir / "input_controls.h5ad", compression="gzip")
    (outdir / "split.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def write_split(
    adata: ad.AnnData,
    outdir: Path,
    *,
    control: str,
    holdout_fraction: float,
    seed: int,
    max_genes: int = 0,
    holdout_targets: list[str] | None = None,
    split_column: str = "target_gene",
    require_targets_on_gene_axis: bool = True,
    required_genes: list[str] | None = None,
) -> None:
    _validate_gene_selection(adata, max_genes, required_genes)
    if outdir.exists() and any(outdir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty directory: {outdir}")
    outdir.mkdir(parents=True, exist_ok=True)
    train, test, inputs, manifest = split_by_target(
        adata, control=control, holdout_fraction=holdout_fraction, seed=seed,
        holdout_targets=holdout_targets,
        split_column=split_column, require_targets_on_gene_axis=require_targets_on_gene_axis,
    )
    train, test, inputs = _select_gene_axis(
        train, test, inputs, manifest, max_genes=max_genes, required_genes=required_genes,
        require_targets_on_gene_axis=require_targets_on_gene_axis,
    )
    _write_split_files(train, test, inputs, manifest, outdir)


def write_sample_disjoint_split(
    training: ad.AnnData,
    heldout: ad.AnnData,
    outdir: Path,
    *,
    control: str = CONTROL,
    seed: int = 0,
    max_genes: int = 0,
    required_genes: list[str] | None = None,
) -> None:
    """Write genetic splits with disjoint targets, samples and control cells.

    A sample is identified by its (context, batch) metadata pair. The caller
    must ensure those labels correspond to independent samples.
    """
    _validate_gene_selection(training, max_genes, required_genes)
    if outdir.exists() and any(outdir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty directory: {outdir}")
    validate_counts(training)
    validate_counts(heldout)
    if training.var_names.tolist() != heldout.var_names.tolist():
        raise ValueError("Training and heldout gene axes must match exactly")
    if "ensembl_ids" in training.var or "ensembl_ids" in heldout.var:
        if (
            "ensembl_ids" not in training.var or "ensembl_ids" not in heldout.var
            or training.var["ensembl_ids"].isna().any()
            or heldout.var["ensembl_ids"].isna().any()
            or training.var["ensembl_ids"].astype(str).tolist()
            != heldout.var["ensembl_ids"].astype(str).tolist()
        ):
            raise ValueError("Training and heldout Ensembl gene mappings differ")
    if training.uns.get("c3vcell_genome_assembly") != heldout.uns.get("c3vcell_genome_assembly"):
        raise ValueError("Training and heldout genome assemblies differ")
    check_cell_id_schemes(training.obs_names, heldout.obs_names)
    if set(training.obs_names.astype(str)) & set(heldout.obs_names.astype(str)):
        raise ValueError("Training and heldout cell identifiers overlap")

    labels: list[np.ndarray] = []
    sample_keys: list[list[tuple[str, str]]] = []
    for role, data in (("training", training), ("heldout", heldout)):
        for column in ("target_gene", "context", "batch"):
            if column not in data.obs or data.obs[column].isna().any():
                raise ValueError(f"{role} requires non-null {column!r} metadata")
            if data.obs[column].astype(str).str.strip().eq("").any():
                raise ValueError(f"{role} requires nonempty {column!r} metadata")
        role_labels = data.obs["target_gene"].astype(str).to_numpy()
        keys = list(zip(data.obs["context"].astype(str), data.obs["batch"].astype(str)))
        controls: dict[tuple[str, str], list[int]] = {}
        for row, (label, key) in enumerate(zip(role_labels, keys)):
            if label == control:
                controls.setdefault(key, []).append(row)
        minimum = 1 if role == "training" else 2
        missing = sorted(
            key for key in set(keys) if len(controls.get(key, [])) < minimum
        )
        if missing:
            raise ValueError(
                f"{role} sample groups need at least {minimum} control cells: {missing[:10]}"
            )
        labels.append(role_labels)
        sample_keys.append(keys)

    if set(sample_keys[0]) & set(sample_keys[1]):
        raise ValueError("Training and heldout sample groups overlap")
    train_targets = set(labels[0]) - {control}
    test_targets = set(labels[1]) - {control}
    if not train_targets or not test_targets:
        raise ValueError("Training and heldout data each require perturbed targets")
    if train_targets & test_targets:
        raise ValueError("Training and heldout targets overlap")
    missing_targets = sorted((train_targets | test_targets) - set(training.var_names.astype(str)))
    if missing_targets:
        raise ValueError(f"Target genes absent from the model gene axis: {missing_targets[:20]}")

    controls_by_sample: dict[tuple[str, str], list[int]] = {}
    for row, (label, key) in enumerate(zip(labels[1], sample_keys[1])):
        if label == control:
            controls_by_sample.setdefault(key, []).append(row)
    rng = np.random.default_rng(seed)
    input_rows: list[int] = []
    reference_rows: list[int] = []
    for key in sorted(controls_by_sample):
        controls = rng.permutation(controls_by_sample[key])
        n_input = len(controls) // 2
        input_rows.extend(controls[:n_input].tolist())
        reference_rows.extend(controls[n_input:].tolist())
    reference_mask = labels[1] != control
    reference_mask[reference_rows] = True
    reference = heldout[reference_mask].copy()
    inputs = heldout[np.sort(input_rows)].copy()
    manifest = {
        "method": "sample-and-target-disjoint-v1",
        "seed": seed,
        "control": control,
        "sample_columns": ["context", "batch"],
        "train_samples": [list(key) for key in sorted(set(sample_keys[0]))],
        "test_samples": [list(key) for key in sorted(set(sample_keys[1]))],
        "train_targets": sorted(train_targets),
        "test_targets": sorted(test_targets),
        "train_cells": training.n_obs,
        "test_cells": reference.n_obs,
        "train_controls": int(np.count_nonzero(labels[0] == control)),
        "input_controls": len(input_rows),
        "reference_controls": len(reference_rows),
        "train_cell_ids": training.obs_names.astype(str).tolist(),
        "test_cell_ids": reference.obs_names.astype(str).tolist(),
        "input_control_cell_ids": inputs.obs_names.astype(str).tolist(),
    }
    training, reference, inputs = _select_gene_axis(
        training, reference, inputs, manifest, max_genes=max_genes,
        required_genes=required_genes, require_targets_on_gene_axis=True,
    )
    outdir.mkdir(parents=True, exist_ok=True)
    _write_split_files(training, reference, inputs, manifest, outdir)
