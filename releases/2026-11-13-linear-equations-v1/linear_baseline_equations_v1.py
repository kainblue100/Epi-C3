"""Independent implementations of the two linear-baseline equations in STATE.

The paper's displayed two-sided formula and its stated ridge objective are
different estimators. Neither implementation asserts equivalence to the
authors' released code or to the Figure 2 benchmark. Inputs must already be
on the same frozen, library-normalized log1p expression axis.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


PAPER_FORMULA = "paper-formula"
STATED_OBJECTIVE = "stated-objective"


def _labels(values: Sequence[str], name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be a sequence, not a scalar string")
    try:
        result = tuple(values)
    except TypeError as exc:
        raise ValueError(f"{name} must be a sequence of strings") from exc
    if not result or any(
        not isinstance(value, str) or not value or value != value.strip()
        for value in result
    ):
        raise ValueError(f"{name} must contain nonempty, unpadded strings")
    return result


def _axis(values: Sequence[str], name: str) -> tuple[str, ...]:
    result = _labels(values, name)
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must be unique")
    return result


def _contains_masked_values(values) -> bool:
    pending = [values]
    seen = set()
    while pending:
        item = pending.pop()
        if isinstance(item, np.ma.MaskedArray) and np.ma.is_masked(item):
            return True
        if isinstance(item, (list, tuple)) and id(item) not in seen:
            seen.add(id(item))
            pending.extend(item)
    return False


def _matrix(values, name: str) -> np.ndarray:
    if _contains_masked_values(values):
        raise ValueError(f"{name} must not contain masked values")
    raw = np.asarray(values)
    if raw.dtype.kind not in "biuf":
        raise ValueError(f"{name} must contain real numeric values")
    result = np.asarray(raw, dtype=np.float64)
    if result.ndim != 2 or not all(result.shape) or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a nonempty finite matrix")
    return result


def _aligned_embeddings(
    wanted: Sequence[str], available: Sequence[str], values, name: str
) -> np.ndarray:
    source = _axis(available, f"{name} identifiers")
    matrix = _matrix(values, name)
    if len(source) != matrix.shape[0]:
        raise ValueError(f"{name} identifiers and rows must agree")
    lookup = {value: index for index, value in enumerate(source)}
    missing = [value for value in wanted if value not in lookup]
    if missing:
        raise ValueError(f"{name} is missing {len(missing)} required identifiers")
    return matrix[[lookup[value] for value in wanted]].copy()


def _positive_gram_eigensystem(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    eigenvalues, eigenvectors = np.linalg.eigh(matrix.T @ matrix)
    tolerance = 1e-10 * max(1.0, float(np.max(np.abs(eigenvalues))))
    if np.min(eigenvalues) < -tolerance:
        raise ValueError("A Gram matrix is unexpectedly non-positive-semidefinite")
    return np.maximum(eigenvalues, 0.0), eigenvectors


def solve_weights(
    changes, gene_embeddings, perturbation_embeddings, ridge: float, variant: str
) -> tuple[np.ndarray, np.ndarray]:
    """Return coefficients and the paper-defined per-gene mean intercept.

    ``changes`` has one row per gene and one column per training perturbation.
    ``paper-formula`` follows the displayed formula using uncentered changes.
    ``stated-objective`` minimizes the displayed objective with that fixed
    intercept. The variants do not claim to reproduce the released implementation.
    """
    changes = _matrix(changes, "changes")
    genes = _matrix(gene_embeddings, "gene embeddings")
    perts = _matrix(perturbation_embeddings, "perturbation embeddings")
    if changes.shape != (genes.shape[0], perts.shape[0]):
        raise ValueError("Change and embedding axes do not match")
    if not np.isfinite(ridge) or ridge <= 0:
        raise ValueError("ridge must be finite and strictly positive")
    bias = changes.mean(axis=1)
    if variant == PAPER_FORMULA:
        left = genes.T @ genes + ridge * np.eye(genes.shape[1])
        right = perts.T @ perts + ridge * np.eye(perts.shape[1])
        middle = np.linalg.solve(left, genes.T @ changes @ perts)
        weights = np.linalg.solve(right, middle.T).T
    elif variant == STATED_OBJECTIVE:
        gene_values, gene_vectors = _positive_gram_eigensystem(genes)
        pert_values, pert_vectors = _positive_gram_eigensystem(perts)
        centered = changes - bias[:, None]
        right_hand_side = genes.T @ centered @ perts
        rotated = gene_vectors.T @ right_hand_side @ pert_vectors
        denominator = np.outer(gene_values, pert_values) + ridge
        weights = gene_vectors @ (rotated / denominator) @ pert_vectors.T
        residual = (
            (genes.T @ genes) @ weights @ (perts.T @ perts)
            + ridge * weights
            - right_hand_side
        )
        scale = max(1.0, float(np.linalg.norm(right_hand_side)))
        if np.linalg.norm(residual) > 1e-8 * scale:
            raise ArithmeticError("The ridge solution failed its normal-equation check")
    else:
        raise ValueError(f"Unknown estimator variant: {variant}")
    if not np.isfinite(weights).all() or not np.isfinite(bias).all():
        raise ArithmeticError("The fitted coefficients are nonfinite")
    return weights, bias


@dataclass(frozen=True)
class LinearBaseline:
    genes: tuple[str, ...]
    perturbations: tuple[str, ...]
    control_label: str
    offsets: np.ndarray
    training_counts: tuple[int, ...]
    ridge: float
    variant: str

    def predict(
        self, controls, gene_ids: Sequence[str], perturbation_labels: Sequence[str]
    ) -> np.ndarray:
        if _axis(gene_ids, "prediction gene axis") != self.genes:
            raise ValueError("Prediction gene axis differs from the frozen training axis")
        controls = _matrix(controls, "prediction controls")
        labels = _labels(perturbation_labels, "prediction perturbation labels")
        if controls.shape != (len(labels), len(self.genes)):
            raise ValueError("Prediction controls, labels, and gene axis do not match")
        if np.any(controls < 0):
            raise ValueError("Prediction controls must be nonnegative log1p expression")
        lookup = {label: index for index, label in enumerate(self.perturbations)}
        result = controls.copy()
        for index, label in enumerate(labels):
            if label == self.control_label:
                continue
            if label not in lookup:
                raise ValueError(f"Unseen perturbation at prediction row {index}")
            result[index] += self.offsets[lookup[label]]
        if not np.isfinite(result).all():
            raise ArithmeticError("Predictions contain nonfinite values")
        return result


def fit(
    perturbed,
    matched_controls,
    gene_ids: Sequence[str],
    labels: Sequence[str],
    gene_embedding_ids: Sequence[str],
    gene_embeddings,
    perturbation_embedding_ids: Sequence[str],
    perturbation_embeddings,
    *,
    control_label: str,
    ridge: float,
    variant: str,
) -> LinearBaseline:
    """Fit on paired training rows; pairing and split provenance are caller duties.

    This function does not normalize or clip expression, infer gene aliases, or
    inspect test data. The caller must freeze training/control identity, original
    library-depth normalization, embeddings, feature axes, and the variant before
    evaluation. Controls in ``labels`` are excluded from the perturbation means.
    """
    genes = _axis(gene_ids, "training gene axis")
    if not isinstance(control_label, str) or not control_label.strip():
        raise ValueError("control_label must be a nonempty string")
    if control_label != control_label.strip():
        raise ValueError("control_label must not be padded")
    treated = _matrix(perturbed, "training perturbed expression")
    controls = _matrix(matched_controls, "matched training controls")
    labels = _labels(labels, "training labels")
    if treated.shape != controls.shape or treated.shape != (len(labels), len(genes)):
        raise ValueError("Training rows, matched controls, labels, and genes do not match")
    if np.any(treated < 0) or np.any(controls < 0):
        raise ValueError("Training expression must be nonnegative log1p values")
    targets = tuple(sorted(set(labels) - {control_label}))
    if not targets:
        raise ValueError("At least one training perturbation is required")
    gene_matrix = _aligned_embeddings(
        genes, gene_embedding_ids, gene_embeddings, "gene embeddings"
    )
    pert_matrix = _aligned_embeddings(
        targets, perturbation_embedding_ids, perturbation_embeddings,
        "perturbation embeddings",
    )
    labels_array = np.asarray(labels)
    changes = np.empty((len(genes), len(targets)), dtype=np.float64)
    counts = []
    for column, target in enumerate(targets):
        selected = labels_array == target
        counts.append(int(selected.sum()))
        changes[:, column] = (treated[selected] - controls[selected]).mean(axis=0)
    weights, bias = solve_weights(changes, gene_matrix, pert_matrix, ridge, variant)
    offsets = (gene_matrix @ weights @ pert_matrix.T).T + bias
    offsets.setflags(write=False)
    return LinearBaseline(genes, targets, control_label, offsets, tuple(counts), ridge, variant)
