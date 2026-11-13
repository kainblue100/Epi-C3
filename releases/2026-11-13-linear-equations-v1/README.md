# Independent Linear equation verifier

This is a **synthetic, independent equation check**, not the original
Figure 2 Linear implementation, a trained perturbation model, a benchmark
result, or an official cell-eval score. No real cells, embeddings, weights,
predictions or scores are included.

The supplied independent source and tests had SHA-256 values
`7c1119263fa8e79151c5830bd16375a634594cd7cc5286fcc2e3b8d387278a81` and
`dd6dfa8c9f8e52b24cbac924a4d5b4a34ef46e8eb17961f8a1e3717cce2928ab`.
This packaged version keeps their equations and original tests, but adds
input validation and regression tests: scalar strings cannot stand in for
identifier sequences, and complex, object, string or actually masked
matrices cannot be silently converted to ordinary real matrices.
`SHA256SUMS.txt` records the packaged file bytes; `.gitattributes` preserves
those bytes, including their line endings, in a fresh Git checkout.

## Two different estimators

Let `D` be genes by training perturbations, `G` be gene embeddings, `P` be
training perturbation embeddings, `b = mean(D, axis=perturbations)` and
`lambda > 0`. The modes are deliberately distinct:

```text
paper-formula:
    (G.T G + lambda I) W (P.T P + lambda I) = G.T D P

stated-objective:
    minimize ||D - G W P.T - b 1.T||_F^2 + lambda ||W||_F^2,
    with b fixed to the per-gene mean of D
```

Both use `G W P.T + b 1.T` for training-target offsets. The first uses
uncentered `D` in the displayed two-sided formula; the second centers `D`
by the fixed bias and solves the single-ridge normal equations using Gram
eigensystems. In general they are not equivalent, even if the bias is zero.
The tests compare against independently constructed small linear systems
and cover rank-deficient embeddings.

The [author's separate reproduction code at the fixed
revision](https://github.com/ArcInstitute/state-reproduce/blob/1a29ba1aa57ae996a57f77ec273e8547e824835b/baselines/state_sets_reproduce/models/low_rank_linear.py#L330-L346)
has another centering/bias behavior: when `center_Y` is enabled it centers
the responses before taking the bias mean, making that bias approximately
zero. This independent verifier does not claim to match that implementation.
The separate reproduction repository did not declare a root license at that
revision; no author source or embedding weights are included here.

## Boundaries before any real evaluation

`fit` expects already paired training treatment/control rows on the same
nonnegative, library-normalized log1p gene axis; pairing, control identity,
normalization, feature provenance, and split isolation are caller duties.
It aligns embeddings only by exact identifiers and excludes control rows
from perturbation means. `predict` accepts only training perturbation
labels or the control label; it **rejects unseen targets** and therefore
cannot evaluate a genuinely unseen-target split. It does not clip outputs.

Before any real run, independently select and freeze the estimator variant,
pairing and held-out split, embedding provenance and usage permission,
normalization, gene axis, output convention and comparison protocol.
Neither current synthetic output nor any future result should be mixed with
paper Figure 2 or the existing STATE pilot as if it were the same protocol.
The shared foundation remains a prerequisite for the requested new models;
this tool neither completes nor substitutes for that pretraining.

The separate STATE/scVI baseline requires raw counts: the existing log1p
pilot matrix must not be supplied as counts. The Replogle runner's scGPT
perturbation embedding and the paper text's GEARS embedding also require
independent configuration verification before any same-setup claim.

## Run synthetic tests

The verifier requires Python and NumPy; it does not install dependencies or
contact external services. From this directory:

```bash
sha256sum -c SHA256SUMS.txt
python -B -m unittest -v test_linear_baseline_equations_v1
```

The repository supports Python 3.11–3.12. Tests use synthetic matrices
only; passing them does not establish biological validity, fair splitting,
licensed model inputs, or official baseline equivalence.
