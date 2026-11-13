# Tri-modal masked-RNA input audit

This is a synthetic audit of the published v13 prototype, not a model,
training run, real-data leakage finding, or production-ready masking API.
There is no known real training caller of `masked_rna_loss` in the checked-in
release. Do not modify the published prototype or treat the helper's name as
proof that its targets were hidden from the encoder.

The tests exercise the complete RNA/ATAC/contact input, joint encoder,
reconstruction and held-out loss path in two configurations: the untrained
RNA stand-in and a randomly initialized, synthetic exploratory foundation
checkpoint created in a temporary directory. They demonstrate that:

1. Passing a held-out RNA value to the encoder as measured changes its
   reconstruction when that value changes, while `masked_rna_loss` accepts
   both inputs.
2. Zeroing hidden counts before encoding and passing a visible mask makes the
   reconstruction and its hidden-input gradient invariant in the tested
   examples. The loss still depends on the hidden labels. This is a candidate
   input convention, not a validated trainer.
3. The lower-level `GeneEncoder` computes its denominator from all supplied
   counts before applying `visible`; a visibility mask alone therefore does
   not prevent hidden counts from affecting visible-gene normalization.

The candidate test keeps three disjoint states outside the encoder:
unmeasured, measured-and-hidden, and measured-and-visible. The existing RNA
encoder receives only values and a visibility mask, so it cannot distinguish
hidden from unmeasured internally. A future trainer must retain and audit all
three states, aligned gene order, independent mask selection, and its complete
data flow; the tests do not enforce that future callers use this convention.
Other modalities are held fixed in these examples, so correlated signals in
real paired ATAC or contacts are not assessed.

For the candidate inputs, raw hidden counts are removed **before** the
existing encoder sums the visible raw counts and applies log1p(CPM 10,000).
The original full-library depth is not sent to the encoder, including as an
auxiliary feature. In one synthetic loss example, full-library depth is used
only to construct detached supervised log1p(CPM 10,000) targets. A real
pipeline must verify the original depth independently, including genes
outside the selected panel, and freeze its target scale rather than infer
full depth from a projected panel. This input policy differs from the
original foundation training convention, which normalizes with the supplied
full count library before applying visibility; the distribution shift and
any alternative normalization require controlled evaluation.

No shared foundation has completed, and at least three eligible authorized
human same-cell tri-modal cell lines have not been established. These tests
do not start or change training, evaluate perturbations, or run cell-eval.

## Reproduce

Use Python 3.11 or 3.12 with the project's PyTorch dependency and run from the
repository root. The test creates and removes only a synthetic temporary
checkpoint; it does not read cells or model weights.

```sh
CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m unittest discover \
  -s releases/2026-11-13-trimodal-mask-audit-v1/tests -p 'test_*.py' -v
```

The test imports the immutable
`releases/2026-11-12-foundation-v13` source and its `src` directory. No
optimizer step is performed.
