# Legacy K562 comparison

This version recomputes exploratory, post-hoc population-mean metrics for
previously saved Epi-C3 and adapted GEARS predictions on one frozen K562
test reference. It is not a STATE underrepresented-context benchmark,
official cell-eval score, new-model result, or independent biological
validation. No raw cells, private identities, trained weights, STATE code,
or STATE outputs are distributed here.

The original test outcomes had been examined before GEARS was added. GEARS
uses an additional transductive GO prior and one seed; three legacy Epi-C3
response seeds share one encoder. Controls and batches are reused. The two
new shared-foundation branches remain unfinished, and STATE, PerturbMean,
ContextMean, Linear, scVI, CPA, and scGPT architecture were not evaluated
on the shared protocol.

## Metrics

All metrics omit each target gene. MSE is averaged over genes and then
targets. Response Pearson correlates `(prediction - input control)` with
`(reference - reference control)` across genes and averages finite target
correlations. Constant vectors are undefined. Discrimination compares each
predicted effect to every reference effect by Euclidean distance, assigns
the mean zero-based rank of ties, and reports `1 - rank / target_count`.
It is a custom population-mean calculation inspired by, but not identical
to, cell-eval. Per-run macros are averaged with their support counts
reported; deterministic baselines have no training seed, and the no-effect
Pearson is undefined.

DE p-value AUPRC, log-fold-change Spearman, DE overlap, and DE-count
effect-size correlation require information not available in the saved
population means. They are intentionally not reported. The candidate graph
directly covers only two test targets; the shuffled-graph control is needed
when interpreting any difference.

## Reproduce with authorized private artifacts

The original private artifacts are not included. With their frozen directory
structure and NumPy installed, run in a separate output directory:

```text
python -m unittest -v test_compare_legacy_k562_means
python compare_legacy_k562_means.py --root <private-run-root> --output <new-summary.json>
python audit_legacy_k562_means.py --root <private-run-root> --summary <new-summary.json> --output <new-audit.json>
python render_legacy_comparison.py --comparison <new-summary.json> --output-dir <new-figure-directory>
```

The scorer checks frozen state and array hashes, test-plan identity, shared
gene/target/control axes, expected selection indices, and independently
audited historical MSEs. The second script recomputes aggregates with an
independent per-target implementation. It verifies arithmetic, not
experimental validity. The renderer additionally requires Matplotlib; all
outputs must be new paths.

Published [figure and evidence](../../docs/benchmark/2026-11-13-v1/).
Sources for the planned full comparison: [STATE](https://github.com/ArcInstitute/state),
[cell-eval](https://github.com/ArcInstitute/cell-eval),
and [GEARS](https://github.com/snap-stanford/GEARS). STATE code is
CC BY-NC-SA 4.0; its model license and acceptable-use policy apply separately.
The user confirmed that this project is noncommercial; applicable license
and attribution terms must still be followed when running STATE.
