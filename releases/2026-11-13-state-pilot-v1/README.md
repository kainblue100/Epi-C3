# Exploratory STATE pilot

This is a **single two-cell-line, one-split feasibility experiment**, not the
Cell Figure 2 benchmark or an evaluation of either new Epi-C3 model. See the
[Chinese report and six-panel figure](../../docs/benchmark/2026-11-13-v3/).
The author-saved results and original-paper interpretation are
[separate](../../docs/benchmark/2026-11-13-v2/).

## What was run

The scPerturb [Zenodo record 7041849](https://doi.org/10.5281/zenodo.7041849)
contains the Replogle K562 genome-wide and RPE1 CRISPRi files. Their SHA-256
hashes are recorded in the public JSON. The collection record declares
CC BY 4.0; original study terms also apply. Cite the collection and
[Replogle et al.](https://doi.org/10.1016/j.cell.2022.05.013).

Before reading expression values, the input preparation froze 40 of 565
eligible shared targets and selected 100 cells per target and context.
K562 contributes all 40 targets to training; RPE1 contributes 10 training,
2 validation, and 28 test targets. Thus the RPE1 test targets are observed
in K562 training and this is not a globally unseen-target benchmark.
Counts were divided by each original measured-library total, scaled to
10,000, and log1p-transformed. The 2,000 Seurat HVGs were selected only
from the 5,000 training treatment cells and 3,230 training input controls.

The three arms use the same test cells, feature axis, and frozen parent
assignments. The 280 RPE1 treatment-parent controls, 280 predicted-control
parents, and 280 independent real reference controls have disjoint
identities. A parent can recur across perturbations, though not within a
perturbation; 33 of the 2,800 treatment assignments cross batches. The
held-out treatment expression is replaced with the parent expression before
`predict_step`. The prediction and identity files are private and are not
distributed.

All three methods use official [STATE code at commit
`9bbfe78a434a55205e4de834e1ea99f85f7a3add`](https://github.com/ArcInstitute/state/tree/9bbfe78a434a55205e4de834e1ea99f85f7a3add).
STATE was randomly initialized with hidden dimension 64, one encoder and
one decoder layer, eight cells per set, and seed 42, then trained for
512 CPU updates. The official PerturbMean and ContextMean calculate training
statistics and each checkpoint records one global step. The prediction
manifest's `training_max_steps` is the configuration upper bound of 512, not
the mean baselines' actual step count; the public exporter reads and checks
the checkpoint's `global_step` separately.

The official prediction convention clips outputs to `[0, 14]`. PerturbMean
had 1,666,400 negative values clipped among 3,080 × 2,000 outputs (27.05%).
ContextMean returns one identical treatment vector within each of the 28
conditions; STATE outputs within a set are attention-coupled. Repeated
predictions are not independent biological replicates, and DE significance
must be interpreted accordingly. DE does not adjust for batch.

## Metrics and limitations

`cell-eval==0.8.2` and `pdex==0.3.0` evaluated Mann–Whitney/Wilcoxon rank
sum tests with BH FDR < 0.05, `is_log1p=True`, `geometric_mean=True`, and
`epsilon=0`. The first five panels are means over conditions finite for all
three models. Four conditions with no real significant genes are excluded
from AUPRC and overlap; two further conditions with only one real significant
gene are excluded from logFC Spearman. DE-count Spearman is recomputed from
all 28 real and predicted DE counts, including zeros. ContextMean's predicted
counts are constant, so its correlation is undefined.

The installed cell-eval L1 rank operates on mean *effects* and returns
`1 - rank / N`; it is not the paper's normalized PDS. Its Pearson delta
does not exclude the target gene. AUPRC is average precision against real
DE labels; overlap ranks significant predicted genes by absolute logFC.
With epsilon zero, 3 real and 704 STATE-predicted logFC entries were infinite
across the 28 × 2,000 gene-condition table and are ranked as extremes by the
current implementation. A shared-finite-gene sensitivity analysis excludes
16 gene-condition entries across seven conditions and is reported separately.

The independent audit recalculates the six metric families, verifies the BH
adjustments, spot-checks rank-sum p-values, and tests shared-finite logFC
sensitivity. It does not repair the limited sample, repeated-parent,
single-split, batch, or training-budget limitations. No confidence interval
or statistical superiority claim is justified.

The other paper baselines and both Epi-C3 branches are not evaluated here.
Potential overlap between the existing Epi-C3 foundation pretraining and
these test cells has not been audited. Do not mix this pilot with the
authors' scores or present it as a complete model comparison.

## Reproduce the workflow

Use the pinned STATE source and a separate environment. Observed versions
are recorded in `pilot-results.json`, including Python 3.12.3, arc-state
0.11.3, cell-load 0.10.4, cell-eval 0.8.2 and pdex 0.3.0; this is not a
complete lockfile. Matching package versions, backend and random seeds does
not guarantee bit-identical training. Source files are not included. The
scripts preserve the frozen plan and input identities locally rather than
publishing them; the inputs need over 201 GiB free because of the existing
shared-disk safety reserve.

With the two original source H5ADs already downloaded and verified, run
from this directory on a Linux host:

```bash
export D=/path/to/runs
export P="$D/state-replogle-pilot-v1"
export PY=/path/to/state-environment/bin/python
export STATE=/path/to/state-environment/bin/state
export K562=/path/to/ReplogleWeissman2022_K562_gwps.h5ad
export RPE1=/path/to/ReplogleWeissman2022_rpe1.h5ad
mkdir -p "$P"

"$PY" prepare-state-replogle-pilot-v1.py plan \
  --k562 "$K562" --rpe1 "$RPE1" --plan "$P/frozen-plan.json"
"$PY" prepare-state-replogle-pilot-v1.py prepare \
  --k562 "$K562" --rpe1 "$RPE1" --plan "$P/frozen-plan.json" --output "$P/inputs"
"$PY" convert-state-pilot-cellload-index-v1.py \
  --inputs "$P/inputs" --plan "$P/frozen-plan.json" --output "$P/inputs-cellload-v2"
"$PY" audit-state-replogle-pilot-splits-v1.py \
  --inputs "$P/inputs-cellload-v2" --plan "$P/frozen-plan.json" \
  --output "$P/split-audit-cellload-v2.json"
bash run-state-replogle-pilot-training-v2.sh
"$PY" freeze-state-pilot-parents-v1.py \
  --inputs "$P/inputs-cellload-v2" --plan "$P/frozen-plan.json" \
  --split-audit "$P/split-audit-cellload-v2.json" --output "$P/frozen-parent-plan-v1.json"
for model in state perturb_mean context_mean; do
  "$PY" predict-state-replogle-pilot-v1.py \
    --run-dir "$P/runs-v2/$model" --model "$model" --inputs "$P/inputs-cellload-v2" \
    --parent-plan "$P/frozen-parent-plan-v1.json" --output "$P/predictions-v1/$model"
done
"$PY" score-state-replogle-pilot-v1.py \
  --prediction-root "$P/predictions-v1" --parent-plan "$P/frozen-parent-plan-v1.json" \
  --output "$P/scores-v1"
"$PY" audit-state-pilot-metrics-v1.py \
  --scores "$P/scores-v1" --predictions "$P/predictions-v1" --output "$P/metric-audit-v1.json"
```

Only export the whitelisted aggregates, then render the figure:

```bash
mkdir -p "$P/public"
"$PY" export-state-pilot-aggregate-v1.py \
  --scores "$P/scores-v1/summary.json" --audit "$P/metric-audit-v1.json" \
  --plan "$P/frozen-plan.json" --runs "$P/runs-v2" --predictions "$P/predictions-v1" \
  --output "$P/public/pilot-results.json"
"$PY" render-state-pilot-v1.py --data "$P/public/pilot-results.json" --output-dir "$P/public"
"$PY" -m unittest -v test-export-render-state-pilot-v1.py \
  test-prepare-state-replogle-pilot-v1.py test-convert-state-pilot-cellload-index-v1.py \
  test-freeze-state-pilot-parents-v1.py test-predict-state-replogle-pilot-v1.py \
  test-score-state-replogle-pilot-v1.py
```

For the complete checked-in webpage, run `python validate-pilot-page.py
--page /path/to/2026-11-13-v3` and `node test-pilot-web.js
/path/to/2026-11-13-v3`.

The exported aggregate validates this exact pilot's source hashes, seed,
split, software scoring versions, actual steps and independent-audit hash.
If rerunning the pipeline changes the frozen diagnostics, audit the new
results as a new experiment rather than publishing them under this snapshot.
Intermediate outputs refuse to overwrite existing paths. The CellLoad index
conversion was required for the observed AnnData version and verifies that
all values and non-index HDF5 metadata remain unchanged. The training shell
is the executed recipe with only its private server paths parameterized;
release text files use normalized line endings.

STATE's source is under
[CC BY-NC-SA 4.0](https://github.com/ArcInstitute/state/blob/9bbfe78a434a55205e4de834e1ea99f85f7a3add/LICENSE);
the separate [STATE model license](https://github.com/ArcInstitute/state/blob/9bbfe78a434a55205e4de834e1ea99f85f7a3add/MODEL_LICENSE.md)
and acceptable-use policy may also apply. The user identified this use as
noncommercial. This release does not distribute STATE source, model weights,
raw cells, per-cell predictions, or identity/parent plans.

For the paper, cite Adduri et al., *Predicting cellular responses to
perturbation across diverse contexts with State*, Cell 189, 5914–5931
(2026), [doi:10.1016/j.cell.2026.07.052](https://doi.org/10.1016/j.cell.2026.07.052).
