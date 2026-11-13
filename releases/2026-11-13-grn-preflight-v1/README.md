# GRN / STATE-pilot preflight

This is a post-hoc compatibility and leakage audit, **not a trained model,
held-out benchmark, or official cell-eval result**. The aggregate-only
[report](../../docs/benchmark/2026-11-13-grn-preflight-v1/) is a snapshot;
the separate [ENCODE candidate-prior release](../2026-11-13-grn-encode-v1/)
does not establish graph efficacy.

## What was checked

The frozen pilot has 40 K562 training targets and 10/2/28 RPE1
training/validation/test targets. Consequently, every RPE1 test target also
appears in K562 training: the experiment is cross-context, not globally
unseen-target. Of its 2,000 training-selected HVGs, 1,997 have exact Ensembl
matches on the foundation axis; only 5 of the 28 RPE1 test targets are on
the HVG axis. Of 7,094 shared features, 7,090 have exact foundation-axis
matches. Exact IDs, not symbol-only fallback, define these matches.

The legacy K562 candidate graph contains 4,141 real and 4,141 shuffled
edges on its original axis, but no direct source edges from any of the 40
pilot targets. Projecting it to the pilot HVGs yields 626 real and 650
shuffled edges, with different in-degrees at 163 nodes and out-degrees at
7 nodes. Among all 565 shared single-gene candidates with at least 100 cells
in **each** source context, only one sources a legacy edge, and that source
was in the legacy training split. These are metadata feasibility results,
not a prospective split.

Of the pilot's 4,000 K562 training-treatment inputs, 426, 189, and 128
source rows also occur in the legacy graph's training, validation, and test
treatment inputs, respectively. Row membership does not establish that each
row was actually consumed by an optimizer, but old pilot and graph scores
cannot be combined as an independent comparison.

The pilot stores original-library normalized log1p values; the planned
foundation path expects counts and performs its own normalization. For the
first 64 RPE1 training controls in stored order, the HVG panel accounts for
33.84% of the original library on average. Re-normalizing the panel rather
than the original library changes the mean absolute value by 0.335014;
mistaking log1p values for counts changes it by 0.281491 relative to correct
panel normalization. These are input diagnostics, not model scores. No
held-out treatment expression was read for this diagnostic.

At the 2026-11-13 snapshot, foundation stage one exited on a count or
one-to-one mapping validation error after 106,750 updates; its last persisted
state was stale and still said `running`. Stage two had not started. A
separate audit found that **if** the original Tahoe continuation were fully
consumed, at least 7,562,326 paper-test perturbation *row occurrences* would
be exposed; this is not a count of distinct cells. Candidate five-context
isolation had not been connected to that continuation, and stage-one and
replay cross-source exposure remained unexcluded. Target overlap alone does
not prove identity-level overlap. Recovery, a fair checkpoint, and a
leakage-controlled common split remain prerequisites for model training.

## Frozen evidence

The JSON files contain counts and hashes only; they do not contain raw cell
data, identities, or graph edges.

| Artifact | SHA-256 |
| --- | --- |
| `audit.json` | `a30750c88e4070f1e17a925c3825a7757f20e811f73594afee5c6d6e013763d8` |
| `eligibility.json` | `453456f3dbe4d6753464132c97da9134f4eced5a042e8a170060738d47b0df67` |
| Frozen pilot plan | `eb12e237bc459d24630c7399099393a17fe9bde2616b59508c78765790b6dbe3` |
| Stage-one foundation plan | `77192b5b29c5b096b1b64dd17c3ec9d989acfba41ca97f4c5b7aebbcaf8550e9` |
| Legacy graph plan | `e22d57934f3f661361f266b350906eecaef875521f1686d1cd9ee1d9ea5d40da` |
| `audit_grn_pilot.py` | `ed9ad36c80a35c0a195e40ea64063645f0d4af5b3db99beb1a81c1d68e36a47a` |
| `audit_graph_eligibility.py` | `5eccf4f598048e7e65e135d9e279a4b447a9cc102949a14686412e828bd69854` |

`audit.json` verifies the pilot input and graph artifact hashes and compares
source hashes recorded in the plans; it does not rehash the original source
H5AD files. `eligibility.json` independently rehashes **both** original
sources and reconstructs the frozen pilot's target selection and shared
axis before counting graph coverage.

## Reproduce

Use the authorized private frozen runs, original Replogle source files and
the exact pilot preparation script. The observed environment was Python
3.12.3, NumPy 2.5.4 and h5py 3.16.0; it is not a complete lockfile. The
auditors require more than 201 GiB free on the output filesystem, refuse
existing outputs, and may read large source files to check their hashes.
Run on a Linux host with at most one CPU thread and an approximately 12-GiB
address-space limit, for example:

```bash
cd /path/to/checkout/releases/2026-11-13-grn-preflight-v1
export ROOT=/path/to/private/frozen-runs
export K562=/path/to/ReplogleWeissman2022_K562_gwps.h5ad
export RPE1=/path/to/ReplogleWeissman2022_rpe1.h5ad
export PREP=/path/to/prepare-state-replogle-pilot-v1.py
export PY=/path/to/state-environment/bin/python
export OUT=/path/to/new-private-preflight-directory
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
umask 077
mkdir -m 700 -- "$OUT"
ulimit -v 12582912
nice -n 19 ionice -c2 -n7 "$PY" -B audit_grn_pilot.py \
  --root "$ROOT" --output "$OUT/audit.json"
nice -n 19 ionice -c2 -n7 "$PY" -B audit_graph_eligibility.py \
  --root "$ROOT" --k562 "$K562" --rpe1 "$RPE1" --preparation "$PREP" \
  --output "$OUT/eligibility.json"
sha256sum "$OUT/audit.json" "$OUT/eligibility.json"
```

The hashes above match the recorded environment. Software-version fields
can change the aggregate bytes in another environment; investigate any
differences rather than treating a new run as the same snapshot. The
publisher requires the pinned aggregate hashes and a **new** output path:

```bash
"$PY" -B render_grn_preflight.py --output /path/to/new-index.html
"$PY" -B -m unittest -v test_audit_grn_pilot test_audit_graph_eligibility test_render_grn_preflight
```

The original Replogle data are from [scPerturb / Zenodo
7041849](https://doi.org/10.5281/zenodo.7041849); cite the collection and
[Replogle et al.](https://doi.org/10.1016/j.cell.2022.05.013), and respect
the source terms. This release does not distribute source cells, per-cell
records, or pretrained weights.
