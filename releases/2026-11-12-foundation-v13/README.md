# Epi-C3 shared RNA foundation · research snapshot v13

This is an independently developed research prototype, not the official STATE model.
It contains no patient-level data, source datasets, trained weights or credentials.
The accompanying dashboard is a timestamped, non-live research snapshot.

## What is included

- The existing `c3vcell` base-package source under `src/c3vcell`; no installed
  copy of the server workspace is needed to import it.
- Shared RNA gene alignment and measured-feature masks, interleaved representation
  training, and a Linux stage-one driver for Arc, curated Tabula and Orion.
- A new Tahoe streaming adapter that excludes frozen protected and validation
  identities, counts duplicate source rows correctly, and refuses incomplete or
  stale isolation audits. It has passed a real-data loader smoke test but has
  **not yet been used** for shared training. The older v1 adapter is retained
  for reference and rejects protected identities instead of skipping them.
- Two additional candidate control-only streams for the downloaded KHP and CMO
  matrices. Their full sparse count layers were audited and the streams were
  exhausted once for verification, without training the foundation. The KHP
  stream excludes every CD8T row, including frozen input and evaluation rows;
  the CMO stream holds out an entire donor. All noncontrol rows remain excluded.
  The 5,000-feature source panels may themselves have been selected using
  held-out cells. Eight ambiguous KHP gene symbols are dropped rather than
  assigned to potentially incorrect genes. The surviving features use the
  existing measurement mask; absent features are not observed zero expression.
  Within-file row protection does not prove cross-dataset identity, donor or
  chemical isolation. These streams are candidates only and must not be used
  to claim independent final-model evaluation until those checks are resolved.
  Row-index plans are private and are not included in this release.
- An experimental, resumable stage-two driver that gates on a completed,
  provenance-matched stage one and an audited Tahoe stream. It alternates one
  pass of eligible Tahoe rows with approximately as many replay exposures from
  the earlier sources; replay is allocated in proportion to their eligible
  sizes. Replay records are not new independent cells, and balanced row counts
  do not imply equal optimizer-step or gradient weights when batches differ.
  The driver has not been used for real stage-two training and its output is
  deliberately marked incomplete, not `full`.
- A read-only live preflight constructed the actual stage-two streams against
  the frozen stage-one plan and Tahoe isolation audit. It checked the first
  batch and exact cursor replay for each of the four earlier sources and
  Tahoe, without optimizer updates; it did not inspect every expression row.
  A single-instance dispatcher waits for the stage-one completion state and
  released trainer lock, guards selected source and code hashes, and requires
  204 GiB free before starting the existing stage-two driver. The driver
  independently validates the finished checkpoint before training. The
  dispatcher reports errors explicitly and never alters the stage-one run.
- Separate prototype heads for candidate regulatory graphs and jointly encoded,
  strictly same-cell RNA, accessibility and chromatin-contact data. Neither is a
  completed perturbation model. Their interfaces refuse incomplete foundations
  for final use; explicitly enabled exploratory use is not final validation.
  The GRN supervised loss now requires separate boolean control and treated
  measurement masks and scores only their non-target intersection. Every cell
  must have at least one such gene. The loss is averaged over eligible gene-cell
  entries, so cells with larger shared panels contribute more entries. The
  response head may emit values outside the control mask, but those values have
  no observed control baseline and must not be interpreted as measured effects.
- `epi_c3_trimodal_response_v1.py` adds an **untrained exploratory** gene
  perturbation response head to the joint encoder. It requires a frozen shared
  RNA encoder, a matching local gene axis, human cell records with the same
  declared assembly, all three modalities, and at least one observed contact
  per cell. It conditions on a gene target and inhibitory or activating
  intervention, and returns a measurement mask with log1p(CPM) predictions.
  The caller must independently verify the provenance and row ordering of the
  paired data: matching metadata and hashes alone do not prove biological
  pairing. The caller must separately validate at least three training cell
  lines across the collection; minibatch checks do not enforce that threshold.
  The `log1p(CPM)` denominator sums the local measured panel, not necessarily
  the full foundation axis. Checkpoint provenance and assay compatibility
  require independent verification; the checkpoint format does not itself
  bind species or assembly. An explicit opt-in is required because the
  joint encoder is not trained. No real-data model or performance claim follows.
- The new target-excluded population-mean loss accepts different control and
  treated cell counts instead of inventing one-to-one pairs. It requires the
  same measured gene panel throughout a comparison; the caller must separately
  verify the same panel normalization, cell line, condition and data split.
  Unmeasured genes are excluded, not interpreted as zero. CPM normalization is
  compositional, so exclusion of the perturbed gene from the score does not
  remove its effect on the library denominator. The mean loss is not a
  distributional metric or a substitute for official cell-eval.
- Scripts for arithmetic auditing of a prior population-level GRN experiment and
  for preparing and evaluating a prior Tahoe pilot with the separately installed
  `cell-eval` package. These are post-hoc experiments, not official leaderboard
  scores or evaluations of the new foundation model.
- An independent recomputation of the full validation strength grid and all
  14,574 saved test target-level GRN scores, including both direct graph-source
  targets and their candidate-neighbor metrics. It operates on saved group means,
  not raw cells; the results do not constitute an independent biological test.
  The original frozen inputs are required to rerun it and are not distributed.
- A read-only audit compares the original encoder's full-library normalization
  with visible-only normalization for the saved stage-one checkpoint. The
  original training and masked diagnostics use total counts that include masked
  genes, so the hidden portion contributes aggregate library information.
  Visible-only inference is an input ablation, not retraining, and changes the
  encoder's input distribution. The audit requires the original, hash-matched
  server stage-one driver and source datasets, which are not distributed.
- Separate scripts freeze known historical Tahoe evaluation identities, audit all
  source shards, independently verify overlap counts, and exercise the v2 loader.
  Their private identity files, source data and evaluation artifacts are not
  distributed. The checks cover the explicitly inventoried artifacts, not every
  possible benchmark or cross-dataset duplicate.
- A streaming CHARM mESC source audit reports direct RNA, QC, accessibility
  and optional contact-file identifier intersections without publishing private
  identity maps. The completed audit found 805 matching RNA/QC/ATAC/contact
  identifiers in one mouse mESC sample. The outer contact archive and all 958
  nested gzip streams passed integrity checks. Local hashes bind the downloaded
  files and are not independent source-supplied checksums. Fractional RNA values
  must not be rounded into integer-count inputs. The ATAC versus QC count
  conventions, the disagreement between the reported QC histone label and source
  description, and mouse-to-human gene-axis compatibility require clarification. Identity
  agreement alone does not establish training readiness.
- `audit-charm-inner-gzip-v1.py` verifies all nested contact gzip streams by
  reading them to EOF, binds the archive to the frozen pairing-audit SHA-256,
  and refuses unexpected archive members. A full-stream run verified all 958
  contact files, and its aggregate report is in the accompanying dashboard.
  Gzip integrity does not validate the semantics or coordinates of every contact
  or independently establish biological pairing.
- `audit-charm-contacts-coordinates-v1.py` streams the archived contact files,
  checks the reported chromosome lengths against a hash-pinned mm10 reference,
  validates the documented nine-column schema and tests mapped coordinates
  under the 1-based pairs convention. It separately aggregates all contacts
  and the previously matched identities without publishing those identities.
  The full run completed with exit status 0: all 958 files and 1,607,375,529
  records were checked, including 1,406,651,183 records from the 805 matched
  identities; the implemented schema and coordinate checks found no issue.
  Two distinct chromosome declaration sets were observed. A bounded inspection
  of 51 headers found both sets, with or without chrY; their full frequency and
  the reason for the difference were not determined. The header probe is
  `audit-charm-contact-header-probe-v1.py`; it inherits the archive hash from
  the full audit rather than rehashing the archive. Matching chromosome lengths
  do not prove reference-sequence identity, and the producer's precise
  coordinate implementation remains unverified. This audit does not establish
  biological pairing, contact uniqueness, mapping quality or assay completeness.
- `audit-k562-reference-network-v1.py` compares the downloaded RN119 candidate
  network with a hash-pinned K562 graph and evaluation plan. It reports exact,
  case-sensitive gene-symbol overlaps without distributing the candidate source
  network. The candidate has 2,656 directed pairs with opposite activation and
  repression labels, including 70 of 157 pairs whose endpoints are both on the
  frozen 1,861-gene axis. The candidate is not used for training or evaluation;
  its original experimental sources, licensing and benchmark overlap remain
  unverified. Opposite labels are not collapsed to a single sign.

The strictly paired branch still lacks verified data from at least three cell
lines. GSE267117 cell-line RNA and DNA are explicitly unpaired and are excluded.
Candidate bulk-derived graph edges are not validated causal regulation.
SC-MO-GRN-DB's DS010 and DS011 list RNA, accessibility and contacts together,
but their primary publications respectively describe parallel assays and
RNA/ATAC multiome with bulk Hi-C. Neither is evidence of three assays measured
in the same cell. The database's K562 reference networks are candidates only;
their original sources, usage rights and evaluation independence need review.

## Running and limitations

Use Python 3.11 or 3.12 with the required scientific dependencies
(`torch`, `numpy`, `scipy`, `pandas`,
`h5py`, `pyarrow` and `anndata`). `pytest` is required for the tests. The optional
Tahoe evaluation requires separately installed `cell-eval` and `pdex`; check all
dataset, model and dependency licenses independently. This snapshot does not
include a dependency lockfile or grant rights to the external datasets.

From this directory, the synthetic tests can be run with:

```sh
PYTHONPATH="$PWD:$PWD/src${PYTHONPATH:+:$PYTHONPATH}" python -m pytest -q tests
```

With the same `PYTHONPATH`, the stage-one driver provides
`python train-rna-foundation-stage1-v1.py --help`.
Its required `--data-root` points to a separately prepared local data tree with
the source datasets, metadata scans, frozen target splits and protected reference
identities expected by the code. These are not distributed here; this is not a
one-command dataset download or a ready-to-reproduce benchmark. The driver
enforces its 200 GiB free-space reserve and verifies source and scanner hashes.

The public driver accepts a configurable data root and loads its helpers beside
the script. Consequently its source hash differs from the original in-progress
server job. **Do not use this export to resume or modify that existing run.**
Any new run requires compatible, newly verified metadata plans and a separate
output directory. No training should start until evaluation-cell exclusions,
data permissions and disk headroom have been checked.

The stage-two driver provides
`python train-rna-foundation-stage2-v1.py --help`. It checks that stage one
finished, that all planned training rows and provenance match, and that its
saved diagnostics are finite and not completely collapsed. It also checks
the Tahoe isolation audit before building the fresh stream. These checks are
necessary but do not establish representation quality, prevent forgetting, or
substitute for independent biological evaluation. No real stage-two run has
started. `smoke-balanced-replay-live-stage1-v1.py` records the exact script
used to check two batches from each running stage-one source; it requires the
original, hash-matched server driver rather than this public stage-one export.
`check-rna-foundation-stage2-preflight-v1.py --help` describes the read-only
first-batch preflight, and `wait-rna-foundation-stage2-v1.py --help` describes
the Linux-only dispatcher. They require the original, hash-matched private
stage-one inputs and a compatible preflight report. The optional
`launch-shared-foundation-stage2-v1.sh DATA_ROOT RUN_ROOT RELEASE_DIR` wrapper
sets conservative CPU and memory limits and writes a private dispatcher log.
It is a site-specific research convenience, not a general installation command;
it must be launched at most once and must not overwrite an existing run.
Failure of a gate requires investigation rather than bypassing it. The preflight
and dispatcher do not prove cross-source independence or predictive accuracy.

On the current source snapshot, the completed isolation audit found 5,027
historical evaluation rows eligible under the old Tahoe filter and 32 eligible
validation rows; the future v2 stream has 92,090,669 remaining eligible rows.
The nine-condition interim cell-eval reference and input-control identities
had zero overlap with the old filter. These are eligibility findings, not proof
that a particular older checkpoint consumed or excluded all historical rows.
The existing Tahoe job and its checkpoints were not changed.

The Tahoe scripts require the original, hash-matched metadata plans, source
shards, historical evaluation artifacts, `c3vcell` code and software versions.
The frozen audit and source-code hashes intentionally make this export
incompatible with resuming an existing run. Protected sample/barcode mappings
must remain private; do not publish them with the aggregate audit reports.

`prepare-scperturbbench-foundation-v1.py --help` describes the source-audit
arguments; it requires the original verified source tree, frozen foundation
plan and KHP evaluation split. `verify-scperturbbench-foundation-v1.py --help`
describes the independent projection, complete-stream and read-only compatibility
checks. The verifier uses the existing 14,500-update exploratory checkpoint and
performs no optimizer updates; a finite loss is not a performance result.

The source-specific measurement masks do not establish that every annotated
feature was measured in every cell. Across-dataset duplication and all benchmark
holdouts are not yet exhaustively verified. The current first stage excludes
Tahoe training records and other eligible expression sources, so it is not a
complete foundation model. Three flow seeds sharing one encoder are not
independent biological replicates.
