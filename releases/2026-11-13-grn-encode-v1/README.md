# ENCODE K562 promoter graph: candidate prior

This release freezes an **untrained, post-hoc graph candidate**, three
degree-matched structural controls and aggregate audits. It is not a
perturbation predictor, a causal network, a blind benchmark, or an official
cell-eval result. The [Chinese report](../../docs/benchmark/2026-11-13-grn-encode-v1/)
and [pilot preflight](../2026-11-13-grn-preflight-v1/) describe the scope and
limitations. Full graph edges, original cell records, cell identities,
and raw metadata are not distributed here.

## Frozen inputs and construction

On 2026-11-13 the serial, low-rate official ENCODE queries froze 1,600
released GRCh38 optimal/conservative IDR peak files and 430 K562 experiments
with the exact untreated biosample summary and `perturbed=false`. These
are separate catalogs, not 1,600 files belonging to 430 experiments. Of
565 metadata-eligible shared K562/RPE1 single-gene perturbation targets,
32 have a label in the full file catalog. Requiring exact source
symbol/Ensembl identity, released K562 experiment metadata, no recorded
replicate treatment or genetic modification, at least two verified
biological replicates, and one preferred file per experiment leaves
25 sources, 35 experiments and 35 conservative IDR files, totaling
5,438,114 bytes. ENCODE target labels may include cofactors, not only
validated transcription factors.

Each selected file was checked against official size and MD5, an independently
recorded SHA-256, full gzip CRC and narrowPeak row structure. Exclude the
47 ChIP rows whose summit lies outside the half-open peak interval.
The separately checked ATAC source is ENCODE experiment `ENCSR868FGK`,
file [`ENCFF855PCP`](https://www.encodeproject.org/files/ENCFF855PCP/),
with three biological replicates and 172,732 peaks. ChIP and ATAC are
unpaired bulk K562 experiments. The report's independent source manifest
and verified-file manifest contain each exact experiment/file accession,
producing lab, and checksums; retain them in authorized storage. Public
metadata can contain signed download URLs and must be inspected before
sharing.

Use GENCODE v44 GRCh38 primary chromosomes and exact shared symbol/Ensembl
mapping, stripping only GENCODE ID version suffixes. Of 7,090 pilot features
also present on the foundation axis, 6,939 have unambiguous primary-locus
transcripts. For each transcript TSS, construct the 0-based half-open
window `[max(0, TSS - 1000), TSS + 1001)`. A source → target candidate is
created only when the source's ChIP peak, an ATAC peak and that target's
promoter window have an actual three-way overlap. Remove self-edges,
exclude non-primary ChIP rows and do not assign regulatory sign.

The resulting graph has **51,031 edges**, 25 source genes and 6,846 target
genes. Of these edges, 16,899 have support from multiple distinct ENCODE
experiments involving 8 sources; distinct experiments do not necessarily
mean independent biological cultures. An independent implementation
checked all graph/control structure and re-evaluated three-way overlap for
50 sampled edges across all 25 sources: 50 passed. This is a spot check,
not independent recomputation of every edge.

The 20261113, 20261114 and 20261115 controls each perform 255,155 successful
degree-preserving double-edge swaps. They replace 17,913 (35.10%), 18,027
(35.33%) and 17,935 (35.15%) real edges, respectively, and exactly preserve
each node's directed in- and out-degree. The source-degree-only upper bound
with self-edges forbidden is 61.80%; satisfying both degree sequences may
lower the achievable maximum. Finite swap chains do not guarantee uniform
sampling or convergence, and the controls are not biological replicates.

## Scope and leakage limitations

The graph was selected after the pilot targets and prior graph results were
known; it is not a blinded candidate. Pilot RPE1 train/validation/test
partitions have only 0/0/1 targets with direct source edges. All 40 pilot
targets appear in K562 training, so RPE1 test is cross-context rather than
globally unseen-target. A K562 graph used for RPE1 is cross-cell-line
transfer. Metadata indicating no perturbation does not prove biological
independence from benchmark cultures, and promoter binding/accessibility
does not prove a causal regulatory interaction.

At this snapshot, shared RNA foundation stage one had exited on validation
and stage two had not started. The proposed unisolated Tahoe continuation,
if fully consumed, would expose at least 7,562,326 paper-test perturbation
row occurrences; stage-one and replay cross-source exposure are also
unexcluded. No new perturbation model was trained, no official cell-eval
analysis was run on this graph, and the old real/shuffled graph MSEs
0.015187105/0.015193985 do **not** establish a graph effect. Before any
model claim, finish and audit the shared foundation, choose a genuinely
independent common split and compatible expression axis, and compare
against RNA-only, no-effect, shuffled-graph, and STATE or same-protocol
predictions using official cell-eval where applicable.

## Frozen hashes

| Artifact | SHA-256 |
| --- | --- |
| ENCODE metadata snapshot | `f9ccb328805da7a6ba127f772dc1ca086a2807ac7f4bd073281929c2885c83cf` |
| Selected source manifest | `29c279b55674873f88eeae62545f89aeeb748387bee23270de78daa04c1b85b4` |
| Verified ChIP manifest | `0dde35a9d78a832f3ab6227517d6c3e60f679f1a9ddce3710fb1c25c8f7a6dad` |
| Verified ATAC source report | `c05143058178703c5ea72ac9604ce5c9cd9ca099b5a8332ebad5936efe69dfd0` |
| ATAC peak file | `47a71eaac0bfbad586a14d89971398c8922bf8864b5eeab4cb58d37ae925e0e9` |
| GENCODE v44 GTF | `01f817afed65feee863361b4baf30a95e722ee5c5d508ff77b04106ef7ba20d3` |
| `selection-summary.json` | `74ff4e0e8be77a4cef81486d6e52a6d47031b8636c605c05c2205a03124098b0` |
| `graph-summary.json` | `06538583dcbef58718e9a80dc207f5a38633ae91466481d6bdfabc7435d94382` |
| `controls-summary.json` | `cba775e15d6416876aa9791d1cc56470e842947ea9b9ce528bcfe09922341c23` |
| `graph-v3-independent-audit.json` | `407bc1385437981ee8bc2906586860efd209935a4d380950f7ae5cfa68f71b67` |

The aggregate JSON also pins source H5ADs, frozen plans, scripts and every
graph/control output hash. Frozen catalog and source files must be obtained
from authorized storage. Re-querying dynamic ENCODE metadata creates a
**new** snapshot and is not an exact reproduction; do not silently replace
the original inputs.

## Reproduce

Use the pinned manifests, the 35 verified ChIP files, the ATAC peak, GENCODE
GTF and the frozen pilot/foundation plans from authorized storage. The
observed environment was Linux/Python 3.12.3 with NumPy 2.5.4 and h5py
3.16.0 available to the companion preflight modules. Intermediate outputs
are private, require more than 201 GiB free on the output filesystem and
refuse overwrites. For the original manifests' layout, run:

```bash
cd /path/to/checkout/releases/2026-11-13-grn-encode-v1
export ROOT=/path/to/private/frozen-runs
export PRIOR=/path/to/private/encode-artifacts
export ATAC=/path/to/ENCFF855PCP.bed.gz
export GTF=/path/to/gencode.v44.annotation.gtf.gz
export PY=/path/to/state-environment/bin/python
export OUT=/path/to/new-private-graph-directory
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
umask 077
mkdir -m 700 -- "$OUT"
ulimit -v 12582912
nice -n 19 ionice -c2 -n7 "$PY" -B build_encode_promoter_graph.py \
  --root "$ROOT" --selection "$PRIOR/selection-v2/selected-files.json" \
  --selection-sha256 29c279b55674873f88eeae62545f89aeeb748387bee23270de78daa04c1b85b4 \
  --peaks-dir "$PRIOR/verified-peaks-v2" \
  --verified-peaks-sha256 0dde35a9d78a832f3ab6227517d6c3e60f679f1a9ddce3710fb1c25c8f7a6dad \
  --atac "$ATAC" --atac-source "$PRIOR/verified-atac-v1/atac-source.json" \
  --atac-source-sha256 c05143058178703c5ea72ac9604ce5c9cd9ca099b5a8332ebad5936efe69dfd0 \
  --gencode "$GTF" --output-dir "$OUT/graph"
nice -n 19 ionice -c2 -n7 "$PY" -B build_degree_controls.py \
  --graph-dir "$OUT/graph" \
  --summary-sha256 06538583dcbef58718e9a80dc207f5a38633ae91466481d6bdfabc7435d94382 \
  --output-dir "$OUT/controls"
nice -n 19 ionice -c2 -n7 "$PY" -B audit_encode_graph.py \
  --root "$ROOT" --graph-dir "$OUT/graph" \
  --summary-sha256 06538583dcbef58718e9a80dc207f5a38633ae91466481d6bdfabc7435d94382 \
  --peaks-dir "$PRIOR/verified-peaks-v2" --atac "$ATAC" --gencode "$GTF" \
  --controls-dir "$OUT/controls" \
  --controls-summary-sha256 cba775e15d6416876aa9791d1cc56470e842947ea9b9ce528bcfe09922341c23 \
  --output "$OUT/independent-audit.json"
sha256sum "$OUT/graph/graph-summary.json" "$OUT/controls/controls-summary.json" \
  "$OUT/independent-audit.json"
```

Changing the Python/runtime or inputs can change bytes or fail pinned
checks; investigate differences as a new analysis. To reselect from the
**original** snapshot, use `select_encode_prior.py --help` with the
frozen catalog and the two source H5ADs. `freeze_encode_catalogs.py` creates
a fresh, time-dependent public snapshot, not the original. The files are
downloaded serially by `download_encode_peaks.py`; `verify_atac_prior.py`
performs a fresh metadata query and creates a timestamped report, so its
new report hash is not the frozen report hash.

To render from the pinned public aggregates or run focused tests:

```bash
"$PY" -B render_encode_report.py --output /path/to/new-index.html
"$PY" -B -m unittest -v test_freeze_encode_catalogs test_select_encode_prior \
  test_download_encode_peaks test_verify_atac_prior test_build_encode_promoter_graph \
  test_build_degree_controls test_audit_encode_graph test_render_encode_report
```

Use and cite [ENCODE](https://www.encodeproject.org/about/data-use-policy/)
according to its [citation guidance](https://www.encodeproject.org/help/citing-encode/),
including the producing labs and all accessions in the retained manifest,
and cite [GENCODE v44](https://www.gencodegenes.org/human/release_44.html).
The Replogle source is [scPerturb / Zenodo
7041849](https://doi.org/10.5281/zenodo.7041849); cite the collection and
[Replogle et al.](https://doi.org/10.1016/j.cell.2022.05.013). Respect all
applicable source and model terms.
