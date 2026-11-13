# scHiCAR source-prefix QA

This is a bounded **data-handling audit**, not a trained model or a performance
result. It checks the source-provided RNA-to-DNA barcode map for the public
mouse frontal-cortex dataset GSE305439, then counts matching ATAC and contact
records in initial compressed-file prefixes. Mouse tissue is not an eligible
human cell-line training collection and is not mixed into the human RNA axis.

The source metadata maps RNA barcodes to DNA barcodes. Joining the two assays
to the RNA axis by identical strings instead can associate a DNA observation
with another cell: the audit reports such collisions separately. It emits only
aggregate counts and local hashes, never cell barcodes or raw assay records.
Identity consistency does not independently prove physical pairing; the
[source protocol](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM9177145)
and [authors' dataset description](https://github.com/DiaoLab/scHiCAR/blob/2ece10efca321d73e21046e9c8426c0b2d3c1f3e/GEO_datasets.md)
are needed to interpret the mapping.

## Reproduce

With Python 3.11 or 3.12 and internet access, run:

```sh
python probe-schicar-brain-v1.py --output /path/to/new-audit.json
python -m pytest -q tests/test_probe_schicar_brain_v1.py
```

The output must not already exist. The script completely reads the small RNA
barcode and metadata gzip files and requests only the first 1 MiB of each
compressed ATAC and contact file. The optional `--prefix-bytes` limit is at
most 8 MiB. Prefixes are not random or representative; gzip integrity for
the complete large files is not checked. SHA-256 digests are computed locally
over downloaded bytes and are not independent source-supplied checksums.
The RNA expression matrix, genomic coordinate semantics, assay quality,
source permissions and cross-source leakage are not checked.

The published aggregate snapshot is
[`audit.json`](../../docs/trimodal/2026-11-13-data-qa-v1/audit.json), with a
[static report](../../docs/trimodal/2026-11-13-data-qa-v1/index.html).

The existing shared foundation is not complete. The coordinator reports that
fully consuming its planned Tahoe continuation would expose at least 7,562,326
paper test-perturbation records to pretraining; these are rows, not deduplicated
cells, and are not a count of exposure already observed. Isolation of all five
test cell lines is only a candidate, is not connected to the existing waiter,
and does not resolve possible cross-source exposure in stage one or replay.
The coordinator's preliminary check also flags RPE1 target-gene incompatibility
between the old evaluation and stage-one axes; a reproducible audit is pending.
This release does not change the axis; if it is changed, all comparisons must
be rerun on a new preregistered split rather than ranking old trial scores.
This audit neither changes existing training jobs nor supplies perturbation
predictions or official cell-eval scores.
