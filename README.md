# C3-VCell prototype

**Research status:** in a sample- and target-disjoint HCT116 experiment,
all three calibrated model arms selected zero perturbation response. The
epigenomic prior has not been shown to improve predictive accuracy, and the
local metrics are not official challenge leaderboard scores.

## Getting started

- [中文快速入门：合成数据到预测和评分](docs/quickstart.zh-CN.md)
- [数据、表观组网络与真实数据工作流](docs/data-and-epigenomics.zh-CN.md)
- [训练、推理与命令参考](docs/training-and-cli.zh-CN.md)
- [评估、挑战赛格式与已知结果](docs/evaluation.zh-CN.md)
- [测试方法与故障排查](docs/testing.zh-CN.md)
- [Interactive synthetic-data notebook](examples/quickstart.ipynb)

The documentation organization is informed by the installation, command,
example and testing surfaces of [ArcInstitute/state](https://github.com/ArcInstitute/state)
at commit `9bbfe78a434a55205e4de834e1ea99f85f7a3add`. C3-VCell is an
independent research prototype, not STATE or a reproduction of its results.
No STATE model weights are included. Public data and pretrained resources
retain their own licenses.

C3-VCell predicts a population of raw single-cell counts from independent
control cells after a **genetic perturbation** or a Tahoe drug treatment.
This repository implements bounded research prototypes, not a validated
general-purpose virtual cell.

The count-aware encoder uses masked JEPA-style representation learning, a
negative-binomial reconstruction objective and an EMA teacher. A conditional
latent flow is trained on unpaired control and perturbed cells; target-gene
embeddings share encoder weights, with an optional sparse gene-graph prior.
The output is sampled as nonnegative integer counts. The latent interpolation
is a statistical transport, not a measured time trajectory; graph edges are
priors, not automatically causal evidence. Small-subset results are
feasibility checks and do not establish accuracy on unseen contexts.

## Installation

Use Python 3.11 or 3.12 in a separate environment. Install a PyTorch build
compatible with the machine's NVIDIA driver if GPU training is wanted; the
default device is CPU. The evaluation package defaults to the explicit
`pdex` CPU differential-expression engine.

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[test]'
pytest -q
c3-vcell --help
```

For a prespecified genetic holdout, `prepare --holdout-targets GENE1 GENE2`
accepts explicit targets. They must be present in the sampled panel; set
`--max-targets 0` to sample all available targets, or use
`--include-targets ...` to specify the entire panel explicitly.

For a CPU-only environment, `bash scripts/bootstrap-cpu.sh` installs the
CPU PyTorch wheel and the project. Set `PYTHON` to the desired 3.11/3.12
interpreter. If it lacks `ensurepip`, also set `PIP_BOOTSTRAP_PYTHON` to a
Python installation that already has pip; it will install pip into the new
environment without modifying the existing one.

Dependencies are from `pyproject.toml`; a CUDA-compatible PyTorch wheel and
the public `cell-eval2` and `pdex` packages must be accessible from the host.
Installation can require several gigabytes; preserve the storage reserve and
do not install into a shared system Python.

For the separately installed official offline validator, use
`python -m pip install -e '.[vcc]'`. The exporter was exercised with
`vcc-cli==0.3.0` and synthetic challenge-shaped inputs; validation of real
submissions still requires the organizer's actual control bundle.

## Local holdout workflow

Input must be a **completed** H5AD with `.X` containing raw nonnegative
integer counts, unique cell and gene names, and a perturbation column. The
default control label is `non-targeting`. Every targeted gene must occur on
the gene axis. The train and holdout sets are split by gene, and three
nonoverlapping sets of control cells are reserved for training, prediction
and evaluation. Input batches are matched when a `batch` column is present.

For an initial pipeline check, `c3-vcell fetch-example --output
runs/H1-VCC-2025-training.h5ad` downloads the public 600-cell, 1,000-gene Arc
2025 example distributed by cell-eval2. Its pinned Git blob hash is checked;
only one of its five perturbation targets appears on its 1,000-gene axis, so
it can check data handling or the scorer but cannot satisfy this model's
strict unseen-target train/holdout split. Use a compatible Orion subset or
the complete Arc training dataset for that workflow.

```bash
c3-vcell prepare --input /path/to/complete_counts.h5ad \
  --outdir runs/example/split --label-column target_gene \
  --max-targets 20 --max-per-target 100 --max-controls 500 \
  --max-genes 4096 --holdout-fraction 0.25 --seed 42

c3-vcell train --train runs/example/split/train.h5ad \
  --outdir runs/example/model --device cpu \
  --pretrain-steps 300 --flow-steps 500 --batch-size 64 --seed 42

c3-vcell predict --checkpoint runs/example/model/model.pt \
  --reference runs/example/split/reference.h5ad \
  --controls runs/example/split/input_controls.h5ad \
  --output runs/example/prediction.h5ad --cells-per-target 100 --seed 42

c3-vcell score --prediction runs/example/prediction.h5ad \
  --reference runs/example/split/reference.h5ad \
  --outdir runs/example/score

c3-vcell control-baseline --reference runs/example/split/reference.h5ad \
  --controls runs/example/split/input_controls.h5ad \
  --output runs/example/control-baseline.h5ad --cells-per-target 100 --seed 42

c3-vcell score --prediction runs/example/control-baseline.h5ad \
  --reference runs/example/split/reference.h5ad \
  --outdir runs/example/control-score
```

Use `--device cuda:0` or `cuda:1` only after checking shared GPU availability.
For strict matching, training requires controls in every context/batch with
perturbed cells. `--allow-unmatched-batch` deliberately falls back to
context-only matching and records that choice in the checkpoint. During local
prediction, controls are sampled to match the held-out target's observed
context and batch composition without reading its expression values. The
prediction command also requires `--allow-unmatched-batch` to fall back.
The model rejects held-out target genes already seen during training and
overlapping training, input or reference cell identifiers.

For inference when no held-out expression file exists, provide a CSV with
unique `context,target_gene` pairs and independent raw-count controls:

```bash
c3-vcell predict-targets --checkpoint runs/example/model/model.pt \
  --targets requests.csv --controls runs/example/split/input_controls.h5ad \
  --output runs/example/requested-prediction.h5ad \
  --sampling residual-rounding \
  --effect-calibration runs/example/effect-calibration/effect-calibration.json \
  --cells-per-target 400 --seed 42
```

This command samples only from the supplied controls, not a held-out batch
distribution. It rejects training targets, missing target genes, overlapping
training cells, and incompatible contexts or epigenomic gene axes. A
calibration file must have been fitted with `--control-sampling
input-controls` and matching solver steps and cell count. Input file hashes
are recorded in the prediction metadata. A missing target cannot be inferred
on the current gene axis and is not silently replaced by a control.

`scripts/compare-external-bulk.py` can compare one such target's predicted
population log-fold changes with replicated bulk TPM from another study. It
uses the recorded source-control IDs, excludes the perturbed gene before
normalizing both assays on their shared panel, and reports an all-common and a
predefined expressed-gene panel. This exploratory cross-assay comparison is
not a VCC score, and constant-vector correlations are reported as undefined.

Representation training records sample-based variation and rank diagnostics
in the log and checkpoint. A relative mean per-dimension standard deviation
below 0.05 triggers a near-constant warning. With at least ten sampled cells
and more than one latent dimension, participation rank below two or a first
principal component explaining over 90% of variation triggers a low-rank
warning. These thresholds are heuristics, not biological validity or
prediction-accuracy tests. Loading a checkpoint with a recorded warning
repeats it; older checkpoints may not contain any diagnostic.

Both `train` and `train-drug` accept `--variance-weight`,
`--covariance-weight`, and `--regularization-view masked|unmasked`.
Covariance regularization penalizes correlated latent dimensions;
unmasked regularization evaluates it and the variance term on complete
measured views rather than the random training masks. The defaults remain
the historical variance weight 0.1, covariance weight 0 and masked view;
nested calibration inherits all three settings, including those defaults
when loading old checkpoints. Nondefault settings remain experimental.

Existing checkpoints can be diagnosed against their exact training file:

```bash
c3-vcell diagnose-representation --checkpoint runs/example/model/model.pt \
  --source runs/example/split/train.h5ad --output runs/example/representation.json \
  --max-cells 512 --seed 42
python scripts/run_representation_ablation.py --source runs/example/split/train.h5ad \
  --outdir runs/example/representation-ablation --weights 5 \
  --covariance-weights 0 0.01 0.05 0.1 --regularization-views unmasked
python scripts/audit_representation_ablation.py \
  --source runs/example/split/train.h5ad --run runs/example/representation-ablation \
  --output runs/example/representation-audit.json
```

The diagnostic checks cell IDs and gene order and, for new checkpoints, the
training-file SHA-256. Older checkpoints without a fingerprint produce a
warning because matching IDs alone cannot verify the historical counts.
The ablation reserves diagnostic cells within the supplied training set,
trains representation-only states with identical seeds, and compares
unmasked variation, reconstruction, and within-cell variation under random
masks. The optional audit examines latent rank and exploratory associations
with batch, sequencing depth, and target labels. These cells may share
targets with training cells; neither the diagnostic nor a larger variance
establishes held-out perturbation accuracy, and the associations are not
causal. Keep outer evaluation targets and their expression out of model
selection.

Newly prepared Orion and Tahoe subsets use deterministic, namespaced source
cell identifiers: Orion hashes context, sample and barcode; Tahoe hashes
sample and `BARCODE_SUB_LIB_ID`. Thus the same source cell has the same ID
across different preparations. Older `orion-000...` and `tahoe-000...`
identifiers were assigned by position: cross-file independence cannot be
established from those IDs alone, and comparisons emit a warning. Mixing
legacy and stable IDs from the same source is rejected; reprepare the data
and retrain legacy checkpoints for reliable cross-preparation comparisons.

The control-only baseline resamples the same independent controls and batch
composition without modeling any response. It records `source_control_id`
and writes integer counts even if raw input counts were stored as floats.
It is not the challenge's mean-response calibration baseline.

For a stronger local comparator, `fit-mean-response` computes the mean
normalized expression response across training perturbations, weighting each
eligible target equally within each context/batch. `predict-mean-response`
applies that same training-derived profile to every held-out target:

```bash
c3-vcell fit-mean-response --train runs/example/split/train.h5ad \
  --output runs/example/mean-response-profile.h5ad --min-group-cells 3
c3-vcell predict-mean-response --profile runs/example/mean-response-profile.h5ad \
  --reference runs/example/split/reference.h5ad \
  --controls runs/example/split/input_controls.h5ad \
  --output runs/example/mean-response-prediction.h5ad \
  --cells-per-target 400 --seed 42
```

The comparator fits only training counts, requires matching context/batch
profiles by default, records training cells and targets, and rejects
overlapping held-out cells or targets. It is target-independent and is not
the challenge organizer's private mean-response calibration baseline. Like
residual rounding below, it cannot create expression in genes absent from an
input control cell. Explicit `--allow-unmatched-batch` enables and records
context-only profile/control fallback.

The experimental shared-residual predictor combines that shared profile with
the target-conditioned C3-VCell flow. Its conditional component is the
decoder log-fold difference between the requested target and the **mean
training-target embedding**; this anchor is an approximation, not the
average of each training-target prediction:

```bash
c3-vcell predict-shared-residual --profile runs/example/mean-response-profile.h5ad \
  --checkpoint runs/example/model/model.pt \
  --reference runs/example/split/reference.h5ad \
  --controls runs/example/split/input_controls.h5ad \
  --output runs/example/shared-residual-prediction.h5ad \
  --shared-strength 1 --conditional-strength 0 --steps 8 --seed 42
```

The checkpoint and profile must use the same training cells, targets, and
gene order; new checkpoints also bind the exact training H5AD SHA-256 to the
profile's source. External representation-reference cells must not overlap
input or held-out cells. Conditional strength defaults to zero, which exactly
recovers the shared comparator for the same seed. Older checkpoints lack
training-file fingerprints and cannot detect altered counts with unchanged
cell IDs; using them with a profile emits a warning.

To select a nonzero conditional strength, first fit the final profile and
checkpoint from the same training file, then use nested target-disjoint folds:

```bash
c3-vcell calibrate-shared-residual --source runs/example/split/train.h5ad \
  --checkpoint runs/example/model/model.pt \
  --profile runs/example/mean-response-profile.h5ad \
  --outdir runs/example/shared-calibration --folds 3 \
  --strengths 0 0.25 0.5 1 --include-null-control --control-sampling input-controls \
  --cells-per-target 400 --prediction-steps 8
c3-vcell predict-shared-residual --profile runs/example/mean-response-profile.h5ad \
  --checkpoint runs/example/model/model.pt \
  --reference runs/example/split/reference.h5ad \
  --controls runs/example/split/input_controls.h5ad \
  --output runs/example/shared-calibrated.h5ad --cells-per-target 400 --steps 8 \
  --conditional-calibration runs/example/shared-calibration/shared-residual-calibration.json \
  --control-sampling input-controls
```

Each fold trains a new model and refits the profile without its validation
targets, using separate training, inference, and scoring control cells. The
prespecified plan records the candidate grid and fold targets before fitting.
The default selection objective is the raw normalized VCC expression metric
averaged across contexts and folds, **not** the official six-metric score. A nonzero
strength must improve every fold and exceed the relative improvement
threshold; otherwise the selected strength is zero. Undefined metrics fail
the calibration and produce an error record instead of a fallback score.
With `--include-null-control`, the selected response must additionally beat
unmodified input controls in every fold and by the relative threshold on
average; otherwise inference returns those controls exactly. Omitting the flag
retains the original shared-response-only selection. This single-objective
training-fold rule does not guarantee improvement on unseen targets.
The explicitly optional `--objective expr_mse_unbiased_capped` instead uses
the unnormalized numerator diagnostic described below; it does not silently
replace an undefined normalized metric.
`--control-sampling input-controls` samples within each context using only
the available input controls, without using the held-out targets' batch
distribution. The reference file is still required for the local target and
context list and later scoring, but its expression and target-batch
frequencies do not determine these predictions. The historical default,
`reference-batch`, instead matches the held-out target batch distribution and
can be optimistic when that metadata will not be available at inference.
Calibration and inference must use the same sampling mode. `control-baseline`,
`predict-mean-response`, direct genetic flow prediction, genetic effect
calibration, and ensemble prediction accept the same option. Drug prediction
and its batch-stratified calibration still use reference-batch sampling.
The output is bound to the exact checkpoint, profile, source file, prediction
cell count, solver steps and matching settings; old checkpoints must be
retrained to use this calibration. A full, externally fixed gene axis avoids
leaking validation expression through feature selection. Keep the final test
targets and their expression out of both training and calibration.

`scripts/run_nested_arc.py` reproduces an Arc experiment end to end: it
checks the supplied source SHA-256, excludes targets from specified prior
evaluation manifests, optionally excludes prior training targets with
`--exclude-training-targets-manifests`, writes a target-selection protocol
and code snapshot before training, runs nested calibration, and scores
control, shared, fixed-strength diagnostic, and calibrated predictions.
Pass `--include-null-control` to enable the same no-response candidate in
this script; the recorded result includes the selected shared strength.
The script also supports `--control-sampling input-controls` and
`--exclude-test-training-targets-manifests` for excluding prior training
targets from its new test panel without excluding them from training.
The fixed-strength result is not used to select the calibrated model.
Run `python scripts/run_nested_arc.py --help` for its required
source, exclusion manifests, and output directory. This can require
substantial CPU time and storage. Target-disjoint splits within the new run
do not erase earlier exposure to the same dataset.

For an alternative to random control/perturbed-cell pairing, genetic `train`
accepts `--flow-target population-shift --min-group-cells 5`. After
representation learning, it computes a latent mean difference for each
training target/context/batch against matched training controls and learns
to transport a control by that population shift. Underpopulated groups and
targets are reported and recorded; this remains a statistical estimate, not
a matched-cell trajectory. The default `paired` behavior is unchanged.

The default `negative-binomial` output sampler can change a control
distribution even when the predicted effect is small. As an explicitly
experimental alternative, `predict` and `export-vcc` accept `--sampling
residual-rounding`: multiplicative gene effects are applied to observed
control counts and stochastically rounded. Zero effect preserves each
control exactly, but this sampler cannot introduce expression in genes that
were zero in that input cell. It must be compared against both the default
sampler and control-only baseline rather than assumed better.

`predict-ensemble` can mix predictions from two or more independent training
seeds that use the same training cells and targets, gene axis, reference
provenance, architecture, training parameters and graph:

```bash
c3-vcell predict-ensemble \
  --checkpoints runs/example/model-1/model.pt runs/example/model-2/model.pt \
  --reference runs/example/split/reference.h5ad \
  --controls runs/example/split/input_controls.h5ad \
  --output runs/example/ensemble.h5ad \
  --disagreement runs/example/disagreement.h5ad \
  --cells-per-target 400 --sampling residual-rounding --seed 42
```

Each member predicts the same sampled input control cells. The output mixes
members as evenly as possible for each target/context; `source_control_id`
and `ensemble_member` identify the source. The disagreement H5AD has one row
per target/context, `X` contains the mean of member pseudobulk count means,
and layers contain each member's mean and their sample standard deviation.
This standard deviation also includes finite-cell output-sampling noise; it
is **not** a calibrated confidence interval, coverage guarantee, or proof of
biological uncertainty. Use an independently validated holdout to calibrate
or assess uncertainty. Ensemble inference takes a fixed effect strength; an
individual checkpoint's calibration cannot be applied to the ensemble.
New checkpoint metadata also require the same training-file fingerprint.
Ensembles of older checkpoints emit a warning because matching cell IDs alone
cannot establish identical training counts; old and new checkpoints cannot be
mixed in one ensemble.

For an explicitly narrower uncertainty target, `calibrate-intervals` fits
split-conformal intervals for **observed pseudobulk means**, using targets
that were excluded from every ensemble member's training. It takes the
maximum standardized residual across all genes and contexts for each
calibration target. The scale is the model disagreement with a positive
floor; the finite-sample quantile is chosen at target level, not cell level.
For example:

```bash
c3-vcell calibrate-intervals \
  --prediction runs/example/calibration-prediction.h5ad \
  --disagreement runs/example/calibration-disagreement.h5ad \
  --reference runs/example/calibration-reference.h5ad \
  --controls runs/example/calibration-controls.h5ad \
  --output runs/example/interval-calibration.json --alpha 0.1 --scale-floor 1
c3-vcell apply-intervals --calibration runs/example/interval-calibration.json \
  --prediction runs/example/evaluation-prediction.h5ad \
  --disagreement runs/example/evaluation-disagreement.h5ad \
  --controls runs/example/evaluation-controls.h5ad \
  --output runs/example/intervals.h5ad
c3-vcell evaluate-intervals --calibration runs/example/interval-calibration.json \
  --prediction runs/example/evaluation-prediction.h5ad \
  --disagreement runs/example/evaluation-disagreement.h5ad \
  --controls runs/example/evaluation-controls.h5ad \
  --intervals runs/example/intervals.h5ad \
  --reference runs/example/evaluation-reference.h5ad \
  --output runs/example/interval-coverage.json
```

The calibration and evaluation target and cell sets must be disjoint; all
members and inference settings must match. `intervals.h5ad` contains
target/context mean counts in `X` and gene-wise `lower`/`upper` layers.
At least nine calibration targets are needed for finite intervals at 90%
nominal coverage. The usual target-level joint coverage interpretation
requires exchangeability of calibration and future target blocks, including
their sampling procedures; biological targets and batches may violate this.
These are not intervals for single cells or underlying biological means.
The ensemble pipeline can use input-only control sampling, but it still uses
the local reference for its target/context list and interval evaluation; it
is not an official challenge export. Interval calibration also binds the
control sampling mode.

An optional **effect-strength** calibration selects how much of the flow
model's predicted log-fold change to use. It trains new models on disjoint
folds of the final checkpoint's training targets, evaluates a prespecified
grid against the VCC preset's expression metric only, and requires a
prespecified improvement over zero effect. It never uses the external holdout
expression for selection. It binds the result to the exact final checkpoint
and training IDs. Prediction must use the same flow solver step count and
number of cells per target as calibration. Calibration can legitimately select zero, in which case
predictions are the control baseline rather than a learned response.

```bash
c3-vcell calibrate-effect --source runs/example/split/train.h5ad \
  --checkpoint runs/example/model/model.pt \
  --outdir runs/example/effect-calibration --folds 3 --seed 2026
c3-vcell predict --checkpoint runs/example/model/model.pt \
  --reference runs/example/split/reference.h5ad \
  --controls runs/example/split/input_controls.h5ad \
  --output runs/example/prediction-calibrated.h5ad \
  --sampling residual-rounding \
  --effect-calibration runs/example/effect-calibration/effect-calibration.json \
  --cells-per-target 400 --seed 42
```

For new checkpoints, `--source` must be the exact training H5AD recorded at
training. The fixed gene axis is reused within folds; avoid deriving a
reduced panel from validation-target expression. For old checkpoints without
training-file fingerprints, source identity cannot be fully verified. For
checkpoints that do not record training settings, pass the matching
`--pretrain-steps`, `--flow-steps`, `--batch-size` and `--learning-rate`.
The default selection objective is one raw VCC metric, **not** the six-metric
calibrated challenge score. When a small training fold has a nonpositive
real-effect denominator, the normalized metric cannot be computed and
calibration fails. As an explicit alternative, both `calibrate-effect` and
`calibrate-shared-residual` accept
`--objective expr_mse_unbiased_capped`. This uses the VCC preset's
unnormalized capped expression-error numerator, not a scored challenge
metric. Its estimates can be negative, depend on the gene and perturbation
panel, and are averaged equally over contexts and folds; they are not
comparable across panels or equivalent to averaging normalized scores.
No automatic fallback is performed, and final evaluation still reports
undefined challenge metrics as undefined. `calibrate-effect` also accepts
`--require-every-fold-improvement`: it first keeps nonzero strengths that
strictly improve every training fold and meet the requested mean relative
improvement, then chooses the feasible strength with lowest mean loss.
If none qualifies it selects zero; the default remains the historical
mean-only rule. Repeated decisions informed by the outer holdout can still
bias that evaluation; use a fresh, prespecified holdout for final claims.

`--max-genes` is optional (zero retains the full axis). A reduced panel keeps
all supplied target genes and selects the remaining genes by count totals from
training cells only. The same ordered panel is applied to held-out cells and
input controls; excluded genes are not treated as measured zeros. This is a
local reduced-panel experiment, not a full-gene official challenge result.

If the scorer cannot define a metric because the reference panel has
insufficient signal, `score --metrics METRIC ...` can evaluate the remaining
named metrics. This does not turn the missing metric into a valid six-metric
score; retain the failure and report the omitted metric as undefined.

`split.json` records cell IDs, targets and seed; `model/model.pt` contains the
weights and target list; `training.jsonl` records losses. Commands refuse to
overwrite existing output files or nonempty run directories. These IDs can be
sensitive in other datasets: protect the files accordingly.

To supply a graph, use `--graph edges.csv` with columns
`source,target,weight`, using gene symbols from the training axis. Unknown
genes are reported; a graph with no matching edges is rejected.

## Orion subset

Use only already completed and verified parquet shards. First inspect labels,
contexts and samples rather than guessing the control label:

```bash
c3-vcell inspect-orion --shards /path/to/completed/*.parquet \
  --output runs/orion-label-counts.csv
c3-vcell prepare-orion --shards /path/to/completed/*.parquet \
  --gene-metadata /path/to/metadata/gene_metadata.parquet \
  --targets GENE1 GENE2 GENE3 --control-label ACTUAL_CONTROL_LABEL \
  --max-per-target 100 --max-controls 500 \
  --output runs/orion-subset.h5ad
```

Replace the placeholder labels with values actually present in the inspected
shards. `prepare-orion` preserves `sample` as `batch`, filters for passing
guides, and aggregates duplicate gene symbols while retaining their Ensembl
IDs. It refuses missing genes, duplicate cells and invalid raw counts. Run
`prepare` on the resulting H5AD, then train and score as above. The Orion
dataset is CC BY-NC-SA 4.0; check its terms before downstream use.

For a reproducible panel selected only by passing-cell counts, use
`python scripts/select-orion-targets.py --groups runs/orion-label-counts.csv
--gene-metadata /path/to/metadata/gene_metadata.parquet --control-label
ACTUAL_CONTROL_LABEL --targets 20 --min-cells 25 --output
runs/orion-targets.txt`. The accompanying JSON records counts and matching
control coverage. It does not inspect held-out expression values. Preserve the
list of completed input shards with the run; downloading additional files
after selection should not silently change the experiment's inputs.

For the bounded CPU example, `bash scripts/run-orion-example.sh
runs/orion-subset.h5ad runs/orion-demo` creates a new run directory, performs
the target-disjoint split, trains and predicts, scores a control-only
baseline, and attempts raw and locally calibrated evaluation. The command
stops on errors and preserves each stage's log. `C3_MAX_PER_TARGET`,
`C3_MAX_CONTROLS`, `C3_PREDICTION_CELLS` and `C3_SAMPLING` override the
default sampling limits and strategy. Local calibration can be undefined for
small datasets.

## Tahoe drug responses

The Tahoe adapter accepts only completed parquet shards, removes the dataset's
CLS marker, verifies raw integer UMI counts and merges duplicate symbols while
retaining Ensembl IDs. It parses the sample-level drug concentration in
micromolar units and preserves cell line, plate, sample, and canonical SMILES.
`DMSO_TF` is the solvent control; controls must match both cell line and plate.
Current parsing requires a single drug per sample and a usable canonical
SMILES for each selected drug.

```bash
c3-vcell inspect-tahoe --shards /path/to/completed/*.parquet \
  --output runs/tahoe-groups.csv
python scripts/select-tahoe-drugs.py --groups runs/tahoe-groups.csv \
  --cell-line ACTUAL_CELL_LINE_ID --plate ACTUAL_PLATE \
  --drugs 20 --min-cells 100 --output runs/tahoe-drugs.txt
c3-vcell prepare-tahoe --shards /path/to/completed/*.parquet \
  --gene-metadata /path/to/metadata/gene_metadata.parquet \
  --sample-metadata /path/to/metadata/sample_metadata.parquet \
  --drugs DRUG1 DRUG2 DRUG3 --cell-lines ACTUAL_CELL_LINE_ID \
  --plates ACTUAL_PLATE --output runs/tahoe-subset.h5ad
c3-vcell split-tahoe --input runs/tahoe-subset.h5ad \
  --outdir runs/tahoe/split --max-genes 4096 --seed 2026
c3-vcell train-drug --train runs/tahoe/split/train.h5ad \
  --outdir runs/tahoe/model --pretrain-steps 200 --flow-steps 300 --seed 2026
c3-vcell calibrate-drug-effect --source runs/tahoe/split/train.h5ad \
  --checkpoint runs/tahoe/model/model.pt \
  --outdir runs/tahoe/effect-calibration --folds 3 --seed 2026
c3-vcell predict-drug --checkpoint runs/tahoe/model/model.pt \
  --reference runs/tahoe/split/reference.h5ad \
  --controls runs/tahoe/split/input_controls.h5ad \
  --effect-calibration runs/tahoe/effect-calibration/effect-calibration.json \
  --output runs/tahoe/prediction.h5ad --cells-per-target 400 --seed 2026
```

The selector uses counts and matched-control availability, not expression.
Replace placeholders with observed metadata values, and preserve the exact
shard list. Splitting groups all doses and names with the same canonical
SMILES, so the exact recorded compound is absent from training; it is not a
scaffold-disjoint test. Drug effect calibration likewise holds out entire
compounds inside the training set and uses the explicitly non-genetic version
of the expression metric; it may select the zero-effect control baseline.
The drug conditioner encodes SMILES **text** and
concentration; it is not a chemically invariant molecular graph model and
has not established dose extrapolation. Doses in a single plate cannot
validate dose response. If different doses are confined to different plates,
dose and plate can be confounded even with matched DMSO controls.

Tahoe is CC0. Its drug labels are not VCC genetic target genes. Running
`c3-vcell score --no-target-genes --stratify-by-batch` declares that there is
no genetic target to exclude and evaluates each cell-line/plate group against
its own controls. Drug calibration also scores per plate. These adapted
metrics may be useful for diagnostics, but they are not the official
genetic-perturbation challenge or a leaderboard score.

## Reference pretraining

Both `train` and `train-drug` accept `--reference /path/complete.h5ad
--reference-fraction 0.5 --reference-max-cells 10000`. This independently
samples bounded raw-count cells and alternates their representation-learning
batches with primary training batches; the conditional flow still uses the
primary training data. Genes are aligned by their unique symbols, and a
measurement mask excludes absent genes rather than interpreting them as
zeros. `--reference-min-coverage` rejects insufficient overlap; sampled
cells without counts on the shared panel are rejected.

The checkpoint records the sampled reference cell IDs, gene coverage and a
count fingerprint. Local prediction rejects reference cells overlapping
held-out or input cells. Genetic effect calibration verifies the same
reference fingerprint and rejects overlap with its inner-fold source.
Identifier checks cannot discover duplicate biological cells renamed across
datasets, and matching by symbol does not eliminate all assay or annotation
differences. Complete Tabula or Arc H5ADs can be supplied when available;
this code does not imply they have finished downloading.

## Experimental epigenomic regulatory network

The version-2 Epi-C3 prior constructs a context-specific **candidate gene
network before training**, using external bulk reference assays rather than
the held-out response. It combines ATAC accessibility and CTCF, SP1 and USF1
ChIP-seq peak overlaps with chromatin-contact anchors. Promoter contacts make
candidate gene-gene edges; ATAC and TF peaks at promoters or linked anchors
can support TF-to-gene edges. Each gene also has promoter CpG methylation and
methylation at ATAC-accessible contact-linked elements. Covered CpG inputs
are weighted by read coverage; inputs without per-site coverage use the
separately specified equal-CpG procedure described below.
The continuous fractions and explicit availability masks are encoded into
gene embeddings and propagated along the row-normalized network before
conditioning the perturbation flow. Missing CpGs have mask zero; a measured
zero is not missing. These associations do not establish regulatory
direction or causality, and methylation is not assumed to be inhibitory.
The older five-feature version-1 bundles and checkpoints retain their
original behavior.

The design is inspired by the integration of prior gene knowledge in
[GeneCompass](https://www.nature.com/articles/s41422-024-01034-y) and
mechanism-related knowledge in
[MAP](https://www.nature.com/articles/s42256-026-01286-w). It is **not a
reproduction** of either paper: it does not implement GeneCompass's full
cross-species priors or MAP's drug, protein and text alignment, and does not
establish zero-shot drug prediction.

For the HCT116 example, the data scripts validate source metadata, pinned
checksums and chromosome assembly. The RRBS inputs
`ENCSR000DFS/ENCFF001TMM` and `ENCFF001TMN` are hg19, so both CpG bases must
map uniquely and adjacently to GRCh38 and remain `CG` in the checked target
reference. CpGs below the input coverage threshold are excluded; overlapping
observations are pooled by coverage. Source percentages are rounded and the
two nominal replicates are associated with the same biosample in the ENCODE
audit, so pooling does not constitute independent validation. RRBS is
sparse; unmeasured sites are not imputed.

The HCT116 RAD21 contacts `ENCFF625LBL` and `ENCFF769MZQ` are **ChIA-PET,
not Hi-C**. Two anchors are retained only when each has at least 50% mutual
overlap across the two called-loop files. Both experiments are labeled
unreplicated and have ENCODE audit warnings; overlapping coordinates do not
prove independent replication. The HCT116 bulk assays come from different
experiments and are not paired multimodal measurements. An HFFc6 4DN Hi-C
preview is a different cell context and must not be used as if it were
HCT116. `extract-mcool-contacts` can extract explicitly labeled
distance-normalized cis contact candidates from a suitable local Hi-C or
Micro-C file; these candidates are not statistically called loops.

Install optional preprocessing dependencies, download the verified HCT116
inputs and create a source specification. The base specification uses the
separately downloaded, verified GENCODE v44 annotation and HCT116 ENCODE
ATAC/TF peaks:

```bash
python -m pip install -e '.[test,epigenomics]'
python scripts/download-hct116-priors.py --outdir /path/hct116-multimodal
python scripts/make-hct116-epi-spec.py --encode-dir /path/hct116-peaks \
  --annotation-dir /path/gencode --output /path/hct116-sources-v1.json
python scripts/make-hct116-network-spec.py --base-spec /path/hct116-sources-v1.json \
  --raw-dir /path/hct116-multimodal --outdir /path/hct116-network-sources \
  --min-coverage 5
c3-vcell build-epigenetic-bundle --gene-data runs/example/split/train.h5ad \
  --spec /path/hct116-network-sources/sources.json \
  --outdir runs/example/epigenome
c3-vcell shuffle-epigenetic-bundle --bundle runs/example/epigenome \
  --gene-data runs/example/split/train.h5ad \
  --outdir runs/example/epigenome-shuffled --seed 42
c3-vcell shuffle-epigenetic-bundle --bundle runs/example/epigenome \
  --gene-data runs/example/split/train.h5ad \
  --outdir runs/example/epigenome-graph-shuffled --seed 42 --graph-only
c3-vcell train --train runs/example/split/train.h5ad --outdir runs/example/epi-model \
  --epigenetic-bundle runs/example/epigenome --seed 42
```

The default shuffle randomizes both feature assignments and graph targets.
`--graph-only` instead preserves the complete feature and mask file
byte-for-byte while shuffling only graph targets; this isolates a graph
topology control, not a fully independent experimental replicate.

The network builder reads the gene axis and context, not expression, and
requires explicit matching HCT116 context, GRCh38 assembly and unique
Ensembl identifiers. Source and bundle files are hashed and the checkpoint
binds the feature axis and graph. For an explicitly **partial** chr21/chr22
methylation preview while the full reference is pending, run
`python scripts/download-hct116-priors.py --outdir /path/hct116-multimodal --preview-reference`
and add, for example,
`--preview-chromosomes chr21 chr22` to `make-hct116-network-spec.py`;
one available chromosome may be specified on its own. All other source
chromosomes are skipped, not treated as unmethylated. This is separate from
the 4DN Hi-C preview.

For the separately collected HCT116
[intact Hi-C loops](https://www.encodeproject.org/experiments/ENCSR477GZK/)
and [WGBS](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM3317488)
inputs, `make-hct116-matched-spec.py` verifies their local provenance
sidecars, cell context and assembly before preparing another v2 specification:

```bash
python scripts/make-hct116-matched-spec.py --base-spec /path/hct116-sources-v1.json \
  --matched-dir /path/hct116-matched-v1 --reference-dir /path/hct116-multimodal/ucsc \
  --outdir /path/hct116-matched-sources --preview-chromosomes chr21 chr22
```

The Hi-C loop file has anchor widths of 1, 2, 5 and 10 kb; it is not a
single-resolution contact matrix. The downloaded loop file's ENCODE MD5 is
verified, including when retrieved from the GEO mirror. The WGBS bedGraph
contains 0–100 percentages without read counts; its locally recorded SHA-256
is checked, but GEO did not provide an independent source checksum. The
preparer checks each selected position against the reference sequence,
excludes and counts non-reference CpGs, infers the reference CpG dyad for
each strand, and averages the available strands equally per dyad. Region
features then average observed dyads equally. Neither stage applies a
coverage filter or invents read depth. Reference allele differences may
cause true sample CpGs to be excluded.

The preview validates only the named chromosomes, leaving all others
unmeasured. To use all primary chromosomes, omit `--preview-chromosomes`
only after the complete, checksum-verified `hg38.2bit` and its provenance
sidecar are present; a `.part` file is not sufficient. The Hi-C and WGBS
samples are bulk assays from different experiments, not paired multiomics,
and their candidate links do not establish causal regulation.

`scripts/run-epi-hct116.py` can evaluate a fresh target panel from completed
Orion shards. It excludes recorded historical train and test targets, and
its optional `--prioritized-holdouts` selects measured-methylation targets
with outgoing edges from a verified prior bundle built from the same
specification. Supply both `--priority-selection-data` and
`--priority-selection-bundle`; these use gene identities, prior masks,
edges, and passing-cell counts, not held-out perturbation expression.
The prior bundle's outgoing neighbors are explicitly retained on the reduced
gene axis, and the rebuilt network must retain their prespecified links.
Such holdouts are deliberately enriched and remain from the same dataset,
not an external or population-representative validation set.

For a stronger within-Orion test, `scripts/run-epi-sample-holdout.py` fixes
disjoint training and held-out sample labels as well as fresh targets. The
priority-selection data must contain only selected training samples. The
training-only development evidence is also checked against its source
fingerprints, sample labels and checkpoint cell identities. The script
records sample/target panels, input hashes, settings and a code snapshot
before decoding held-out perturbation expression; each held-out sample's
controls are split into disjoint prediction and scoring controls.
It compares no prior, the real prior and the graph-only control, and reports
general and deliberately prior-enriched target panels separately.

```bash
python scripts/run-epi-sample-holdout.py \
  --orion-root /path/X-Atlas-Orion --groups /path/hct116-groups.csv \
  --history-root /path/prior-runs --spec /path/hct116-sources.json \
  --priority-selection-data /path/training-only/split/train.h5ad \
  --priority-selection-bundle /path/training-only/network \
  --development-evidence /path/training-only/result.json \
  --outdir runs/new-sample-holdout
```

Its population-shift training excludes target/batch groups with fewer than
five perturbed or matching control cells; checkpoints record omitted groups
and targets. The reduced gene axis is selected using outer training counts
and reused in inner target folds, so it is not independently selected within
each fold. Distinct sample labels do not establish independent biological
replicates or external-dataset generalization.

The script records a code snapshot and a plan before fitting, and accepts
the representation regularization and calibration options described above.
Prediction and calibration cell counts must agree. For final evaluation,
it checks the VCC real-effect denominator: if nonpositive, the script
reports normalized expression MSE as undefined and still evaluates the other
available metrics, explicitly recording the reason. This does not replace
the metric with an invented value or produce an official challenge score.

Use the same fixed training/target split for no-prior, real-prior and
shuffled-prior comparisons. The shuffle preserves measured-feature masks and
values' marginal distribution, per-source outdegree and the multiset of
target indegrees, using constrained target-label transpositions; it is not a
uniform sample of all possible graphs. Its evidence labels no longer have
biological meaning. Tune only on nested training targets; do not choose
parameters based on a reused outer holdout. The previously observed HCT116
inner-fold expression denominator was nonpositive, so its effect calibration
failed and the separately fixed-strength run is an uncalibrated diagnostic,
not evidence that the prior improves prediction. All local raw VCC metrics
remain distinct from official leaderboard scores.

`run-network-diagnostic.py` writes a machine-readable `research-review.json`
after checking the split, bundle, checkpoint, prediction and scoring
fingerprints. It compares all six raw metrics with the no-response, no-prior
and shuffled controls, retains undefined metrics, reports held-out-target
methylation reachability, and blocks validation claims for this previously
scored, uncalibrated split. This bounded evidence gate is inspired by the
task-analysis and deterministic-verification stages of
[CellForge](https://github.com/gersteinlab/CellForge/tree/6585ff93e452a8a0f2ad6ed8144ab6d998a3b400);
it is not a CellForge installation or an independent multi-expert review.
The standalone `scripts/review-network-diagnostic.py` can review an existing
compatible run without overwriting its artifacts.

## Evaluation and VCC compatibility

`score` runs **cell-eval2 0.22.0** with the official `vcc2026` preset for each
context, including the six scored metrics. It validates raw nonnegative
integer counts, checks genetic targets against the gene axis, rejects
recorded input controls that overlap reference cells, and fingerprints both
scoring inputs. Individual zero-count predictions are allowed, but an
all-zero context/perturbation group or a cell over one million counts is
rejected. For the scorer's required
matching label sets, independent reference controls are appended to a
temporary evaluation copy, never to the prediction file. The preset uses
real controls for its comparisons. It limits CPU scoring to two
threads by default (`--threads` changes the cap). `--calibrate` additionally builds
the official-style mean-response baseline and five split-half anchors from
the **local held-out reference**, then applies the corresponding scoring
procedure. This uses reference labels/expressions only after prediction.
Small datasets can yield undefined differential-expression metrics or
degenerate anchors; the scoring program fails or reports those values rather
than inventing a score. CPU `pdex` values can differ from GPU-DE results.
Local results are **not official leaderboard scores** and cannot reproduce
private challenge reference anchors.

The `export-vcc` command can generate a challenge-shaped H5AD from three
control H5ADs and genuine challenge-provided `gene_names.csv` and
`pert_counts.csv` files:

```bash
c3-vcell export-vcc --checkpoint runs/example/model/model.pt \
  --controls A=/path/A.h5ad B=/path/B.h5ad C=/path/C.h5ad \
  --gene-names /path/gene_names.csv --pert-counts /path/pert_counts.csv \
  --output runs/example/vcc_prediction.h5ad --device cpu
```

It generates 400 cells for each of 300 targets in each context and checks the
18,533-gene order, nonnegative count limits, the stored-entry cap and a
200-GiB disk reserve. On-disk concatenation uses bounded fan-in. Missing
targets are rejected unless `--allow-missing-target` explicitly enables a
control-only baseline, which is recorded in the accompanying JSON; targets
seen during training are also recorded. Export refuses input control cell IDs
that overlap the checkpoint's training or external reference cells. This does
not fetch private challenge controls or constitute a submission: use `vcc prep` for
final official format validation. The official 2026 submission deadline has
passed.
Direct-flow challenge export always samples from input controls. Any supplied
effect calibration must therefore have been fitted with `calibrate-effect
--control-sampling input-controls`; older calibrations without a recorded mode
are treated as reference-batch and are rejected for challenge export.

`export-shared-vcc` separately exports the calibrated shared-residual model
without constructing a reference expression file:

```bash
c3-vcell export-shared-vcc \
  --checkpoint runs/example/model/model.pt --profile runs/example/profile.h5ad \
  --calibration runs/example/shared-calibration/shared-residual-calibration.json \
  --controls A=/path/A.h5ad B=/path/B.h5ad C=/path/C.h5ad \
  --gene-names /path/gene_names.csv --pert-counts /path/pert_counts.csv \
  --profile-mapping /path/explicit-profile-mapping.json \
  --steps 8 --output runs/example/shared_vcc_prediction.h5ad
```

The calibration must have used `--control-sampling input-controls`, 400 cells
per target, the same solver steps, shared strength and unmatched-batch
setting. A nonzero shared response requires a JSON list explicitly mapping
every input context/batch group to a training-profile context/batch:

```json
[
  {"input_context": "A", "input_batch": "", "profile_context": "TRAIN_CONTEXT_A", "profile_batch": "TRAIN_BATCH_A"},
  {"input_context": "B", "input_batch": "", "profile_context": "TRAIN_CONTEXT_B", "profile_batch": "TRAIN_BATCH_B"},
  {"input_context": "C", "input_batch": "", "profile_context": "TRAIN_CONTEXT_C", "profile_batch": "TRAIN_BATCH_C"}
]
```

These are placeholders, not inferred equivalences. A missing input batch
column is represented by the empty string. Missing, duplicate, unavailable or
unused mappings are rejected. Mapping is unnecessary when calibration selects
the exact no-response control, which is copied without modification. For any
nonzero response all model genes must be present in the challenge gene axis;
additional challenge genes retain their input counts. If a target is missing
from the model and the conditional response is nonzero, export fails unless
`--allow-missing-target` explicitly selects the unchanged-control fallback.
Inputs and mappings are hashed in the output provenance. Mapping groups and
transferring calibration across contexts do not establish biological
equivalence or predict challenge performance; actual organizer-provided inputs
and `vcc prep` are still required.

## Current limitations

The drug and reference-pretraining paths are prototypes, not validated
biological simulators. Longitudinal or spatial dynamics, a molecular graph
encoder, single-cell uncertainty, ensemble challenge export, comprehensive
multi-dataset harmonization, distributed training and established competitor
baselines are not implemented. The model does not claim zero-shot performance on the
challenge contexts. Static, unpaired perturbation observations do not identify
true temporal paths or prove causal regulatory edges.
