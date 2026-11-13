"""Command-line entry points for the C3-VCell prototype."""

from __future__ import annotations

import argparse
from pathlib import Path

from .data import CONTROL


def build_parser() -> argparse.ArgumentParser:
    from .calibration import CALIBRATION_OBJECTIVES, OBJECTIVE
    from .scoring import AVAILABLE_METRICS

    parser = argparse.ArgumentParser(prog="c3-vcell")
    commands = parser.add_subparsers(dest="command", required=True)

    example = commands.add_parser("fetch-example", help="download a verified small Arc dataset")
    example.add_argument("--output", type=Path, required=True)

    prep = commands.add_parser("prepare", help="sample a completed H5AD and split by target")
    prep.add_argument("--input", type=Path, required=True)
    prep.add_argument("--outdir", type=Path, required=True)
    prep.add_argument("--label-column", default="target_gene")
    prep.add_argument("--control", default=CONTROL)
    prep.add_argument("--max-per-target", type=int, default=100)
    prep.add_argument("--max-controls", type=int, default=500)
    prep.add_argument("--max-targets", type=int, default=20)
    prep.add_argument(
        "--include-targets", nargs="+",
        help="explicit target panel; overrides random --max-targets selection",
    )
    prep.add_argument("--max-genes", type=int, default=0)
    prep.add_argument("--holdout-fraction", type=float, default=0.25)
    prep.add_argument(
        "--holdout-targets", nargs="+",
        help="explicit target names to hold out; they must be present in the sampled panel",
    )
    prep.add_argument("--seed", type=int, default=0)

    inspect = commands.add_parser("inspect-orion", help="count labels in completed Orion shards")
    inspect.add_argument("--shards", type=Path, nargs="+", required=True)
    inspect.add_argument("--output", type=Path, required=True)

    orion = commands.add_parser("prepare-orion", help="sample completed Orion parquet shards")
    orion.add_argument("--shards", type=Path, nargs="+", required=True)
    orion.add_argument("--gene-metadata", type=Path, required=True)
    orion.add_argument("--output", type=Path, required=True)
    orion.add_argument("--targets", nargs="+", required=True)
    orion.add_argument("--control-label", required=True)
    orion.add_argument("--max-per-target", type=int, default=100)
    orion.add_argument("--max-controls", type=int, default=500)
    orion.add_argument("--seed", type=int, default=0)

    epi = commands.add_parser(
        "build-epigenetic-bundle",
        help="construct a context-matched epigenomic feature and graph prior",
    )
    epi.add_argument("--gene-data", type=Path, required=True)
    epi.add_argument("--spec", type=Path, required=True)
    epi.add_argument("--outdir", type=Path, required=True)
    epi.add_argument("--promoter-flank", type=int, default=2000)

    epi_shuffle = commands.add_parser(
        "shuffle-epigenetic-bundle", help="create a labeled randomization control",
    )
    epi_shuffle.add_argument("--bundle", type=Path, required=True)
    epi_shuffle.add_argument("--gene-data", type=Path, required=True)
    epi_shuffle.add_argument("--outdir", type=Path, required=True)
    epi_shuffle.add_argument("--seed", type=int, default=0)
    epi_shuffle.add_argument(
        "--graph-only", action="store_true",
        help="shuffle graph targets while preserving the original feature values and masks",
    )

    contacts = commands.add_parser(
        "extract-mcool-contacts",
        help="extract distance-normalized cis contact candidates from a local .mcool",
    )
    contacts.add_argument("--mcool", type=Path, required=True)
    contacts.add_argument("--source-metadata", type=Path, required=True)
    contacts.add_argument("--chrom-sizes", type=Path, required=True)
    contacts.add_argument("--outdir", type=Path, required=True)
    contacts.add_argument("--resolution", type=int, required=True)
    contacts.add_argument("--chromosomes", nargs="+", required=True)
    contacts.add_argument("--max-distance", type=int, default=2_000_000)
    contacts.add_argument("--min-count", type=int, default=10)
    contacts.add_argument("--min-observed-expected", type=float, default=2.0)
    contacts.add_argument("--allow-assembly-inference", action="store_true")

    tahoe_inspect = commands.add_parser(
        "inspect-tahoe", help="count drug, plate, and cell-line groups in completed Tahoe shards"
    )
    tahoe_inspect.add_argument("--shards", type=Path, nargs="+", required=True)
    tahoe_inspect.add_argument("--output", type=Path, required=True)

    tahoe = commands.add_parser("prepare-tahoe", help="sample raw Tahoe drug-response counts")
    tahoe.add_argument("--shards", type=Path, nargs="+", required=True)
    tahoe.add_argument("--gene-metadata", type=Path, required=True)
    tahoe.add_argument("--sample-metadata", type=Path, required=True)
    tahoe.add_argument("--output", type=Path, required=True)
    tahoe.add_argument("--drugs", nargs="+", required=True)
    tahoe.add_argument("--cell-lines", nargs="+", required=True)
    tahoe.add_argument("--plates", nargs="+", required=True)
    tahoe.add_argument("--max-per-condition", type=int, default=200)
    tahoe.add_argument("--max-controls-per-group", type=int, default=1000)
    tahoe.add_argument("--seed", type=int, default=0)

    tahoe_split = commands.add_parser(
        "split-tahoe", help="split Tahoe by canonical SMILES and disjoint DMSO controls"
    )
    tahoe_split.add_argument("--input", type=Path, required=True)
    tahoe_split.add_argument("--outdir", type=Path, required=True)
    tahoe_split.add_argument("--max-genes", type=int, default=0)
    tahoe_split.add_argument("--holdout-fraction", type=float, default=0.25)
    tahoe_split.add_argument("--seed", type=int, default=0)

    drug_train = commands.add_parser(
        "train-drug", help="train the Tahoe SMILES- and dose-conditioned model"
    )
    drug_train.add_argument("--train", type=Path, required=True)
    drug_train.add_argument("--outdir", type=Path, required=True)
    drug_train.add_argument("--latent-dim", type=int, default=128)
    drug_train.add_argument("--hidden-dim", type=int, default=256)
    drug_train.add_argument("--pretrain-steps", type=int, default=300)
    drug_train.add_argument("--flow-steps", type=int, default=500)
    drug_train.add_argument("--batch-size", type=int, default=64)
    drug_train.add_argument("--learning-rate", type=float, default=1e-3)
    drug_train.add_argument("--variance-weight", type=float, default=0.1)
    drug_train.add_argument("--covariance-weight", type=float, default=0.0)
    drug_train.add_argument("--regularization-view", choices=["masked", "unmasked"], default="masked")
    drug_train.add_argument("--seed", type=int, default=0)
    drug_train.add_argument("--device", default="cpu")
    drug_train.add_argument("--reference", type=Path)
    drug_train.add_argument("--reference-fraction", type=float, default=0.5)
    drug_train.add_argument("--reference-max-cells", type=int, default=10_000)
    drug_train.add_argument("--reference-seed", type=int)
    drug_train.add_argument("--reference-min-coverage", type=float, default=0.2)

    drug_predict = commands.add_parser(
        "predict-drug", help="predict drug responses for held-out canonical SMILES"
    )
    drug_predict.add_argument("--checkpoint", type=Path, required=True)
    drug_predict.add_argument("--reference", type=Path, required=True)
    drug_predict.add_argument("--controls", type=Path, required=True)
    drug_predict.add_argument("--output", type=Path, required=True)
    drug_predict.add_argument("--cells-per-target", type=int, default=400)
    drug_predict.add_argument("--steps", type=int, default=16)
    drug_strength = drug_predict.add_mutually_exclusive_group()
    drug_strength.add_argument("--effect-strength", type=float)
    drug_strength.add_argument("--effect-calibration", type=Path)
    drug_predict.add_argument(
        "--sampling", choices=["negative-binomial", "residual-rounding"],
        default="residual-rounding",
    )
    drug_predict.add_argument("--seed", type=int, default=0)
    drug_predict.add_argument("--device", default="cpu")

    fitting = commands.add_parser("train", help="train representation and flow models")
    fitting.add_argument("--train", type=Path, required=True)
    fitting.add_argument("--outdir", type=Path, required=True)
    fitting.add_argument("--control", default=CONTROL)
    fitting.add_argument("--latent-dim", type=int, default=128)
    fitting.add_argument("--hidden-dim", type=int, default=256)
    fitting.add_argument("--pretrain-steps", type=int, default=300)
    fitting.add_argument("--flow-steps", type=int, default=500)
    fitting.add_argument("--batch-size", type=int, default=64)
    fitting.add_argument("--learning-rate", type=float, default=1e-3)
    fitting.add_argument("--variance-weight", type=float, default=0.1)
    fitting.add_argument("--covariance-weight", type=float, default=0.0)
    fitting.add_argument("--regularization-view", choices=["masked", "unmasked"], default="masked")
    fitting.add_argument("--seed", type=int, default=0)
    fitting.add_argument("--device", default="cpu")
    prior = fitting.add_mutually_exclusive_group()
    prior.add_argument("--graph", type=Path)
    prior.add_argument("--epigenetic-bundle", type=Path)
    fitting.add_argument("--allow-unmatched-batch", action="store_true")
    fitting.add_argument(
        "--flow-target", choices=["paired", "population-shift"], default="paired",
    )
    fitting.add_argument("--min-group-cells", type=int, default=5)
    fitting.add_argument("--reference", type=Path)
    fitting.add_argument("--reference-fraction", type=float, default=0.5)
    fitting.add_argument("--reference-max-cells", type=int, default=10_000)
    fitting.add_argument("--reference-seed", type=int)
    fitting.add_argument("--reference-min-coverage", type=float, default=0.2)

    diagnostics = commands.add_parser(
        "diagnose-representation", help="measure variation of training-cell latent encodings"
    )
    diagnostics.add_argument("--checkpoint", type=Path, required=True)
    diagnostics.add_argument("--source", type=Path, required=True)
    diagnostics.add_argument("--output", type=Path, required=True)
    diagnostics.add_argument("--max-cells", type=int, default=512)
    diagnostics.add_argument("--batch-size", type=int, default=64)
    diagnostics.add_argument("--seed", type=int, default=0)
    diagnostics.add_argument("--device", default="cpu")

    for command, help_text in (
        ("calibrate-effect", "select flow strength using held-out training targets"),
        ("calibrate-drug-effect", "select flow strength using held-out training compounds"),
    ):
        effect = commands.add_parser(command, help=help_text)
        effect.add_argument("--source", type=Path, required=True)
        effect.add_argument("--checkpoint", type=Path, required=True)
        effect.add_argument("--outdir", type=Path, required=True)
        effect.add_argument("--folds", type=int, default=3)
        effect.add_argument(
            "--strengths", type=float, nargs="+", default=[0, 0.01, 0.05, 0.1, 0.25, 0.5, 1],
        )
        effect.add_argument("--min-relative-improvement", type=float, default=0.02)
        effect.add_argument(
            "--require-every-fold-improvement", action="store_true",
            help="select a nonzero effect only when it improves every training fold",
        )
        effect.add_argument(
            "--objective", choices=CALIBRATION_OBJECTIVES, default=OBJECTIVE,
            help="explicit calibration metric; the unnormalized option is diagnostic, not scored",
        )
        effect.add_argument("--pretrain-steps", type=int)
        effect.add_argument("--flow-steps", type=int)
        effect.add_argument("--batch-size", type=int)
        effect.add_argument("--learning-rate", type=float)
        effect.add_argument("--cells-per-target", type=int, default=400)
        effect.add_argument("--prediction-steps", type=int, default=16)
        effect.add_argument("--seed", type=int, default=0)
        effect.add_argument("--device", default="cpu")
        effect.add_argument("--threads", type=int, default=2)
        if command == "calibrate-effect":
            effect.add_argument(
                "--control-sampling", choices=["reference-batch", "input-controls"],
                default="reference-batch",
            )

    shared_calibration = commands.add_parser(
        "calibrate-shared-residual",
        help="select shared-response residual strength on target-disjoint training folds",
    )
    shared_calibration.add_argument("--source", type=Path, required=True)
    shared_calibration.add_argument("--checkpoint", type=Path, required=True)
    shared_calibration.add_argument("--profile", type=Path, required=True)
    shared_calibration.add_argument("--outdir", type=Path, required=True)
    shared_calibration.add_argument("--folds", type=int, default=3)
    shared_calibration.add_argument("--strengths", type=float, nargs="+", default=[0, 0.25, 0.5, 1])
    shared_calibration.add_argument("--min-relative-improvement", type=float, default=0.02)
    shared_calibration.add_argument(
        "--objective", choices=CALIBRATION_OBJECTIVES, default=OBJECTIVE,
        help="explicit calibration metric; the unnormalized option is diagnostic, not scored",
    )
    shared_calibration.add_argument("--shared-strength", type=float, default=1.0)
    shared_calibration.add_argument(
        "--include-null-control", action="store_true",
        help="allow a no-response control fallback selected on training folds",
    )
    shared_calibration.add_argument("--cells-per-target", type=int, default=400)
    shared_calibration.add_argument("--prediction-steps", type=int, default=8)
    shared_calibration.add_argument("--seed", type=int, default=0)
    shared_calibration.add_argument("--device", default="cpu")
    shared_calibration.add_argument("--threads", type=int, default=2)

    predict = commands.add_parser("predict", help="predict held-out targets from independent controls")
    predict.add_argument("--checkpoint", type=Path, required=True)
    predict.add_argument("--reference", type=Path, required=True)
    predict.add_argument("--controls", type=Path, required=True)
    predict.add_argument("--output", type=Path, required=True)
    predict.add_argument("--cells-per-target", type=int, default=100)
    predict.add_argument("--device", default="cpu")
    predict.add_argument("--seed", type=int, default=0)
    predict.add_argument("--steps", type=int, default=16)
    predict.add_argument("--allow-unmatched-batch", action="store_true")
    predict.add_argument(
        "--sampling", choices=["negative-binomial", "residual-rounding"],
        default="negative-binomial",
    )
    predict_strength = predict.add_mutually_exclusive_group()
    predict_strength.add_argument("--effect-strength", type=float)
    predict_strength.add_argument("--effect-calibration", type=Path)

    predict_targets = commands.add_parser(
        "predict-targets", help="predict requested unseen targets without reference expression",
    )
    predict_targets.add_argument("--checkpoint", type=Path, required=True)
    predict_targets.add_argument("--targets", type=Path, required=True)
    predict_targets.add_argument("--controls", type=Path, required=True)
    predict_targets.add_argument("--output", type=Path, required=True)
    predict_targets.add_argument("--cells-per-target", type=int, default=100)
    predict_targets.add_argument("--device", default="cpu")
    predict_targets.add_argument("--seed", type=int, default=0)
    predict_targets.add_argument("--steps", type=int, default=16)
    predict_targets.add_argument(
        "--sampling", choices=["negative-binomial", "residual-rounding"],
        default="negative-binomial",
    )
    target_strength = predict_targets.add_mutually_exclusive_group()
    target_strength.add_argument("--effect-strength", type=float)
    target_strength.add_argument("--effect-calibration", type=Path)

    ensemble = commands.add_parser(
        "predict-ensemble", help="mix held-out predictions and report model disagreement"
    )
    ensemble.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    ensemble.add_argument("--reference", type=Path, required=True)
    ensemble.add_argument("--controls", type=Path, required=True)
    ensemble.add_argument("--output", type=Path, required=True)
    ensemble.add_argument("--disagreement", type=Path, required=True)
    ensemble.add_argument("--cells-per-target", type=int, default=400)
    ensemble.add_argument("--device", default="cpu")
    ensemble.add_argument("--seed", type=int, default=0)
    ensemble.add_argument("--steps", type=int, default=16)
    ensemble.add_argument("--allow-unmatched-batch", action="store_true")
    ensemble.add_argument(
        "--sampling", choices=["negative-binomial", "residual-rounding"],
        default="residual-rounding",
    )
    ensemble.add_argument("--effect-strength", type=float, default=1.0)

    interval_calibration = commands.add_parser(
        "calibrate-intervals", help="calibrate target-level pseudobulk prediction intervals"
    )
    interval_calibration.add_argument("--prediction", type=Path, required=True)
    interval_calibration.add_argument("--disagreement", type=Path, required=True)
    interval_calibration.add_argument("--reference", type=Path, required=True)
    interval_calibration.add_argument("--controls", type=Path, required=True)
    interval_calibration.add_argument("--output", type=Path, required=True)
    interval_calibration.add_argument("--alpha", type=float, default=0.1)
    interval_calibration.add_argument("--scale-floor", type=float, default=1.0)

    intervals = commands.add_parser(
        "apply-intervals", help="apply independently calibrated pseudobulk intervals"
    )
    intervals.add_argument("--calibration", type=Path, required=True)
    intervals.add_argument("--prediction", type=Path, required=True)
    intervals.add_argument("--disagreement", type=Path, required=True)
    intervals.add_argument("--controls", type=Path, required=True)
    intervals.add_argument("--output", type=Path, required=True)

    interval_evaluation = commands.add_parser(
        "evaluate-intervals", help="measure empirical target-level interval coverage"
    )
    interval_evaluation.add_argument("--calibration", type=Path, required=True)
    interval_evaluation.add_argument("--prediction", type=Path, required=True)
    interval_evaluation.add_argument("--disagreement", type=Path, required=True)
    interval_evaluation.add_argument("--controls", type=Path, required=True)
    interval_evaluation.add_argument("--intervals", type=Path, required=True)
    interval_evaluation.add_argument("--reference", type=Path, required=True)
    interval_evaluation.add_argument("--output", type=Path, required=True)

    baseline = commands.add_parser(
        "control-baseline", help="predict held-out targets by resampling independent controls"
    )
    baseline.add_argument("--reference", type=Path, required=True)
    baseline.add_argument("--controls", type=Path, required=True)
    baseline.add_argument("--output", type=Path, required=True)
    baseline.add_argument("--control", default=CONTROL)
    baseline.add_argument("--cells-per-target", type=int, default=100)
    baseline.add_argument("--seed", type=int, default=0)
    baseline.add_argument("--allow-unmatched-batch", action="store_true")

    mean_fit = commands.add_parser(
        "fit-mean-response", help="fit an empirical matched-batch mean-response comparator"
    )
    mean_fit.add_argument("--train", type=Path, required=True)
    mean_fit.add_argument("--output", type=Path, required=True)
    mean_fit.add_argument("--control", default=CONTROL)
    mean_fit.add_argument("--min-group-cells", type=int, default=3)
    mean_fit.add_argument("--pseudocount", type=float, default=0.1)
    mean_fit.add_argument("--logfold-limit", type=float, default=2.0)

    mean_predict = commands.add_parser(
        "predict-mean-response", help="predict held-out targets with the empirical comparator"
    )
    mean_predict.add_argument("--profile", type=Path, required=True)
    mean_predict.add_argument("--reference", type=Path, required=True)
    mean_predict.add_argument("--controls", type=Path, required=True)
    mean_predict.add_argument("--output", type=Path, required=True)
    mean_predict.add_argument("--cells-per-target", type=int, default=400)
    mean_predict.add_argument("--seed", type=int, default=0)
    mean_predict.add_argument("--effect-strength", type=float, default=1.0)
    mean_predict.add_argument("--allow-unmatched-batch", action="store_true")
    mean_predict.add_argument("--device", default="cpu")

    hybrid = commands.add_parser(
        "predict-shared-residual",
        help="combine a training-derived shared response with a target-conditioned residual",
    )
    hybrid.add_argument("--profile", type=Path, required=True)
    hybrid.add_argument("--checkpoint", type=Path, required=True)
    hybrid.add_argument("--reference", type=Path, required=True)
    hybrid.add_argument("--controls", type=Path, required=True)
    hybrid.add_argument("--output", type=Path, required=True)
    hybrid.add_argument("--cells-per-target", type=int, default=400)
    hybrid.add_argument("--seed", type=int, default=0)
    hybrid.add_argument("--shared-strength", type=float, default=1.0)
    hybrid_strength = hybrid.add_mutually_exclusive_group()
    hybrid_strength.add_argument(
        "--conditional-strength", type=float,
        help="experimental target-specific residual strength (default: 0)",
    )
    hybrid_strength.add_argument(
        "--conditional-calibration", type=Path,
        help="use a strength fitted on independent training-target folds",
    )
    hybrid.add_argument("--steps", type=int, default=8)
    hybrid.add_argument("--allow-unmatched-batch", action="store_true")
    hybrid.add_argument("--device", default="cpu")

    for command in (shared_calibration, baseline, mean_predict, hybrid, predict, ensemble):
        command.add_argument(
            "--control-sampling", choices=["reference-batch", "input-controls"],
            default="reference-batch",
            help="sample by held-out batch distribution or only from input controls",
        )

    score = commands.add_parser("score", help="score with the official VCC 2026 metric preset")
    score.add_argument("--prediction", type=Path, required=True)
    score.add_argument("--reference", type=Path, required=True)
    score.add_argument("--outdir", type=Path, required=True)
    score.add_argument("--control", default=CONTROL)
    score.add_argument("--backend", choices=["pdex", "scanpy"], default="pdex")
    score.add_argument("--calibrate", action="store_true")
    score.add_argument("--threads", type=int, default=2)
    score.add_argument(
        "--metrics", nargs="+", choices=AVAILABLE_METRICS,
        help="evaluate an explicit scored or diagnostic metric subset",
    )
    score.add_argument(
        "--no-target-genes", action="store_true",
        help="declare non-genetic treatments such as drugs; not a VCC genetic score",
    )
    score.add_argument(
        "--stratify-by-batch", action="store_true",
        help="evaluate each context/batch combination against its own controls",
    )

    export = commands.add_parser("export-vcc", help="generate one VCC 2026-shaped H5AD")
    export.add_argument("--checkpoint", type=Path, required=True)
    export.add_argument("--controls", nargs=3, required=True, metavar="CONTEXT=PATH")
    export.add_argument("--gene-names", type=Path, required=True)
    export.add_argument("--pert-counts", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)
    export.add_argument("--device", default="cpu")
    export.add_argument("--seed", type=int, default=0)
    export.add_argument("--steps", type=int, default=16)
    export.add_argument("--allow-missing-target", action="store_true")
    export.add_argument(
        "--sampling", choices=["negative-binomial", "residual-rounding"],
        default="negative-binomial",
    )
    export_strength = export.add_mutually_exclusive_group()
    export_strength.add_argument("--effect-strength", type=float)
    export_strength.add_argument("--effect-calibration", type=Path)

    shared_export = commands.add_parser(
        "export-shared-vcc", help="export a calibrated shared-residual VCC 2026 prediction"
    )
    shared_export.add_argument("--checkpoint", type=Path, required=True)
    shared_export.add_argument("--profile", type=Path, required=True)
    shared_export.add_argument("--calibration", type=Path, required=True)
    shared_export.add_argument("--controls", nargs=3, required=True, metavar="CONTEXT=PATH")
    shared_export.add_argument("--gene-names", type=Path, required=True)
    shared_export.add_argument("--pert-counts", type=Path, required=True)
    shared_export.add_argument("--output", type=Path, required=True)
    shared_export.add_argument("--profile-mapping", type=Path)
    shared_export.add_argument("--shared-strength", type=float, default=1.0)
    shared_export.add_argument("--allow-unmatched-batch", action="store_true")
    shared_export.add_argument("--allow-missing-target", action="store_true")
    shared_export.add_argument("--device", default="cpu")
    shared_export.add_argument("--seed", type=int, default=0)
    shared_export.add_argument("--steps", type=int, default=8)
    return parser


def _control_paths(items: list[str]) -> dict[str, Path]:
    controls = {}
    for item in items:
        context, separator, path = item.partition("=")
        if not separator or not context or not path or context in controls:
            raise ValueError(f"Expected a unique CONTEXT=PATH argument, got {item!r}")
        controls[context] = Path(path)
    return controls


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "fetch-example":
        from .example import fetch_example

        fetch_example(args.output)
    elif args.command == "prepare":
        from .data import read_h5ad_subset, write_split

        dataset = read_h5ad_subset(
            args.input,
            label_column=args.label_column,
            control=args.control,
            max_per_target=args.max_per_target,
            max_controls=args.max_controls,
            max_targets=args.max_targets,
            seed=args.seed,
            include_targets=args.include_targets,
        )
        write_split(
            dataset, args.outdir, control=args.control,
            holdout_fraction=args.holdout_fraction, seed=args.seed, max_genes=args.max_genes,
            holdout_targets=args.holdout_targets,
        )
    elif args.command == "inspect-orion":
        from .orion import inspect_shards

        if args.output.exists():
            raise FileExistsError(f"Refusing to overwrite {args.output}")
        frame = inspect_shards(args.shards)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(args.output, index=False)
        print(f"Saved {len(frame)} context/target/sample groups to {args.output}")
    elif args.command == "prepare-orion":
        from .orion import prepare_orion

        prepare_orion(
            args.shards, args.gene_metadata, args.output, targets=args.targets,
            control_label=args.control_label, max_per_target=args.max_per_target,
            max_controls=args.max_controls, seed=args.seed,
        )
    elif args.command == "build-epigenetic-bundle":
        from .epigenetics import build_epigenetic_bundle

        manifest = build_epigenetic_bundle(
            args.gene_data, args.spec, args.outdir, promoter_flank=args.promoter_flank,
        )
        print(f"Saved epigenomic bundle: {manifest}")
    elif args.command == "shuffle-epigenetic-bundle":
        from .epigenetics import shuffle_epigenetic_bundle

        manifest = shuffle_epigenetic_bundle(
            args.bundle, args.gene_data, args.outdir, seed=args.seed,
            shuffle_features=not args.graph_only,
        )
        print(f"Saved shuffled control bundle: {manifest}")
    elif args.command == "extract-mcool-contacts":
        from .mcool import extract_contact_candidates

        source = extract_contact_candidates(
            args.mcool, args.source_metadata, args.chrom_sizes, args.outdir,
            resolution=args.resolution, chromosomes=args.chromosomes,
            max_distance=args.max_distance, min_count=args.min_count,
            min_observed_expected=args.min_observed_expected,
            allow_assembly_inference=args.allow_assembly_inference,
        )
        print(f"Saved candidate contacts and metadata: {source}")
    elif args.command == "inspect-tahoe":
        from .tahoe import inspect_tahoe

        if args.output.exists():
            raise FileExistsError(f"Refusing to overwrite {args.output}")
        frame = inspect_tahoe(args.shards)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(args.output, index=False)
        print(f"Saved {len(frame)} drug/plate/cell-line groups to {args.output}")
    elif args.command == "prepare-tahoe":
        from .tahoe import prepare_tahoe

        prepare_tahoe(
            args.shards, args.gene_metadata, args.sample_metadata, args.output,
            drugs=args.drugs, cell_lines=args.cell_lines, plates=args.plates,
            max_per_condition=args.max_per_condition,
            max_controls_per_group=args.max_controls_per_group, seed=args.seed,
        )
    elif args.command == "split-tahoe":
        import anndata as ad

        from .data import write_split
        from .drug import validate_tahoe

        data = ad.read_h5ad(args.input)
        validate_tahoe(data)
        write_split(
            data, args.outdir, control=CONTROL,
            holdout_fraction=args.holdout_fraction, seed=args.seed, max_genes=args.max_genes,
            split_column="compound", require_targets_on_gene_axis=False,
        )
    elif args.command == "train-drug":
        from .drug import train_drug

        checkpoint = train_drug(
            args.train, args.outdir, latent_dim=args.latent_dim, hidden_dim=args.hidden_dim,
            pretrain_steps=args.pretrain_steps, flow_steps=args.flow_steps,
            batch_size=args.batch_size, learning_rate=args.learning_rate,
            variance_weight=args.variance_weight,
            covariance_weight=args.covariance_weight, regularization_view=args.regularization_view,
            seed=args.seed, device_name=args.device,
            reference_path=args.reference, reference_fraction=args.reference_fraction,
            reference_max_cells=args.reference_max_cells, reference_seed=args.reference_seed,
            reference_min_coverage=args.reference_min_coverage,
        )
        print(f"Saved drug checkpoint: {checkpoint}")
    elif args.command == "predict-drug":
        from .drug import predict_drug

        predict_drug(
            args.checkpoint, args.reference, args.controls, args.output,
            cells_per_target=args.cells_per_target, steps=args.steps,
            effect_strength=args.effect_strength, sampling=args.sampling,
            calibration_path=args.effect_calibration,
            seed=args.seed, device_name=args.device,
        )
    elif args.command == "train":
        from .training import train

        checkpoint = train(
            args.train, args.outdir, control=args.control, latent_dim=args.latent_dim,
            hidden_dim=args.hidden_dim, pretrain_steps=args.pretrain_steps,
            flow_steps=args.flow_steps, batch_size=args.batch_size,
            learning_rate=args.learning_rate, variance_weight=args.variance_weight,
            covariance_weight=args.covariance_weight, regularization_view=args.regularization_view,
            seed=args.seed, device_name=args.device,
            graph_path=args.graph, epigenetic_bundle=args.epigenetic_bundle,
            allow_unmatched_batch=args.allow_unmatched_batch,
            flow_target=args.flow_target, min_group_cells=args.min_group_cells,
            reference_path=args.reference, reference_fraction=args.reference_fraction,
            reference_max_cells=args.reference_max_cells, reference_seed=args.reference_seed,
            reference_min_coverage=args.reference_min_coverage,
        )
        print(f"Saved checkpoint: {checkpoint}")
    elif args.command == "diagnose-representation":
        from .diagnostics import diagnose_checkpoint

        diagnose_checkpoint(
            args.checkpoint, args.source, args.output, max_cells=args.max_cells,
            batch_size=args.batch_size, seed=args.seed, device_name=args.device,
        )
    elif args.command in {"calibrate-effect", "calibrate-drug-effect"}:
        from .calibration import calibrate_effect

        calibrate_effect(
            args.source, args.checkpoint, args.outdir, folds=args.folds,
            strengths=tuple(args.strengths), min_relative_improvement=args.min_relative_improvement,
            pretrain_steps=args.pretrain_steps, flow_steps=args.flow_steps,
            batch_size=args.batch_size, learning_rate=args.learning_rate,
            cells_per_target=args.cells_per_target, prediction_steps=args.prediction_steps,
            seed=args.seed, device_name=args.device, threads=args.threads,
            kind="drug" if args.command == "calibrate-drug-effect" else "genetic",
            control_sampling=getattr(args, "control_sampling", "reference-batch"),
            objective=args.objective,
            require_every_fold_improvement=args.require_every_fold_improvement,
        )
    elif args.command == "calibrate-shared-residual":
        from .shared_calibration import calibrate_shared_residual

        calibrate_shared_residual(
            args.source, args.checkpoint, args.profile, args.outdir, folds=args.folds,
            strengths=tuple(args.strengths), min_relative_improvement=args.min_relative_improvement,
            shared_strength=args.shared_strength, include_no_response=args.include_null_control,
            control_sampling=args.control_sampling,
            cells_per_target=args.cells_per_target,
            prediction_steps=args.prediction_steps, seed=args.seed, device_name=args.device,
            threads=args.threads, objective=args.objective,
        )
    elif args.command == "predict":
        from .predict import predict_holdout

        predict_holdout(
            args.checkpoint, args.reference, args.controls, args.output,
            cells_per_target=args.cells_per_target, device_name=args.device,
            seed=args.seed, steps=args.steps,
            allow_unmatched_batch=args.allow_unmatched_batch, sampling=args.sampling,
            effect_strength=args.effect_strength, calibration_path=args.effect_calibration,
            control_sampling=args.control_sampling,
        )
    elif args.command == "predict-targets":
        from .predict import predict_targets

        predict_targets(
            args.checkpoint, args.targets, args.controls, args.output,
            cells_per_target=args.cells_per_target, device_name=args.device,
            seed=args.seed, steps=args.steps, sampling=args.sampling,
            effect_strength=args.effect_strength, calibration_path=args.effect_calibration,
        )
    elif args.command == "predict-ensemble":
        from .ensemble import predict_ensemble

        predict_ensemble(
            args.checkpoints, args.reference, args.controls, args.output, args.disagreement,
            cells_per_target=args.cells_per_target, device_name=args.device,
            seed=args.seed, steps=args.steps,
            allow_unmatched_batch=args.allow_unmatched_batch, sampling=args.sampling,
            effect_strength=args.effect_strength, control_sampling=args.control_sampling,
        )
    elif args.command == "calibrate-intervals":
        from .uncertainty import calibrate_intervals

        calibrate_intervals(
            args.prediction, args.disagreement, args.reference, args.controls, args.output,
            alpha=args.alpha, scale_floor=args.scale_floor,
        )
    elif args.command == "apply-intervals":
        from .uncertainty import apply_intervals

        apply_intervals(
            args.calibration, args.prediction, args.disagreement, args.controls, args.output,
        )
    elif args.command == "evaluate-intervals":
        from .uncertainty import evaluate_intervals

        evaluate_intervals(
            args.calibration, args.prediction, args.disagreement, args.controls,
            args.intervals, args.reference, args.output,
        )
    elif args.command == "control-baseline":
        from .predict import predict_control_baseline

        predict_control_baseline(
            args.reference, args.controls, args.output, control=args.control,
            cells_per_target=args.cells_per_target, seed=args.seed,
            allow_unmatched_batch=args.allow_unmatched_batch, control_sampling=args.control_sampling,
        )
    elif args.command == "fit-mean-response":
        from .mean_response import fit_mean_response

        fit_mean_response(
            args.train, args.output, control=args.control,
            min_group_cells=args.min_group_cells, pseudocount=args.pseudocount,
            logfold_limit=args.logfold_limit,
        )
    elif args.command == "predict-mean-response":
        from .mean_response import predict_mean_response

        predict_mean_response(
            args.profile, args.reference, args.controls, args.output,
            cells_per_target=args.cells_per_target, seed=args.seed,
            effect_strength=args.effect_strength,
            allow_unmatched_batch=args.allow_unmatched_batch, control_sampling=args.control_sampling,
            device_name=args.device,
        )
    elif args.command == "predict-shared-residual":
        from .mean_response import predict_mean_response

        predict_mean_response(
            args.profile, args.reference, args.controls, args.output,
            cells_per_target=args.cells_per_target, seed=args.seed,
            effect_strength=args.shared_strength,
            allow_unmatched_batch=args.allow_unmatched_batch, control_sampling=args.control_sampling,
            device_name=args.device,
            checkpoint_path=args.checkpoint, conditional_strength=args.conditional_strength,
            conditional_calibration_path=args.conditional_calibration,
            steps=args.steps,
        )
    elif args.command == "score":
        from .scoring import SCORED_METRICS, score_local

        score_local(
            args.prediction, args.reference, args.outdir, control=args.control,
            backend=args.backend, calibrate=args.calibrate, threads=args.threads,
            metrics=tuple(args.metrics) if args.metrics else SCORED_METRICS,
            no_target_genes=args.no_target_genes,
            stratify_by_batch=args.stratify_by_batch,
        )
    elif args.command == "export-vcc":
        from .predict import export_vcc_h5ad

        export_vcc_h5ad(
            args.checkpoint, _control_paths(args.controls), args.gene_names, args.pert_counts, args.output,
            device_name=args.device, seed=args.seed, steps=args.steps,
            allow_missing_target=args.allow_missing_target, sampling=args.sampling,
            effect_strength=args.effect_strength, calibration_path=args.effect_calibration,
        )
    elif args.command == "export-shared-vcc":
        from .shared_export import export_shared_vcc_h5ad

        export_shared_vcc_h5ad(
            args.checkpoint, args.profile, args.calibration, _control_paths(args.controls),
            args.gene_names, args.pert_counts, args.output,
            profile_mapping_path=args.profile_mapping, shared_strength=args.shared_strength,
            allow_unmatched_batch=args.allow_unmatched_batch,
            allow_missing_target=args.allow_missing_target, device_name=args.device,
            seed=args.seed, steps=args.steps,
        )


if __name__ == "__main__":
    main()
