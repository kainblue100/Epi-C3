"""Reproduce an explicitly scoped figure from the authors' saved STATE scores."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path


REVISION = "ef2f1e13966a2a8f61fb3e8d178fae2b46aad195"
BASE_URL = f"https://huggingface.co/arcinstitute/ST-HVG-Tahoe/resolve/{REVISION}/benchmark"
SOURCE_HASHES = {
    "contexts": "ad015f3bdc22079296cb1a892bc2b08fbdce3ccee7a9ae34ea9dca06b7aa740c",
    "datasets": "81bd577d96dad435dfad7d9fe9e9e5434cfb9e41507276818603d85ba57dce75",
    "index": "ce3532090837241d9253aa3cff5648e59cfa1e99faa57bfae74b9bbf1bc66f42",
}
CONTEXTS = {
    "Tahoe": ("C32", "HOP62", "HepG2-C3A", "Hs 766T", "PANC-1"),
    "Parse": ("B_Intermediate_Memory", "B_Naive", "CD14_Mono"),
    "Replogle": ("hepg2", "jurkat", "k562", "rpe1"),
}
DATASET_LABELS = {
    "Tahoe": "Tahoe-100M\nChemical",
    "Parse": "Parse-PBMC\nSignaling",
    "Replogle": "Replogle-Nadig\nGenetic",
}
METRICS = (
    ("discrimination_score_l1", "PDS L1\nsaved field"),
    ("pearson_delta", "Pearson\nexpression delta"),
    ("pr_auc", "DE significance\nAUPRC"),
    ("de_spearman_lfc_sig", "DE logFC\nSpearman"),
    ("overlap_at_N", "DE overlap\nat N"),
    ("de_spearman_sig", "DE-count\nSpearman"),
)
REPRESENTATIONS = ("X_hvg", "X_state")


def verify_source(path: Path, expected: str) -> None:
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise ValueError(f"Source SHA-256 mismatch for {path.name}: {actual}")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _finite(value: str, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Non-finite {label}")
    return result


def _check_index(index: dict) -> None:
    for dataset, expected in CONTEXTS.items():
        for representation in REPRESENTATIONS:
            models = [
                model for model in index["models"]
                if model["dataset"] == dataset
                and model["regime"] == "fewshot"
                and model["representation"] == representation
            ]
            if not models or any(model["origin"] != "original fewshot" for model in models):
                raise ValueError(f"Unexpected run origin: {dataset}/{representation}")
            available = [
                evaluation["context"] for model in models
                for evaluation in model["evaluations"]
            ]
            if len(available) != len(set(available)) or not set(expected).issubset(available):
                raise ValueError(f"Unexpected saved contexts: {dataset}/{representation}")
            if dataset != "Parse" and set(available) != set(expected):
                raise ValueError(f"Unexpected extra contexts: {dataset}/{representation}")
            if dataset == "Parse" and representation == "X_state":
                if set(available) != set(expected) or not any(
                    "CD4_Memory" in model["missing_contexts"] for model in models
                ):
                    raise ValueError("The documented Parse CD4 Memory gap changed")
            if dataset == "Parse" and representation == "X_hvg":
                if set(available) != set(expected) | {"CD4_Memory"}:
                    raise ValueError("Unexpected Parse HVG contexts")


def build_summary(context_rows: list[dict], dataset_rows: list[dict], index: dict) -> dict:
    _check_index(index)
    datasets = []
    for dataset, expected_contexts in CONTEXTS.items():
        panels = []
        for metric, title in METRICS:
            rows = [
                row for row in context_rows
                if row["dataset"] == dataset and row["regime"] == "fewshot"
                and row["metric"] == metric
            ]
            by_context = {row["context"]: row for row in rows}
            if len(rows) != len(by_context) or set(by_context) != set(expected_contexts):
                raise ValueError(f"Unexpected contexts for {dataset}/{metric}")
            aggregate = [
                row for row in dataset_rows
                if row["dataset"] == dataset and row["regime"] == "fewshot"
                and row["metric"] == metric
            ]
            if len(aggregate) != 1:
                raise ValueError(f"Unexpected aggregate rows for {dataset}/{metric}")
            reference = aggregate[0]
            if int(reference["n_contexts"]) != len(expected_contexts):
                raise ValueError(f"Context count mismatch for {dataset}/{metric}")
            paired_counts = [int(by_context[context]["n_pairs"]) for context in expected_contexts]
            if min(paired_counts) < 1 or sum(paired_counts) != int(reference["n_pairs"]):
                raise ValueError(f"Paired count mismatch for {dataset}/{metric}")
            scores = {}
            for representation in REPRESENTATIONS:
                values = [
                    _finite(by_context[context][representation], f"{dataset}/{metric}/{context}")
                    for context in expected_contexts
                ]
                mean = statistics.fmean(values)
                if not math.isclose(
                    mean, _finite(reference[representation], "aggregate"), rel_tol=0.0, abs_tol=1e-12
                ):
                    raise ValueError(f"Aggregate mismatch for {dataset}/{metric}/{representation}")
                scores[representation] = {
                    "mean": mean,
                    "population_sd": statistics.pstdev(values),
                    "by_context": [
                        {"context": context, "score": value, "n_pairs": count}
                        for context, value, count in zip(expected_contexts, values, paired_counts, strict=True)
                    ],
                }
            panels.append({
                "metric": metric,
                "title": title.replace("\n", " "),
                "n_contexts": len(expected_contexts),
                "n_pairs_total": sum(paired_counts),
                "scores": scores,
            })
        datasets.append({"id": dataset, "label": DATASET_LABELS[dataset], "panels": panels})
    return {
        "format": "epi-c3-author-state-reference-v2",
        "citation": "Adduri et al., Cell 189, 5914-5931 (2026), doi:10.1016/j.cell.2026.07.052",
        "project_same_protocol_evaluation": False,
        "article_figure_2_reproduction": False,
        "scope": (
            "Authors' saved CellEval v1 fewshot results; paired finite perturbations, "
            "equal-weight context means. Population SD was computed here across context means."
        ),
        "caveats": [
            "These are authors' saved results, not an Epi-C3 experiment.",
            "Parse has three matched contexts because the SE CD4 Memory result is missing; the paper reports four splits.",
            "The original saved L1 field is not transformed to the paper's normalized rank.",
            "HVG and SE runs are descriptive saved experiments with differing recipes, not a controlled causal comparison.",
            "The requested non-STATE baselines and both new Epi-C3 branches are not included.",
        ],
        "source": {
            "repository": "arcinstitute/ST-HVG-Tahoe",
            "revision": REVISION,
            "updated": index["updated"],
            "context_csv": f"{BASE_URL}/context_metrics.csv",
            "dataset_csv": f"{BASE_URL}/dataset_metrics.csv",
            "model_index": f"{BASE_URL}/model_index.json",
            "author_recomputation": f"{BASE_URL}/recompute_benchmark.py",
            "sha256": SOURCE_HASHES,
        },
        "datasets": datasets,
    }


def render(summary: dict, output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import Patch

    matplotlib.rcParams.update({
        "font.family": "DejaVu Sans",
        "svg.fonttype": "none",
        "svg.hashsalt": "epi-c3-author-state-reference-v2",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    colors = {"X_hvg": "#1685b8", "X_state": "#d49238"}
    fig, axes = plt.subplots(3, 6, figsize=(18.0, 9.5), sharey=True)
    fig.patch.set_facecolor("#f8faf8")
    fig.subplots_adjust(left=0.13, right=0.99, top=0.78, bottom=0.29, wspace=0.3, hspace=0.34)
    fig.text(0.035, 0.96, "EPI-C3 / PAPER REFERENCE / 2026-11-13", color="#087c68", weight="bold", fontsize=10)
    fig.text(0.035, 0.922, "Authors' saved STATE results: three datasets, six metrics", weight="bold", fontsize=18)
    fig.text(
        0.035, 0.885,
        "Reference only: not an Epi-C3 benchmark and not an exact reproduction of Cell Figure 2.",
        color="#526861", fontsize=10.5,
    )
    fig.legend(
        handles=[
            Patch(facecolor=colors["X_hvg"], label="ST + HVG (paper Figure 2 representation)"),
            Patch(facecolor=colors["X_state"], label="ST + SE (additional authors' saved run)"),
        ],
        loc="upper left", bbox_to_anchor=(0.035, 0.866), ncol=2, frameon=False, fontsize=9.5,
    )
    for row_index, dataset in enumerate(summary["datasets"]):
        for col_index, panel in enumerate(dataset["panels"]):
            ax = axes[row_index, col_index]
            ax.set_facecolor("#f8faf8")
            ax.set_ylim(-0.02, 1.18)
            ax.set_yticks([0, 0.5, 1.0])
            ax.grid(axis="y", color="#e0e8e3", linewidth=0.7)
            ax.set_axisbelow(True)
            ax.tick_params(axis="both", labelsize=8, length=0, colors="#526861")
            ax.set_xticks([0, 1], ["HVG", "SE"])
            if row_index == 0:
                ax.set_title(METRICS[col_index][1], fontsize=9.5, weight="bold", pad=10)
            for x, representation in enumerate(REPRESENTATIONS):
                score = panel["scores"][representation]
                mean, sd = score["mean"], score["population_sd"]
                ax.bar(x, mean, width=0.62, color=colors[representation], zorder=2)
                ax.errorbar(x, mean, yerr=sd, color="#263b36", capsize=3, linewidth=1, fmt="none", zorder=3)
                points = score["by_context"]
                jitter = np.linspace(-0.10, 0.10, len(points))
                ax.scatter(
                    x + jitter, [point["score"] for point in points],
                    s=12, color="#263b36", alpha=0.72, zorder=4,
                )
                ax.text(x, mean + sd + 0.035, f"{mean:.3f}", ha="center", va="bottom", fontsize=7.6)
            if col_index == 0:
                box = ax.get_position()
                fig.text(
                    0.035, (box.y0 + box.y1) / 2,
                    f"{dataset['label']}\n{panel['n_contexts']} contexts",
                    va="center", fontsize=9.5, weight="bold", color="#29463d",
                )
    footnotes = [
        "Bars: equal-weight context mean; whiskers: population SD across available contexts; dots: individual context means.",
        "Parse has 3 matched contexts (CD4 Memory SE output missing); the paper's Figure 2 reports 4 splits.",
        "PDS is the saved L1 field, not converted to the paper's normalized rank; CellEval versions can differ.",
        "HVG and SE recipes differ. Other paper baselines and both new Epi-C3 models are not evaluated in this figure.",
        "Source: Adduri et al., Cell 189, 5914-5931 (2026), doi:10.1016/j.cell.2026.07.052.",
    ]
    for i, line in enumerate(footnotes):
        fig.text(0.035, 0.235 - i * 0.035, line, fontsize=9, color="#425c51")
    fig.savefig(output / "state-author-reference.svg", bbox_inches="tight", metadata={"Date": None})
    fig.savefig(output / "state-author-reference.png", bbox_inches="tight", dpi=200, metadata={"Software": "Matplotlib"})
    fig.savefig(output / "state-author-reference.pdf", bbox_inches="tight", metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contexts", required=True, type=Path)
    parser.add_argument("--datasets", required=True, type=Path)
    parser.add_argument("--index", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    for key in ("contexts", "datasets", "index"):
        verify_source(getattr(args, key), SOURCE_HASHES[key])
    summary = build_summary(
        read_csv(args.contexts),
        read_csv(args.datasets),
        json.loads(args.index.read_text(encoding="utf-8")),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = [
        args.output_dir / name for name in (
            "state-author-reference.json", "state-author-reference.js",
            "state-author-reference.svg", "state-author-reference.png", "state-author-reference.pdf",
        )
    ]
    if any(path.exists() for path in outputs):
        raise FileExistsError("Refusing to overwrite an existing reference snapshot")
    render(summary, args.output_dir)
    with outputs[0].open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    with outputs[1].open("x", encoding="utf-8", newline="\n") as stream:
        stream.write("/* Generated from authors' saved reference scores; do not edit. */\n")
        stream.write("const stateAuthorReference = ")
        json.dump(summary, stream, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        stream.write(";\nif (typeof module !== 'undefined') module.exports = stateAuthorReference;\n")
        stream.write("if (typeof window !== 'undefined') window.stateAuthorReference = stateAuthorReference;\n")
    print("Verified author sources and rendered the reference figure and aggregate evidence.")


if __name__ == "__main__":
    main()
