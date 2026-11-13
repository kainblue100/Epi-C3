"""Render an original, clearly scoped comparison from aggregated results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


COLORS = {
    "legacy-epi-graph": "#087c68",
    "legacy-epi-shuffled": "#5c9c91",
    "legacy-epi-nodes": "#80aea3",
    "legacy-rna": "#405c5c",
    "gears": "#7f5a90",
    "training-mean": "#bc8a45",
    "no-effect": "#8d9297",
}


def render(summary_path: Path, output_dir: Path) -> None:
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    if (
        data.get("format") != "epi-c3-posthoc-legacy-k562-comparison-v1"
        or data.get("official_state_benchmark") is not False
        or data.get("official_cell_eval_result") is not False
        or set(item["id"] for item in data["methods"]) != set(COLORS)
    ):
        raise ValueError("Unexpected comparison format or scope")
    methods = data["methods"]
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = [output_dir / f"legacy-k562-comparison.{ext}" for ext in ("svg", "png", "pdf")]
    data_js = output_dir / "comparison-data.js"
    if any(path.exists() for path in [*outputs, data_js]):
        raise FileExistsError("Refusing to overwrite an existing figure or data script")

    matplotlib.rcParams.update({
        "font.family": "DejaVu Sans",
        "svg.fonttype": "none",
        "svg.hashsalt": "epi-c3-legacy-k562-comparison-v1",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
    })
    fig, axes = plt.subplots(1, 3, figsize=(15.4, 7.4), sharey=True)
    fig.patch.set_facecolor("#fbfbf8")
    fig.subplots_adjust(left=0.235, right=0.975, top=0.75, bottom=0.31, wspace=0.32)
    fig.text(0.035, 0.955, "EPI-C3 / COMPARISON SNAPSHOT", color="#087c68", fontsize=11, weight="bold")
    fig.text(0.035, 0.905, "Legacy K562: shared evaluation, not a STATE benchmark", fontsize=19, weight="bold")
    fig.text(
        0.035, 0.865,
        "Exploratory post-hoc analysis  |  347 held-out targets  |  1,860 non-target genes per target",
        fontsize=10.5, color="#4d5d5a",
    )
    y = np.arange(len(methods))[::-1]
    configs = [
        ("mse", "Mean-expression MSE", "Lower is better", ".6f"),
        ("delta_pearson", "Response Pearson", "Higher is better", ".4f"),
        ("discrimination", "Perturbation discrimination", "Higher is better", ".4f"),
    ]
    for ax, (metric, title, direction, number_format) in zip(axes, configs, strict=True):
        ax.set_facecolor("#fbfbf8")
        ax.set_title(title + "\n" + direction, loc="left", fontsize=11, weight="bold", pad=15)
        values = [method["metrics"][metric]["mean_of_run_macros"] for method in methods]
        finite = [value for value in values if value is not None]
        if not finite:
            raise ValueError(f"No defined values for {metric}")
        low, high = min(finite), max(finite)
        width = max(high - low, abs(high) * 0.02, 1e-6)
        lower = low - width * 0.2
        upper = high + width * 0.52
        ax.set_xlim(lower, upper)
        ax.set_ylim(-0.6, len(methods) - 0.4)
        ax.grid(axis="x", color="#dde6e1", lw=0.8)
        ax.set_axisbelow(True)
        for row, value, method in zip(y, values, methods, strict=True):
            if value is None:
                ax.text(lower + (upper - lower) * 0.03, row, "undefined", color="#777f7b", va="center", fontsize=9)
            else:
                ax.scatter(value, row, color=COLORS[method["id"]], s=65, zorder=3)
                ax.annotate(format(value, number_format), (value, row), xytext=(7, 0),
                            textcoords="offset points", va="center", fontsize=8.5, color="#1e3530")
        ax.tick_params(axis="x", labelsize=8, length=0, pad=5, colors="#51635e")
        ax.tick_params(axis="y", length=0, labelsize=9.4, pad=8, colors="#263a35")
        ax.ticklabel_format(axis="x", style="plain", useOffset=False)
    axes[0].set_yticks(y, [method["label"] for method in methods])
    for ax in axes[1:]:
        ax.tick_params(axis="y", labelleft=False)

    fig.text(
        0.035, 0.23,
        "NOT EVALUATED HERE: STATE, PerturbMean, ContextMean, Linear, scVI, CPA, scGPT architecture, "
        "and the two new Epi-C3 branches.",
        fontsize=9.5, weight="bold", color="#7a4c24",
    )
    fig.text(
        0.035, 0.195,
        "DE p-value AUPRC, log-fold-change correlation, DEG overlap and effect-size correlation: unavailable from saved means.",
        fontsize=9.2, color="#354a43",
    )
    fig.text(
        0.035, 0.151,
        "Single K562 context; previously examined test; GEARS has an additional GO prior and one seed. "
        "Epi-C3 has three response seeds sharing one encoder.",
        fontsize=9.1, color="#354a43",
    )
    fig.text(
        0.035, 0.119,
        "All scores are custom population-mean metrics, not official cell-eval scores. "
        "A constant predicted effect has undefined Pearson correlation.",
        fontsize=9.1, color="#354a43",
    )
    fig.text(
        0.035, 0.075,
        "Point estimates only: no independent biological replicate or confidence interval is claimed. "
        "Axis ranges are truncated; read numerical labels.",
        fontsize=9.1, color="#354a43",
    )
    fig.savefig(outputs[0], bbox_inches="tight", metadata={"Date": None})
    fig.savefig(outputs[1], dpi=220, bbox_inches="tight", metadata={"Software": "Matplotlib"})
    fig.savefig(outputs[2], bbox_inches="tight", metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)
    with data_js.open("x", encoding="utf-8", newline="\n") as file:
        file.write("/* Generated from k562-comparison.json; do not edit. */\n")
        file.write("const comparisonData = ")
        json.dump(data, file, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        file.write(";\nif (typeof module !== 'undefined') module.exports = comparisonData;\n")
        file.write("if (typeof window !== 'undefined') window.comparisonData = comparisonData;\n")
    print("Rendered original SVG, PNG, PDF, and data script")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    render(args.comparison, args.output_dir)


if __name__ == "__main__":
    main()
