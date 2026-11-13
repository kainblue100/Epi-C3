"""Render six exploratory STATE pilot metrics from the public aggregate."""

import argparse
import json
import math
from pathlib import Path


MODELS = ("state", "perturb_mean", "context_mean")
PANELS = (
    ("discrimination_score_l1", "L1 rank\n(cell-eval raw)"),
    ("pearson_delta", "Pearson\nexpression delta"),
    ("pr_auc", "DE AUPRC\n(average precision)"),
    ("de_spearman_lfc_sig", "DE logFC\nSpearman"),
    ("overlap_at_N", "DE overlap\nat N"),
    ("de_count_spearman_all_conditions", "DE-count\nSpearman"),
)
LABELS = {"state": "STATE", "perturb_mean": "PerturbMean", "context_mean": "ContextMean"}
COLORS = {"state": "#087c68", "perturb_mean": "#c47b2d", "context_mean": "#456c9b"}


def validate(data):
    if data.get("format") != "epi-c3-state-replogle-pilot-public-v1":
        raise ValueError("Unexpected public aggregate format")
    if [row["key"] for row in data["metrics"]] != [key for key, _ in PANELS]:
        raise ValueError("Missing or reordered metrics")
    for row in data["metrics"]:
        if set(row["scores"]) != set(MODELS) or not 1 <= row["common_conditions"] <= 28:
            raise ValueError("Unexpected model or common-condition support")
        for value in row["scores"].values():
            if value is not None and (not math.isfinite(value) or not -1 <= value <= 1):
                raise ValueError("A metric is nonfinite or outside its possible range")


def render(data, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    validate(data)
    paths = [output / f"state-pilot-comparison.{ext}" for ext in ("svg", "png", "pdf")]
    data_script = output / "pilot-results.js"
    if any(path.exists() for path in [*paths, data_script]):
        raise FileExistsError("Refusing to overwrite an existing rendered artifact")
    matplotlib.rcParams.update({
        "font.family": "DejaVu Sans",
        "svg.fonttype": "none",
        "svg.hashsalt": "epi-c3-state-replogle-pilot-v1",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    fig, axes = plt.subplots(1, 6, figsize=(18, 5.3))
    fig.patch.set_facecolor("#f8faf8")
    fig.subplots_adjust(left=0.045, right=0.99, top=0.69, bottom=0.29, wspace=0.45)
    fig.text(0.035, 0.96, "EPI-C3 / EXPLORATORY PILOT / 2026-11-13",
             fontsize=10, weight="bold", color="#087c68")
    fig.text(0.035, 0.908, "STATE and mean baselines on one held-out RPE1 split",
             fontsize=17, weight="bold", color="#17362f")
    fig.text(0.035, 0.865,
             "Two cell lines; 40 targets; 28 test perturbations. Not the paper benchmark or Epi-C3 models.",
             fontsize=9.7, color="#526861")
    fig.legend(
        handles=[Patch(color=COLORS[m], label=LABELS[m]) for m in MODELS],
        loc="upper left", bbox_to_anchor=(0.035, 0.842), ncol=3, frameon=False, fontsize=9,
    )
    for panel_index, (ax, (_, title), row) in enumerate(zip(axes, PANELS, data["metrics"], strict=True)):
        values = [row["scores"][model] for model in MODELS]
        defined = [value for value in values if value is not None]
        span_low = min(0.0, min(defined))
        span_high = max(0.0, max(defined))
        margin = max(span_high - span_low, 0.06)
        lower = span_low - margin * 0.24 if span_low < 0 else 0
        upper = span_high + margin * 0.35
        ax.set_ylim(lower, upper)
        ax.set_facecolor("#f8faf8")
        ax.grid(axis="y", color="#dce8e1", linewidth=0.7)
        ax.set_axisbelow(True)
        ax.axhline(0, color="#8ba298", linewidth=0.7)
        ax.tick_params(axis="both", labelsize=7.5, length=0, colors="#526861")
        ax.set_xticks(range(3), ["ST", "PM", "CM"])
        ax.set_title(f"{chr(65 + panel_index)}  {title}\n(n={row['common_conditions']})",
                     fontsize=9.5, weight="bold", pad=10)
        for index, (model, value) in enumerate(zip(MODELS, values, strict=True)):
            if value is None:
                ax.text(index, lower + (upper - lower) * 0.07, "N/A",
                        ha="center", va="bottom", fontsize=8.5, color="#526861", weight="bold")
                continue
            ax.bar(index, value, width=0.65, color=COLORS[model])
            offset = (upper - lower) * 0.03
            ax.text(index, value + offset if value >= 0 else value - offset,
                    f"{value:.3f}", ha="center", va="bottom" if value >= 0 else "top",
                    fontsize=8, color="#17362f")
        ax.set_xlim(-0.5, 2.5)
    fig.text(0.035, 0.16,
             "Bars are equally weighted means over common valid conditions, except DE-count Spearman",
             fontsize=9, color="#526861")
    fig.text(0.035, 0.12,
             "(one correlation across 28 conditions). N/A = undefined; no uncertainty bars; panel axes differ.",
             fontsize=9, color="#526861")
    fig.text(0.035, 0.075,
             "STATE: random initialization, 512 CPU steps. Repeated predictions and clipping limit DE inference.",
             fontsize=9, color="#526861")
    fig.savefig(paths[0], facecolor=fig.get_facecolor(), metadata={"Date": None})
    fig.savefig(paths[1], dpi=180, facecolor=fig.get_facecolor(), metadata={"Software": "Matplotlib"})
    fig.savefig(paths[2], facecolor=fig.get_facecolor(), metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)
    svg = paths[0].read_text(encoding="utf-8")
    paths[0].write_text(
        "\n".join(line.rstrip(" \t") for line in svg.splitlines()) + "\n",
        encoding="utf-8", newline="\n",
    )
    with data_script.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write("window.EPI_C3_STATE_PILOT = ")
        json.dump(data, stream, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        stream.write(";\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    render(json.loads(args.data.read_text(encoding="utf-8")), args.output_dir)


if __name__ == "__main__":
    main()
