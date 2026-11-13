"use strict";

const metricLabels = {
  discrimination_score_l1: "L1 区分度（保存字段）",
  pearson_delta: "表达变化 Pearson",
  pr_auc: "DE AUPRC",
  de_spearman_lfc_sig: "logFC Spearman",
  overlap_at_N: "DE overlap at N",
  de_spearman_sig: "DE 基因数 Spearman",
};

function populateResults(data) {
  const body = document.getElementById("author-results");
  body.replaceChildren();
  for (const dataset of data.datasets) {
    for (const panel of dataset.panels) {
      const row = document.createElement("tr");
      const entries = [
        dataset.id,
        metricLabels[panel.metric],
        panel.n_contexts,
        panel.n_pairs_total,
        ...["X_hvg", "X_state"].map((key) => {
          const value = panel.scores[key];
          return `${value.mean.toFixed(4)} ± ${value.population_sd.toFixed(4)}`;
        }),
      ];
      for (const [index, value] of entries.entries()) {
        const cell = document.createElement(index === 0 ? "th" : "td");
        if (index === 0) cell.scope = "row";
        if (index >= 2) cell.className = "numeric";
        cell.textContent = String(value);
        row.appendChild(cell);
      }
      body.appendChild(row);
    }
  }
}

if (typeof document !== "undefined" && typeof stateAuthorReference !== "undefined") {
  populateResults(stateAuthorReference);
}
if (typeof module !== "undefined") module.exports = { metricLabels, populateResults };
