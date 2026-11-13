"use strict";

(function () {
  const labels = {
    "legacy-epi-graph": "旧 Epi-C3 · 候选图",
    "legacy-epi-shuffled": "旧 Epi-C3 · 打乱图",
    "legacy-epi-nodes": "旧 Epi-C3 · 表观节点",
    "legacy-rna": "旧 Epi-C3 · RNA-only",
    "gears": "GEARS · 适配版",
    "training-mean": "训练均值",
    "no-effect": "无效应"
  };
  const body = document.getElementById("comparison-body");
  const data = window.comparisonData;
  if (!body) return;
  if (!data || data.format !== "epi-c3-posthoc-legacy-k562-comparison-v1") {
    body.textContent = "";
    const row = body.insertRow();
    const cell = row.insertCell();
    cell.colSpan = 6;
    cell.textContent = "比较数据无法读取，请查看 JSON。";
    return;
  }
  const count = (metric) => {
    const support = metric.finite_targets_by_run;
    return support.every((value) => value === support[0]) ? String(support[0]) : support.join(",");
  };
  const format = (metric, digits) => metric.mean_of_run_macros === null
    ? "未定义" : metric.mean_of_run_macros.toFixed(digits);
  body.textContent = "";
  for (const method of data.methods) {
    const row = body.insertRow();
    const label = document.createElement("th");
    label.scope = "row";
    label.textContent = labels[method.id] || method.label;
    row.appendChild(label);
    const values = [
      format(method.metrics.mse, 6),
      format(method.metrics.delta_pearson, 4),
      format(method.metrics.discrimination, 4),
      [count(method.metrics.mse), count(method.metrics.delta_pearson), count(method.metrics.discrimination)]
        .join(" / ") + "（共 " + data.target_count + "）",
      method.seed_count === null ? "—" : String(method.seed_count)
    ];
    for (const value of values) {
      const cell = row.insertCell();
      cell.className = "numeric";
      cell.textContent = value;
    }
  }
}());
