(() => {
  const data = window.EPI_C3_STATE_PILOT;
  const body = document.getElementById("pilot-results");
  if (!data || data.format !== "epi-c3-state-replogle-pilot-public-v1" || !body) return;

  const labels = {
    discrimination_score_l1: "L1 排名（cell-eval 原始值）",
    pearson_delta: "表达变化 Pearson",
    pr_auc: "DE AUPRC（average precision）",
    de_spearman_lfc_sig: "DE logFC Spearman",
    overlap_at_N: "DE overlap at N",
    de_count_spearman_all_conditions: "DE 基因数 Spearman",
  };
  const models = ["state", "perturb_mean", "context_mean"];
  body.replaceChildren();
  for (const metric of data.metrics) {
    const row = document.createElement("tr");
    const label = document.createElement("th");
    label.scope = "row";
    label.textContent = labels[metric.key];
    row.appendChild(label);
    const count = document.createElement("td");
    count.textContent = `${metric.common_conditions} / 28`;
    row.appendChild(count);
    for (const model of models) {
      const cell = document.createElement("td");
      cell.className = "numeric";
      const value = metric.scores[model];
      cell.textContent = value === null ? "未定义" : value.toFixed(4);
      row.appendChild(cell);
    }
    body.appendChild(row);
  }
  const sensitivity = data.independent_metric_audit.finite_lfc_sensitivity;
  const note = document.getElementById("finite-sensitivity");
  if (note) {
    const values = models.map((model) => sensitivity.mean_spearman[model].toFixed(4));
    note.textContent = `敏感性分析在每个条件中同时排除真实值或任一模型预测 logFC 非有限的基因，三模型使用相同剩余基因。${sensitivity.affected_conditions} 个条件共排除 ${sensitivity.removed_condition_gene_occurrences} 个基因—条件项；${sensitivity.common_conditions} 个共同有效条件的均值依次为 STATE ${values[0]}、PerturbMean ${values[1]}、ContextMean ${values[2]}。这不是原始 cell-eval 分数。`;
  }
})();
