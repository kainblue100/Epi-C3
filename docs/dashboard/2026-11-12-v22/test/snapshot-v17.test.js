"use strict";

const assert = require("node:assert/strict");
const crypto = require("node:crypto");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const snapshot = require("../snapshot-data.js");

const root = path.resolve(__dirname, "..");
const evidence = path.join(root, "evidence", "foundation-and-grn-v1");
const html = fs.readFileSync(path.join(root, "index.html"), "utf8");
const readme = fs.readFileSync(path.join(root, "README.md"), "utf8");
const report = JSON.parse(fs.readFileSync(path.join(evidence, "grn-full-score-audit.json"), "utf8"));
const hash = (name) => crypto.createHash("sha256").update(fs.readFileSync(path.join(evidence, name))).digest("hex");

test("full saved-aggregate score audit and frozen validation selection agree", () => {
  assert.equal(report.status, "verified-saved-aggregates");
  assert.equal(report.test_targets, 347);
  assert.equal(report.genes, 1861);
  assert.equal(report.validated_test_rows, 14574);
  assert.ok(report.max_validation_per_target_absolute_difference < 1e-12);
  assert.ok(report.max_test_per_target_absolute_difference < 1e-12);
  assert.ok(report.max_neighbor_metric_absolute_difference < 1e-12);
  assert.equal(hash("audit-grn-full-scores-v1.py"), report.script_sha256);
  assert.equal(hash("grn-target-differences.csv"), report.target_differences_sha256);
  for (const [arm, values] of Object.entries(report.validation_objectives_by_arm_and_strength_order)) {
    assert.equal(values.indexOf(Math.min(...values)), 2, arm);
    assert.equal(report.selected_strengths[arm], 0.5);
  }
  assert.ok(report.limitations.some((item) => item.includes("not independently reconstructed from raw cells")));
  assert.match(html, /14,574 条测试逐靶点评分/);
  assert.match(readme, /不是从原始细胞独立重建实验/);
});

test("target differences and direct-neighbor MSEs match the audit", () => {
  const rows = fs.readFileSync(path.join(evidence, "grn-target-differences.csv"), "utf8").trim().split(/\r?\n/);
  assert.equal(rows.length, 348);
  const deltas = rows.slice(1).map((row) => Number(row.split(",")[1]));
  assert.ok(deltas.every(Number.isFinite));
  assert.equal(deltas.filter((value) => value < 0).length,
    report.real_minus_shuffled_mse.targets_real_lower_after_seed_averaging);
  assert.equal(deltas.filter((value) => value > 0).length,
    report.real_minus_shuffled_mse.targets_shuffled_lower_after_seed_averaging);
  const average = (values) => values.reduce((sum, item) => sum + item.neighbor_mse, 0) / values.length;
  for (const [target, count] of [["SPI1", 795], ["ZNF318", 19]]) {
    const graph = report.graph_source_targets[target];
    const real = graph["population-v2-epi-real-graph-selected"];
    const shuffled = graph["population-v2-epi-shuffled-graph-selected"];
    assert.equal(real.length, 3);
    assert.equal(shuffled.length, 3);
    assert.ok(real.every((row) => row.neighbor_count === count));
    assert.ok(shuffled.every((row) => row.neighbor_count === count));
    assert.ok(average(real) > average(shuffled));
    assert.ok(html.includes(average(real).toFixed(9)));
    assert.ok(html.includes(average(shuffled).toFixed(9)));
  }
  assert.match(html, /不能推断因果或统计显著性/);
});

test("displayed training timestamp matches the frozen data", () => {
  assert.equal(snapshot.foundation.snapshotUtc, "2026-11-12T17:27:54.394429+00:00");
  assert.match(html, /2026-11-12 17:27:54 UTC/);
  assert.match(readme, /2026-11-12 17:27:54 UTC/);
  assert.match(readme, /1,210,610 \/ 8,703,689/);
  assert.doesNotMatch(html, /2026-11-12 15:23:24 UTC/);
  assert.doesNotMatch(readme, /2026-11-12 15:23:24 UTC/);
});

test("saved-checkpoint diagnostic predates the frozen state and shares its plan", () => {
  const state = JSON.parse(fs.readFileSync(path.join(evidence, "foundation-state.json"), "utf8"));
  const diagnostic = JSON.parse(fs.readFileSync(path.join(evidence, "foundation-diagnostic-14500.json"), "utf8"));
  assert.equal(diagnostic.status, "complete");
  assert.equal(diagnostic.scope, "early-stage1-read-only-diagnostic");
  assert.equal(diagnostic.updates, 14500);
  assert.ok(diagnostic.updates < state.updates);
  assert.equal(diagnostic.plan_sha256, state.plan_sha256);
  for (const source of Object.keys(diagnostic.trained_cells)) {
    assert.ok(diagnostic.trained_cells[source] <= state.trained_cells[source], source);
  }
  assert.equal(diagnostic.snapshot_sha256, "f28885d6a7cb5d424c9588c80ebff2d9a1a87f1917a8aebc8b0d2b451577c34c");
  assert.equal(hash("audit-foundation-diagnostic.py"), "32919be0fc4661533193a7be2ef2384ea8bfb2b1ea5566223bfb38f57d6971f1");
  for (const source of ["arc", "tabula", "orion-HCT116", "orion-HEK293T"]) {
    const before = diagnostic.before_training[source];
    const after = diagnostic.at_snapshot[source];
    assert.equal(before.cells, 32);
    assert.equal(after.cells, 32);
    assert.equal(before.masked_detected_entries, after.masked_detected_entries);
    assert.equal(after.latent_dimensions_std_gt_0_1, 64);
    assert.ok(after.masked_detected_mse < before.masked_detected_mse);
    assert.ok(html.includes(before.masked_detected_mse.toFixed(6)));
    assert.ok(html.includes(after.masked_detected_mse.toFixed(6)));
  }
  assert.match(html, /Arc 和 Orion 的验证并非供体独立/);
  assert.match(html, /该检查点早于上方 26,250 次更新的状态快照/);
  assert.match(html, /不能证明扰动预测、泛化能力或训练已完成/);
});

test("strict visible-library ablation remains distinct from retraining", () => {
  const diagnostic = JSON.parse(fs.readFileSync(path.join(evidence, "foundation-diagnostic-14500.json"), "utf8"));
  const audit = JSON.parse(fs.readFileSync(path.join(evidence, "foundation-masked-library-audit-v1.json"), "utf8"));
  assert.equal(audit.status, "complete");
  assert.equal(audit.updates, diagnostic.updates);
  assert.equal(audit.snapshot_sha256, diagnostic.snapshot_sha256);
  assert.equal(audit.previous_diagnostic_sha256, hash("foundation-diagnostic-14500.json"));
  assert.equal(audit.audit_script_sha256, hash("audit-foundation-masked-library-v1.py"));
  for (const [source, row] of Object.entries(audit.sources)) {
    assert.equal(row.cells, 32, source);
    assert.equal(row.visible_only_latent_unchanged_after_hidden_input_alteration, true, source);
    assert.ok(row.max_full_library_latent_change_after_hidden_input_alteration > 1e-7, source);
    assert.equal(row.full_library.masked_detected_mse, diagnostic.at_snapshot[source].masked_detected_mse);
    assert.ok(html.includes(row.full_library.masked_detected_mse.toFixed(6)), source);
    assert.ok(html.includes(row.visible_only_library.masked_detected_mse.toFixed(6)), source);
    assert.ok(html.includes(row.mean_absolute_latent_change_between_protocols.toFixed(3)), source);
  }
  assert.match(html, /包含被遮盖基因的计数/);
  assert.match(html, /更严格输入消融，不是重新训练/);
  assert.match(html, /不能证明扰动预测、泛化能力或训练已完成/);
});
