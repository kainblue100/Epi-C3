"use strict";

const assert = require("node:assert/strict");
const crypto = require("node:crypto");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");

const root = path.resolve(__dirname, "..");
const comparison = require("../comparison-data.js");
const read = (file) => fs.readFileSync(path.join(root, file));
const digest = (content) => crypto.createHash("sha256").update(content).digest("hex");
const audit = JSON.parse(read("k562-comparison-audit.json").toString("utf8"));
const html = read("index.html").toString("utf8");

test("the browser data and arithmetic audit bind the same aggregate JSON", () => {
  const bytes = read("k562-comparison.json");
  assert.deepEqual(comparison, JSON.parse(bytes.toString("utf8")));
  assert.equal(audit.summary_sha256, digest(bytes));
  const release = path.resolve(root, "../../../releases/2026-11-13-benchmark-v1");
  assert.equal(comparison.script_sha256, digest(fs.readFileSync(path.join(release, "compare_legacy_k562_means.py"))));
  assert.equal(audit.audit_script_sha256, digest(fs.readFileSync(path.join(release, "audit_legacy_k562_means.py"))));
  assert.equal(audit.status, "complete");
  assert.ok(audit.maximum_aggregate_absolute_difference < 1e-12);
});

test("legacy methods use the frozen shared axes and preserve undefined Pearson", () => {
  assert.equal(comparison.target_count, 347);
  assert.equal(comparison.gene_count, 1861);
  assert.equal(comparison.scored_genes_per_target, 1860);
  assert.equal(comparison.official_state_benchmark, false);
  assert.equal(comparison.official_cell_eval_result, false);
  assert.equal(comparison.methods.length, 7);
  for (const method of comparison.methods) {
    for (const metric of Object.values(method.metrics)) {
      assert.equal(metric.total_targets, 347);
      assert.equal(metric.by_run.length, method.prediction_runs);
      assert.equal(metric.finite_targets_by_run.length, method.prediction_runs);
    }
  }
  const control = comparison.methods.find((item) => item.id === "no-effect");
  assert.equal(control.seed_count, null);
  assert.equal(control.metrics.delta_pearson.mean_of_run_macros, null);
  assert.deepEqual(control.metrics.delta_pearson.finite_targets_by_run, [0]);
});

test("figure and page disclose the actual scope and missing methods", () => {
  const svg = read("legacy-k562-comparison.svg").toString("utf8");
  for (const text of ["NOT EVALUATED HERE: STATE", "not a STATE benchmark", "custom population-mean metrics"]) {
    assert.ok(svg.includes(text), text);
  }
  for (const method of comparison.methods) {
    for (const [name, digits] of [["mse", 6], ["delta_pearson", 4], ["discrimination", 4]]) {
      const value = method.metrics[name].mean_of_run_macros;
      if (value !== null) assert.ok(svg.includes(value.toFixed(digits)), `${method.id} ${name}`);
    }
  }
  for (const text of ["不是 STATE 论文排行榜", "PerturbMean", "ContextMean", "scVI", "CPA", "scGPT",
    "新调控网络分支", "严格同细胞三模态分支", "无效应", "未定义", "不是官方 cell-eval"]) {
    assert.ok(html.includes(text), text);
  }
});
