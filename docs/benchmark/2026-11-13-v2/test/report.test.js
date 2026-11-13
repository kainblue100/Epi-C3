"use strict";

const assert = require("node:assert/strict");
const crypto = require("node:crypto");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");

const root = path.resolve(__dirname, "..");
const read = (file) => fs.readFileSync(path.join(root, file));
const data = require("../state-author-reference.js");
const { metricLabels } = require("../report.js");

test("the browser data match the source-bound aggregate JSON", () => {
  assert.deepEqual(data, JSON.parse(read("state-author-reference.json").toString("utf8")));
  assert.equal(data.source.revision, "ef2f1e13966a2a8f61fb3e8d178fae2b46aad195");
  assert.equal(data.source.sha256.contexts, "ad015f3bdc22079296cb1a892bc2b08fbdce3ccee7a9ae34ea9dca06b7aa740c");
  assert.equal(data.project_same_protocol_evaluation, false);
  assert.equal(data.article_figure_2_reproduction, false);
  assert.deepEqual(data.datasets.map((dataset) => dataset.panels[0].n_contexts), [5, 3, 4]);
  assert.deepEqual(data.datasets.map((dataset) => dataset.panels.length), [6, 6, 6]);
  for (const dataset of data.datasets) {
    for (const panel of dataset.panels) {
      assert.ok(metricLabels[panel.metric]);
      assert.ok(panel.n_pairs_total > 0);
      for (const score of Object.values(panel.scores)) {
        assert.equal(score.by_context.length, panel.n_contexts);
        assert.ok(Number.isFinite(score.mean));
        assert.ok(Number.isFinite(score.population_sd));
        assert.equal(score.by_context.reduce((total, item) => total + item.n_pairs, 0), panel.n_pairs_total);
        const values = score.by_context.map((item) => item.score);
        const mean = values.reduce((total, value) => total + value, 0) / values.length;
        const sd = Math.sqrt(values.reduce((total, value) => total + (value - mean) ** 2, 0) / values.length);
        assert.ok(Math.abs(mean - score.mean) < 1e-12);
        assert.ok(Math.abs(sd - score.population_sd) < 1e-12);
      }
    }
  }
});

test("all plotted values and essential scope disclaimers are present", () => {
  const svg = read("state-author-reference.svg").toString("utf8");
  for (const dataset of data.datasets) {
    for (const panel of dataset.panels) {
      for (const score of Object.values(panel.scores)) {
        assert.ok(svg.includes(score.mean.toFixed(3)), `${dataset.id}/${panel.metric}`);
      }
    }
  }
  for (const text of [
    "not an Epi-C3 benchmark", "not an exact reproduction", "Parse has 3 matched contexts",
    "not converted to the paper", "Other paper baselines", "Adduri et al.",
  ]) assert.ok(svg.includes(text), text);
  const html = read("index.html").toString("utf8");
  for (const text of [
    "正式比较尚未完成", "不是本项目的 STATE 实测", "CD4 Memory", "66%", "+170%",
    "48%", "+58%", "真实 DE 数也为零", "3 / 9", "未定义", "预训练测试暴露风险",
    "Cell 189", "10.1016/j.cell.2026.07.052",
  ]) assert.ok(html.includes(text), text);
  const exposure = JSON.parse(read("tahoe-state-exposure.json").toString("utf8"));
  assert.equal(exposure.matched_official_test_rows, 8133621);
  assert.equal(exposure.minimum_matched_rows_in_planned_eligible_stream, 4599956);
  assert.equal(exposure.stage2_has_been_run_by_this_audit, false);
});

test("the archived figure formats and relative page resources exist", () => {
  assert.ok(read("state-author-reference.png").subarray(0, 8).equals(Buffer.from("89504e470d0a1a0a", "hex")));
  assert.equal(read("state-author-reference.pdf").subarray(0, 5).toString("ascii"), "%PDF-");
  const html = read("index.html").toString("utf8");
  for (const match of html.matchAll(/(?:src|href)="([^"]+)"/g)) {
    const url = match[1];
    if (/^(?:https?:|#)/.test(url)) continue;
    const local = path.resolve(root, url);
    assert.ok(fs.existsSync(local), url);
  }
  for (const line of read("SHA256SUMS.txt").toString("utf8").trim().split(/\r?\n/)) {
    const match = /^([a-f0-9]{64})  (.+)$/.exec(line);
    assert.ok(match, line);
    assert.equal(crypto.createHash("sha256").update(read(match[2])).digest("hex"), match[1], match[2]);
  }
});
