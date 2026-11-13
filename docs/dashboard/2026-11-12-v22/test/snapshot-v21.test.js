"use strict";

const assert = require("node:assert/strict");
const crypto = require("node:crypto");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const snapshot = require("../snapshot-data.js");

const root = path.resolve(__dirname, "..");
const evidence = path.join(root, "evidence", "foundation-and-grn-v1");
const read = (file) => JSON.parse(fs.readFileSync(path.join(evidence, file), "utf8"));
const hash = (file) => crypto.createHash("sha256").update(fs.readFileSync(path.join(evidence, file))).digest("hex");
const html = fs.readFileSync(path.join(root, "index.html"), "utf8");

test("RN119 findings are tied to the audited script and remain unused", () => {
  const report = read("k562-reference-network-audit-v1.json");
  assert.equal(hash("k562-reference-network-audit-v1.json"), snapshot.provenance.k562ReferenceAuditSha256);
  assert.equal(hash("audit-k562-reference-network-v1.py"), report.script_sha256);
  assert.equal(report.status, "complete");
  assert.equal(report.reference.rows, 29298);
  assert.equal(report.reference.unique_directed_gene_pairs, 26642);
  assert.equal(report.reference.pairs_with_both_signs, 2656);
  assert.equal(report.frozen_axis_comparison.reference_unique_edges_both_ends_on_axis, 157);
  assert.equal(report.frozen_axis_comparison.conflicting_edges_both_ends_on_axis, 70);
  assert.equal(report.frozen_axis_comparison.exact_directed_gene_pair_overlap, 0);
  assert.equal(report.used_for_training_or_evaluation, false);
  assert.ok(report.limitations.some((item) => item.includes("aliases")));
  for (const text of ["29,298", "26,642", "2,656", "157", "70", "未用于训练或评价", "不是预测准确度提升"]) {
    assert.ok(html.includes(text), text);
  }
  assert.equal(fs.existsSync(path.join(evidence, "RN119.tsv")), false);
});
