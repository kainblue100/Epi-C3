"use strict";

const assert = require("node:assert/strict");
const crypto = require("node:crypto");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const snapshot = require("../snapshot-data.js");

const root = path.resolve(__dirname, "..");
const evidence = path.join(root, "evidence", "foundation-and-grn-v1");
const read = (name) => JSON.parse(fs.readFileSync(path.join(evidence, name), "utf8"));
const hash = (name) => crypto.createHash("sha256").update(fs.readFileSync(path.join(evidence, name))).digest("hex");
const html = fs.readFileSync(path.join(root, "index.html"), "utf8");
const readme = fs.readFileSync(path.join(root, "README.md"), "utf8");

test("additional benchmark count audits match the displayed, candidate-only source table", () => {
  const report = read("scperturbbench-foundation-audit.json");
  const verification = read("scperturbbench-foundation-verification.json");
  assert.equal(hash("scperturbbench-foundation-audit.json"), snapshot.provenance.scPerturbBenchAuditSha256);
  assert.equal(hash("scperturbbench-foundation-verification.json"), snapshot.provenance.scPerturbBenchVerificationSha256);
  assert.equal(verification.source_audit_sha256, snapshot.provenance.scPerturbBenchAuditSha256);
  assert.equal(report.state, "complete");
  assert.equal(verification.state, "complete");
  assert.match(report.training_eligibility, /candidate-only/);
  assert.equal(verification.training_updates, 0);
  assert.equal(report.private_sha256, verification.private_sha256);
  assert.equal(hash("prepare-scperturbbench-foundation-v1.py"), report.preparation_script_sha256);
  assert.equal(hash("epi_c3_scperturbbench_foundation_v1.py"), report.source_module_sha256);
  assert.equal(hash("verify-scperturbbench-foundation-v1.py"), verification.verification_script_sha256);
  assert.equal(fs.existsSync(path.join(evidence, "sources.private.json")), false);
  for (const [name, rows, train, mapped, heldout] of [
    ["KHP", 127595, 9354, 4992, 23898],
    ["CMO", 10410, 1419, 5000, 4853]
  ]) {
    const source = report.sources[name];
    const checked = verification.sources[name];
    assert.equal(source.numeric_audit.shape[0], rows);
    assert.equal(source.numeric_audit.shape[1], 5000);
    assert.equal(source.split.training_controls, train);
    assert.equal(source.split.validation_controls, 32);
    assert.equal(source.mapping.mapped_features, mapped);
    assert.equal(source.split[name === "KHP" ? "target_cd8_rows_excluded" : "heldout_donor_rows_excluded"], heldout);
    assert.equal(source.numeric_audit.empty_mapped_rows, 0);
    assert.equal(checked.training_controls_streamed, train);
    assert.equal(checked.validation_controls, 32);
    assert.equal(checked.measured_gene_count, mapped);
    assert.equal(checked.directly_reconstructed_rows, 5);
    assert.equal(checked.cursor_reproduced_first_batch, true);
    assert.equal(checked.readonly_representation_loss_finite, true);
    assert.ok(html.includes(train.toLocaleString("en-US")), name);
    assert.ok(html.includes(mapped.toLocaleString("en-US")), name);
    assert.ok(html.includes(heldout.toLocaleString("en-US")), name);
  }
  assert.equal(report.sources.KHP.mapping.status_counts.ambiguous_symbol, 8);
  assert.equal(report.sources.KHP.numeric_audit.unsorted_major_vectors, 4237);
  assert.equal(report.sources.CMO.numeric_audit.unsorted_major_vectors, 0);
  assert.match(html, /所有非对照记录暂不纳入/);
  assert.match(html, /两者尚未用于共享基础模型训练/);
  assert.match(html, /未产生训练更新/);
  assert.match(readme, /跨数据集细胞、供体和药物隔离尚未证明/);
});

test("source chart includes both matrices without presenting candidates as trained cells", () => {
  const chart = snapshot.charts.find((item) => item.id === "source-cells");
  assert.equal(chart.items.find((item) => item.id === "khp").value, 127595);
  assert.equal(chart.items.find((item) => item.id === "cmo").value, 10410);
  const resource = snapshot.resources.find((item) => item.name === "scPerturbBench · KHP / CMO");
  assert.match(resource.status, /尚未训练/);
  assert.match(resource.detail, /9,354.*1,419/);
});
