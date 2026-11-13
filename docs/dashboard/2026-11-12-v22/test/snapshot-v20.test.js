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

test("first contact file probe is bound to the exact auditor and remains explicitly partial", () => {
  const report = read("charm-coordinate-first-file-probe-v1.json");
  assert.equal(hash("charm-coordinate-first-file-probe-v1.json"), snapshot.provenance.charmCoordinateProbeSha256);
  assert.equal(report.script_sha256, hash("audit-charm-contacts-coordinates-v1.py"));
  assert.equal(report.records_checked, 2192041);
  assert.equal(report.declared_chromosomes, 20);
  assert.equal(report.header_problems, 0);
  assert.deepEqual(report.record_findings, {});
  assert.match(report.scope, /first contact member only/);
  assert.ok(report.limitations.some((item) => item.includes("remaining 957")));
  assert.match(html, /全量接触坐标审计/);
  assert.match(html, /完成 · 1,607,375,529 条记录/);
  assert.match(html, /不能外推至其余文件/);
});

test("catalog candidates are not represented as paired data or trained networks", () => {
  const report = read("sc-mo-grn-dataset-screen-v1.json");
  assert.equal(hash("sc-mo-grn-dataset-screen-v1.json"), snapshot.provenance.scMoGrnScreenSha256);
  assert.deepEqual(report.triple_assay_candidates.map((item) => item.database_id), ["DS010", "DS011"]);
  assert.ok(report.triple_assay_candidates.every((item) => item.eligible_as_same_cell_trimodal === false));
  assert.deepEqual(report.k562_reference_network_candidates.map((item) => item.catalog_edges), [1435720, 236425, 29298]);
  assert.match(html, /目录数字尚未独立核验/);
  assert.match(html, /未用于模型训练或评价/);
  assert.match(html, /不是同细胞三模态/);
});
