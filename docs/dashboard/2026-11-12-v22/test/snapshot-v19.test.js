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

test("live-source preflight and queued dispatch bind the frozen plan without training", () => {
  const state = read("foundation-state.json");
  const report = read("stage2-source-preflight.json");
  const queue = read("stage2-dispatch-snapshot.json");
  assert.equal(hash("stage2-source-preflight.json"), snapshot.provenance.stage2PreflightSha256);
  assert.equal(hash("stage2-dispatch-snapshot.json"), snapshot.provenance.stage2DispatchSnapshotSha256);
  assert.equal(queue.preflight_sha256, snapshot.provenance.stage2PreflightSha256);
  assert.equal(queue.dispatcher_sha256, hash("wait-rna-foundation-stage2-v1.py"));
  assert.equal(report.preflight_script_sha256, hash("check-rna-foundation-stage2-preflight-v1.py"));
  assert.equal(report.stage2_driver_sha256, hash("train-rna-foundation-stage2-v1.py"));
  assert.equal(report.tahoe_isolation_audit_sha256, hash("tahoe-isolation-audit.json"));
  assert.equal(report.stage1_plan_sha256, state.plan_sha256);
  assert.equal(queue.stage1_plan_sha256, state.plan_sha256);
  assert.equal(report.optimizer_updates, 0);
  assert.equal(queue.stage, "waiting-for-stage1");
  assert.equal(queue.state, "running");
  assert.equal(queue.minimum_free_gib, 204);
  assert.equal(queue.guarded_file_count, 19);
  const expected = [
    ["arc", 108808, 48, 18080], ["tabula", 998796, 48, 60606],
    ["orion-HCT116", 3260365, 14, 38606], ["orion-HEK293T", 4335720, 13, 38606],
    ["tahoe", 92090669, 16, 62710]
  ];
  assert.equal(Object.keys(report.sources).length, expected.length);
  for (const [name, rows, batch, genes] of expected) {
    const source = report.sources[name];
    assert.equal(source.expected_training_rows, rows);
    assert.equal(source.first_batch_rows, batch);
    assert.equal(source.measured_genes, genes);
    assert.equal(source.cursor_replay_exact, true);
    assert.ok(html.includes(rows.toLocaleString("en-US")), name);
    assert.ok(html.includes(genes.toLocaleString("en-US")), name);
  }
  assert.match(html, /仅检查首批数值、测量掩码及游标重放/);
  assert.match(html, /此静态快照时尚未进行真实第二阶段训练/);
  assert.doesNotMatch(JSON.stringify(queue), /\/mnt\/storage|guarded_files_sha256/);
});
