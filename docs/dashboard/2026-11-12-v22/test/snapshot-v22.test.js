"use strict";

const assert = require("node:assert/strict");
const crypto = require("node:crypto");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const snapshot = require("../snapshot-data.js");

const root = path.resolve(__dirname, "..");
const evidence = path.join(root, "evidence", "foundation-and-grn-v1");
const hash = (file) => crypto.createHash("sha256").update(fs.readFileSync(path.join(evidence, file))).digest("hex");
const read = (file) => JSON.parse(fs.readFileSync(path.join(evidence, file), "utf8"));
const html = fs.readFileSync(path.join(root, "index.html"), "utf8");

test("the tri-modal response is tied to exact source and described as untrained", () => {
  assert.equal(hash("epi_c3_trimodal_response_v1.py"), snapshot.provenance.trimodalResponseSha256);
  assert.equal(hash("test_epi_c3_trimodal_response_v1.py"), snapshot.provenance.trimodalResponseTestSha256);
  for (const text of [
    "未训练的基因扰动响应头", "没有真实数据性能结果", "元数据一致不证明",
    "至少三个细胞系", "虚构成逐一配对", "同一测量基因面板",
    "文库分母", "不能替代 cell-eval"
  ]) {
    assert.ok(html.includes(text), text);
  }
});

test("full contact audit and bounded header probe remain separately scoped", () => {
  const full = read("charm-contact-coordinate-audit-v1.json");
  const pairing = read("charm-mesc-trimodal-audit.json");
  const probe = read("charm-contact-header-probe-v1.json");
  assert.equal(hash("charm-contact-coordinate-audit-v1.json"), snapshot.provenance.charmCoordinateAuditSha256);
  assert.equal(hash("charm-contact-header-probe-v1.json"), snapshot.provenance.charmHeaderProbeSha256);
  assert.equal(full.script_sha256, hash("audit-charm-contacts-coordinates-v1.py"));
  assert.equal(probe.script_sha256, hash("audit-charm-contact-header-probe-v1.py"));
  assert.equal(probe.full_audit_sha256, snapshot.provenance.charmCoordinateAuditSha256);
  assert.equal(full.pairing_audit_sha256, snapshot.provenance.charmAuditSha256);
  assert.equal(full.archive_sha256, pairing.inputs.contacts.sha256);
  assert.equal(probe.archive_sha256_from_full_audit, full.archive_sha256);
  assert.equal(full.status, "complete");
  assert.equal(full.mapped_record_check, "pass");
  assert.equal(full.contact_files, 958);
  assert.equal(full.matched_contact_files, 805);
  assert.equal(full.all_contacts.records, 1607375529);
  assert.equal(full.matched_contacts.records, 1406651183);
  assert.equal(full.files_with_header_problems, 0);
  assert.equal(full.files_without_records, 0);
  assert.equal(full.distinct_declared_chromosome_sets, 2);
  for (const [name, value] of Object.entries(full.all_contacts)) {
    if (name !== "records") assert.equal(value, 0, name);
  }
  assert.equal(probe.status, "complete");
  assert.equal(probe.sampled_files, 51);
  assert.deepEqual(probe.groups.map((group) => group.sampled_files), [50, 1]);
  assert.deepEqual(probe.groups[1].chromosomes.filter((chromosome) => !probe.groups[0].chromosomes.includes(chromosome)),
    ["chrY"]);
  assert.ok(probe.limitations.some((item) => item.includes("not archive-wide frequencies")));
  assert.ok(probe.limitations.some((item) => item.includes("not independently recomputed")));
  for (const text of ["1,607,375,529", "1,406,651,183", "抽查 51 个文件头", "全量频率和原因尚未确定"]) {
    assert.ok(html.includes(text), text);
  }
});
