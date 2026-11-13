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
const html = fs.readFileSync(path.join(root, "index.html"), "utf8");

test("foundation progress and interactive chart match their frozen state", () => {
  const state = read("foundation-state.json");
  const foundation = snapshot.foundation;
  assert.equal(foundation.snapshotUtc, state.updated_at_utc);
  assert.equal(foundation.status, state.status);
  assert.equal(foundation.scope, state.scope);
  assert.equal(foundation.updates, state.updates);
  assert.deepEqual(foundation.trained, state.trained_cells);
  assert.deepEqual(foundation.expected, state.expected_cells);
  assert.equal(Object.values(foundation.trained).reduce((a, b) => a + b, 0), 1210610);
  assert.equal(Object.values(foundation.expected).reduce((a, b) => a + b, 0), 8703689);
  const chart = snapshot.charts.find((item) => item.id === "foundation-trained");
  assert.deepEqual(chart.items.map((item) => item.value), [
    state.trained_cells.arc, state.trained_cells.tabula,
    state.trained_cells["orion-HCT116"], state.trained_cells["orion-HEK293T"]
  ]);
  assert.match(chart.note, /不是实时进度/);
  assert.match(chart.note, /Tahoe 尚未接入/);
  assert.match(html, /不是实时监控/);
});

test("CHARM RNA, QC, accessibility and contact identities are matched with explicit limits", () => {
  const partial = read("charm-mesc-rna-atac-audit.json");
  const auditPath = path.join(evidence, "charm-mesc-trimodal-audit.json");
  const report = read("charm-mesc-trimodal-audit.json");
  assert.equal(partial.state, "partial-no-contact-archive");
  assert.equal(partial.overlap.rna_positive_qc_atac_and_contacts, null);
  assert.equal(report.state, "complete");
  assert.equal(report.species, "Mus musculus");
  assert.equal(report.rna.cells, 959);
  assert.equal(report.rna.genes, 41020);
  assert.equal(report.rna.fractional_entries, 501341);
  assert.equal(report.atac.fragment_rows, 50879408);
  assert.equal(report.atac.cells_with_fragments, 959);
  assert.equal(report.contacts.cells_with_contact_files, 958);
  assert.equal(report.overlap.rna_positive_qc_and_atac, 805);
  assert.equal(report.overlap.atac_fragment_count_equals_qc, 0);
  assert.equal(report.overlap.rna_positive_qc_atac_and_contacts, 805);
  assert.equal(report.final_training_ready, false);
  assert.match(report.contacts.inspection, /full inner CRCs were not checked/);
  assert.equal(crypto.createHash("sha256").update(fs.readFileSync(auditPath)).digest("hex"),
    snapshot.provenance.charmAuditSha256);
  const script = path.join(evidence, "audit-charm-mesc-pairing-v1.py");
  assert.equal(crypto.createHash("sha256").update(fs.readFileSync(script)).digest("hex"),
    report.audit_script_sha256);
  const innerPath = path.join(evidence, "charm-inner-gzip-audit-v1.json");
  const inner = read("charm-inner-gzip-audit-v1.json");
  assert.equal(inner.status, "complete");
  assert.equal(inner.inner_gzip_streams_verified, report.contacts.cells_with_contact_files);
  assert.equal(inner.archive_sha256, report.inputs.contacts.sha256);
  assert.equal(inner.pairing_audit_sha256, snapshot.provenance.charmAuditSha256);
  assert.equal(crypto.createHash("sha256").update(fs.readFileSync(innerPath)).digest("hex"),
    snapshot.provenance.charmInnerAuditSha256);
  assert.equal(crypto.createHash("sha256").update(fs.readFileSync(path.join(evidence, "audit-charm-inner-gzip-v1.py"))).digest("hex"),
    inner.script_sha256);
  assert.ok(inner.limitations.some((item) => item.includes("does not validate all contact records")));
  assert.equal(fs.existsSync(path.join(evidence, report.private_identity_file)), false);
  for (const text of ["50,879,408", "501,341", "958", "RNA / QC / ATAC / 接触身份交集", "全部内层 gzip 的 CRC 均通过", "完成逐条格式及坐标审计", "至少三个细胞系"]) {
    assert.ok(html.includes(text), text);
  }
  const card = snapshot.modalities.find((item) => item.title === "CHARM 三模态候选");
  assert.equal(card.number, "805");
  assert.match(card.detail, /完整读取并按一基坐标检查/);
  const resource = snapshot.resources.find((item) => item.name === "CHARM mESC · GSE303006");
  assert.match(resource.detail, /805 个同时匹配/);
  assert.match(resource.detail, /逐条坐标审计已完成/);
});

test("balanced replay evidence binds the exact real-source smoke script", () => {
  const report = read("balanced-real-source-smoke.json");
  const state = read("foundation-state.json");
  assert.equal(report.state, "complete");
  assert.equal(report.stage1_plan_sha256, state.plan_sha256);
  assert.deepEqual(Object.keys(report.sources).sort(), ["arc", "orion-HCT116", "orion-HEK293T", "tabula"]);
  assert.ok(Object.values(report.sources).every((item) => item.initial_and_checkpoint_replay_exact));
  const script = path.join(evidence, "smoke-balanced-replay-live-stage1-v1.py");
  assert.equal(crypto.createHash("sha256").update(fs.readFileSync(script)).digest("hex"), report.script_sha256);
  assert.match(html, /尚未进行真实第二阶段训练/);
  assert.match(html, /记录数平衡并不等于梯度或优化步数平衡/);
});

test("Tahoe isolation counts, verification and loader evidence agree", () => {
  const audit = read("tahoe-isolation-audit.json");
  const verification = read("tahoe-isolation-verification.json");
  const overlap = read("tahoe-overlap-summary.json");
  const smoke = read("tahoe-loader-smoke.json");
  const hash = crypto.createHash("sha256").update(fs.readFileSync(path.join(evidence, "tahoe-isolation-audit.json"))).digest("hex");
  assert.equal(audit.state, "complete");
  assert.equal(audit.files_scanned, 3388);
  assert.equal(audit.source_rows_checked, 95624334);
  assert.equal(audit.protected_unique_identities, 6213);
  assert.equal(audit.protected_eligible_rows, 5027);
  assert.equal(audit.validation_eligible_rows, 32);
  assert.equal(audit.eligible_training_rows_after_exclusions,
    audit.eligible_rows_before_exclusions - audit.protected_eligible_rows - audit.validation_eligible_rows);
  assert.equal(audit.eligible_training_rows_after_exclusions, 92090669);
  assert.equal(verification.audit_sha256, hash);
  assert.equal(verification.per_identity_counts_match, true);
  assert.equal(verification.protected_eligible_rows, audit.protected_eligible_rows);
  assert.equal(overlap.audit_sha256, hash);
  const interim = overlap.evaluations.filter((row) => row.evaluation.startsWith("interim/"));
  assert.deepEqual(interim.map((row) => row.evaluation), ["interim/reference", "interim/input_controls"]);
  assert.ok(interim.every((row) => row.eligible_rows_under_old_filter === 0));
  assert.equal(smoke.audit_sha256, hash);
  assert.equal(smoke.cursor_replay_exact, true);
  assert.equal(smoke.targeted_exclusion_matches_independent_identity_filter, true);
  assert.equal(smoke.targeted_batch_eligible_before_exclusions - smoke.targeted_batch_emitted_rows, 1);
  for (const text of ["6,213", "5,027", "92,090,669", "尚未用于共享训练"]) assert.ok(html.includes(text), text);
});

test("GRN neighbor counts and direction results match the independent audit", () => {
  const report = read("grn-neighbor-audit.json");
  assert.equal(report.audited_rows, 52);
  assert.equal(report.value_comparisons_against_csv_and_summary, 832);
  assert.ok(report.maximum_absolute_difference < 1e-12);
  for (const [target, count, eligible, correct] of [
    ["SPI1", 795, 353, [193, 199, 205]], ["ZNF318", 19, 3, [3, 2, 1]]
  ]) {
    const rows = report.results.filter((row) => row.target === target &&
      row.method === "population-v2-epi-real-graph-selected").sort((a, b) => a.seed_index - b.seed_index);
    assert.equal(rows.length, 3);
    assert.ok(rows.every((row) => row.neighbor_count === count && row.eligible_effect_count === eligible));
    assert.deepEqual(rows.map((row) => row.correct_direction_count), correct);
    assert.ok(html.includes(correct.join(" / ")));
  }
});

test("foundation parameter row matches the strictly loaded recovery audit", () => {
  const state = read("foundation-state.json");
  const parameters = read("foundation-parameters.json");
  const row = snapshot.modelRows.find((item) => item.model === "Epi-C3 · 共享 RNA 基础模型");
  assert.equal(parameters.plan_sha256, state.plan_sha256);
  assert.equal(parameters.checkpoint_loaded_strictly, true);
  assert.equal(parameters.scope, "incomplete-stage1-not-a-full-foundation");
  assert.equal(parameters.model_config.n_genes, snapshot.foundation.genes);
  assert.equal(parameters.model_trainable_at_construction, row.trainable);
  assert.equal(parameters.representation_optimizer_parameters, snapshot.foundation.representationOptimizer);
  assert.equal(parameters.model_trainable_at_construction - parameters.representation_optimizer_parameters, 57792);
  assert.equal(row.flow, null);
  assert.match(row.status, /未训练扰动 flow/);
});

test("post-hoc GRN and old-model cell-eval tables match their audits", () => {
  const grn = read("grn-audit.json");
  const cellEval = read("cell-eval-summary.json");
  const cellAudit = read("cell-eval-audit.json");
  assert.equal(grn.checked_target_scores, 9022);
  assert.equal(grn.targets, 347);
  assert.equal(grn.targets_with_lower_real_mse, 194);
  assert.equal(grn.targets_with_higher_real_mse, 153);
  for (const key of [
    "population-v2-epi-real-graph-selected", "population-v2-epi-shuffled-graph-selected",
    "population-v2-epi-nodes-only-selected", "population-v2-rna-only-selected", "training-mean"
  ]) {
    assert.ok(html.includes(grn.macro_mse[key].mean.toFixed(9)), key);
  }
  assert.equal(cellEval.cell_eval_version, "0.8.2");
  assert.equal(cellEval.official_leaderboard_result, false);
  assert.equal(cellAudit.control_and_prediction_parents_verified_disjoint, true);
  for (const arm of ["full", "noeffect"]) {
    for (const metric of ["mse", "mse_delta"]) {
      const value = cellEval.summary[arm][metric];
      assert.equal(value.finite_conditions, 9);
      assert.ok(html.includes(value.mean_of_finite.toFixed(6)), `${arm}/${metric}`);
    }
  }
  assert.match(html, /不是新基础模型或两种最终模型的成绩/);
});

test("pairing exclusion and architecture assets remain explicit and local", () => {
  const pairing = read("schicar-pairing-evidence.json");
  const limca = read("limca-screening-evidence.json");
  assert.equal(pairing.dataset, "GSE267117");
  assert.match(pairing.decision, /exclude/);
  assert.match(pairing.reason, /different batches/);
  assert.deepEqual(limca.reported_human_cell_lines, ["GM12878", "K562", "eHAP", "BJ"]);
  assert.deepEqual(limca.reported_same_cell_modalities, ["RNA", "chromatin contacts"]);
  assert.equal(limca.strict_rna_atac_contact_eligible, false);
  assert.equal(limca.source_file_cell_pairing_verified, false);
  for (const relativePath of [
    "assets/architecture-v2.svg",
    "evidence/foundation-and-grn-v1/foundation-state.json",
    "evidence/foundation-and-grn-v1/foundation-parameters.json",
    "evidence/foundation-and-grn-v1/audit-foundation-parameters.py",
    "evidence/foundation-and-grn-v1/grn-audit.json",
    "evidence/foundation-and-grn-v1/cell-eval-summary.json",
    "evidence/foundation-and-grn-v1/cell-eval-audit.json",
    "evidence/foundation-and-grn-v1/schicar-pairing-evidence.json",
    "evidence/foundation-and-grn-v1/limca-screening-evidence.json",
    "evidence/foundation-and-grn-v1/grn-neighbor-audit.json",
    "evidence/foundation-and-grn-v1/audit-grn-neighbors.py",
    "evidence/foundation-and-grn-v1/tahoe-isolation-audit.json",
    "evidence/foundation-and-grn-v1/tahoe-isolation-verification.json",
    "evidence/foundation-and-grn-v1/tahoe-overlap-summary.json",
    "evidence/foundation-and-grn-v1/tahoe-loader-smoke.json"
  ]) {
    assert.ok(fs.existsSync(path.join(root, relativePath)), relativePath);
    assert.ok(html.includes(relativePath), relativePath);
  }
  const figure = fs.readFileSync(path.join(root, "assets", "architecture-v2.svg"), "utf8");
  assert.match(figure, /TWO DISTINCT PERTURBATION BRANCHES/);
  assert.match(figure, /DATA GATE NOT MET/);
  assert.match(figure, /DASHED = PLANNED \/ BLOCKED/);
});
