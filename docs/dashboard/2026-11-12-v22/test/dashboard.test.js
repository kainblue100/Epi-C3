"use strict";

const assert = require("node:assert/strict");
const crypto = require("node:crypto");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const snapshot = require("../snapshot-data.js");

const root = path.resolve(__dirname, "..");
const auditPath = path.join(root, "evidence", "parameters-audit.json");
const audit = JSON.parse(fs.readFileSync(auditPath, "utf8"));
const snapshotAuditPath = path.join(root, "evidence", "snapshot-audit-v2.json");
const snapshotAudit = JSON.parse(fs.readFileSync(snapshotAuditPath, "utf8"));
const sha256 = (file) => crypto.createHash("sha256").update(fs.readFileSync(file)).digest("hex");
const chart = (id) => {
  const found = snapshot.charts.find((item) => item.id === id);
  assert.ok(found, `missing chart ${id}`);
  return found;
};
const total = (item) => item.items.reduce((sum, segment) => sum + segment.value, 0);

test("chart units and exact totals stay separate", () => {
  assert.equal(total(chart("source-cells")), 3795459);
  assert.equal(total(chart("target-split")), 1735);
  assert.equal(total(chart("graph-shuffle")), 4141);
  assert.equal(chart("source-cells").unit, "细胞记录");
  assert.equal(chart("target-split").unit, "靶点");
  assert.equal(chart("graph-shuffle").unit, "条候选边");
  assert.match(chart("source-cells").note, /未去重|去重后/);
  assert.match(chart("source-cells").note, /可能重叠/);
  assert.match(chart("graph-shuffle").note, /较弱/);
  for (const item of snapshot.charts) {
    assert.ok(item.items.every((segment) => Number.isInteger(segment.value) && segment.value > 0));
    assert.equal(new Set(item.items.map((segment) => segment.id)).size, item.items.length);
  }
});

test("additional dataset and assay quantities match the snapshot evidence", () => {
  assert.equal(sha256(snapshotAuditPath), snapshot.provenance.snapshotAuditSha256);
  assert.equal(snapshotAudit.format, "epi-c3-dashboard-snapshot-audit-v2");
  assert.equal(chart("source-cells").items.find((item) => item.id === "arc").value, snapshotAudit.arc.source_cells);
  assert.equal(snapshot.provenance.arcSourceCells, snapshotAudit.arc.source_cells);
  assert.equal(snapshot.provenance.arcSourceGenes, snapshotAudit.arc.source_genes);
  assert.equal(snapshot.provenance.arcTrainCells, snapshotAudit.arc.train_cells);
  assert.equal(snapshot.provenance.orionHct116TrainedRows, snapshotAudit.orion.hct116.trained_rows);
  assert.equal(snapshot.provenance.orionHek293tTrainedRows, snapshotAudit.orion.hek293t.trained_rows);
  assert.equal(snapshotAudit.orion.hct116.epochs, 1);
  assert.equal(snapshotAudit.orion.hek293t.epochs, 1);
  assert.equal(snapshot.provenance.k562AtacPeakRows, snapshotAudit.k562_assays.atac_peak_rows);
  assert.equal(snapshot.provenance.k562SelectedTfTracks, snapshotAudit.k562_assays.selected_tf_chip_tracks.length);
  assert.equal(snapshotAudit.k562_assays.atac_accession, "ENCFF855PCP");
  const assays = snapshot.modalities.find((item) => item.tag === "ACCESSIBILITY + TF");
  assert.equal(assays.number, "172,732 / 7");
  assert.match(assays.unit, /ATAC 峰.*TF ChIP 轨道/);
  assert.match(assays.detail, /未配对/);
  const arc = snapshot.resources.find((resource) => resource.name === "Arc virtual cell dataset");
  const orion = snapshot.resources.find((resource) => resource.name === "X-Atlas-Orion");
  assert.match(arc.detail, /221,273.*18,080.*198,701/);
  assert.match(orion.detail, /3,373,853.*4,487,649/);
  assert.match(orion.detail, /不是.*去重/);
});

test("dashboard parameter counts agree with the verified checkpoint audit", () => {
  assert.equal(audit.state, "complete");
  assert.equal(sha256(auditPath), snapshot.provenance.parameterAuditSha256);
  const rna = audit.arms.find((arm) => arm.arm === "rna-only");
  const epi = audit.arms.find((arm) => arm.arm === "epi-real-graph");
  const rnaRow = snapshot.modelRows.find((row) => row.model === "Epi-C3 · RNA-only");
  const epiRow = snapshot.modelRows.find((row) => row.model === "Epi-C3 · 表观图臂");
  assert.equal(rna.model_trainable_at_construction, rnaRow.trainable);
  assert.equal(rna.flow_optimizer_parameters, rnaRow.flow);
  assert.equal(epi.model_trainable_at_construction, epiRow.trainable);
  assert.equal(epi.flow_optimizer_parameters, epiRow.flow);
  assert.equal(epi.frozen_teacher_parameters, 255104);
  assert.equal(epi.nonparameter_buffer_elements, 26054);
  assert.ok(audit.arms.every((arm) => arm.seeds_checked === 3));
  assert.equal(snapshot.modelRows.find((row) => row.model === "STATE").trainable, null);
});

test("comparison model parameters match their loaded checkpoints and remain scoped", () => {
  const directory = path.join(root, "evidence", "model-parameters");
  const reports = {};
  for (const [file, expectedHash] of Object.entries(snapshot.provenance.comparisonAuditSha256)) {
    const reportPath = path.join(directory, file);
    assert.equal(sha256(reportPath), expectedHash);
    const report = JSON.parse(fs.readFileSync(reportPath, "utf8"));
    assert.equal(report.state, "complete");
    assert.equal(report.checkpoint_values_verified, true);
    assert.equal(report.not_a_model_performance_comparison, true);
    assert.equal(report.audit_script_sha256, snapshot.provenance.comparisonAuditScriptSha256);
    reports[file] = report;
  }
  assert.equal(sha256(path.join(directory, "audit-parameters.py")), snapshot.provenance.comparisonAuditScriptSha256);
  const mappings = [
    ["Epi-C3 · KHP RNA-only", "c3-khp.json"],
    ["GEARS", "gears-k562.json"],
    ["scGen", "scgen-khp.json"],
    ["scVIDR", "scvidr-khp.json"]
  ];
  for (const [model, file] of mappings) {
    const row = snapshot.modelRows.find((item) => item.model === model);
    assert.ok(row, `missing model ${model}`);
    assert.equal(row.trainable, reports[file].trainable_at_construction);
    assert.match(row.context, new RegExp(reports[file].genes === 5000 ? "5,000 基因" : "1,861 基因"));
  }
  assert.equal(reports["c3-khp.json"].frozen_at_construction, 656896);
  assert.equal(reports["c3-khp.json"].separate_learned_condition_vector_parameters, 64);
  assert.equal(snapshot.modelRows.find((row) => row.model === "Epi-C3 · KHP RNA-only").flow,
    reports["c3-khp.json"].flow_optimizer_parameters_including_condition);
  assert.equal(reports["gears-k562.json"].perturbation_identities, 1735);
  assert.equal(reports["scvidr-khp.json"].separate_two_source_regression_excluded, true);
  assert.match(reports["scvidr-khp.json"].preprocessing_warning, /non-integer/);
});

test("K562 post-hoc GEARS, zero-effect C3 and KHP values stay scoped", () => {
  const k562 = snapshot.benchmarks.find((benchmark) => benchmark.id === "k562");
  const khp = snapshot.benchmarks.find((benchmark) => benchmark.id === "khp");
  assert.ok(k562 && khp);
  assert.equal(k562.rows[0].metric, 0.01604481739394287);
  assert.equal(k562.rows[1].metric, 0.015711020985759664);
  assert.ok(k562.rows.filter((row) => row.kind === "model").every((row) => row.metric === k562.rows[0].metric));
  assert.match(k562.note, /强度均为 0/);
  const gears = k562.rows.find((row) => row.name === "GEARS · 后验适配");
  assert.ok(gears);
  assert.equal(gears.kind, "posthoc");
  assert.equal(gears.metric, 0.01583080899741956);
  assert.match(k562.note, /原结果已被观察后加入/);
  assert.match(k562.note, /额外公开 GO/);
  assert.match(k562.note, /一个种子/);
  assert.match(k562.note, /191\/347、130\/347/);
  assert.match(k562.note, /插值后表达条目被投影为零/);
  assert.match(k562.note, /不是外部盲测/);
  assert.match(snapshot.modelRows.find((row) => row.model === "GEARS").status, /测试已完成（后验）/);
  const posthoc = khp.rows.find((row) => row.kind === "posthoc");
  assert.deepEqual([posthoc.metric, posthoc.delta, posthoc.mmd], [0.009871, 0.591462, 0.076248]);
  assert.match(khp.note, /后加入/);
  assert.match(khp.note, /未独立重算/);
});

test("GEARS test results, selection and figure match their audited evidence", () => {
  const directory = path.join(root, "evidence", "gears-k562");
  const files = [
    ["test-summary.json", "gearsTestSummarySha256"],
    ["test-audit.json", "gearsTestAuditSha256"],
    ["validation-summary.json", "gearsValidationSummarySha256"],
    ["selection.json", "gearsSelectionSha256"],
    ["figure-manifest.json", "gearsFigureManifestSha256"],
    ["figure-audit.json", "gearsFigureAuditSha256"]
  ];
  const evidence = {};
  for (const [file, hashKey] of files) {
    const filePath = path.join(directory, file);
    assert.equal(sha256(filePath), snapshot.provenance[hashKey]);
    evidence[file] = JSON.parse(fs.readFileSync(filePath, "utf8"));
  }
  const result = evidence["test-summary.json"];
  const audit = evidence["test-audit.json"];
  const validation = evidence["validation-summary.json"];
  const selection = evidence["selection.json"];
  const figure = evidence["figure-manifest.json"];
  const figureAudit = evidence["figure-audit.json"];
  const k562 = snapshot.benchmarks.find((benchmark) => benchmark.id === "k562");
  assert.equal(result.targets, 347);
  assert.equal(result.genes, 1861);
  assert.equal(k562.rows.find((row) => row.name === "GEARS · 后验适配").metric, result.gears_macro_mse);
  assert.equal(k562.rows.find((row) => row.name === "无效应").metric, result.no_effect_macro_mse);
  assert.equal(k562.rows.find((row) => row.name === "训练均值").metric, result.training_mean_log_expression_macro_mse);
  assert.equal(result.gears_targets_better_than_no_effect, 191);
  assert.equal(result.gears_targets_better_than_training_mean, 130);
  assert.equal((result.gears_negative_projection_fraction * 100).toFixed(1), "15.9");
  assert.equal(result.c3_selected_arms_equal_no_effect, true);
  assert.equal(result.post_hoc, true);
  assert.equal(result.official_challenge_score, false);
  assert.equal(audit.target_scores_recomputed, 1041);
  assert.equal(audit.maximum_absolute_difference, 0);
  assert.equal(audit.gears_macro_mse, result.gears_macro_mse);
  assert.equal(selection.strength, 0.5);
  assert.equal(selection.validation_summary_sha256, snapshot.provenance.gearsValidationSummarySha256);
  assert.equal(selection.validation_candidate_macro_mse[2], validation.candidates[2].macro_mse);
  assert.equal(validation.validation_selected_strength, 0.5);
  assert.equal(figure.test_summary_sha256, snapshot.provenance.gearsTestSummarySha256);
  assert.equal(figure.test_audit_sha256, snapshot.provenance.gearsTestAuditSha256);
  assert.equal(figureAudit.figure_manifest_sha256, snapshot.provenance.gearsFigureManifestSha256);
  assert.equal(figureAudit.test_target_mse_values_checked, 1041);
  assert.equal(figureAudit.post_hoc, true);
  assert.equal(figureAudit.official_challenge_score, false);
  for (const extension of ["svg", "pdf"]) {
    assert.equal(sha256(path.join(root, "assets", `k562-gears-test.${extension}`)),
      figure.artifacts[`figure-18-replogle-gwps-gears-posthoc.${extension}`]);
  }
});

test("training progress is explicitly a timestamped row snapshot", () => {
  const tahoePath = path.join(root, "evidence", "tahoe-representation-snapshot.json");
  const tahoeState = JSON.parse(fs.readFileSync(tahoePath, "utf8"));
  assert.equal(sha256(tahoePath), snapshot.provenance.tahoeSnapshotSha256);
  assert.equal(tahoeState.state, "running");
  assert.equal(tahoeState.stage, "representation");
  assert.equal(snapshot.provenance.tahoeTrainedRows, 13959398);
  assert.equal(snapshot.provenance.tahoeRowsPerEpoch, 92095728);
  assert.equal(snapshot.provenance.tahoeSnapshotUtc, "2026-11-12T16:24:11.284708+00:00");
  assert.equal(snapshot.provenance.tahoeTrainedRows, tahoeState.trained_rows);
  assert.equal(snapshot.provenance.tahoeRowsPerEpoch, tahoeState.expected_rows_per_epoch);
  assert.equal(snapshot.provenance.tahoeSnapshotUtc, tahoeState.updated_at_utc);
  assert.ok(tahoeState.trained_rows > snapshotAudit.tahoe.trained_rows);
  assert.equal(snapshot.provenance.curatedTabulaCells, 1136218);
  assert.equal(snapshot.provenance.curatedTabulaGenes, 60606);
  const tahoe = snapshot.resources.find((resource) => resource.name === "Tahoe-100M");
  assert.match(tahoe.detail, /2026-11-12 16:24:11 UTC.*13,959,398/);
  assert.match(tahoe.detail, /不是去重细胞数/);
  assert.match(tahoe.detail, /不是实时进度/);
});

test("v30 manuscript and redrawn figures match their scoped audits", () => {
  const manuscript = path.join(root, "manuscripts", "epi-c3-manuscript-draft-v30.md");
  assert.equal(sha256(manuscript), snapshot.provenance.manuscriptSha256);
  assert.match(fs.readFileSync(manuscript, "utf8"), /not ready for submission/);
  const configs = [
    {
      dir: "tahoe-pilot",
      asset: "tahoe-diagnostic",
      stem: "figure-02-tahoe-interim-diagnostic",
      manifestHash: snapshot.provenance.tahoePilotFigureManifestSha256,
      auditHash: snapshot.provenance.tahoePilotFigureAuditSha256
    },
    {
      dir: "khp-four-method",
      asset: "khp-transfer",
      stem: "figure-03-khp-transfer",
      manifestHash: snapshot.provenance.khpFourFigureManifestSha256,
      auditHash: snapshot.provenance.khpFourFigureAuditSha256
    }
  ];
  for (const config of configs) {
    const directory = path.join(root, "evidence", config.dir);
    const manifestPath = path.join(directory, "figure-manifest.json");
    const auditPath = path.join(directory, "figure-audit.json");
    assert.equal(sha256(manifestPath), config.manifestHash);
    assert.equal(sha256(auditPath), config.auditHash);
    const figure = JSON.parse(fs.readFileSync(manifestPath, "utf8"));
    const figureAudit = JSON.parse(fs.readFileSync(auditPath, "utf8"));
    assert.equal(figure.state, "complete");
    assert.equal(figure.official_challenge_score, false);
    assert.equal(figureAudit.figure_manifest_sha256, config.manifestHash);
    assert.equal(sha256(path.join(directory, "condition-metrics.csv")), figure.source_table_sha256);
    for (const ext of ["svg", "pdf"]) {
      assert.equal(sha256(path.join(root, "assets", `${config.asset}.${ext}`)),
        figure.artifacts[`${config.stem}.${ext}`]);
    }
  }
  const tahoeDir = path.join(root, "evidence", "tahoe-pilot");
  const tahoeAuditPath = path.join(tahoeDir, "metric-audit.json");
  assert.equal(sha256(tahoeAuditPath), snapshot.provenance.tahoePilotMetricAuditSha256);
  const tahoeAudit = JSON.parse(fs.readFileSync(tahoeAuditPath, "utf8"));
  assert.equal(tahoeAudit.numeric_values_recomputed, 45);
  assert.equal(tahoeAudit.full_flow_greater_mse_conditions, 7);
  assert.equal(tahoeAudit.official_challenge_score, false);
  const khpAudit = JSON.parse(fs.readFileSync(path.join(root, "evidence", "khp-four-method", "figure-audit.json"), "utf8"));
  assert.equal(khpAudit.saved_score_checks.saved_values_compared, 280);
  assert.equal(khpAudit.saved_score_checks.cell_level_metrics_independently_recomputed, false);
  const datlingerDir = path.join(root, "evidence", "datlinger");
  const datlingerManifestPath = path.join(datlingerDir, "figure-manifest.json");
  const datlingerFigureAuditPath = path.join(datlingerDir, "figure-audit.json");
  const datlingerMetricAuditPath = path.join(datlingerDir, "metric-audit.json");
  assert.equal(sha256(datlingerManifestPath), snapshot.provenance.datlingerFigureManifestSha256);
  assert.equal(sha256(datlingerFigureAuditPath), snapshot.provenance.datlingerFigureAuditSha256);
  assert.equal(sha256(datlingerMetricAuditPath), snapshot.provenance.datlingerMetricAuditSha256);
  const datlinger = JSON.parse(fs.readFileSync(datlingerManifestPath, "utf8"));
  const datlingerAudit = JSON.parse(fs.readFileSync(datlingerFigureAuditPath, "utf8"));
  const datlingerMetrics = JSON.parse(fs.readFileSync(datlingerMetricAuditPath, "utf8"));
  assert.equal(datlingerAudit.figure_manifest_sha256, snapshot.provenance.datlingerFigureManifestSha256);
  assert.equal(datlingerAudit.saved_score_checks.saved_values_compared, 64);
  assert.equal(datlingerAudit.saved_score_checks.distribution_metrics_independently_recomputed, false);
  assert.match(datlingerMetrics.scope, /distribution metrics were not independently recomputed/);
  assert.equal(sha256(path.join(datlingerDir, "gene-metrics.csv")), datlinger.source_table_sha256);
  for (const ext of ["svg", "pdf"]) {
    assert.equal(sha256(path.join(root, "assets", `datlinger-test.${ext}`)),
      datlinger.artifacts[`figure-04-datlinger-test.${ext}`]);
  }
  const normanDir = path.join(root, "evidence", "norman-combination");
  const normanManifestPath = path.join(normanDir, "figure-manifest.json");
  const normanFigureAuditPath = path.join(normanDir, "figure-audit.json");
  const normanMetricAuditPath = path.join(normanDir, "metric-audit.json");
  assert.equal(sha256(normanManifestPath), snapshot.provenance.normanFigureManifestSha256);
  assert.equal(sha256(normanFigureAuditPath), snapshot.provenance.normanFigureAuditSha256);
  assert.equal(sha256(normanMetricAuditPath), snapshot.provenance.normanMetricAuditSha256);
  const norman = JSON.parse(fs.readFileSync(normanManifestPath, "utf8"));
  const normanAudit = JSON.parse(fs.readFileSync(normanFigureAuditPath, "utf8"));
  const normanMetrics = JSON.parse(fs.readFileSync(normanMetricAuditPath, "utf8"));
  assert.equal(norman.case_count, 44);
  assert.deepEqual(norman.seen_counts, { "0": 4, "1": 28, "2": 12 });
  assert.equal(normanAudit.figure_manifest_sha256, snapshot.provenance.normanFigureManifestSha256);
  assert.equal(normanAudit.saved_score_checks.saved_values_compared, 528);
  assert.equal(normanAudit.saved_score_checks.previously_independently_recomputed_values, 264);
  assert.equal(normanAudit.saved_score_checks.distribution_metrics_independently_recomputed, false);
  assert.match(normanMetrics.scope, /distribution metrics were not independently recomputed/);
  assert.equal(sha256(path.join(normanDir, "condition-metrics.csv")), norman.source_table_sha256);
  for (const ext of ["svg", "pdf"]) {
    assert.equal(sha256(path.join(root, "assets", `norman-combination.${ext}`)),
      norman.artifacts[`figure-05-norman-combination.${ext}`]);
  }
  const graphDir = path.join(root, "evidence", "norman-graph");
  const graphManifestPath = path.join(graphDir, "figure-manifest.json");
  const graphAuditPath = path.join(graphDir, "figure-audit.json");
  const graphMetricPath = path.join(graphDir, "metric-audit.json");
  assert.equal(sha256(graphManifestPath), snapshot.provenance.normanGraphManifestSha256);
  assert.equal(sha256(graphAuditPath), snapshot.provenance.normanGraphAuditSha256);
  assert.equal(sha256(graphMetricPath), snapshot.provenance.normanGraphMetricAuditSha256);
  const graph = JSON.parse(fs.readFileSync(graphManifestPath, "utf8"));
  const graphAudit = JSON.parse(fs.readFileSync(graphAuditPath, "utf8"));
  const graphMetric = JSON.parse(fs.readFileSync(graphMetricPath, "utf8"));
  assert.equal(graph.condition_count, 44);
  assert.equal(graph.post_hoc, true);
  assert.equal(graph.official_challenge_score, false);
  assert.equal(graphAudit.figure_manifest_sha256, snapshot.provenance.normanGraphManifestSha256);
  assert.equal(graphAudit.saved_score_checks.underlying_seed_stratum_means_recomputed, 57);
  assert.equal(graphAudit.saved_score_checks.condition_difference_values_recomputed, 176);
  assert.equal(graphMetric.scored_method_condition_pairs_checked, 1672);
  for (const file of ["summary-mse.csv", "condition-graph-differences.csv", "graph-contrasts.csv", "graph-coverage.csv"]) {
    assert.equal(sha256(path.join(graphDir, file)), graph.artifacts[file]);
  }
  for (const ext of ["svg", "pdf"]) {
    assert.equal(sha256(path.join(root, "assets", `norman-graph.${ext}`)),
      graph.artifacts[`figure-09-norman-graph.${ext}`]);
  }
  const contrastRows = fs.readFileSync(path.join(graphDir, "graph-contrasts.csv"), "utf8").trim().split(/\r?\n/).slice(1);
  const fixed = contrastRows.map((row) => row.split(",")).filter(([mode]) => mode === "fixed");
  const weights = { seen0: 4, seen1: 28, seen2: 12 };
  assert.equal(fixed.length, 3);
  const weightedDifference = fixed.reduce((sum, [, seen, value]) => sum + weights[seen] * Number(value), 0) / 44;
  assert.ok(Math.abs(weightedDifference - 0.0000014507254478985205) < 1e-14);
});

test("essential-gene figure and tables match the scoped independent audits", () => {
  const directory = path.join(root, "evidence", "replogle-essential");
  const references = [
    ["figure-manifest.json", "essentialFigureManifestSha256"],
    ["figure-audit.json", "essentialFigureAuditSha256"],
    ["test-audit.json", "essentialTestAuditSha256"],
    ["selection-audit.json", "essentialSelectionAuditSha256"]
  ];
  const evidence = {};
  for (const [file, hashKey] of references) {
    const filePath = path.join(directory, file);
    assert.equal(sha256(filePath), snapshot.provenance[hashKey]);
    evidence[file] = JSON.parse(fs.readFileSync(filePath, "utf8"));
  }
  const figure = evidence["figure-manifest.json"];
  const figureAudit = evidence["figure-audit.json"];
  const testAudit = evidence["test-audit.json"];
  const selectionAudit = evidence["selection-audit.json"];
  assert.equal(figure.targets, 127);
  assert.equal(figure.official_challenge_score, false);
  assert.equal(figureAudit.figure_manifest_sha256, snapshot.provenance.essentialFigureManifestSha256);
  assert.equal(figureAudit.saved_score_checks.saved_method_target_mse_values_compared, 381);
  assert.equal(figureAudit.saved_score_checks.saved_paired_differences_compared, 127);
  assert.equal(figureAudit.saved_score_checks.validation_objectives_compared, 5);
  assert.equal(figureAudit.saved_score_checks.training_mean_lower_test_targets, 82);
  assert.equal(testAudit.checked_method_target_pairs, 381);
  assert.equal(testAudit.zero_strength_counts_equal_no_effect, true);
  assert.equal(selectionAudit.checked_candidate_target_pairs, 635);
  assert.equal(selectionAudit.selected_strength, 0);
  assert.equal(selectionAudit.test_treatment_expression_used, false);
  for (const file of ["per-target.csv", "validation-strength.csv"]) {
    assert.equal(sha256(path.join(directory, file)), figure.artifacts[file]);
  }
  for (const ext of ["svg", "pdf"]) {
    assert.equal(sha256(path.join(root, "assets", `replogle-essential.${ext}`)),
      figure.artifacts[`figure-14-replogle-essential.${ext}`]);
  }
  const html = fs.readFileSync(path.join(root, "index.html"), "utf8");
  for (const relativePath of [
    "manuscripts/epi-c3-manuscript-draft-v30.md", "assets/replogle-essential.svg",
    "assets/replogle-essential.pdf", ...references.map(([file]) => `evidence/replogle-essential/${file}`),
    "evidence/replogle-essential/per-target.csv", "evidence/replogle-essential/validation-strength.csv"
  ]) {
    assert.ok(fs.existsSync(path.join(root, relativePath)), `missing ${relativePath}`);
    assert.ok(html.includes(relativePath), `HTML does not reference ${relativePath}`);
  }
});

test("page assets are local, present and match the approved figure", () => {
  const html = fs.readFileSync(path.join(root, "index.html"), "utf8");
  for (const relativePath of ["styles.css", "snapshot-data.js", "app.js", "assets/architecture.svg", "assets/architecture.pdf", "assets/k562-gears-test.svg", "assets/k562-gears-test.pdf", "assets/tahoe-diagnostic.svg", "assets/tahoe-diagnostic.pdf", "assets/khp-transfer.svg", "assets/khp-transfer.pdf", "assets/datlinger-test.svg", "assets/datlinger-test.pdf", "assets/norman-combination.svg", "assets/norman-combination.pdf", "assets/norman-graph.svg", "assets/norman-graph.pdf", "manuscripts/epi-c3-manuscript-draft-v29.md", "evidence/norman-graph/summary-mse.csv", "evidence/norman-graph/condition-graph-differences.csv", "evidence/norman-graph/graph-contrasts.csv", "evidence/norman-graph/graph-coverage.csv", "evidence/norman-graph/figure-manifest.json", "evidence/norman-graph/figure-audit.json", "evidence/norman-graph/metric-audit.json", "evidence/norman-combination/condition-metrics.csv", "evidence/norman-combination/figure-manifest.json", "evidence/norman-combination/figure-audit.json", "evidence/norman-combination/metric-audit.json", "evidence/datlinger/gene-metrics.csv", "evidence/datlinger/figure-manifest.json", "evidence/datlinger/figure-audit.json", "evidence/datlinger/metric-audit.json", "evidence/tahoe-pilot/figure-manifest.json", "evidence/tahoe-pilot/figure-audit.json", "evidence/tahoe-pilot/metric-audit.json", "evidence/tahoe-pilot/condition-metrics.csv", "evidence/khp-four-method/figure-manifest.json", "evidence/khp-four-method/figure-audit.json", "evidence/khp-four-method/condition-metrics.csv", "evidence/parameters-audit.json", "evidence/snapshot-audit-v2.json", "evidence/tahoe-representation-snapshot.json", "evidence/gears-k562/test-summary.json", "evidence/gears-k562/test-audit.json", "evidence/gears-k562/validation-summary.json", "evidence/gears-k562/selection.json", "evidence/gears-k562/figure-audit.json", "evidence/gears-k562/figure-manifest.json", "evidence/model-parameters/c3-khp.json", "evidence/model-parameters/gears-k562.json", "evidence/model-parameters/scgen-khp.json", "evidence/model-parameters/scvidr-khp.json", "evidence/model-parameters/audit-parameters.py"]) {
    assert.ok(fs.existsSync(path.join(root, relativePath)), `missing ${relativePath}`);
    assert.ok(html.includes(relativePath), `HTML does not reference ${relativePath}`);
  }
  assert.doesNotMatch(html, /<(?:script|link)\b[^>]*(?:src|href)=["']https?:/i);
  assert.equal(sha256(path.join(root, "assets", "architecture.svg")), snapshot.provenance.architectureSvgSha256);
  assert.equal(sha256(path.join(root, "assets", "architecture.pdf")), "3154cd66333c2c1358a7702a276dfde784e8136ba7dba6496fe13676a4c14282");
  assert.ok(html.includes(snapshot.provenance.manuscriptSha256));
  for (const file of ["index.html", "styles.css", "app.js", "snapshot-data.js", "README.md", "evidence/parameters-audit.json", "evidence/snapshot-audit-v2.json", "evidence/tahoe-representation-snapshot.json", "evidence/gears-k562/test-summary.json", "evidence/gears-k562/test-audit.json", "evidence/gears-k562/validation-summary.json", "evidence/gears-k562/selection.json", "evidence/gears-k562/figure-audit.json", "evidence/gears-k562/figure-manifest.json", "evidence/model-parameters/c3-khp.json", "evidence/model-parameters/gears-k562.json", "evidence/model-parameters/scgen-khp.json", "evidence/model-parameters/scvidr-khp.json", "evidence/model-parameters/audit-parameters.py"]) {
    const text = fs.readFileSync(path.join(root, file), "utf8");
    assert.doesNotMatch(text, /\b10(?:\.\d{1,3}){3}\b|\/mnt\/storage\/[^/\s]+|[a-z]:\\users\\[^\\\s]+/i);
  }
});

test("text assets do not expose internal server or user paths", () => {
  function walk(directory) {
    return fs.readdirSync(directory, { withFileTypes: true }).flatMap((entry) => {
      const file = path.join(directory, entry.name);
      return entry.isDirectory() ? walk(file) : [file];
    });
  }
  for (const file of walk(root)) {
    if (!/\.(?:html|css|js|json|csv|svg|md|py)$/i.test(file)) continue;
    assert.doesNotMatch(fs.readFileSync(file, "utf8"), /\b10(?:\.\d{1,3}){3}\b|\/mnt\/storage\/[^/\s]+|[a-z]:\\users\\[^\\\s]+/i);
  }
});
