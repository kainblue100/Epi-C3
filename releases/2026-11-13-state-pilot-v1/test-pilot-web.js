"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

const page = process.argv[2];
if (!page) throw new Error("Provide the pilot report directory");
const expected = JSON.parse(fs.readFileSync(path.join(page, "pilot-results.json"), "utf8"));
const element = (tag) => ({
  tag,
  children: [],
  textContent: "",
  appendChild(child) { this.children.push(child); },
  replaceChildren(...children) { this.children = children; },
});
const body = element("tbody");
const note = element("p");
const document = {
  getElementById(id) { return { "pilot-results": body, "finite-sensitivity": note }[id] || null; },
  createElement: element,
};
const context = { window: {}, document };
vm.runInNewContext(fs.readFileSync(path.join(page, "pilot-results.js"), "utf8"), context);
assert.equal(JSON.stringify(context.window.EPI_C3_STATE_PILOT), JSON.stringify(expected));
vm.runInNewContext(fs.readFileSync(path.join(page, "report.js"), "utf8"), context);
assert.equal(body.children.length, 6);
const models = ["state", "perturb_mean", "context_mean"];
expected.metrics.forEach((metric, i) => {
  const cells = body.children[i].children;
  assert.equal(cells.length, 5);
  assert.equal(cells[1].textContent, `${metric.common_conditions} / 28`);
  models.forEach((model, j) => {
    const value = metric.scores[model];
    assert.equal(cells[j + 2].textContent, value === null ? "未定义" : value.toFixed(4));
  });
});
assert.match(note.textContent, /7 个条件共排除 16 个基因—条件项；22 个共同有效条件/);
console.log("Pilot report table, undefined values, and sensitivity text match the public JSON.");
