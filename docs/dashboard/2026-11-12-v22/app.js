(function () {
  "use strict";

  const data = window.EpiC3Snapshot;
  if (!data) {
    throw new Error("The Epi-C3 snapshot data did not load.");
  }

  const number = new Intl.NumberFormat("zh-CN");
  const percent = new Intl.NumberFormat("zh-CN", {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2
  });
  const svgNamespace = "http://www.w3.org/2000/svg";

  function element(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  }

  function svgElement(tag, attributes) {
    const node = document.createElementNS(svgNamespace, tag);
    Object.entries(attributes).forEach(([name, value]) => node.setAttribute(name, String(value)));
    return node;
  }

  function appendCell(row, tag, text, className, label) {
    const cell = element(tag, className, text);
    if (tag === "th") cell.scope = "row";
    if (label) cell.dataset.label = label;
    row.appendChild(cell);
    return cell;
  }

  function renderFoundation() {
    const foundation = data.foundation;
    const trained = Object.values(foundation.trained).reduce((sum, value) => sum + value, 0);
    const expected = Object.values(foundation.expected).reduce((sum, value) => sum + value, 0);
    const progress = document.getElementById("foundation-progress");
    progress.max = expected;
    progress.value = trained;
    document.getElementById("foundation-progress-text").textContent =
      `${number.format(trained)} / ${number.format(expected)} 条计划记录 · ${percent.format(trained / expected * 100)}%`;
    document.getElementById("foundation-updates").textContent = number.format(foundation.updates);
    document.getElementById("foundation-genes").textContent = number.format(foundation.genes);
    document.getElementById("foundation-time").textContent =
      `${foundation.snapshotUtc.slice(0, 10)} ${foundation.snapshotUtc.slice(11, 19)} UTC`;
  }

  function renderModalities() {
    const container = document.getElementById("modality-grid");
    data.modalities.forEach((item) => {
      const card = element("article", "modality-card");
      const top = element("div", "modality-top");
      top.append(element("span", "modality-index", item.index), element("span", "overline", item.tag));
      card.append(
        top,
        element("h4", "", item.title),
        element("p", "modality-number", item.number),
        element("p", "modality-unit", item.unit),
        element("p", "modality-detail", item.detail)
      );
      container.appendChild(card);
    });
  }

  function renderResources() {
    const container = document.getElementById("resource-list");
    data.resources.forEach((item) => {
      const article = element("article", "resource-row");
      const heading = element("div", "resource-title");
      const link = element("a", "", item.name);
      link.href = item.url;
      link.target = "_blank";
      link.rel = "noopener noreferrer";
      heading.append(link, element("span", "resource-status", item.status));
      article.append(heading, element("p", "", item.detail));
      container.appendChild(article);
    });
  }

  function renderModels() {
    const body = document.getElementById("model-body");
    data.modelRows.forEach((item) => {
      const row = element("tr");
      appendCell(row, "th", item.model);
      appendCell(row, "td", item.context, "muted-cell", "对照 / 配置");
      appendCell(row, "td", item.trainable === null ? "未记录" : number.format(item.trainable), "numeric-cell", "初始化时可训练参数");
      appendCell(row, "td", item.flow === null ? "—" : number.format(item.flow), "numeric-cell", "flow 优化器参数");
      appendCell(row, "td", item.status, "muted-cell", "状态");
      body.appendChild(row);
    });
  }

  function setupChart() {
    const select = document.getElementById("chart-select");
    const legend = document.getElementById("chart-legend");
    const donut = document.getElementById("donut");
    const centerValue = document.getElementById("donut-value");
    const centerLabel = document.getElementById("donut-label");
    const reset = document.getElementById("chart-reset");
    let activeChart = data.charts[0];
    let selectedId = null;
    let total = 0;

    data.charts.forEach((chart) => {
      const option = element("option", "", chart.option);
      option.value = chart.id;
      select.appendChild(option);
    });

    function updateSelection() {
      const item = activeChart.items.find((candidate) => candidate.id === selectedId);
      centerValue.textContent = number.format(item ? item.value : total);
      centerLabel.textContent = item
        ? `${item.label} · ${percent.format((item.value / total) * 100)}%`
        : activeChart.centerLabel;
      legend.querySelectorAll("button[data-item-id]").forEach((button) => {
        const selected = button.dataset.itemId === selectedId;
        button.setAttribute("aria-pressed", String(selected));
      });
      donut.querySelectorAll("[data-item-id]").forEach((segment) => {
        segment.classList.toggle("is-muted", selectedId !== null && segment.dataset.itemId !== selectedId);
      });
      reset.disabled = selectedId === null;
    }

    function renderChart(chart) {
      activeChart = chart;
      selectedId = null;
      total = chart.items.reduce((sum, item) => sum + item.value, 0);
      document.getElementById("chart-title").textContent = chart.title;
      document.getElementById("chart-description").textContent = chart.description;
      document.getElementById("chart-note").textContent = chart.note;
      legend.replaceChildren();
      donut.replaceChildren();

      const radius = 82;
      const circumference = 2 * Math.PI * radius;
      const background = svgElement("circle", {
        cx: 110, cy: 110, r: radius, fill: "none", stroke: "#e8eeea", "stroke-width": 20
      });
      donut.appendChild(background);
      const arcs = svgElement("g", { transform: "rotate(-90 110 110)" });
      const animations = [];
      let offset = 0;
      chart.items.forEach((item) => {
        const arcLength = (item.value / total) * circumference;
        const arc = svgElement("circle", {
          cx: 110, cy: 110, r: radius, fill: "none", stroke: item.color, "stroke-width": 20,
          "stroke-dasharray": `${arcLength} ${circumference}`,
          "stroke-dashoffset": -offset,
          "data-item-id": item.id
        });
        arc.style.strokeDasharray = `0 ${circumference}`;
        animations.push({ arc, arcLength });
        arcs.appendChild(arc);
        offset += arcLength;

        const button = element("button", "legend-row");
        button.type = "button";
        button.dataset.itemId = item.id;
        button.setAttribute("aria-pressed", "false");
        const swatch = element("span", "legend-swatch");
        swatch.style.backgroundColor = item.color;
        const labels = element("span", "legend-label");
        labels.append(element("strong", "", item.label), element("small", "", item.detail));
        const values = element("span", "legend-value");
        values.append(
          element("strong", "", number.format(item.value)),
          element("small", "", `${percent.format((item.value / total) * 100)}% · ${chart.unit}`)
        );
        button.append(swatch, labels, values);
        button.addEventListener("click", () => {
          selectedId = selectedId === item.id ? null : item.id;
          updateSelection();
        });
        legend.appendChild(button);
      });
      donut.appendChild(arcs);
      window.requestAnimationFrame(() => {
        window.requestAnimationFrame(() => {
          animations.forEach(({ arc, arcLength }) => {
            arc.style.strokeDasharray = `${arcLength} ${circumference}`;
          });
        });
      });
      updateSelection();
    }

    select.addEventListener("change", () => {
      const chart = data.charts.find((candidate) => candidate.id === select.value);
      if (!chart) throw new Error(`Unknown chart: ${select.value}`);
      renderChart(chart);
    });
    reset.addEventListener("click", () => {
      selectedId = null;
      updateSelection();
    });
    renderChart(activeChart);
  }

  function setupBenchmarks() {
    const tablist = document.getElementById("benchmark-tabs");
    const panel = document.getElementById("benchmark-panel");
    const header = document.getElementById("benchmark-head");
    const body = document.getElementById("benchmark-body");
    const buttons = [];

    function showBenchmark(benchmark) {
      document.getElementById("benchmark-title").textContent = benchmark.title;
      document.getElementById("benchmark-subtitle").textContent = benchmark.subtitle;
      document.getElementById("benchmark-caption").textContent = `${benchmark.title}的内部对照结果`;
      document.getElementById("benchmark-note").textContent = benchmark.note;
      panel.setAttribute("aria-labelledby", `benchmark-tab-${benchmark.id}`);
      buttons.forEach((button) => {
        const selected = button.dataset.benchmark === benchmark.id;
        button.setAttribute("aria-selected", String(selected));
        button.tabIndex = selected ? 0 : -1;
      });

      const headerRow = element("tr");
      benchmark.columns.forEach((column) => {
        const th = element("th", "", column);
        th.scope = "col";
        headerRow.appendChild(th);
      });
      header.replaceChildren(headerRow);
      body.replaceChildren();

      benchmark.rows.forEach((item) => {
        const row = element("tr");
        appendCell(row, "th", item.name, item.kind === "model" ? "model-name" : "");
        appendCell(row, "td", item.metric.toFixed(6), "numeric-cell", benchmark.columns[1]);
        if (benchmark.id === "k562") {
          appendCell(row, "td", item.detail, "muted-cell", benchmark.columns[2]);
        } else {
          appendCell(row, "td", item.delta.toFixed(6), "numeric-cell", benchmark.columns[2]);
          appendCell(row, "td", item.mmd.toFixed(6), "numeric-cell", benchmark.columns[3]);
        }
        body.appendChild(row);
      });
    }

    data.benchmarks.forEach((benchmark, index) => {
      const button = element("button", "tab", benchmark.tab);
      button.type = "button";
      button.id = `benchmark-tab-${benchmark.id}`;
      button.dataset.benchmark = benchmark.id;
      button.setAttribute("role", "tab");
      button.setAttribute("aria-controls", "benchmark-panel");
      button.addEventListener("click", () => showBenchmark(benchmark));
      button.addEventListener("keydown", (event) => {
        let next = index;
        if (event.key === "ArrowRight") next = (index + 1) % buttons.length;
        else if (event.key === "ArrowLeft") next = (index - 1 + buttons.length) % buttons.length;
        else if (event.key === "Home") next = 0;
        else if (event.key === "End") next = buttons.length - 1;
        else return;
        event.preventDefault();
        showBenchmark(data.benchmarks[next]);
        buttons[next].focus();
      });
      tablist.appendChild(button);
      buttons.push(button);
    });
    showBenchmark(data.benchmarks[0]);
  }

  renderFoundation();
  renderModalities();
  renderResources();
  renderModels();
  setupChart();
  setupBenchmarks();
})();
