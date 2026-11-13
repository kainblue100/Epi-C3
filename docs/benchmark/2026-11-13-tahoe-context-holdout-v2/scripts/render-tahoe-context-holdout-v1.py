"""Render the candidate Tahoe context holdout as a self-contained HTML report."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
from pathlib import Path


CONTEXTS = {"C32", "HOP62", "HepG2/C3A", "Hs 766T", "PANC-1"}


def count(value) -> int:
    if type(value) is not int or value < 0:
        raise ValueError("Report counts must be nonnegative integers")
    return value


def render(report, report_sha256: str) -> str:
    if (
        report.get("format") != "epi-c3-tahoe-state-context-holdout-exact-summary-v1"
        or report.get("status") != "candidate-not-enforced"
        or report.get("new_validation_selection_still_required") is not True
        or len(report_sha256) != 64
        or any(char not in "0123456789abcdef" for char in report_sha256)
    ):
        raise ValueError("The exact candidate report is incomplete or incompatible")
    contexts = report["contexts"]
    if not isinstance(contexts, list) or len(contexts) != 5 or {
        row["context"] for row in contexts
    } != CONTEXTS:
        raise ValueError("The report does not contain the five expected contexts")
    source_total = count(report["source_rows"])
    matched_total = count(report["official_test_rows_with_exact_condition_match"])
    blocked_total = count(report["all_source_rows_in_heldout_contexts"])
    blocked_eligible = count(report["eligible_rows_in_heldout_contexts_before_identity_exclusions"])
    remaining_before = count(report["remaining_eligible_rows_before_identity_exclusions"])
    retained_exclusions = count(report["previous_identity_exclusions_outside_heldout_contexts"])
    remaining = count(report["remaining_eligible_rows_after_original_identity_exclusions"])
    if remaining != remaining_before - retained_exclusions:
        raise ValueError("The exact candidate count is inconsistent")
    if not 0 <= matched_total <= blocked_total <= source_total or blocked_eligible > blocked_total:
        raise ValueError("The candidate source-row totals are inconsistent")
    if (
        sum(count(row["source_rows_in_context"]) for row in contexts) != blocked_total
        or sum(count(row["matched_official_test_rows"]) for row in contexts) != matched_total
        or sum(
            count(row["eligible_treated_rows_before_identity_exclusions"])
            + count(row["eligible_training_control_rows_before_identity_exclusions"])
            for row in contexts
        ) != blocked_eligible
    ):
        raise ValueError("Context totals do not match the candidate report")
    for row in contexts:
        if (
            count(row["matched_official_test_rows"]) > count(row["source_rows_in_context"])
            or count(row["eligible_treated_rows_before_identity_exclusions"])
            + count(row["eligible_training_control_rows_before_identity_exclusions"])
            > count(row["source_rows_in_context"])
        ):
            raise ValueError("A context's source-row totals are inconsistent")

    max_rows = max(row["source_rows_in_context"] for row in contexts)
    if max_rows == 0:
        raise ValueError("The report has no source rows")
    table, bars = [], []
    for index, row in enumerate(contexts):
        name = html.escape(row["context"])
        all_rows, matched = row["source_rows_in_context"], row["matched_official_test_rows"]
        eligible = (
            row["eligible_treated_rows_before_identity_exclusions"]
            + row["eligible_training_control_rows_before_identity_exclusions"]
        )
        table.append(
            f"<tr><th scope=\"row\">{name}</th><td>{all_rows:,}</td>"
            f"<td>{matched:,}</td><td>{eligible:,}</td></tr>"
        )
        y = 42 + index * 45
        total_width = 480 * all_rows / max_rows
        matched_width = 480 * matched / max_rows
        bars.append(
            f'<text x="0" y="{y + 17}" class="label">{name}</text>'
            f'<rect x="135" y="{y}" width="{total_width:.2f}" height="22" rx="4" class="total"/>'
            f'<rect x="135" y="{y + 6}" width="{matched_width:.2f}" height="10" rx="3" class="matched"/>'
            f'<text x="630" y="{y + 17}" class="value">{all_rows:,} / {matched:,}</text>'
        )
    table_html = "\n".join(table)
    bars_html = "\n".join(bars)
    safe_sha = html.escape(report_sha256)
    evidence_json = html.escape(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Tahoe 测试情境隔离候选方案 · Epi-C3</title>
<style>
:root{{color-scheme:light;--ink:#17243a;--muted:#52647a;--line:#dce4ef;--blue:#275fa4;--red:#bd4a3e;--bg:#f4f7fb}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--ink);font:16px/1.6 system-ui,-apple-system,"Segoe UI",sans-serif}}
main{{max-width:1100px;margin:auto;padding:32px 22px 60px}} h1{{font-size:clamp(1.7rem,4vw,2.4rem);line-height:1.2;margin:.4rem 0}}
h2{{font-size:1.25rem;margin:0 0 12px}} p{{margin:.55rem 0}} a{{color:#164f93}} .eyebrow{{font-size:.84rem;letter-spacing:.08em;color:var(--blue);font-weight:700}}
.notice{{background:#fff5e8;border-left:5px solid #d27b16;padding:15px 18px;margin:22px 0;border-radius:4px}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:12px;margin:20px 0}}
.card,section{{background:#fff;border:1px solid var(--line);border-radius:10px;padding:18px}}
.card strong{{display:block;font-size:1.45rem;color:var(--blue)}} .card span,.muted{{color:var(--muted);font-size:.92rem}}
section{{margin-top:14px}} .chart{{overflow-x:auto}} svg{{display:block;width:100%;min-width:720px;height:auto}}
svg .label,svg .value{{font:14px system-ui,sans-serif;fill:var(--ink)}} svg .total{{fill:#b8cce4}} svg .matched{{fill:var(--red)}}
.legend{{display:flex;gap:22px;flex-wrap:wrap;font-size:.9rem;color:var(--muted)}} .swatch{{display:inline-block;width:12px;height:12px;margin-right:6px;border-radius:2px}}
.table-wrap{{overflow-x:auto}} table{{border-collapse:collapse;width:100%;font-size:.94rem}} th,td{{padding:9px 10px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}}
th:first-child{{text-align:left}} thead th{{color:var(--muted);font-weight:600}} ul{{padding-left:1.3rem}} li{{margin:.35rem 0}}
code{{overflow-wrap:anywhere;font-size:.85em}} footer{{color:var(--muted);font-size:.85rem;margin-top:22px}}
details{{margin-top:14px}} summary{{cursor:pointer;color:var(--blue);font-weight:600}} pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:var(--bg);padding:12px;border-radius:6px;font-size:.8rem}}
@media(max-width:600px){{main{{padding:22px 14px 40px}} .cards{{grid-template-columns:1fr 1fr}} .card{{padding:13px}} .card strong{{font-size:1.1rem}}}}
</style>
</head>
<body>
<main>
<div class="eyebrow">EPI-C3 · TAHOE / STATE</div>
<h1>测试情境隔离候选方案</h1>
<p class="muted">基于冻结元数据与历史身份排除审计；本次交集核查读取了私有身份元数据，未读取原始表达，也不公开单细胞身份。</p>
<div class="notice"><strong>候选方案，尚未接入训练。</strong> 原有续训任务未改变，仍不受此方案保护；本页不是两个 Epi-C3 模型的预测成绩，也不是完整 STATE 比较。</div>
<div class="cards">
<div class="card"><strong>{blocked_total:,}</strong><span>五个测试细胞系的全部来源记录</span></div>
<div class="card"><strong>{blocked_eligible:,}</strong><span>其中原规则下合格的记录，身份排除前</span></div>
<div class="card"><strong>{remaining:,}</strong><span>保留的合格记录，原有身份排除后</span></div>
<div class="card"><strong>{matched_total:,}</strong><span>原论文测试标签精确匹配的来源记录</span></div>
</div>
<section>
<h2>隔离策略与范围</h2>
<p>候选方案把 Tahoe 中五个 STATE 测试细胞系的<strong>全部记录</strong>从基础模型训练和验证中排除，包括各种药物、剂量及对照；它不依赖药物别名或剂量字符串匹配。下游扰动预测仍需采用事先冻结的共同划分。</p>
<p>图中浅色为各细胞系全部来源记录，红色为原论文测试条件精确匹配的子集。数字均为来源行次，不等同于去重细胞。</p>
<div class="legend"><span><i class="swatch" style="background:#b8cce4"></i>整条细胞系</span><span><i class="swatch" style="background:#bd4a3e"></i>测试条件精确匹配</span></div>
<div class="chart"><svg viewBox="0 0 920 275" role="img" aria-labelledby="chart-title">
<title id="chart-title">五个测试细胞系的全部来源记录与测试条件精确匹配记录</title>
{bars_html}
</svg></div>
<div class="table-wrap"><table>
<thead><tr><th scope="col">细胞系</th><th scope="col">全部来源记录</th><th scope="col">测试条件匹配</th><th scope="col">原规则合格记录*</th></tr></thead>
<tbody>{table_html}</tbody></table></div>
<p class="muted">* 原规则合格记录为旧身份排除之前的处理记录与训练对照之和。</p>
</section>
<section>
<h2>数量如何确定</h2>
<p>冻结的 Tahoe 来源共 {source_total:,} 条记录。原规则下五个测试细胞系有 {blocked_eligible:,} 条合格记录；从其余记录中扣除既有的历史身份留出后，得到 {remaining:,} 条。该值<strong>尚未扣除</strong>新方案需要重新选取的验证身份。</p>
<p>历史留出与五个细胞系的交集通过已验证来源分片的元数据核对；没有向报告导出单细胞身份。分片完整性沿用冻结的大小、修改时间及下载时哈希证据。</p>
</section>
<section>
<h2>尚未解决</h2>
<ul>
<li>需要把候选过滤器同时接入新的训练流和验证选择，保留原有身份排除，并重新核对实际流的完整行数。</li>
<li>第一阶段检查点和重放数据的跨来源重复与测试暴露尚未排除；不能因此宣称整个基础模型无泄漏。</li>
<li>细胞系注释和来源文件仍需独立核实；两个新模型和其他论文基线尚未完成同协议评测。</li>
</ul>
<p><a href="https://github.com/ArcInstitute/state/tree/9bbfe78a434a55205e4de834e1ea99f85f7a3add/state_preprint_toml_files/tahoe_tomls">STATE 作者冻结划分</a> · <a href="https://kainblue100.github.io/Epi-C3/benchmark/2026-11-13-v3/">已发布的独立小规模 STATE 试验</a></p>
<details><summary>查看聚合摘要（JSON）</summary><pre>{evidence_json}</pre></details>
</section>
<footer>聚合报告 SHA-256：<code>{safe_sha}</code>。该摘要不含细胞身份、表达矩阵或模型权重。</footer>
</main>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = args.report.read_bytes()
    report = json.loads(data)
    document = render(report, hashlib.sha256(data).hexdigest())
    descriptor = os.open(args.output, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
        stream.write(document)
        stream.flush()
        os.fsync(stream.fileno())


if __name__ == "__main__":
    main()
