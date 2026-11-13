"""Render the static GRN preflight page from the pinned public aggregates."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


AUDIT_SHA = "a30750c88e4070f1e17a925c3825a7757f20e811f73594afee5c6d6e013763d8"
ELIGIBILITY_SHA = "453456f3dbe4d6753464132c97da9134f4eced5a042e8a170060738d47b0df67"
HERE = Path(__file__).resolve().parent


def read_pinned(path: Path, expected: str) -> dict:
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError(f"Aggregate hash mismatch: {path.name}")
    result = json.loads(data)
    if not isinstance(result, dict):
        raise ValueError(f"Expected an object in {path.name}")
    return result


def render(audit: dict, eligibility: dict) -> str:
    if (
        audit["status"] != "preflight-complete-not-a-model"
        or eligibility["status"] != "metadata-audit-complete-not-a-new-split"
    ):
        raise ValueError("These aggregates do not describe the expected preflight")
    for key in ("pilot_plan", "foundation_stage1_plan", "legacy_graph_plan", "graph_manifest", "real_graph_edges"):
        if audit["input_sha256"][key] != eligibility["input_sha256"][key]:
            raise ValueError(f"Aggregate input provenance disagrees: {key}")
    if audit["input_sha256"]["audit_script"] != eligibility["input_sha256"]["common_preflight_script"]:
        raise ValueError("The eligibility audit used different preflight code")

    axes = audit["axis_and_graph"]
    partitions = axes["pilot_target_partitions"]
    shared = eligibility["eligibility"]
    overlap = audit["same_source_training_row_overlap"]["legacy_graph_partitions"]
    full = axes["graph_full_axis"]
    panel = axes["graph_projected_to_pilot_hvgs"]
    norm = audit["training_control_normalization"]
    missing_test = partitions["test"]["targets"] - partitions["test"]["targets_on_hvg_axis"]
    changed_percent = 100 * full["real_edges_absent_from_shuffled"] / full["real_edges"]

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="description" content="调控网络分支在共同划分上的真实数据兼容性与泄漏预检。">
  <title>Epi-C3 | 调控网络预检</title>
  <style>
    :root {{ color-scheme: light; font-family: system-ui, -apple-system, "Segoe UI", sans-serif; color: #172433; background: #f5f8f7; line-height: 1.65; }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; }}
    a {{ color: #075c68; text-decoration-thickness: .08em; text-underline-offset: .2em; }}
    .wrap {{ width: min(1120px, calc(100% - 2.5rem)); margin: auto; }}
    header {{ border-bottom: 1px solid #d6e2de; background: #fff; }}
    header .wrap {{ display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: .75rem; padding: 1rem 0; }}
    .brand {{ font-weight: 800; letter-spacing: .12em; text-decoration: none; }}
    nav {{ display: flex; flex-wrap: wrap; gap: 1rem; font-size: .9rem; }}
    .hero {{ padding: 3.5rem 0 2rem; }}
    .eyebrow {{ color: #147481; font-size: .75rem; font-weight: 800; letter-spacing: .15em; }}
    h1 {{ font-size: clamp(2.15rem, 5vw, 4rem); line-height: 1.15; letter-spacing: -.04em; max-width: 20ch; margin: .5rem 0 1rem; }}
    h2 {{ font-size: clamp(1.45rem, 3vw, 2rem); line-height: 1.25; }}
    h3 {{ font-size: 1.05rem; }}
    .lead {{ max-width: 72ch; font-size: 1.07rem; }}
    .notice {{ border-left: 4px solid #b65b19; background: #fff4e8; padding: 1rem 1.2rem; border-radius: 0 .4rem .4rem 0; }}
    .stats {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; margin: 1.5rem 0 3rem; }}
    .stat, .card {{ background: white; border: 1px solid #dae7e2; border-radius: .7rem; padding: 1.25rem; }}
    .stat strong {{ display: block; color: #0e6471; font-size: 1.9rem; line-height: 1.2; }}
    .stat span, .note {{ color: #526a70; font-size: .9rem; }}
    section {{ margin-bottom: 3rem; }}
    .grid {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 1rem; }}
    .card h3 {{ margin: 0 0 .5rem; }}
    .card p:last-child {{ margin-bottom: 0; }}
    .table-wrap {{ overflow-x: auto; border: 1px solid #dae7e2; border-radius: .7rem; background: white; }}
    table {{ border-collapse: collapse; width: 100%; text-align: left; min-width: 36rem; }}
    th, td {{ padding: .8rem 1rem; border-bottom: 1px solid #e4edea; }}
    th {{ background: #edf5f2; }}
    tr:last-child td, tr:last-child th {{ border-bottom: 0; }}
    footer {{ border-top: 1px solid #d6e2de; padding: 1.5rem 0 2.5rem; color: #526a70; font-size: .9rem; }}
    @media (max-width: 700px) {{ .stats, .grid {{ grid-template-columns: repeat(2, 1fr); }} .hero {{ padding-top: 2rem; }} }}
    @media (max-width: 450px) {{ .stats, .grid {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
  <header><div class="wrap">
    <a class="brand" href="../../">EPI.C3 / RESEARCH ATLAS</a>
    <nav aria-label="页面导航"><a href="#coverage">图覆盖</a><a href="#leakage">划分与泄漏</a><a href="#input">输入语义</a><a href="#limits">边界</a></nav>
  </div></header>
  <main class="wrap">
    <div class="hero">
      <p class="eyebrow">GRN PREFLIGHT / 2026.11.13 / STATIC SNAPSHOT</p>
      <h1>先核对网络、数据与划分，<br>再训练调控模型。</h1>
      <p class="lead">这次真实数据审计对照冻结的 K562 / RPE1 STATE pilot、共享 RNA 基础模型基因轴和既有 K562 候选图，检查是否具备公平比较条件。</p>
      <div class="notice" role="note"><strong>不是模型结果：</strong>共享基础模型第一阶段发生校验异常退出，第二阶段未启动；没有训练新 GRN 模型、运行新预测或计算官方 cell-eval 分数。页面是静态历史快照，不是服务器实时状态。</div>
    </div>
    <div class="stats" aria-label="主要发现">
      <div class="stat"><strong>{partitions["all_selected"]["targets_directly_sourcing_real_edges"]} / {partitions["all_selected"]["targets"]}</strong><span>已选靶点在既有图中有直接源边</span></div>
      <div class="stat"><strong>{shared["eligible_targets_sourcing_real_graph_edges"]} / {shared["eligible_shared_targets"]}</strong><span>共同合格靶点在既有图中有直接源边</span></div>
      <div class="stat"><strong>{missing_test} / {partitions["test"]["targets"]}</strong><span>RPE1 测试靶点不在 pilot 高变基因轴</span></div>
      <div class="stat"><strong>{overlap["test"]["overlapping_source_rows"]}</strong><span>旧图测试输入与 pilot K562 训练输入重叠行</span></div>
    </div>

    <section id="coverage">
      <p class="eyebrow">01 / GRAPH COVERAGE</p><h2>现有图不能通过简单换靶点解决覆盖不足</h2>
      <p>两份原始来源文件的完整 SHA-256 已重新核验。按冻结方案，仅统计两个细胞系中各有至少 {shared["minimum_single_perturbation_cells_per_context"]} 个单基因扰动细胞、且在共同基因轴上的靶点。{shared["eligible_shared_targets"]} 个合格靶点里，仅 {shared["eligible_targets_sourcing_real_graph_edges"]} 个是既有真实图的直接源节点，共 {shared["candidate_edges_from_eligible_graph_sources"]} 条源边；它属于旧图训练划分。当前 40 靶点的直接源边覆盖为零。</p>
      <p class="note">匹配使用区分大小写的精确基因符号，不做别名映射。图来自 K562 的非配对 bulk 证据，迁移到 RPE1 有情境差异，也不能视为已证实的因果调控。直接源边缺失不等于间接消息传递一定无效。本次是后验元数据检查，不是新的盲测划分。</p>
      <div class="grid">
        <article class="card"><h3>完整图的对照</h3><p>真实与打乱图各 {full["real_edges"]:,} 条边，完整轴上的入度和出度一致；只有 {full["real_edges_absent_from_shuffled"]:,} 条真实边被替换（{changed_percent:.2f}%）。</p></article>
        <article class="card"><h3>投影后的对照</h3><p>投影到 pilot 高变基因后分别剩 {panel["real_edges"]} / {panel["shuffled_edges"]} 条边；{panel["nodes_with_different_out_degree"]} 个节点的出度、{panel["nodes_with_different_in_degree"]} 个节点的入度不同，不能把它当作该轴上严格匹配度数的对照。</p></article>
      </div>
    </section>

    <section id="leakage">
      <p class="eyebrow">02 / SPLITS AND LEAKAGE</p><h2>共同靶点与旧划分并不独立</h2>
      <p>在 RPE1 中，冻结 pilot 有 {partitions["train"]["targets"]} / {partitions["val"]["targets"]} / {partitions["test"]["targets"]} 个训练 / 验证 / 测试靶点；但 K562 的全部 40 个同名靶点均进入训练，因此不是全局未见靶点评测。</p>
      <div class="table-wrap"><table>
        <thead><tr><th scope="col">旧图划分</th><th scope="col">与 pilot K562 训练输入重叠的源行</th><th scope="col">涉及扰动条件</th></tr></thead>
        <tbody>
          <tr><th scope="row">训练</th><td>{overlap["train"]["overlapping_source_rows"]}</td><td>{overlap["train"]["overlapping_perturbation_conditions"]}</td></tr>
          <tr><th scope="row">验证</th><td>{overlap["validation"]["overlapping_source_rows"]}</td><td>{overlap["validation"]["overlapping_perturbation_conditions"]}</td></tr>
          <tr><th scope="row">测试</th><td>{overlap["test"]["overlapping_source_rows"]}</td><td>{overlap["test"]["overlapping_perturbation_conditions"]}</td></tr>
        </tbody>
      </table></div>
      <p class="note">行交集证实这些记录存在于保存的输入，并不证明每一行均参与过梯度更新。不能直接把旧 STATE 和旧图结果拼接为无泄漏比较；需要统一、先冻结的训练 / 测试协议。</p>
    </section>

    <section id="input">
      <p class="eyebrow">03 / INPUT SEMANTICS</p><h2>基因轴和归一化必须一起修正</h2>
      <div class="grid">
        <article class="card"><h3>靶点与轴</h3><p>{axes["hvg_count"]:,} 个 pilot 高变基因中，{axes["hvgs_with_exact_ensembl_id_on_foundation_axis"]:,} 个精确匹配 foundation Ensembl 轴；RPE1 测试靶点只有 {partitions["test"]["targets_on_hvg_axis"]} / {partitions["test"]["targets"]} 个在 pilot 高变轴。按现有同轴索引实现，其余靶点不能直接索引；这不是对其他编码方案的否定。</p></article>
        <article class="card"><h3>原始文库与局部 panel</h3><p>在 {norm["training_control_cells"]} 个训练对照中，高变基因平均占原文库 {100 * norm["hvg_fraction_of_original_library"]["mean"]:.2f}%。局部 panel 与原文库归一化的逐元素平均绝对差为 {norm["local_panel_vs_original_library_mean_absolute_difference"]:.6f}；把已 log1p 矩阵误当计数再次归一化，与正确局部归一化的差为 {norm["naive_log_input_vs_local_panel_mean_absolute_difference"]:.6f}。这些是输入诊断，不是预测分数。</p></article>
      </div>
    </section>

    <section id="limits">
      <p class="eyebrow">04 / BOUNDARIES</p><h2>下一步仍需先解决基础模型与隔离</h2>
      <p>基础模型第一阶段及重放数据的跨来源暴露尚未充分排除。独立紧界审计指出：若既定 Tahoe 续训完整消费原计划数据，至少 7,562,326 条论文测试扰动<strong>行次</strong>会被预训练接触；这不是去重细胞数，也不是断言训练已经消费完。候选五细胞系隔离尚未接入现有等待器。</p>
      <p>最终比较需要先完成且审计共享 RNA foundation，冻结无泄漏划分与独立图来源，并在相同条件下比较 RNA-only、无效应、有效打乱图及 STATE 或同协议预测；之后使用官方 cell-eval。旧图微小的真实图与打乱图差异不构成图有效性的证据。</p>
      <p><a href="../2026-11-13-grn-encode-v1/">查看后续 ENCODE 候选图 ↗</a>　<a href="../2026-11-13-v3/">查看已有 STATE 探索性试验 ↗</a>　<a href="https://github.com/kainblue100/Epi-C3/tree/main/releases/2026-11-13-grn-preflight-v1">查看聚合 JSON、代码与复现说明 ↗</a></p>
    </section>
  </main>
  <footer><div class="wrap">仅发布聚合证据；不包含原始细胞、身份、私有网络或模型权重。网页数值由固定哈希的两份聚合报告生成。</div></footer>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New HTML output file")
    args = parser.parse_args()
    page = render(read_pinned(HERE / "audit.json", AUDIT_SHA), read_pinned(HERE / "eligibility.json", ELIGIBILITY_SHA))
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(page)
    print(hashlib.sha256(page.encode("utf-8")).hexdigest())


if __name__ == "__main__":
    main()
