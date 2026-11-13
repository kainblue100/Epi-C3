"""Render the static ENCODE candidate-prior report from pinned aggregates."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
FILES = {
    "selection-summary.json": "74ff4e0e8be77a4cef81486d6e52a6d47031b8636c605c05c2205a03124098b0",
    "graph-summary.json": "06538583dcbef58718e9a80dc207f5a38633ae91466481d6bdfabc7435d94382",
    "controls-summary.json": "cba775e15d6416876aa9791d1cc56470e842947ea9b9ce528bcfe09922341c23",
    "graph-v3-independent-audit.json": "407bc1385437981ee8bc2906586860efd209935a4d380950f7ae5cfa68f71b67",
}


def load_pinned(name: str) -> dict:
    data = (HERE / name).read_bytes()
    if hashlib.sha256(data).hexdigest() != FILES[name]:
        raise ValueError(f"Aggregate hash mismatch: {name}")
    value = json.loads(data)
    if not isinstance(value, dict):
        raise ValueError(f"Aggregate is not a JSON object: {name}")
    return value


def render(selection: dict, graph: dict, controls: dict, audit: dict) -> str:
    if (
        selection["status"] != "source-selection-only-not-a-graph-or-model"
        or graph["status"] != "candidate-prior-not-a-trained-model"
        or controls["status"] != "structural-controls-only-not-model-results"
        or audit["status"] != "structure-and-sampled-overlaps-verified-not-a-model"
        or selection["input_sha256"]["selected_file_manifest"] != graph["input_sha256"]["selected_peak_manifest"]
        or controls["input_sha256"]["graph_summary"] != FILES["graph-summary.json"]
        or audit["input_sha256"]["graph_summary"] != FILES["graph-summary.json"]
        or audit["input_sha256"]["additional_controls_summary"] != FILES["controls-summary.json"]
        or audit["input_sha256"]["real_edges"] != graph["input_sha256"]["real_edges"]
    ):
        raise ValueError("Aggregate status or provenance does not match the candidate graph")
    selected, result, mapping = selection["selection"], graph["graph"], graph["gene_mapping"]
    checked, spots = audit["structure"], audit["spot_check"]
    extra = audit["additional_control_checks"]["controls"]
    original = graph["degree_preserving_shuffle"]
    if (
        checked["real_edges"] != result["candidate_edges"]
        or checked["real_edges_absent_from_shuffled"] != original["real_edges_absent_from_shuffled"]
        or checked["degree_difference_nodes"] != 0
        or original["nodes_with_different_in_degree"] != 0
        or original["nodes_with_different_out_degree"] != 0
        or len(extra) != len(controls["additional_controls"])
        or any(
            independently_checked["seed"] != recorded["seed"]
            or independently_checked["changed_fraction"] != recorded["changed_fraction"]
            or independently_checked["real_edges_absent_from_shuffled"] != recorded["real_edges_absent_from_shuffled"]
            or independently_checked["degree_difference_nodes"] != 0
            or recorded["nodes_with_different_in_degree"] != 0
            or recorded["nodes_with_different_out_degree"] != 0
            for independently_checked, recorded in zip(extra, controls["additional_controls"], strict=True)
        )
    ):
        raise ValueError("Independent audit and graph or control counts disagree")
    control_rows = "\n".join(
        f'<tr><th scope="row">{item["seed"]}</th><td>{item["real_edges_absent_from_shuffled"]:,}</td>'
        f'<td>{100 * item["changed_fraction"]:.2f}%</td><td>0</td></tr>'
        for item in (original, *extra)
    )

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="description" content="ENCODE K562 ChIP 与 ATAC 构建的调控网络候选先验及结构审计。">
  <title>Epi-C3 | ENCODE 调控网络候选先验</title>
  <style>
    :root {{ color-scheme: light; font-family: system-ui, -apple-system, "Segoe UI", sans-serif; color: #172433; background: #f5f8f7; line-height: 1.65; }}
    * {{ box-sizing: border-box; }} body {{ margin: 0; }}
    a {{ color: #075c68; text-decoration-thickness: .08em; text-underline-offset: .2em; }}
    .wrap {{ width: min(1120px, calc(100% - 2.5rem)); margin: auto; }}
    header {{ border-bottom: 1px solid #d6e2de; background: #fff; }}
    header .wrap {{ display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: .75rem; padding: 1rem 0; }}
    .brand {{ font-weight: 800; letter-spacing: .12em; text-decoration: none; }}
    nav {{ display: flex; flex-wrap: wrap; gap: 1rem; font-size: .9rem; }}
    .hero {{ padding: 3.5rem 0 2rem; }}
    .eyebrow {{ color: #147481; font-size: .75rem; font-weight: 800; letter-spacing: .15em; }}
    h1 {{ font-size: clamp(2.15rem, 5vw, 4rem); line-height: 1.15; letter-spacing: -.04em; max-width: 20ch; margin: .5rem 0 1rem; }}
    h2 {{ font-size: clamp(1.45rem, 3vw, 2rem); line-height: 1.25; }} h3 {{ font-size: 1.05rem; }}
    .lead {{ max-width: 72ch; font-size: 1.07rem; }}
    .notice {{ border-left: 4px solid #b65b19; background: #fff4e8; padding: 1rem 1.2rem; border-radius: 0 .4rem .4rem 0; }}
    .stats {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; margin: 1.5rem 0 3rem; }}
    .stat, .card {{ background: white; border: 1px solid #dae7e2; border-radius: .7rem; padding: 1.25rem; }}
    .stat strong {{ display: block; color: #0e6471; font-size: 1.9rem; line-height: 1.2; }}
    .stat span, .note {{ color: #526a70; font-size: .9rem; }} section {{ margin-bottom: 3rem; }}
    .grid {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 1rem; }}
    .card h3 {{ margin: 0 0 .5rem; }} .card p:last-child {{ margin-bottom: 0; }}
    .table-wrap {{ overflow-x: auto; border: 1px solid #dae7e2; border-radius: .7rem; background: white; }}
    table {{ border-collapse: collapse; width: 100%; text-align: left; min-width: 34rem; }}
    th, td {{ padding: .8rem 1rem; border-bottom: 1px solid #e4edea; }} th {{ background: #edf5f2; }}
    tr:last-child td, tr:last-child th {{ border-bottom: 0; }}
    footer {{ border-top: 1px solid #d6e2de; padding: 1.5rem 0 2.5rem; color: #526a70; font-size: .9rem; }}
    @media (max-width: 700px) {{ .stats, .grid {{ grid-template-columns: repeat(2, 1fr); }} .hero {{ padding-top: 2rem; }} }}
    @media (max-width: 450px) {{ .stats, .grid {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
  <header><div class="wrap"><a class="brand" href="../../">EPI.C3 / RESEARCH ATLAS</a>
    <nav aria-label="页面导航"><a href="#evidence">来源与构图</a><a href="#controls">打乱图</a><a href="#limits">评测边界</a><a href="../2026-11-13-grn-preflight-v1/">前置预检 ↗</a></nav>
  </div></header>
  <main class="wrap">
    <div class="hero">
      <p class="eyebrow">ENCODE PRIOR / 2026.11.13 / STATIC SNAPSHOT</p>
      <h1>把公开表观证据<br>收紧成候选网络。</h1>
      <p class="lead">冻结 K562 ChIP-seq 与 ATAC-seq 峰数据，结合 GRCh38 / GENCODE 注释，在同一基因轴上生成候选图及度数匹配的结构对照。</p>
      <div class="notice" role="note"><strong>候选先验，不是模型成绩：</strong>共享 RNA foundation 第一阶段发生校验异常退出，第二阶段尚未启动。本页没有训练扰动模型，也没有官方 cell-eval 分数；服务器状态可能在快照后变化。</div>
    </div>
    <div class="stats" aria-label="候选图范围">
      <div class="stat"><strong>{result["source_genes"]} / {selected["eligible_shared_targets"]}</strong><span>共同合格靶点产生至少一条候选源边</span></div>
      <div class="stat"><strong>{result["candidate_edges"]:,}</strong><span>非自环 ChIP / ATAC / promoter 候选边</span></div>
      <div class="stat"><strong>{result["target_genes"]:,}</strong><span>有候选入边的基因</span></div>
      <div class="stat"><strong>{result["pilot_target_sources_by_split"]["test"]} / 28</strong><span>RPE1 pilot 测试靶点有直接源边</span></div>
    </div>
    <section id="evidence">
      <p class="eyebrow">01 / FROZEN EVIDENCE</p><h2>从公开峰到候选边</h2>
      <div class="grid">
        <article class="card"><h3>来源与样本</h3><p>冻结的官方查询包含 {selection["catalog_counts"]["files"]:,} 个 released GRCh38 IDR peak 文件和 {selection["catalog_counts"]["experiments"]} 个标称未扰动 K562 实验。按靶点精确 symbol / Ensembl、一致细胞系、无记录的处理或遗传修改、至少两个生物重复过滤，选出 {selected["selected_experiments"]} 个实验的 {selected["selected_peak_files"]} 个 ChIP 文件；标签可包含辅因子，并非全为已验证转录因子。</p></article>
        <article class="card"><h3>校验与注释</h3><p>逐文件校验官方 MD5、大小、SHA-256 和 gzip，复用并核验另一个公开 K562 ATAC 峰文件。{mapping["common_foundation_features"]:,} 个共同 foundation 特征中，{mapping["features_with_unambiguous_primary_gencode_transcripts"]:,} 个有无歧义的 GENCODE v44 主染色体转录本注释；排除 {result["excluded_chip_summit_rows"]} 条峰顶越界的 ChIP 记录。</p></article>
        <article class="card"><h3>三重坐标重叠</h3><p>对每个注释转录本，在 0-based TSS 两侧各 {graph["promoters"]["promoter_radius_bp"]:,} bp 建窗口。仅当 ChIP 峰、ATAC 峰和同一启动子窗口实际三重重叠，才建立来源基因 → 目标基因候选边；去除自环，不赋予激活或抑制符号。</p></article>
        <article class="card"><h3>独立抽查</h3><p>对 {spots["sources_represented"]} 个源基因分层抽取 {spots["sampled_edges"]} 条边，用独立实现重新计算三重重叠，{spots["independent_three_way_overlap_checks_passed"]} 条通过。{result["edges_with_multiple_chip_experiments"]:,} 条边有多个 ENCODE 实验支持，涉及 {checked["sources_with_multiple_experiment_edges"]} 个源基因；不同实验不等于独立生物培养。</p></article>
      </div>
      <p class="note">只审计汇总和抽样边；抽查通过不代表所有边均经独立复算。完整来源清单及网络在受控运行目录，页面不发布细胞身份、原始表达或网络边。</p>
    </section>
    <section id="controls">
      <p class="eyebrow">02 / STRUCTURAL CONTROLS</p><h2>三份打乱图，仍需谨慎解释</h2>
      <p>在完整候选基因轴上，每个固定种子完成每条边 5 次成功的双边交换；三个控制均保持每个节点的入度和出度。下表是各自不再与真实图重合的边数，不是预测表现。</p>
      <div class="table-wrap"><table><thead><tr><th scope="col">种子</th><th scope="col">替换的真实边</th><th scope="col">替换比例</th><th scope="col">度数不同的节点</th></tr></thead><tbody>
      {control_rows}
      </tbody></table></div>
      <p class="note">仅根据源节点出度和无自环约束推得的替换比例上界为 {100 * checked["changed_edge_fraction_upper_bound_from_source_outdegrees"]:.2f}%；同时满足全部入度约束的可达上限可能更低。有限次交换不保证从所有合法图中均匀采样或已收敛，三份图也不是生物重复。</p>
    </section>
    <section id="limits">
      <p class="eyebrow">03 / NOT A BENCHMARK</p><h2>这张图不解决模型评测的先决条件</h2>
      <p>K562 ChIP 与 ATAC 是不配对的 bulk 数据，不是同一单细胞的观测；元数据标注未扰动也不能证明培养物与基准细胞完全独立。RPE1 使用该图属于跨细胞系迁移；结合与开放染色质不证明因果调控。</p>
      <p>此图是在已知 pilot 靶点划分后筛选和构建的候选，不能作为盲选图。当前 pilot 的 K562 训练包含全部 40 个同名靶点；RPE1 训练 / 验证 / 测试分别有 {result["pilot_target_sources_by_split"]["train"]} / {result["pilot_target_sources_by_split"]["val"]} / {result["pilot_target_sources_by_split"]["test"]} 个靶点在此图有直接源边，测试仅覆盖 {result["pilot_target_sources_by_split"]["test"]} / 28，不能据此开展有说服力的图效应比较。</p>
      <p>第一阶段与重放跨来源暴露仍未排除；既定 Tahoe 流若完整消费，独立审计下界为 7,562,326 条论文测试扰动行次，不是去重细胞。</p>
      <p>后续需要先完成并审计共享基础模型，冻结真正独立的共同划分、统一可索引基因轴，然后与 RNA-only、无效应、多个打乱图及 STATE 或同协议预测一起使用官方 cell-eval；现有候选图不是结果优劣的证据。</p>
      <p><a href="../2026-11-13-grn-preflight-v1/">查看先前图覆盖与泄漏预检 ↗</a>　<a href="../2026-11-13-v3/">查看已有 STATE 探索性试验 ↗</a>　<a href="https://github.com/kainblue100/Epi-C3/tree/main/releases/2026-11-13-grn-encode-v1">查看聚合证据和复现代码 ↗</a></p>
    </section>
  </main>
  <footer><div class="wrap">公开数据来源：<a href="https://www.encodeproject.org/about/data-use-policy/">ENCODE 数据政策</a>、<a href="https://www.encodeproject.org/help/citing-encode/">引用指南</a>和 <a href="https://www.gencodegenes.org/human/release_44.html">GENCODE v44</a>。本页仅呈现聚合结果。</div></footer>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = render(*(load_pinned(name) for name in FILES))
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(result)
    print(hashlib.sha256(result.encode("utf-8")).hexdigest())


if __name__ == "__main__":
    main()
