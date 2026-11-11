# 数据与表观组先验

本项目不随代码分发真实单细胞数据、基因组参考或训练权重。下载前检查各数据源的访问权限、使用许可和磁盘空间；只使用已完成且校验过的文件，不能把 `.part` 当作完整输入。

## 单细胞输入

遗传扰动工作流接受 H5AD：`.X` 是原始、非负、有限的整数计数；细胞和基因名称唯一；`obs` 中有非空的扰动标签列，默认 `target_gene`，对照默认 `non-targeting`。建议明确提供 `context` 和 `batch`。所有扰动基因须出现在 `var_names`；使用表观组先验还需可映射的 Ensembl 基因 ID 和匹配的基因组版本。不能把归一化矩阵、TPM 或缺失基因当作原始计数。

`prepare` 按目标留出，并将对照细胞划分为训练、预测输入与评分参考三部分。普通 `prepare` **不保证样本级独立**；需要双重留出时使用 `scripts/run-epi-sample-holdout.py`。缩减基因轴时，程序以外层训练计数和预先指定的目标/邻居选基因；内折沿用该轴，不应声称每个内折重新选过基因。

## Orion

先对已完成的分片检查目标和样本，再替换以下占位符：

```bash
c3-vcell inspect-orion --shards /path/to/complete/*.parquet \
  --output runs/orion-groups.csv
c3-vcell prepare-orion --shards /path/to/complete/*.parquet \
  --gene-metadata /path/to/gene_metadata.parquet \
  --targets GENE1 GENE2 GENE3 --control-label ACTUAL_CONTROL_LABEL \
  --max-per-target 100 --max-controls 500 --output runs/orion-subset.h5ad
c3-vcell prepare --input runs/orion-subset.h5ad --outdir runs/orion/split \
  --max-targets 0 --seed 42
```

控制标签和目标必须从数据中确认；保留选择时使用的分片清单。筛选使用 passing-cell 信息，重复基因符号按适配器规则合并并保留 Ensembl ID。Orion 的使用受其数据集许可约束；该工作流不保证任何特定细胞系都与表观组数据匹配。

## Tahoe 药物响应

`inspect-tahoe`、`prepare-tahoe` 和 `split-tahoe` 检查药物、细胞系、板号和 DMSO 对照，按 canonical SMILES 留出化合物。`train-drug` 和 `predict-drug` 使用 SMILES **文本**和剂量，不是分子图模型；不保证 scaffold 外推或剂量外推。药物标签不是 VCC 遗传扰动标签。完整命令见 [英文参考](../README.md#tahoe-drug-responses)。

其他原始计数参考数据可通过 `train --reference` 或 `train-drug --reference` 参与表示预训练。基因按名称对齐并使用缺失 mask；重命名后的同一生物细胞、不同平台或注释差异仍可能导致隐性泄漏或偏差。

## 构建 HCT116 表观组图

图在训练前由**细胞系和组装版本匹配**的外部来源构造，不使用留出扰动表达。输入来源规格包含文件路径、来源 URL、SHA-256、实验条件、坐标约定及测序类型；builder 重新核对文件和模型基因轴。示例准备命令与真实来源要求见 [英文参考](../README.md#experimental-epigenomic-regulatory-network) 和 `python scripts/make-hct116-matched-spec.py --help`。

有经过验证的来源规格与训练 H5AD 后：

```bash
c3-vcell build-epigenetic-bundle --gene-data runs/example/split/train.h5ad \
  --spec /path/to/hct116-sources.json --outdir runs/example/epigenome
c3-vcell shuffle-epigenetic-bundle --bundle runs/example/epigenome \
  --gene-data runs/example/split/train.h5ad \
  --outdir runs/example/graph-shuffle --graph-only --seed 42
c3-vcell train --train runs/example/split/train.h5ad \
  --outdir runs/example/epi-model --epigenetic-bundle runs/example/epigenome \
  --seed 42
```

这些 `/path/to/...` 是占位符，不能直接运行。网络把 Hi-C/ChIA-PET 接触、ATAC 和相应 TF ChIP 证据组织成候选基因边；启动子及联系区域的 CpG 甲基化以数值和可用性 mask 输入。缺失不是零甲基化，边也不是已证实因果关系。`--graph-only` 维持特征和 mask，打乱图目标；它不是所有可行图的均匀采样，也不是生物学重复。

已有 HCT116 先验使用 [ENCODE ENCSR477GZK](https://www.encodeproject.org/experiments/ENCSR477GZK/) Hi-C 候选环及 [GSM3317488](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM3317488) WGBS。环的锚点有多个宽度；WGBS 文件没有逐位点读深，不能按覆盖度加权。当前已验证的甲基化预览只覆盖 chr21/chr22；其他染色体标为未观测。完整参考下载并校验前，不应声称全基因组甲基化覆盖。

上述不同 bulk 实验不是配对单细胞多组学；HFFc6 的 Hi-C 也不能作为 HCT116 同系证据。外部研究可支持覆盖分析，但不能单凭重叠证明调控因果关系或预测性能。

## 数据来源和许可

数据获取入口包括 [Tabula Sapiens v2](https://figshare.com/articles/dataset/Tabula_Sapiens_v2/27921984)、[Arc 数据集](https://huggingface.co/datasets/cyrilzakka/arc-institute-virtual-cell-dataset)、[Tahoe-100M](https://huggingface.co/datasets/tahoebio/Tahoe-100M)、[CELLxGENE](https://cellxgene.cziscience.com/) 和 [X-Atlas-Orion](https://huggingface.co/datasets/Xaira-Therapeutics/X-Atlas-Orion)。确认各资源最新条款和校验文件；不同整理版本不能未经核验就视为逐字节相同。
