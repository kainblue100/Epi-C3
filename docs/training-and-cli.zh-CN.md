# 训练、推理与命令参考

先运行[合成数据快速入门](quickstart.zh-CN.md)。所有命令均可用 `c3-vcell COMMAND --help` 查看参数；下表是导航，不承诺与 STATE 的命令或检查点兼容。训练、预测和评分必须使用相互独立的细胞；选择超参数不能读取最终留出扰动表达。

## 训练与校准

`train` 包含计数感知表示预训练和条件潜空间群体运输；`--graph` 与 `--epigenetic-bundle` 二选一。`--flow-target population-shift` 使用匹配的训练目标/批次群体均值差，细胞数不足 `--min-group-cells` 的组被排除；检查训练日志及检查点中被排除的组和目标。潜空间插值不是观测到的时间轨迹。

例如，训练后可在**训练目标内部**做两折强度校准：

```bash
c3-vcell calibrate-effect --source runs/example/split/train.h5ad \
  --checkpoint runs/example/model/model.pt \
  --outdir runs/example/effect-calibration --folds 2 \
  --strengths 0 0.1 0.25 0.5 1 --objective expr_mse_unbiased_capped \
  --require-every-fold-improvement --min-relative-improvement 0.02 \
  --control-sampling input-controls --cells-per-target 400 \
  --prediction-steps 8 --seed 42
```

`--source` 必须是用于生成 checkpoint 的**确切训练文件**；各折重新训练模型，可能耗时。示例中的未归一化目标是研究诊断，不是 VCC 计分指标；默认的归一化目标也可能因真实效应分母非正而不可定义。没有合格的非零强度时会选择零，即返回无响应对照，不应解释为模型预测出了扰动效应。

## 不提供真实扰动表达的推理

`requests.csv` 只包含唯一的上下文和目标对：

```csv
context,target_gene
HCT116,GENE1
HCT116,GENE2
```

替换为**模型基因轴上存在、训练时未用作目标**的真实基因，并准备该上下文中与训练细胞不重叠的原始计数对照：

```bash
c3-vcell predict-targets --checkpoint runs/example/model/model.pt \
  --targets requests.csv --controls runs/example/split/input_controls.h5ad \
  --output runs/example/requested.h5ad \
  --effect-calibration runs/example/effect-calibration/effect-calibration.json \
  --cells-per-target 400 --steps 8 --sampling residual-rounding --seed 42
```

预测参数须与校准文件中的细胞数、求解步数和 `input-controls` 采样模式匹配。缺失目标、不同基因轴、上下文不兼容、训练细胞重叠或不匹配的校准文件会报错，不会假造参考细胞或静默替换成零响应。`predict` 用于已经有本地评分参考的情况；`predict-targets` 不需要该文件。

## 命令总览

| 类别 | 命令 | 用途 |
|---|---|---|
| 数据 | `fetch-example` | 下载并校验小型 Arc 管线示例；其目标数不足以完成严格留出训练 |
| 数据 | `prepare` | 抽样原始 H5AD，按目标及对照细胞划分 |
| 数据 | `inspect-orion`, `prepare-orion` | 统计 Orion 分片并抽取真实计数 |
| 数据 | `inspect-tahoe`, `prepare-tahoe`, `split-tahoe` | 检查、抽样、按化合物划分 Tahoe |
| 先验 | `build-epigenetic-bundle`, `shuffle-epigenetic-bundle` | 构造匹配先验及随机化对照 |
| 先验 | `extract-mcool-contacts` | 从已完成的 `.mcool` 提取距离归一化接触候选，不等同统计环调用 |
| 训练 | `train`, `train-drug` | 遗传扰动模型、药物/剂量模型 |
| 诊断 | `diagnose-representation` | 检查训练细胞表示的变化和秩；不证明预测有效 |
| 校准 | `calibrate-effect`, `calibrate-drug-effect` | 在训练内部留出目标或化合物选择效应强度 |
| 校准 | `calibrate-shared-residual` | 校准共享响应加条件残差模型 |
| 预测 | `predict`, `predict-targets`, `predict-drug` | 本地留出、无扰动参考、药物响应预测 |
| 对照 | `control-baseline`, `fit-mean-response`, `predict-mean-response` | 对照重采样和训练集均值响应 |
| 对照 | `predict-shared-residual` | 共享经验响应与条件残差的实验性组合 |
| 不确定性 | `predict-ensemble`, `calibrate-intervals`, `apply-intervals`, `evaluate-intervals` | 模型分歧及目标级伪 bulk 区间；不是单细胞区间 |
| 评估 | `score` | 使用本地 `cell-eval2` VCC preset 的原始指标 |
| 导出 | `export-vcc`, `export-shared-vcc` | 构造挑战赛格式候选文件；仍须官方输入和验证 |

研究脚本 `scripts/run-epi-sample-holdout.py`、`run-epi-hct116.py` 和 `run-network-diagnostic.py` 可运行预指定实验；先查看 `--help` 并固定输入、目标、种子及模型选择规则。`scripts/compare-external-bulk.py` 只用于跨研究 TPM 探索比较，不能替代单细胞挑战赛评分。完整参数和扩展示例见[英文 README](../README.md)。

## 输出和隐私

训练输出包括 `model.pt`、`training.jsonl` 及检查点记录；预测输出为 H5AD，可含源对照 ID 和输入哈希；评分输出包含原始指标与来源记录。`split.json` 也可能包含细胞 ID。真实数据、模型和运行产物可能含敏感信息，发布前需检查权限与许可，不应默认提交到 Git。
