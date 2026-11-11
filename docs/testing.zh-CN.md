# 测试方法与故障排查

## 安装和运行

在 Python 3.11/3.12 环境中，从项目根目录运行：

```bash
python -m pip install -e '.[test,epigenomics,vcc]'
python -m pytest -q
```

完整套件包含合成数据测试、部分命令行集成测试与可选 VCC 格式接口测试；有关原始依赖、平台和包版本以 [`pyproject.toml`](../pyproject.toml) 为准。`.mcool` 和全基因组参考功能需要其相应运行环境及输入；不要因为测试通过而声称已验证每个外部资源。实际下载的数据、权重和私有文件不包含在仓库中。

快速检查示例、拆分、推理和评分：

```bash
python -m pytest -q tests/test_quickstart.py tests/test_documentation.py \
  tests/test_pipeline.py tests/test_scoring.py
```

`test_quickstart.py` 检查合成数据的整数计数、确定性和覆盖保护，并执行 notebook 中的真实 CLI 训练、两种推理、基线、两项本地诊断，以及合成表观组图构建、图打乱和带先验的训练；不需要安装 Jupyter 才能运行测试。`test_documentation.py` 检查 Markdown 相对链接和命令索引。其假数据结果只是软件回归测试。若需要交互式打开 notebook，可另行在自己的环境安装 Jupyter，并从项目根目录启动。

重点模块：

| 目标 | 测试文件 |
|---|---|
| 细胞、样本及目标隔离 | `tests/test_pipeline.py`, `tests/test_sample_disjoint_split.py` |
| 校准与强度绑定 | `tests/test_calibration.py`, `tests/test_shared_calibration.py` |
| 表观组图、甲基化、接触 | `tests/test_epigenetics.py`, `tests/test_methylation.py`, `tests/test_contacts.py`, `tests/test_mcool.py` |
| 真实数据适配器 | `tests/test_orion.py`, `tests/test_tahoe.py`, `tests/test_reference.py` |
| 评分、导出与外部 bulk 比较 | `tests/test_scoring.py`, `tests/test_shared_export.py`, `tests/test_external_bulk.py` |
| 模型分歧与区间 | `tests/test_ensemble.py`, `tests/test_uncertainty.py` |

测试使用临时目录，不应在真实数据上做破坏性写入。真实数据实验需额外检查文件来源、授权、完整性、细胞系/组装匹配、独立控制和输出形状；测试通过不证明生物学准确性。

## 可复现与安全检查

- 保留输入数据的来源和校验值、已完成分片清单、目标/样本划分、基因轴、代码版本、参数、种子和依赖版本。对没有上游独立校验值的数据，应明确区分本地 SHA-256 与上游校验。
- 每个模型与对照使用相同、独立的预测输入；最终留出目标的表达不参与特征筛选或超参数选择。跨数据集时还需检查同一细胞被重命名后的潜在泄漏。
- 检查输出中每个目标的细胞数、基因顺序、原始整数计数和来源控制 ID。挑战赛形状与真实主办方文件必须单独验证。
- 保留所有警告和未定义指标；重复运行应使用新的输出目录。测序深度不足、组样本量过小或真实效应分母非正时，不要补零伪造成绩。
- 检查数据、模型和运行产物的许可证及敏感内容；不要把 `.env`、凭据、原始数据、权重或包含细胞 ID 的运行文件提交到公开仓库。

## 常见问题

| 提示或现象 | 检查方式 |
|---|---|
| `X must contain raw integer counts` | 使用原始 UMI 计数而非 TPM、log1p 或缩放矩阵。 |
| `Target genes absent from the model gene axis` | 校验目标名称和基因轴；不在轴上的目标不能直接预测。 |
| 训练、输入或评分细胞重叠 | 重新划分独立细胞，检查数据集中是否有重命名的重复细胞。 |
| `Calibration source does not match` | 使用生成 checkpoint 的精确训练 H5AD，且预测的采样模式、每目标细胞数及步数与校准一致。 |
| 校准强度为 0 | 非零候选未满足训练折门槛；结果为对照基线，并非程序崩溃或已学到的响应。 |
| 某项指标 `null` 或分母非正 | 报告为未定义；检查样本数量及真实效应，不要把该值替换为零。 |
| 图构建时 assembly/context/hash 不匹配 | 核查原始来源、样本条件、染色体坐标系和文件是否完整，不能绕过哈希或把另一细胞系的数据冒充匹配。 |
| 输出路径已存在 | 选择一个新的空目录，保留已有实验及日志。 |

完整的科研评价步骤和当前结果见[评估指南](evaluation.zh-CN.md)。
