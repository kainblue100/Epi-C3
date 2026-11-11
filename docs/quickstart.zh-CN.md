# 快速入门：从合成数据到预测和评分

本教程参考 [STATE](https://github.com/ArcInstitute/state) 的安装、命令行与示例组织方式，但使用的是 **C3-VCell 自己的接口和模型**。它是尚未验证生物学性能的研究原型。下述数据完全由程序合成，只用于检查软件流程，不能用于报告模型准确率。

## 1. 安装

建议使用 Linux 和 Python 3.11 或 3.12；以下命令从项目根目录执行。Windows 用户可以在 WSL 中运行这些 Bash 命令。可选的全基因组表观组预处理依赖 `py2bit`，不在 Windows 原生环境中安装。

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[test]'
c3-vcell --help
```

机器学习依赖可能较大；需要 GPU 时，请按照本机驱动安装兼容的 PyTorch。默认运行 CPU，不要求下载任何真实数据或权重。安装及其具体约束见 [`pyproject.toml`](../pyproject.toml)。

## 2. 创建示例数据并划分

```bash
python examples/make_toy_data.py --output runs/toy/raw.h5ad

c3-vcell prepare --input runs/toy/raw.h5ad --outdir runs/toy/split \
  --max-targets 0 --max-per-target 40 --max-controls 120 --max-genes 64 \
  --holdout-targets G4 G5 --seed 42
```

生成 360 个细胞、64 个示例基因符号，包含 6 个虚构扰动和 120 个对照；其中虽使用 `SP1`/`USF1` 名称，所有计数均为合成数据。`G4`、`G5` 被明确留出；训练、输入对照、评分参考中的细胞 ID 互不重叠。输出包括 `split/train.h5ad`、`split/input_controls.h5ad`、`split/reference.h5ad` 和 `split/split.json`。

## 3. 训练和预测

```bash
c3-vcell train --train runs/toy/split/train.h5ad --outdir runs/toy/model \
  --latent-dim 8 --hidden-dim 16 --pretrain-steps 5 --flow-steps 5 \
  --batch-size 16 --seed 42

c3-vcell predict --checkpoint runs/toy/model/model.pt \
  --reference runs/toy/split/reference.h5ad \
  --controls runs/toy/split/input_controls.h5ad \
  --output runs/toy/prediction.h5ad --cells-per-target 20 --steps 2 \
  --sampling residual-rounding --control-sampling input-controls \
  --effect-strength 1 --seed 42
```

这些极少的训练步数与固定强度 `1` 仅用于跑通接口，不是模型选择或推荐的生物学训练设置。`predict` 从参考文件读取待预测目标和上下文；使用 `input-controls` 时不使用参考中的扰动表达或参考批次比例生成预测。

没有真实扰动表达时，使用请求 CSV 和独立对照即可：

```bash
c3-vcell predict-targets --checkpoint runs/toy/model/model.pt \
  --targets examples/toy_targets.csv --controls runs/toy/split/input_controls.h5ad \
  --output runs/toy/requested.h5ad --cells-per-target 20 --steps 2 \
  --sampling residual-rounding --effect-strength 1 --seed 42
```

请求文件必须有 `context,target_gene` 两列；目标须在模型基因轴上且未用于训练。以上两种预测使用相同目标、对照和随机种子，计数和来源对照应一致；输出都是 40 × 64 的非负整数矩阵。

## 4. 基线与本地诊断

```bash
c3-vcell control-baseline --reference runs/toy/split/reference.h5ad \
  --controls runs/toy/split/input_controls.h5ad \
  --output runs/toy/control.h5ad --cells-per-target 20 \
  --control-sampling input-controls --seed 42

c3-vcell score --prediction runs/toy/prediction.h5ad \
  --reference runs/toy/split/reference.h5ad --outdir runs/toy/score \
  --metrics pds_cosine expr_mse_unbiased_capped --threads 2

c3-vcell score --prediction runs/toy/control.h5ad \
  --reference runs/toy/split/reference.h5ad --outdir runs/toy/control-score \
  --metrics pds_cosine expr_mse_unbiased_capped --threads 2
```

两项指标分别包含一项 VCC preset 原始指标和一项**不计分的未归一化诊断**；这些不是完整的六指标评估，也不是官方排行榜成绩。每个评分目录中的 `scoring-provenance.json` 记录输入哈希和未定义值。合成数据很小，警告或未定义指标不代表真实任务的效果。

## 5. 可选：合成表观组图

```bash
python examples/make_toy_epigenome.py --source runs/toy/raw.h5ad \
  --outdir runs/toy/epi-inputs
c3-vcell build-epigenetic-bundle --gene-data runs/toy/split/train.h5ad \
  --spec runs/toy/epi-inputs/sources.json --outdir runs/toy/epigenome
c3-vcell shuffle-epigenetic-bundle --bundle runs/toy/epigenome \
  --gene-data runs/toy/split/train.h5ad --outdir runs/toy/graph-shuffle \
  --graph-only --seed 42
c3-vcell train --train runs/toy/split/train.h5ad --outdir runs/toy/epi-model \
  --epigenetic-bundle runs/toy/epigenome --latent-dim 8 --hidden-dim 16 \
  --pretrain-steps 5 --flow-steps 5 --batch-size 16 --seed 42
```

此步骤生成的基因注释、ATAC、TF ChIP、Hi-C 环及甲基化均为**虚构数据**；标注 GRCh38 仅用于演示格式校验，不是真实基因组坐标。实际研究必须重新准备同细胞系、同组装版本且可核验来源的输入。

`examples/quickstart.ipynb` 提供同一流程的交互式版本。教程默认拒绝覆盖已有输出；再次运行请指定新的输出目录，而不是删除未检查的研究结果。

下一步：[真实数据与表观组](data-and-epigenomics.zh-CN.md) · [完整评估](evaluation.zh-CN.md) · [测试方法](testing.zh-CN.md)。
