(function (root) {
  "use strict";

  const foundation = {
    snapshotUtc: "2026-11-12T17:27:54.394429+00:00",
    status: "running",
    scope: "incomplete-stage1-not-a-full-foundation",
    updates: 26250,
    genes: 63079,
    trained: { arc: 15544, tabula: 143342, "orion-HCT116": 456634, "orion-HEK293T": 595090 },
    expected: { arc: 108808, tabula: 998796, "orion-HCT116": 3260365, "orion-HEK293T": 4335720 },
    trainable: 16373966,
    representationOptimizer: 16316174
  };

  const snapshot = {
    snapshotDate: "2026-11-12",
    foundation,
    charts: [
      {
        id: "foundation-trained",
        option: "共享基础模型 · 已处理记录",
        title: "第一阶段已处理的来源记录",
        description: "四个来源在时间点快照中的已处理记录；不是全部计划训练数据。",
        centerLabel: "已处理记录合计",
        unit: "条记录",
        note: "快照为 2026-11-12 17:27:54 UTC，不是实时进度。来源之间可能重复；Tahoe 尚未接入共享训练，已处理行数不是跨来源去重细胞数。",
        items: [
          { id: "foundation-arc", label: "Arc", detail: "共享预训练 · 已处理", value: foundation.trained.arc, color: "#738ec1" },
          { id: "foundation-tabula", label: "Tabula Sapiens", detail: "CELLxGENE 整理版 · 已处理", value: foundation.trained.tabula, color: "#a696b9" },
          { id: "foundation-orion-hct", label: "Orion HCT116", detail: "共享预训练 · 已处理", value: foundation.trained["orion-HCT116"], color: "#0f6b66" },
          { id: "foundation-orion-hek", label: "Orion HEK293T", detail: "共享预训练 · 已处理", value: foundation.trained["orion-HEK293T"], color: "#dcae67" }
        ]
      },
      {
        id: "source-cells",
        option: "来源矩阵 · 细胞记录",
        title: "已记录的单细胞来源规模",
        description: "六个来源矩阵的记录数；仅展示已核验矩阵维度的来源。",
        centerLabel: "来源记录合计（未去重）",
        unit: "细胞记录",
        note: "这是选定来源矩阵记录数的构成，不是所有数据的总量或去重后的独立细胞数；来源之间可能重叠。两个 K562 库有 654 个相同的细胞索引字符串；相同字符串本身不能证明是同一细胞。",
        items: [
          { id: "gwps", label: "K562 全基因组", detail: "Replogle · CRISPRi", value: 1989578, color: "#0f6b66" },
          { id: "tabula-curated", label: "Tabula Sapiens", detail: "CELLxGENE 整理版", value: 1136218, color: "#a696b9" },
          { id: "essential", label: "K562 必需基因", detail: "Replogle · CRISPRi", value: 310385, color: "#dcae67" },
          { id: "arc", label: "Arc virtual cell", detail: "训练来源矩阵", value: 221273, color: "#738ec1" },
          { id: "khp", label: "KHP 全来源", detail: "PBMC、K562 与皮肤相关细胞", value: 127595, color: "#91aeb4" },
          { id: "cmo", label: "CMO", detail: "多供体免疫细胞矩阵", value: 10410, color: "#bd7f70" }
        ]
      },
      {
        id: "target-split",
        option: "K562 冻结划分 · 靶点数",
        title: "全基因组 K562 靶点划分",
        description: "验证和测试靶点在训练前冻结；共使用 1,861 个基因轴。",
        centerLabel: "冻结靶点合计",
        unit: "靶点",
        note: "这是靶点身份的划分，不是细胞划分。三个 flow 随机种子共享一个表征模型，不能当作三个生物学重复。",
        items: [
          { id: "train", label: "训练靶点", detail: "模型拟合", value: 1041, color: "#0f6b66" },
          { id: "validation", label: "验证靶点", detail: "选择预测强度", value: 347, color: "#dcae67" },
          { id: "test", label: "测试靶点", detail: "冻结后评估", value: 347, color: "#91aeb4" }
        ]
      },
      {
        id: "graph-shuffle",
        option: "K562 候选图 · 边数",
        title: "候选基因图打乱情况",
        description: "原图和打乱图比较时，4,141 条候选边中实际改变的边。",
        centerLabel: "候选图边合计",
        unit: "条候选边",
        note: "改变的边仅占 31.25%，打乱对照较弱；测试集只有两个图源靶点。候选边不等于验证过的因果调控。",
        items: [
          { id: "changed", label: "打乱后改变", detail: "候选边", value: 1294, color: "#d98260" },
          { id: "unchanged", label: "打乱后未改变", detail: "候选边", value: 2847, color: "#91aeb4" }
        ]
      }
    ],
    modalities: [
      {
        index: "01",
        tag: "TRANSCRIPTOME",
        title: "单细胞表达与扰动",
        number: "1,989,578 × 8,248",
        unit: "细胞记录 × 基因 · K562 来源矩阵",
        detail: "冻结实验实际使用 159,349 个训练细胞；数值呈非负整数状，但原始计数来源尚未确认。"
      },
      {
        index: "02",
        tag: "3D GENOME",
        title: "染色质空间接触",
        number: "44,219",
        unit: "个候选接触环 · HCT116",
        detail: "来自独立 bulk 测量，用于构建候选先验，并非与训练细胞配对的三维组数据。"
      },
      {
        index: "03",
        tag: "DNA METHYLATION",
        title: "DNA 甲基化",
        number: "53,332,109",
        unit: "条 WGBS 测量记录 · HCT116",
        detail: "在 4,096 基因包中，启动子覆盖 4,081 个基因；关联元件覆盖 2,390 个。两类基因可重叠。"
      },
      {
        index: "04",
        tag: "ACCESSIBILITY + TF",
        title: "染色质可及性与 TF",
        number: "172,732 / 7",
        unit: "个 K562 ATAC 峰 / 个选定 TF ChIP 轨道",
        detail: "分别为 ENCODE ATAC 峰文件和七个选定的 Norman 靶点 TF 轨道，单位不同且并非完整 TF 数据总量。bulk 测量与训练细胞未配对。"
      },
      {
        index: "05",
        tag: "REGULATORY GRAPH",
        title: "候选基因调控图",
        number: "4,141 / 3,913",
        unit: "条候选图边 · K562 / HCT116",
        detail: "这是图模型输入边数，与候选接触环或测量记录不是同一单位，也不证明因果关系。"
      },
      {
        index: "06",
        tag: "ADDITIONAL SCREENS",
        title: "其他扰动训练数据",
        number: "62,113 / 35,309",
        unit: "训练细胞 · Norman K562 / HCT116",
        detail: "HCT116 另有 12,000 个有界参考细胞；不同划分与来源不能简单相加为独立细胞总数。"
      },
      {
        index: "07",
        tag: "PAIRED CANDIDATE",
        title: "CHARM 三模态候选",
        number: "805",
        unit: "个匹配标识 · 一组小鼠 mESC 样本",
        detail: "RNA、QC、ATAC 与接触文件的直接身份交集；958 个接触文件完整读取并按一基坐标检查，不代表训练资格。"
      }
    ],
    resources: [
      {
        name: "Arc virtual cell dataset",
        url: "https://huggingface.co/datasets/cyrilzakka/arc-institute-virtual-cell-dataset",
        status: "已使用 · 快照已核验",
        detail: "训练来源矩阵为 221,273 个细胞记录 × 18,080 个基因；排除留出靶点和独立对照后，198,701 个细胞用于训练。"
      },
      {
        name: "Tahoe-100M",
        url: "https://huggingface.co/datasets/tahoebio/Tahoe-100M",
        status: "表征训练 · 静态快照",
        detail: "2026-11-12 16:24:11 UTC：已处理 13,959,398 行；每 epoch 计划 92,095,728 行。处理行不是去重细胞数，也不是实时进度。"
      },
      {
        name: "X-Atlas-Orion",
        url: "https://huggingface.co/datasets/Xaira-Therapeutics/X-Atlas-Orion",
        status: "两种细胞系表征训练完成",
        detail: "HCT116 为 3,373,853 条、HEK293T 为 4,487,649 条表征训练记录，各完成一轮；这不是跨数据集去重后的细胞总数。"
      },
      {
        name: "Tabula Sapiens v2",
        url: "https://figshare.com/articles/dataset/Tabula_Sapiens_v2/27921984",
        status: "Figshare 原版待获取",
        detail: "CELLxGENE 整理版矩阵为 1,136,218 个细胞记录 × 60,606 个基因；Figshare API 曾返回 HTTP 403，尚未获得原版，也未证明两个版本逐字节相同。"
      },
      {
        name: "CELLxGENE",
        url: "https://cellxgene.cziscience.com/",
        status: "使用整理版 · 非全库",
        detail: "已使用整理版 Tabula 数据；未完成全库下载，也没有全库去重细胞统计。"
      },
      {
        name: "scPerturbBench · KHP / CMO",
        url: "https://doi.org/10.5281/zenodo.20823660",
        status: "对照候选 · 尚未训练",
        detail: "KHP 9,354 条与 CMO 1,419 条训练候选对照已核查，各另留 32 条验证对照；非对照细胞和受保护划分未纳入。跨来源重复及化合物隔离尚未完成。"
      },
      {
        name: "CHARM mESC · GSE303006",
        url: "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE303006",
        status: "候选来源 · 三模态身份核对",
        detail: "959 个 RNA 细胞列、959 个 ATAC 身份及 958 个接触文件身份中，805 个同时匹配 mESC QC；接触 gzip 校验及全量逐条坐标审计已完成，仅是一组小鼠样本。"
      },
      {
        name: "SC-MO-GRN-DB · K562 候选网络",
        url: "https://scmogrndb.psu.edu/browse.php",
        status: "来源筛选 · 未用于训练",
        detail: "RN119 的 29,298 行中有 2,656 对同时标记激活和抑制；原始来源、许可和与冻结测试的重叠尚待核对，未用于训练或评价。DS010、DS011 均不能当作已验证的同细胞三模态数据。"
      }
    ],
    modelRows: [
      {
        model: "Epi-C3 · 共享 RNA 基础模型",
        context: "第一阶段 · 63,079 基因 · 未完成",
        trainable: foundation.trainable,
        flow: null,
        status: "表征优化器 16,316,174；未训练扰动 flow"
      },
      {
        model: "Epi-C3 · RNA-only",
        context: "K562 · 1,861 基因 · 冻结配置",
        trainable: 579722,
        flow: 57792,
        status: "验证选择零强度"
      },
      {
        model: "Epi-C3 · 表观图臂",
        context: "K562 · 1,861 基因 · 节点 / 真实图 / 打乱图",
        trainable: 584842,
        flow: 62912,
        status: "验证选择零强度"
      },
      {
        model: "GEARS",
        context: "K562 · 1,861 基因 · 后验适配",
        trainable: 759567,
        flow: null,
        status: "12,000 步训练完成 · 测试已完成（后验）"
      },
      {
        model: "Epi-C3 · KHP RNA-only",
        context: "KHP case-01 · 5,000 基因",
        trainable: 1389584,
        flow: 57856,
        status: "内部探索对照"
      },
      {
        model: "scGen",
        context: "KHP case-01 · 5,000 基因 · 2.1.0",
        trainable: 9534800,
        flow: null,
        status: "内部探索对照"
      },
      {
        model: "scVIDR",
        context: "KHP case-01 · 5,000 基因 · 两来源适配",
        trainable: 10049800,
        flow: null,
        status: "后验加入"
      },
      {
        model: "STATE",
        context: "未运行",
        trainable: null,
        flow: null,
        status: "许可与使用条件待核验"
      }
    ],
    benchmarks: [
      {
        id: "k562",
        tab: "K562 · 靶点留出",
        title: "K562 全基因组 CRISPRi",
        subtitle: "347 个测试靶点 · 内部冻结划分",
        columns: ["方法", "测试靶点宏 MSE ↓", "解释"],
        note: "先在完整基因轴上归一化，再排除对应靶基因。四个 Epi-C3 臂的验证选择强度均为 0，测试预测因此等同无效应；三个 flow 种子共享一个表征模型。GEARS 在原结果已被观察后加入，使用额外公开 GO 先验和已知留出靶点，只有一个种子；其验证选择强度为 0.5。测试中 GEARS 分别在 191/347、130/347 个靶点优于无效应和训练均值；约 15.9% 的插值后表达条目被投影为零。GEARS 和训练均值输出 log 表达，而非直接计数。控制会复用；这不是外部盲测、同先验比较或官方挑战赛成绩，不能证明图先验有效。",
        rows: [
          { name: "无效应", metric: 0.01604481739394287, detail: "精确输入对照", kind: "baseline" },
          { name: "训练均值", metric: 0.015711020985759664, detail: "简单基线 · log 表达", kind: "baseline" },
          { name: "GEARS · 后验适配", metric: 0.01583080899741956, detail: "额外 GO · 1 个种子 · 验证强度 0.5", kind: "posthoc" },
          { name: "Epi-C3 · RNA-only", metric: 0.01604481739394287, detail: "3 个 flow 种子；强度 0", kind: "model" },
          { name: "Epi-C3 · 表观节点", metric: 0.01604481739394287, detail: "3 个 flow 种子；强度 0", kind: "model" },
          { name: "Epi-C3 · 真实图", metric: 0.01604481739394287, detail: "3 个 flow 种子；强度 0", kind: "model" },
          { name: "Epi-C3 · 打乱图", metric: 0.01604481739394287, detail: "3 个 flow 种子；强度 0", kind: "model" }
        ]
      },
      {
        id: "khp",
        tab: "KHP · CD8 迁移",
        title: "KHP CD8 T 细胞迁移",
        subtitle: "十条件平均 · 内部探索",
        columns: ["方法", "平均表达 MSE ↓", "Delta Pearson ↑", "每基因 RBF MMD² ↓"],
        note: "目标 CD8 对照在拟合和预测中可用，不是完全未知细胞类型的盲测；C3 为 RNA-only。scVIDR 是观察原始结果后加入的两来源适配，其 MMD² 仅检查文件完整性，未独立重算。指标衡量不同属性，不构成单一综合排名。",
        rows: [
          { name: "无效应", metric: 0.017461, delta: 0.172245, mmd: 0.006224, kind: "baseline" },
          { name: "来源均值相加", metric: 0.006194, delta: 0.771672, mmd: 0.009700, kind: "baseline" },
          { name: "Epi-C3 · RNA-only", metric: 0.012311, delta: 0.466169, mmd: 0.005421, kind: "model" },
          { name: "scGen · 2.1.0", metric: 0.007829, delta: 0.632018, mmd: 0.079746, kind: "comparison" },
          { name: "scVIDR · 后验适配", metric: 0.009871, delta: 0.591462, mmd: 0.076248, kind: "posthoc" }
        ]
      }
    ],
    provenance: {
      manuscriptSha256: "933ece34333c5409a2bec984af07096baa526a931bc1d281bb311ef74853ae62",
      parameterAuditSha256: "df9fcc0a1224f99ff12394e4f9da67357740dfcaf048f465bed10f1cb8518631",
      snapshotAuditSha256: "a9d6d915911968d36e35c4abf28a1410a8612e488bb816149dae7c2cc72a124f",
      architectureSvgSha256: "7acd6f875b43ccf608d720071aaba8367bd2aeb88b5067c248bdae81b658a8eb",
      comparisonAuditSha256: {
        "c3-khp.json": "607f1f68d3e52a6d7ae59bf5f00f61f6330f46fdfefa187a0c26cbe20342d67d",
        "gears-k562.json": "c4ac0682dbdcf8cad2d907013b730e5c2b54c6ad0702a9bf012203771f5733c1",
        "scgen-khp.json": "b691a195319154651677716ab519d970c0b9477c35021d45aa5b7cda66efb9ba",
        "scvidr-khp.json": "f3170bd7b706da2a723b344e05c8fb3c35836068ccca63acceabd9edf9d272a4"
      },
      comparisonAuditScriptSha256: "5900b577f58aaef0517d16b07357275abca9d308de8fb7ad47b24d0a17cb95e5",
      tahoeSnapshotUtc: "2026-11-12T16:24:11.284708+00:00",
      tahoeSnapshotSha256: "0119cb0a47ff78a818ceefa23abe1700fe9b068dc5ca59ec0910719fd1273edc",
      tahoeTrainedRows: 13959398,
      tahoeRowsPerEpoch: 92095728,
      scPerturbBenchAuditSha256: "94514a5c47b48f4fe273d588609a29e97f4d122401d05c57575e73e65f445a22",
      scPerturbBenchVerificationSha256: "64f1154d0a08c437da23a85f52a6c9e2886f7111f7e8913ef033f003307d4681",
      stage2PreflightSha256: "39c0557507ff1b54c8e0525108df3f03197253b70826059c81ec7551ee5579de",
      stage2DispatchSnapshotSha256: "fd6820a08194647a0dd448305d30687fcadbae94903d2e9029ac2ad39e619c7a",
      charmAuditSha256: "6ea045d9edd0ab97c977ef4142c590f32c414dffb4ec0a1fcc13e3a2cdac8ef1",
      charmInnerAuditSha256: "c7000f6ca752768d078f462202b1720b603b023c7489aff1ade5b6c8c9ed044b",
      charmCoordinateProbeSha256: "efe7ba3ae45844a7e674f423c1cbfb531f684d8360f6ca5b8b8c67f93a01b86d",
      charmCoordinateAuditSha256: "07b6a4ec3c86c13cfcf4af688d4d95a9a969d37524c6b1aafec2fc6a8fa1b723",
      charmHeaderProbeSha256: "2398ceb8e3bf7de7a88640e95bc95528567e5dd682c491d5533aff3f4f154b3b",
      trimodalResponseSha256: "a6aae043c02d8d981d6d2d0288b0d69297b9041d3830af71c58b6c18b3140dd0",
      trimodalResponseTestSha256: "b2fc2f02f7e4207cbbf731cb1117928faec89e6725d56eb3b93baf2f53b49e76",
      scMoGrnScreenSha256: "ec0c2f41121998cf732c92a5f9f9245bf60a1b9030dc41f95293c06901199481",
      k562ReferenceAuditSha256: "c6ba33b993cf28f594abe72e00207eb425b6291cc077a7fe6c7118cf9c170a87",
      gearsTestSummarySha256: "cdbfd82ffbd7d987dab31b51fe8e553e36c265d1db007112b67fd197977a4d09",
      gearsTestAuditSha256: "edc95005298065ccb87af533765b4c0bbd19a5f2714444edf87d68f5ff207ff5",
      gearsValidationSummarySha256: "3c35d10eeb652ea87f186537cc281e03b763f6b72cbf4a5389427e82162107ab",
      gearsSelectionSha256: "373a7d7a319de3225049e65f91ffd20ca567fa17efb6a270b6429bcda7054c6b",
      gearsFigureManifestSha256: "f797e14fa601917bfe1620042b68cc113b4a191dbc532c188bc7327f5c1b5259",
      gearsFigureAuditSha256: "fd80893bd1a5c4ba398f722be297d8652ea5cbe85d0f2a71ab0d3b0623738637",
      tahoePilotFigureManifestSha256: "da297c8abe3ec6152f9edb572b216213663cf4139767f35868c01055107e4782",
      tahoePilotFigureAuditSha256: "df19b2890a4b27e0f22c48e3aab463e38b2e9439d083724476d6c9b91add6ea8",
      tahoePilotMetricAuditSha256: "d48877f46319267f1286eb3c847cfeb4276f8bc42aea1454e2767951e68f1c25",
      khpFourFigureManifestSha256: "286745024c3ede7488cd8b80524cffe259e3d02e4da594c4493cd23864b1b852",
      khpFourFigureAuditSha256: "0f7601659d264ae566e5bcc0272045fae1b9a04ce4a6f55df03a7fe515771929",
      datlingerFigureManifestSha256: "78a86f066f73d86e6b5decc89f22da08ceb060caa68bcdfe8945fe59277b7cd5",
      datlingerFigureAuditSha256: "1d3495b6179624aa3ebdecf2aa502b175425c795324504845e251ca986a374ef",
      datlingerMetricAuditSha256: "8776d814d35ccaeca2503b8ad050c4b4165e98efa396cc5292733df0be4db2cd",
      normanFigureManifestSha256: "e70125d8c85c7fe6ff7f00dc3e1af0fb491ff01c7dd466707f1b626979870d27",
      normanFigureAuditSha256: "94001737b6d23ecdebed1c8cc6590d01cc1ac0211abce1c0533da42a4b4321f0",
      normanMetricAuditSha256: "54f9115900d97d2081f2b1d269f9b572a3da782d13ec6812eda14bb710f3001d",
      normanGraphManifestSha256: "0b87adfb3d7588126d37eead51ecd0548f7fef529d3210548b9a1b68981312ff",
      normanGraphAuditSha256: "c59eb3d5e2f47bf2ba651b8b72afe83c97ef61bf86d366da6c7f92d62b1e24c3",
      normanGraphMetricAuditSha256: "a40d4b5312ca3f026cf5b332b7d57fae51194869ba86dc49ef900714227a571e",
      essentialFigureManifestSha256: "5a58257446bfc41a5c3ce528d9a351c07d2cad9212e0ba36b057140741c36a47",
      essentialFigureAuditSha256: "c7f83dac76aa34064c13420f62ba3f8b179fcdf9fe338925b4d4b9483f07fa40",
      essentialTestAuditSha256: "faacf568bd33e5c34327dfdb99a29fc172528a2fa58525b38db61823e9130431",
      essentialSelectionAuditSha256: "7ede0be29a33b0f9b74a1bee216e805295406228ff0975cc6b14ae87ba0229be",
      arcSourceCells: 221273,
      arcSourceGenes: 18080,
      arcTrainCells: 198701,
      orionHct116TrainedRows: 3373853,
      orionHek293tTrainedRows: 4487649,
      k562AtacPeakRows: 172732,
      k562SelectedTfTracks: 7,
      curatedTabulaCells: 1136218,
      curatedTabulaGenes: 60606,
      curatedTabulaFileBytes: 45013733785
    }
  };

  if (typeof module !== "undefined" && module.exports) {
    module.exports = snapshot;
  } else {
    root.EpiC3Snapshot = snapshot;
  }
})(globalThis);
