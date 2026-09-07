# WARP-G 相关工作与对比方法

更新日期：2026-09-06。实验表协议以 `design.md` 为准。这里的“官方代码”只指论文作者、作者团队或项目维护方
公开的实现；第三方复现不算。“未找到”表示截至 2026-08-23 检查了论文页面、正文代码声明和公开 GitHub 搜索后，
未找到可确认的作者仓库，不等于作者将来不会发布。正式投稿前应再复核一次。

## 1. 本方法是什么

WARP-G（Workload-Aware Regional Physical Graph materialization）解决的是**离线预算下应当为哪些语料区域构建昂贵
GraphRAG 索引**的问题。它先用 design workload 构建文档共访问图并做 Leiden 分区，只构建少量 HippoRAG2 probe
graphs，学习每个 region 的预期检索收益，再按

```text
query_frequency × max(predicted_gain, 0) / estimated_graph_cost
```

选择区域。在线查询先经过共享 BM25 + Dense router，只访问已物化且与问题相关的区域图。

WARP 自己在每个数据集的 1,000 条 released queries 上做确定性 5-fold cross-fitting：每折 800 条只用于物理设计，
200 条只用于测试，每条问题恰好被 held out 一次。主表其他系统构图不读取 query，全库建一次、评全部 1,000 题；
逐题结果按 `query_id` 与 WARP 对齐。

WARP-G 与大多数 GraphRAG 工作的根本区别不是“发明另一种 KG 抽取器”，而是把 GraphRAG 看成一个 workload-aware、
budgeted physical-design 问题。当前区域图后端来自锁定 commit 的官方 HippoRAG2，贡献点位于 region 形成、低成本
probe、收益预测、预算选择和在线路由。

主实验是系统级成本–质量比较，不是同后端 matched-budget 曲线。对照方法各按作者自己的完整方法运行，用同一套
质量指标和成本记账比较，不把 WARP 的 Hybrid Base 注入它们，也不把它们的核心 KG 统一换成 HippoRAG2。

## 2. 主实验七个系统

协议细节见 `design.md` 第 3–4 节。每个数据集一张主表。

| 方法 | 主表角色 | 实现 | 与 WARP-G 的关键区别 |
|---|---|---|---|
| Base | 普通 RAG / WARP 零图点 | 本仓库：BM25 + NV-Embed-v2 + RRF + pinned BGE CrossEncoder | 无 KG，无 workload 设计。 |
| HippoRAG2 | 官方全图 | **官方代码**：[OSU-NLP-Group/HippoRAG](https://github.com/OSU-NLP-Group/HippoRAG)，commit `c617143f01477243992a63b2e2151cc003dd3b21` | 全语料一张 KG，官方图检索通路；不按 workload 选择，不强制再融 Base。 |
| KET-RAG | 作者原生稀疏图 | **官方代码**：[waetr/KET-RAG](https://github.com/waetr/KET-RAG)；commit 写入 `configs/official_baselines.yaml` 后冻结。主表 β=0.8（篇数） | 按 PageRank 只为 ceil(β\|V\|) 篇建骨架 KG，全库 keyword 二分图补覆盖；不读历史 workload，不 probe 区域收益。 |
| G2ConS | 论文复现稀疏图 | **未找到官方代码**（截至 2026-08-23）。按论文复现 concept 图 + κ=0.8 选核 + λ=0.6 双路。核心 KG 用 HippoRAG2（论文为 MS-GraphRAG） | 按 concept 中心性选核，不是 workload region。主表必须写成 *G2ConS-style*，不得写成官方 G2ConS。 |
| LinearRAG | 作者原生全库轻量图 | **官方代码**：[DEEP-PolyU/LinearRAG](https://github.com/DEEP-PolyU/LinearRAG) `bcc94e66c221f798801255efba09311d6fbcd8d6` | 每个文档都进 Tri-Graph，优化的是图本身便宜，不是哪些区域该物化。官方通路无独立 BM25+dense Base。 |
| LightRAG | 作者原生全库图 | **官方代码**：[HKUDS/LightRAG](https://github.com/HKUDS/LightRAG) `d49112fb7548ee14cb727d43bd68e34da0a2c942`。主表 `hybrid` 是图上 local+global | 完整 GraphRAG 系统；官方 `hybrid` 不是 BM25+dense。 |
| WARP-G | 主方法 | 本仓库 | 只物化预测正收益的 workload regions，检索走 Base 路由 + 区域 HippoRAG2。 |

这四个对照方法作者论文里都没有 WARP 的 Hybrid Base。主表比较的是「各系统自己的完整方法」，不是「Base + 它们的图」。

主表不出现 BM25-only、Dense-only、HippoRAG2+Base 融合、旧的 KET/G² HippoRAG 适配版（剩余 token 卡核心 KG），
以及四个选择器消融。

### WARP 内部消融（不进主表）

| 方法 | 角色 | 说明 |
|---|---|---|
| Random-region / Frequency-only / Gain-only / Cost-only | 选择器消融 | 复用同一折 regions 和 HippoRAG2，只换排序公式。回答「三信号公式是否有效」，不回答「KET 强不强」。 |
| query / semantic / random partition | 分区消融 | RQ4。 |
| BM25、Dense、Base+Full Graph | 附录参考 | 诊断 reranker 与融合上界；不是 RQ3 主比较。 |

旧的六点 `budget_fraction × T_full` 曲线若仍计算，只画 WARP 自己，不把 KET/G²/Linear/Light 标上去。

## 3. LinearRAG 与 LightRAG

### LinearRAG（ICLR 2026）— 有官方代码，已接入

- 论文：[LinearRAG](https://arxiv.org/abs/2510.10114)
- 官方代码：[DEEP-PolyU/LinearRAG](https://github.com/DEEP-PolyU/LinearRAG)
- 本仓库锁定 commit：`bcc94e66c221f798801255efba09311d6fbcd8d6`

LinearRAG 用轻量实体识别和语义连接构建 relation-free Tri-Graph，强调线性构建复杂度和零 LLM 构图 token，再通过
实体激活和全局重要性聚合取回 passage。它与 WARP-G 都关注效率，但 LinearRAG 优化的是**每个文档都进入图时，图本身
如何便宜地构建与检索**；WARP-G 优化的是**昂贵图后端只应在哪些 workload regions 上物化**。两者理论上也可组合：
将 WARP-G 的 regional backend 从 HippoRAG2 换成 LinearRAG。主表跑官方 `index` + `qa`，不融 WARP Base。

### LightRAG（EMNLP 2025）— 有官方代码，已接入

- 论文：[LightRAG](https://arxiv.org/abs/2410.05779)
- 官方代码：[HKUDS/LightRAG](https://github.com/HKUDS/LightRAG)
- 本仓库锁定 commit：`d49112fb7548ee14cb727d43bd68e34da0a2c942`

LightRAG 建立实体—关系图，并提供 local/global/hybrid/mix 多粒度检索。它的主要目标是轻量、增量、通用的完整
GraphRAG 系统。主表使用官方 SDK 的 `hybrid` 模式（图通道混合）。构图单元和 LLM 与 WARP 不同，因此和其余系统
一起放到**实测构建 USD–质量**平面上比较，不强行映射到 region token-proxy 预算。

## 4. KET-RAG 与 G2ConS

### KET-RAG — 有官方代码，主实验跑作者仓库

- 论文：[KET-RAG](https://arxiv.org/abs/2502.09304)
- 官方代码：[waetr/KET-RAG](https://github.com/waetr/KET-RAG)

原方法通过 chunk KNN 图和 PageRank 取 ceil(β·|V|) 个核心 chunks 建 KG skeleton，再构建全库
text-keyword bipartite graph。主表 β=0.8（论文默认，按篇数）。必须跑锁定的作者仓库和官方检索/生成，
禁止把骨架换成 HippoRAG2、禁止先扣轻量 token 再把剩余预算当核心 KG 配额。仓库里若仍保留旧的 matched-backend
适配，只可作附录诊断，不得写成主结果。

### G2ConS — 未找到官方代码，按论文复现

- 论文：[G2ConS](https://arxiv.org/abs/2510.24120)
- 官方代码：**截至 2026-08-23 未找到**

G2ConS 以 sentence-level concepts 建语义过滤共现图，用 Dice / PageRank 按 κ 篇数比例选核心 chunks
（论文默认 κ=0.8，双路融合 λ=0.6），再以 concept graph 与 core-KG 并行检索。论文中的
core-KG 由 MS-GraphRAG 构建；本仓库用 HippoRAG2 作为可运行的核心 KG 后端。因此主表必须标为
paper-based reimplementation / G2ConS-style，并公开全部超参。核心集按 κ 截断，不按 WARP 的
剩余 token-proxy 截断。

## 5. 近期最相关的效率、路由与索引工作

### EA-GraphRAG / Use Graph When It Needs（2026）— 未找到官方代码

- 论文：[arXiv:2602.03578](https://arxiv.org/abs/2602.03578)
- 官方代码：**截至 2026-08-23 未找到**

它根据 query 的句法复杂度，在 dense RAG、GraphRAG 和二者融合之间在线路由。它与 WARP-G 最接近的地方是“图不应对
所有查询无条件启用”，但决策层次不同：EA-GraphRAG 是**per-query 在线检索路由**，WARP-G 是**per-region 离线物理
设计**。最有价值的后续消融，是在 WARP-G 已物化 regions 之上增加 query-level graph gate；这属于组合实验，不应冒充
EA-GraphRAG 官方复现。

### TIGRAG（2026）— 未找到官方代码

- 论文：[arXiv:2606.30093](https://arxiv.org/abs/2606.30093)
- 官方代码：**截至 2026-08-23 未找到**

TIGRAG 用 token co-occurrence graph 避免昂贵的 LLM KG 抽取，重点是用更廉价的图表示替换传统知识图。WARP-G 不改变
HippoRAG2 的知识抽取方式，而是选择哪些区域值得承担该成本；一个优化构图算子，一个优化物化范围。

### A2RAG（2026）— 未找到官方代码

- 论文：[arXiv:2601.21162](https://arxiv.org/abs/2601.21162)
- 官方代码：**截至 2026-08-23 未找到**

A2RAG 让 agent 根据证据充分性逐步升级检索强度，并把图信号映射回来源文本，重点是 online adaptive reasoning。
WARP-G 的选择在部署前完成，在线检索是确定性的 Base routing + 图融合，不使用 agent，也没有按题多轮升级成本。

### E²GraphRAG（2025）— 有作者代码

- 论文：[arXiv:2505.24226](https://arxiv.org/abs/2505.24226)
- 作者代码：[yibozhao624/e-2graphrag](https://github.com/yibozhao624/e-2graphrag)

E²GraphRAG 用顺序层次摘要树和轻量 NLP 实体图服务超长文档检索，降低层次结构和实体抽取成本。它优化的是单个超长
文档的多粒度索引结构；WARP-G 面向共享多文档 corpus 和重复 workload，学习 regional materialization policy。

### Towards Practical GraphRAG（2025）— 未找到官方代码

- 论文：[arXiv:2507.03226](https://arxiv.org/abs/2507.03226)
- 官方代码：**截至 2026-08-23 未找到**

该工作以 dependency parsing 代替大量 LLM 抽取，并融合 entity/chunk/relation embeddings 与图遍历，面向企业级可扩展
构图。与 TIGRAG/LinearRAG 类似，它首先降低全量图的单位构建成本；WARP-G 首先减少需要构建昂贵图的语料范围。

### Core-based Hierarchies for Efficient GraphRAG（KDD 2026）— 有作者代码 artifact

- 论文：[arXiv:2603.05207](https://arxiv.org/abs/2603.05207)
- 作者代码 artifact：[Zenodo DOI 10.5281/zenodo.20500254](https://doi.org/10.5281/zenodo.20500254)

该工作以确定性的 k-core hierarchy 替代稀疏 KG 上不稳定的 Leiden 社区，并做 token-budget-aware sampling，重点是全局
sensemaking 社区层次的稳定性和摘要成本。WARP-G 当前使用 seeded Leiden 划分 workload coaccess graph，目标是区域物化；
它提示了一个很有价值的 partition ablation，但任务和评测（global sensemaking vs evidence retrieval）并不相同。

### 最相似工作对照矩阵

下面的表不只按标题中的 `GraphRAG` 检索，而是按“选择性构图、workload、预算、收益学习、物化、路由”六个机制检查。
“部分重合”不表示论文解决了同一个问题，而是指出审稿人最可能用来质疑 WARP-G 新颖性的先验工作。

| 工作                                                         | 主要决策对象                             |                          使用历史 workload |           显式构图预算 |       学习边际收益 | 与 WARP-G 最相似处                                           | 仍然缺少的 WARP-G 环节                                       |
| ------------------------------------------------------------ | ---------------------------------------- | -----------------------------------------: | ---------------------: | -----------------: | ------------------------------------------------------------ | ------------------------------------------------------------ |
| [KET-RAG](https://arxiv.org/abs/2502.09304)                  | 哪些 core chunks 进入 KG skeleton        |                                         否 |                     是 |                 否 | 在有限成本下只为部分 corpus 构建昂贵 KG                      | 不按 workload 共访问分区，不 probe 区域真实收益，也不对 held-out workload 学习物化策略 |
| [G2ConS](https://arxiv.org/abs/2510.24120)                   | 哪些高中心性 chunks 进入 core KG         |                                         否 |                     是 |                 否 | cost-constrained partial GraphRAG construction，是当前最直接的 GraphRAG 近邻 | 依据 concept graph/PageRank 的 corpus importance，而不是历史需求、区域收益和收益成本比 |
| [EA-GraphRAG](https://arxiv.org/abs/2602.03578)              | 每个 query 使用 Dense、Graph 还是 Fusion |                                         否 |             查询时成本 |     是，复杂度分数 | 同样认为图不应无条件服务所有问题                             | 做 per-query online routing，不决定哪些 corpus regions 需要离线构图 |
| [RAGRouter-Bench](https://arxiv.org/abs/2602.00296)          | 每个 query 选择哪种 RAG 范式             |                                         否 |           评测资源成本 |      路由模型/规则 | 研究 query-corpus compatibility 和效果—效率折中              | 是 benchmark 与 query router，不做 GraphRAG physical design  |
| [When Should Active RAG Retrieve?](https://arxiv.org/abs/2607.24010) | 每个 query/生成步骤是否检索              |                       校准集可视为过去数据 |                     是 | 是，检索边际正确性 | 用 held-out budget frontier、realized usage 和 harm audit 评估预算策略 | 预算花在在线 evidence usage，不是离线 regional graph construction |
| [SubQRAG](https://arxiv.org/abs/2510.07718)                  | 当前 sub-question 是否需要补充图事实     |                                         否 | 未形成离线全局物化预算 | 基于在线充分性判断 | 避免一开始就假定静态 KG 完整，按需要补图                     | 在 query time 动态抽取三元组，不学习可复用的 workload-level regional layout |
| [QCG-RAG](https://arxiv.org/abs/2509.21237)                  | 如何用 query nodes 构建检索图            | 否；使用从 chunks 生成的 synthetic queries |                     否 |                 否 | 名称和结构上最容易与 workload-driven query graph 混淆        | synthetic Doc2Query nodes 不是观察到的历史 workload，也没有预算化区域物化 |
| [GRiever](https://aclanthology.org/2025.emnlp-industry.174/) | 如何低延迟执行多跳图检索                 |                                         否 |       以运行资源为目标 |                 否 | 强调低资源 graph-based retriever 和 partial-triple retrieval | 假定 passage/triple indices 已存在，不选择昂贵 KG 的物化范围 |
| [LogicRAG](https://ojs.aaai.org/index.php/AAAI/article/view/40278) | query time 构建怎样的逻辑 DAG            |                                         否 |           隐式在线成本 |                 否 | 直接质疑预构建全局图的必要性                                 | 每题临时构建推理结构，属于 no-prebuilt-graph 路线，不做 workload amortization |

从 GraphRAG 文献本身看，KET-RAG 与 G2ConS 已经覆盖了“昂贵 KG 不必覆盖全语料”；EA-GraphRAG 与 Active RAG 已经覆盖
了“根据问题和预算分配检索资源”。因此 WARP-G 不能宣称首次提出 selective、adaptive、budget-aware 或 cost-efficient
GraphRAG。仍未找到被上述工作覆盖的，是以下完整链路：

```text
observed design workload
  -> document coaccess regions
  -> selective real GraphRAG probes
  -> learned regional marginal utility
  -> global-budget regional materialization
  -> Base-routed held-out workload evaluation
```

### 数据库物理设计与图分区先验

如果只检索 GraphRAG，会高估 WARP-G 的算法原创性。数据库、RDF 和 learned index 已长期研究根据过去 query workload
选择分区、索引和物化视图；它们不是直接 GraphRAG baseline，但必须在论文相关工作中承认。

| 工作                                                         | 原问题                                                       | 与 WARP-G 的共同抽象                                  | 与 WARP-G 的根本区别                                         | 对 novelty claim 的影响                                      |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ----------------------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| [Query Workload-based RDF Graph Fragmentation and Allocation](https://arxiv.org/abs/1508.07845) | 根据 SPARQL workload 将 RDF 图切分并分配到机器               | query coaccess/frequent patterns 影响 graph partition | 输入已经是结构化 RDF，目标是减少 crossing matches 和通信，不衡量 QA graph gain | workload-aware graph partition 不是新概念                    |
| [WawPart](https://arxiv.org/abs/2203.14888)                  | 按 workload 划分大型 KG，减少 distributed joins              | 从查询集合提取关键访问特征，再聚类 query 和 graph     | 不决定哪些原始文档值得进行昂贵 LLM KG extraction             | 不能宣称 first workload-aware graph system                   |
| [WISK](https://arxiv.org/abs/2302.14287)                     | 为 spatial-keyword queries 学习 workload-aware index         | 使用已知 query distribution 学习数据分区和索引结构    | 优化空间关键词查询成本，不预测 GraphRAG 对 evidence retrieval 的边际质量收益 | “workload + learned partition/index”本身不是新算法框架       |
| [Dynamic Materialized View Management using GNN](https://dbgroup.cs.tsinghua.edu.cn/ligl/papers/dynamic-view-icde23.pdf) | 对动态 SQL workload 预测 view benefit，并在空间预算下维护 MVs | 从 workload 学习候选物化对象的 benefit，再受预算选择  | benefit 是查询执行时间下降；候选是 SQL views，而不是必须真实构图才能测量的 regional KGs | “learned benefit + budgeted materialization”已有直接先验     |
| [Workload-Aware Materialization of Junction Trees](https://arxiv.org/abs/2110.03475) | 为概率查询选择 junction-tree 物化结果                        | 根据 workload 选择可复用结构以加速未来查询            | 面向 Bayesian inference，并给出专用优化算法和近似分析        | workload-aware materialization 术语及总体目标不能声称首次提出 |
| [Materialized View Selection for Regular Path Queries](https://doi.org/10.1145/3654955) | 在存储预算下为 graph RPQ workload 选择共享子查询视图         | 频率、收益、物化成本和预算共同决定选择                | 优化已有图上的 path-query execution，不构建文本 GraphRAG，也不优化答案/evidence 质量 | WARP-G 的选择问题属于已有 physical-design 家族               |

这些工作说明，WARP-G 的单个组成部分——Leiden、LightGBM、probe sampling、greedy benefit/cost ranking——都不应独立
包装为算法首创。合理的新颖性来自把数据库 physical-design 视角引入昂贵 GraphRAG indexing，并解决该场景特有的
区域效用观测、图后端构建、跨区域损失、查询路由和 QA 质量—成本联合评测问题。

### 新颖性结论与安全声明

截至 2026-08-23，没有检索到同时实现“真实历史 QA workload、原始文档共访问分区、少量真实 GraphRAG probes、区域
边际收益学习、全局预算区域图物化、held-out 查询路由评测”的公开工作。这支持的是**问题定义与系统闭环的新颖性**，
而不是每个优化组件的新颖性。

论文可谨慎写：

> To our knowledge, WARP-G is the first framework to study workload-aware regional physical design for expensive
> GraphRAG indexing: it estimates regional marginal graph utility from selective probes and materializes regional
> graph indices under a global construction budget.

论文不应写成 `first cost-efficient GraphRAG`、`first selective graph construction`、`first adaptive/budget-aware RAG`、
`first workload-aware graph partitioning` 或 `first learned materialization method`。正式投稿前还应按投稿截止日期重新检索，
因为 2026 年 GraphRAG 与 Active RAG 文献更新很快。

## 6. 其他已讨论的公开 GraphRAG 系统

### Microsoft GraphRAG — 有官方代码

- 项目与代码：[microsoft/graphrag](https://github.com/microsoft/graphrag)

Microsoft GraphRAG 从文本抽取实体/关系，构建社区和社区摘要，支持 local/global/DRIFT 等查询，面向 corpus-wide global
sensemaking。WARP-G 当前用的是 HippoRAG2 后端和 QA evidence retrieval，不使用社区摘要；主表中的 HippoRAG2 是官方全图通路，不能写成 Microsoft GraphRAG。

### LazyGraphRAG — 没有公开官方实现

- 项目背景：[Microsoft GraphRAG research](https://www.microsoft.com/en-us/research/project/graphrag/)
- 公开代码状态：**Microsoft 明确称其为 internal-only experimental fork；没有独立公开官方实现**

LazyGraphRAG 将较重的分析推迟到查询期，并结合 vector/graph search，降低前置索引成本。WARP-G 仍在查询前物化选定
区域，换取重复 workload 下稳定低延迟；二者分别代表 deferred computation 和 workload-amortized materialization。

### FastGraphRAG（Circlemind）— 有维护方代码，但不是同名论文官方复现

- 项目：[circlemind-ai/fast-graphrag](https://github.com/circlemind-ai/fast-graphrag)

这是一个独立开源产品/框架，使用实体图、personalized PageRank、增量更新和 agent-oriented retrieval，不能与 Microsoft
GraphRAG 的 `fast` indexing 配置混为一谈。它没有与名称一一对应的同行评审论文，因此更适合作为工程系统对照，而非
严格 paper baseline。WARP-G 的主张是预算化物理设计，不是通用 GraphRAG SDK。

### LeanRAG（AAAI 2026）— 有官方代码

- 论文：[arXiv:2508.10391](https://arxiv.org/abs/2508.10391)
- 官方代码：[KnowledgeXLab/LeanRAG](https://github.com/KnowledgeXLab/LeanRAG)

LeanRAG 通过语义聚合和层次检索组织 KG，目标是用更紧凑的上下文提升答案质量和效率。它优化图内知识组织及 online
retrieval；WARP-G 决定图在语料空间中的部署范围，且依赖真实 workload frequency/gain。

### PolyG（2025）— 有官方代码

- 论文：[arXiv:2504.02112](https://arxiv.org/abs/2504.02112)
- 官方代码：[Liu-rj/PolyG](https://github.com/Liu-rj/PolyG)

PolyG 先分类问题类型，再为不同类型选择图遍历策略，本质是 GraphRAG query planner。与 EA-GraphRAG 类似，它做在线
per-query traversal planning；WARP-G 做离线 per-region construction planning。

### RAPTOR（ICLR 2024）— 有官方代码

- 论文：[arXiv:2401.18059](https://arxiv.org/abs/2401.18059)
- 官方代码：[parthsarthi03/raptor](https://github.com/parthsarthi03/raptor)

RAPTOR 递归聚类并摘要 chunks，形成从局部文本到全局摘要的树。它不是知识图谱区域物化方法；构建成本主要来自递归
摘要，检索单位是多层树节点。WARP-G 的 regions 是 workload coaccess communities，且输出仍是原始 evidence documents。

### KGP（AAAI 2024）— 有官方代码

- 论文：[arXiv:2308.11730](https://arxiv.org/abs/2308.11730)
- 官方代码：[YuWVandy/KG-LLM-MDQA](https://github.com/YuWVandy/KG-LLM-MDQA)

KGP 为每个多文档 QA 实例构图，并让训练后的 traversal agent 在 KG 上形成 prompt。WARP-G 面向一个被大量查询共享的
corpus，学习可复用的物理布局；不存在 per-question KG，也不训练 traversal LLM。

### DALK（Findings of EMNLP 2024）— 有官方代码

- 论文：[arXiv:2405.04819](https://arxiv.org/abs/2405.04819)
- 官方代码：[David-Li0406/DALK](https://github.com/David-Li0406/DALK)

DALK 面向阿尔茨海默病文献，使 LLM 与领域 KG 动态共增强。它是领域专用、强调 KG/LLM 交互推理；WARP-G 是通用的
图索引物理设计层，当前四个数据集也不是该生物医学设置。

### PathRAG（2025）— 有官方代码

- 论文：[arXiv:2502.14902](https://arxiv.org/abs/2502.14902)
- 官方代码：[BUPT-GAMMA/PathRAG](https://github.com/BUPT-GAMMA/PathRAG)

PathRAG 通过 flow-based relational-path pruning 减少图检索上下文噪声和 token。它假定图已经存在，优化 query-time
subgraph/path extraction；WARP-G 优化图是否先被构建，两者作用在不同生命周期阶段。

### HiRAG（Findings of EMNLP 2025）— 有官方代码

- 论文：[arXiv:2503.10150](https://arxiv.org/abs/2503.10150)
- 官方代码：[hhy-huang/HiRAG](https://github.com/hhy-huang/HiRAG)

这里指 “Retrieval-Augmented Generation with Hierarchical Knowledge”，不是其他同名 RAG。HiRAG 同时组织知识图和
文本层次，支持局部/全局信息检索；WARP-G 不构建新的知识层次，而是在 workload regions 上部署 HippoRAG2。

### ArchRAG（AAAI 2026）— 有作者代码

- 论文：[arXiv:2502.09891](https://arxiv.org/abs/2502.09891)
- 作者代码：[sam234990/ArchRAG](https://github.com/sam234990/ArchRAG)

ArchRAG 用 attributed communities、LLM-based hierarchical clustering 和分层 ANN 索引提升检索并降低在线 token。
它仍建立 corpus-wide hierarchical community index；WARP-G 的 region 是物化和预算单位，不生成 community summaries。

### RoG（ICLR 2024）— 有官方代码

- 论文：[arXiv:2310.01061](https://arxiv.org/abs/2310.01061)
- 官方代码：[RManLuo/reasoning-on-graphs](https://github.com/RManLuo/reasoning-on-graphs)

RoG 先生成 relation-path plan，再从已有结构化 KG 取回匹配路径并推理，通常涉及模型训练。WARP-G 从非结构化文本构建
HippoRAG2 索引，在线不生成 relation plan；它们的数据前提和研究问题均不同。

### Think-on-Graph / ToG（ICLR 2024）— 有官方代码

- 论文：[arXiv:2307.07697](https://arxiv.org/abs/2307.07697)
- 官方代码：[IDEA-FinAI/ToG](https://github.com/IDEA-FinAI/ToG)

ToG 让 LLM agent 在 Freebase/Wikidata 等已有 KG 上执行多轮 beam search。WARP-G 的图来自输入 corpus，在线路径固定且
无 agent 循环；ToG 的主要成本在 per-query reasoning，WARP-G 的主要受控成本在 offline graph construction。

### Graph-CoT（Findings of ACL 2024）— 有官方代码

- 论文：[arXiv:2404.07103](https://arxiv.org/abs/2404.07103)
- 官方代码：[PeterGriffinJin/Graph-CoT](https://github.com/PeterGriffinJin/Graph-CoT)

Graph-CoT 让 LLM 在外部 textual graph 上反复执行 reasoning、interaction 和 execution。它是 agentic graph reasoning
框架；WARP-G 是非 agentic graph-index design，正式评测不允许测试题影响已物化布局。

### G-Retriever（NeurIPS 2024）— 有官方代码

- 论文：[NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/efaf1c9726648c8ba363a5c927440529-Abstract-Conference.html)
- 官方代码：[XiaoxinHe/G-Retriever](https://github.com/XiaoxinHe/G-Retriever)

G-Retriever 面向已有 textual graph，用 Prize-Collecting Steiner Tree 取回相关子图，并可通过 soft prompting 训练。
WARP-G 的输入是文档 corpus，其问题是选择构建哪些 regional KGs；它不解决任意输入图的子图优化。

## 7. 哪些方法适合放进哪张实验表

| 实验表 | 应放方法 | 原因 |
|---|---|---|
| 主表（成本–质量） | Base、HippoRAG2、KET-RAG、G2ConS-style、LinearRAG、LightRAG、WARP-G | 同一语料与 1,000 题；各按作者方法运行；横轴实测构建 USD，纵轴 CE@10 / EM。见 `design.md`。 |
| WARP 内部消融 | 四个选择器、分区变体、特征/probe、路由诊断、first-run | 只拆 WARP，不和 KET/G²/Linear/Light 画在同一条曲线上。 |
| 附录参考 | BM25、Dense、Base+Full Graph、旧 matched-backend KET/G² | 诊断用；不是 RQ3。 |
| 后续机制消融 | EA-GraphRAG-style query gate、k-core partition | 与 WARP 正交，不能写成官方复现。 |
| 相关工作、不进当前主表 | RAPTOR、KGP、DALK、RoG、ToG、Graph-CoT、G-Retriever 等 | 输入图、任务、训练或 online agent 与当前 shared-corpus QA 协议差异太大。 |

## 8. 官方对照运行方式

先下载已锁定的作者仓库：

```bash
python scripts/prepare_official_baselines.py
```

LinearRAG 和 LightRAG 的依赖跨度较大，应按各自官方 `requirements.txt` / `pyproject.toml` 建立独立环境；两边都需要
`OPENAI_API_KEY`。使用兼容代理时应同时设置 `OPENAI_BASE_URL`（LinearRAG 官方实现读取）和
`OPENAI_API_BASE`（LightRAG 官方实现读取）。LinearRAG 还需安装官方指定的 spaCy 模型。KET-RAG 同样使用独立
环境与锁定 commit（写入 `configs/official_baselines.yaml` 后冻结）。准备好后：

```bash
python scripts/run_official_baseline.py \
  --method linearrag \
  --config configs/paper/hotpotqa.yaml \
  --output outputs/official/hotpotqa-linearrag.json

python scripts/run_official_baseline.py \
  --method lightrag \
  --config configs/paper/hotpotqa.yaml \
  --output outputs/official/hotpotqa-lightrag.json
```

KET-RAG 官方入口接入后走同一套输出约定。四个数据集：

```bash
python scripts/run_official_suite.py
```

G2ConS 没有官方仓库，走本仓库论文复现，不经过上述 official 入口。

每份 JSON 保存官方仓库 URL/commit（或 G2 复现超参）、输入文件 SHA-256、文档和问题数量、多维构建/查询成本、
逐题检索或答案、标准化 EM/F1。该入口只运行正式的完整 1,000-query 实验，没有 smoke test、mock backend 或失败
降级路径。成本与质量如何对齐到主表，见 `design.md` 第 4 节。
