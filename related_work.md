# WARP-G 相关工作与对比方法

更新日期：2026-08-26。这里的“官方代码”只指论文作者、作者团队或项目维护方公开的实现；第三方复现不算。
“未找到”表示截至该日期检查了论文页面、论文正文中的代码声明和公开 GitHub 搜索后，未找到可确认的作者仓库，
不等于作者将来不会发布。论文与代码仍可能继续更新，因此正式投稿前应再复核一次。

## 1. 本方法是什么

WARP-G（Workload-Aware Regional Physical Graph materialization）解决的是**离线预算下应当为哪些语料区域构建昂贵
GraphRAG 索引**的问题。它先用 design workload 构建文档共访问图并做 Leiden 分区，只构建少量 HippoRAG2 probe
graphs，学习每个 region 的预期检索收益，再按

```text
query_frequency × max(predicted_gain, 0) / estimated_graph_cost
```

选择区域。在线查询先经过共享 BM25 + Dense router，只访问已物化且与问题相关的区域图。正式协议在每个数据集的
1,000 条 released queries 上做确定性 5-fold cross-fitting；每折 800 条只用于物理设计，200 条只用于测试，每条问题
恰好被 held out 一次。

WARP-G 与大多数 GraphRAG 工作的根本区别不是“发明另一种 KG 抽取器”，而是把 GraphRAG 看成一个 workload-aware、
budgeted physical-design 问题。当前图能力来自锁定 commit 的官方 HippoRAG2，贡献点位于 region 形成、低成本 probe、
收益预测、预算选择和在线路由。

## 2. 当前代码中已经进入正式实验的方法

| 方法 | 在本仓库中的角色 | 实现来源 | 与 WARP-G 的关键区别 |
|---|---|---|---|
| BM25 | 无图参考线 | 本仓库实现 | 只利用词项匹配，不构图、无 workload 设计。 |
| Dense | 无图参考线 | HippoRAG2 passage encoder + 本仓库检索器 | 只利用向量相似度，不利用图传播。 |
| Hybrid | WARP-G 的共享 Base | 本仓库 BM25 + Dense + RRF | 覆盖全语料但没有 KG；也是零图预算点。 |
| HippoRAG2 graph-only | 全图参考线 | **官方代码**：[OSU-NLP-Group/HippoRAG](https://github.com/OSU-NLP-Group/HippoRAG) | 全语料构建一个 KG，只走图检索；不按 workload 选择。 |
| Base + Full Graph | 全图上界参考线 | 官方 HippoRAG2 + 本仓库融合/rerank | 全语料单图保留跨 region 边，不受预算约束。 |
| KET-RAG | 匹配预算的全局稀疏图 baseline | 本仓库适配性复现；论文有官方代码 | 先用全局 chunk KNN/PageRank 选 core chunks，再建 skeleton KG，并用轻量 keyword graph 补覆盖；不是 workload region 选择。 |
| G2ConS | 匹配预算的全局稀疏图 baseline | 本仓库适配性复现；**未找到官方代码** | 用 concept co-occurrence graph、Dice/PageRank 选 core chunks，并以 concept graph 补覆盖；不学习 region-level workload gain。 |
| Random-region | 新增选择器消融 | 本仓库实现 | 相同 regions、后端和预算，固定 seed 随机排序，检验 WARP-G 是否优于随机物化。 |
| Frequency-only | 新增选择器消融 | 本仓库实现 | 只按 design workload 中的 query frequency 排序，不看预测收益/成本比。 |
| Gain-only | 新增选择器消融 | 本仓库实现 | 只按预测增益排序，不做 workload frequency 与 cost normalization。 |
| Cost-only | 新增选择器消融 | 本仓库实现 | 从最便宜的 region 开始填预算，不看需求与质量收益。 |
| WARP-G | 主方法 | 本仓库实现 | 同时建模访问频率、预测收益和物化成本，并只部署预测为正收益的区域。 |

四个选择器消融已经加入四份 `configs/paper/*.yaml` 的六点预算曲线，并自动进入逐题配对显著性检验。它们复用
WARP-G 形成的同一组 regions，因而回答的是“选择公式是否有效”，不是“另一套 GraphRAG 后端是否更强”。

## 3. 两个新增的作者官方端到端对照

### LinearRAG（ICLR 2026）— 有官方代码，已接入

- 论文：[LinearRAG](https://arxiv.org/abs/2510.10114)
- 官方代码：[DEEP-PolyU/LinearRAG](https://github.com/DEEP-PolyU/LinearRAG)
- 本仓库锁定 commit：`bcc94e66c221f798801255efba09311d6fbcd8d6`

LinearRAG 用轻量实体识别和语义连接构建 relation-free Tri-Graph，强调线性构建复杂度和零 LLM 构图 token，再通过
实体激活和全局重要性聚合取回 passage。它与 WARP-G 都关注效率，但 LinearRAG 优化的是**每个文档都进入图时，图本身
如何便宜地构建与检索**；WARP-G 优化的是**昂贵图后端只应在哪些 workload regions 上物化**。两者理论上也可组合：
将 WARP-G 的 regional backend 从 HippoRAG2 换成 LinearRAG。

### LightRAG（EMNLP 2025）— 有官方代码，已接入

- 论文：[LightRAG](https://arxiv.org/abs/2410.05779)
- 官方代码：[HKUDS/LightRAG](https://github.com/HKUDS/LightRAG)
- 本仓库锁定 commit：`d49112fb7548ee14cb727d43bd68e34da0a2c942`

LightRAG 建立实体—关系图，并提供 local/global/hybrid/mix 多粒度检索。它的主要目标是轻量、增量、通用的完整
GraphRAG 系统；WARP-G 则把固定图后端的物化范围作为 workload-aware 预算决策。当前接入使用官方 SDK 的 `hybrid`
模式，在同一完整 corpus 和全部 1,000 条问题上报告答案 EM/F1、构建时间和查询时间。

这两个系统不会混入 WARP-G 的 `actual_cost_fraction` 曲线，因为它们的构图单元、embedding/LLM 调用和检索器都不同，
不存在诚实的一一对应 region budget。它们作为独立的 end-to-end 表更合理。

## 4. KET-RAG 与 G2ConS：论文方法、代码状态和当前适配差异

### KET-RAG — 有官方代码，但主实验使用匹配后端适配版

- 论文：[KET-RAG](https://arxiv.org/abs/2502.09304)
- 官方代码：[waetr/KET-RAG](https://github.com/waetr/KET-RAG)

原方法通过 chunk KNN 图和 PageRank 找到核心 chunks，仅为核心部分抽取 KG skeleton，再构建廉价的 text-keyword
bipartite graph 保持全局覆盖。当前仓库保留这些关键机制，但把核心 KG 统一换成与 WARP-G 相同的 HippoRAG2 builder，
并统一 Base、candidate depth、RRF、CrossEncoder、预算分母和成本记账。因此它是适合做**受控算法比较**的 matched-backend
适配，不是作者仓库逐行复现。论文中应写成 “KET-RAG-style, adapted to the shared HippoRAG2 backend”，同时可把作者
官方端到端结果放进补充材料，避免混称。

### G2ConS — 未找到官方代码，当前只能按论文复现

- 论文：[G2ConS](https://arxiv.org/abs/2510.24120)
- 官方代码：**截至 2026-08-23 未找到**

G2ConS 以 sentence-level concepts 构建语义过滤的共现图，用 Dice edge weight 和 concept PageRank 选 core chunks，
再以 concept graph expansion 弥补稀疏 KG 的覆盖损失。当前仓库按论文机制实现，并同样统一为 HippoRAG2 core KG 和共享
检索/rerank/成本口径。由于没有作者代码可对照，必须明确标为 paper-based reimplementation，并在论文中公开所有参数。

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

### CrossAug（arXiv 2026-05）— 有官方代码，2026-08 复核新增，与 WARP-G 最接近的新工作

- 论文：[arXiv:2605.28004](https://arxiv.org/abs/2605.28004)
- 官方代码：[DonFinliani/CrossAug](https://github.com/DonFinliani/CrossAug)

CrossAug 用自监督图损坏（mask fact edges / delete entity nodes）训练一个 topology-aware GNN，为每个局部子图打
“缺失度”分数，然后在显式 LLM 调用预算内只对高分、低重叠的子图做 evidence-grounded LLM 补全，补出的三元组持久化
回图索引，查询期零额外成本。它已经占据“learned scoring + budgeted selective LLM graph construction”这条线，因此
WARP-G 不能把“学习收益 + 预算构图”本身写成首创。但与 WARP-G 有四点根本区别：(1) 监督信号来自图拓扑的自监督
损坏，不是历史 workload 的 QA 收益标签；(2) 它补全的是已经全量建好的 base graph，不决定哪些语料区域值得建图；
(3) 没有 workload 共访问分区，候选子图按实体随机游走采样；(4) 评测是三个框架上的 EM/F1 提升（MuSiQue 上
HippoRAG2 +1.33 F1），不做 held-out cross-fitting 和成本分账曲线。两者理论上可组合：WARP-G 决定物化范围，
CrossAug 增强已物化区域的图质量。

### PropRAG（EMNLP 2025）— 有官方代码，2026-08 复核新增

- 论文：[arXiv:2504.18070](https://arxiv.org/abs/2504.18070)
- 官方代码：[ReLink-Inc/PropRAG](https://github.com/ReLink-Inc/PropRAG)

PropRAG 用自包含 proposition 替代三元组构图，离线 LLM 抽取一次，在线用无 LLM 的 beam search 沿 proposition path
检索。Recall@5：PopQA 55.3、2Wiki 93.7、HotpotQA 97.0、MuSiQue 77.3，索引成本约 $4。它代表“全语料轻量构图”路线，
与 WARP-G 互补但竞争：PropRAG 把全语料的单位构图成本压到极低，WARP-G 只给部分区域构昂贵图。若 PropRAG 的
MuSiQue 77.3 在共享协议下复现，会进一步压缩 Full HippoRAG2 的价值叙事，讨论部分应正视而不是回避。

### SAG（arXiv 2026-08）— 有 benchmark 代码，2026-08 复核新增

- 论文：[arXiv:2608.12129](https://arxiv.org/abs/2608.12129)
- 代码：[Zleap-AI/SAG-Benchmark](https://github.com/Zleap-AI/SAG-Benchmark)

SAG 完全不建离线 KG：把每个 chunk 做成“event + entity”索引，查询时把共享实体当作 SQL join key 动态连出
query-scoped 超边，证据始终是原始 chunk。MuSiQue Recall@5 80.36（超最强 baseline 11.5 分）、Recall@2 64.1
（HippoRAG2 为 49.5），声称已部署到 5 亿条生产数据。它是“无图”路线目前最强的公开结果，对 WARP-G 的 motivation
构成最直接质疑：如果无图方法在多跳证据上已超过 HippoRAG2 全图，WARP-G 需要论证的不是“区域图省成本”，而是
“区域图相对共享 Base 的增量仍然值得付”。论文 discussion 必须回应这条线。

### HCG-RAG（arXiv 2026-07）— 2026-08 复核新增

- 论文：[arXiv:2607.22592](https://arxiv.org/abs/2607.22592)

用固定 schema 的因果变量约束开放式抽取，构图 LLM 调用减少 8–135 倍、节点减少 3–20 倍。与 TIGRAG/LinearRAG 同类：
降低全量图的单位构建成本，不解决物化范围选择。

### MeshRAG（ACL 2026）— 2026-08 复核新增

- 论文：[ACL Anthology](https://aclanthology.org/2026.acl-long.1156/)

hash 碰撞构图，零 LLM token、零 GPU，10k+ chunk 分钟级成图。属于“彻底消除构图成本”路线；WARP-G 研究的是仍有
LLM 抽取的昂贵后端，两者成本结构不同，但若 MeshRAG 检索质量成立，同样压缩“昂贵图值得建”的空间。

### The Commercial Tax（arXiv 2026-08）— 元审计，2026-08 复核新增

- 论文：[arXiv:2608.16096](https://arxiv.org/abs/2608.16096)

审计发现 HippoRAG-2、PropRAG、SAG、KET-RAG 的最佳 recall 都依赖 NV-Embed-v2，且多数论文未披露其非商用 license。
本仓库 Base 与 Dense 也用 NV-Embed-v2：不影响学术对照，但正式论文应披露模型 license 与替代方案，避免同样的批评。

### The Reasoning Bottleneck in Graph-RAG（arXiv 2026-03）— 独立评测，2026-08 复核新增

- 论文：[arXiv:2603.14045](https://arxiv.org/abs/2603.14045)

对 KET-RAG 的独立评测：HotpotQA/MuSiQue/2Wiki 的 context coverage 77–91%，但 73–84% 的错误是 reasoning 失败而非
检索失败；结构化提示 + 图游走压缩让 8B 模型追平 70B baseline（约 12× 成本）。对 WARP-G 的两点意义：(1) 检索指标
与答案指标之间可能存在巨大 gap，固定 reader 口径（design.md 第 10 节）是对的选择；(2) KET-RAG 高 coverage 的独立
证据支持把它作为强 baseline。

### 最相似工作对照矩阵

下面的表不只按标题中的 `GraphRAG` 检索，而是按“选择性构图、workload、预算、收益学习、物化、路由”六个机制检查。
“部分重合”不表示论文解决了同一个问题，而是指出审稿人最可能用来质疑 WARP-G 新颖性的先验工作。

| 工作 | 主要决策对象 | 使用历史 workload | 显式构图预算 | 学习边际收益 | 与 WARP-G 最相似处 | 仍然缺少的 WARP-G 环节 |
|---|---|---:|---:|---:|---|---|
| [KET-RAG](https://arxiv.org/abs/2502.09304) | 哪些 core chunks 进入 KG skeleton | 否 | 是 | 否 | 在有限成本下只为部分 corpus 构建昂贵 KG | 不按 workload 共访问分区，不 probe 区域真实收益，也不对 held-out workload 学习物化策略 |
| [G2ConS](https://arxiv.org/abs/2510.24120) | 哪些高中心性 chunks 进入 core KG | 否 | 是 | 否 | cost-constrained partial GraphRAG construction，是当前最直接的 GraphRAG 近邻 | 依据 concept graph/PageRank 的 corpus importance，而不是历史需求、区域收益和收益成本比 |
| [EA-GraphRAG](https://arxiv.org/abs/2602.03578) | 每个 query 使用 Dense、Graph 还是 Fusion | 否 | 查询时成本 | 是，复杂度分数 | 同样认为图不应无条件服务所有问题 | 做 per-query online routing，不决定哪些 corpus regions 需要离线构图 |
| [RAGRouter-Bench](https://arxiv.org/abs/2602.00296) | 每个 query 选择哪种 RAG 范式 | 否 | 评测资源成本 | 路由模型/规则 | 研究 query-corpus compatibility 和效果—效率折中 | 是 benchmark 与 query router，不做 GraphRAG physical design |
| [When Should Active RAG Retrieve?](https://arxiv.org/abs/2607.24010) | 每个 query/生成步骤是否检索 | 校准集可视为过去数据 | 是 | 是，检索边际正确性 | 用 held-out budget frontier、realized usage 和 harm audit 评估预算策略 | 预算花在在线 evidence usage，不是离线 regional graph construction |
| [SubQRAG](https://arxiv.org/abs/2510.07718) | 当前 sub-question 是否需要补充图事实 | 否 | 未形成离线全局物化预算 | 基于在线充分性判断 | 避免一开始就假定静态 KG 完整，按需要补图 | 在 query time 动态抽取三元组，不学习可复用的 workload-level regional layout |
| [QCG-RAG](https://arxiv.org/abs/2509.21237) | 如何用 query nodes 构建检索图 | 否；使用从 chunks 生成的 synthetic queries | 否 | 否 | 名称和结构上最容易与 workload-driven query graph 混淆 | synthetic Doc2Query nodes 不是观察到的历史 workload，也没有预算化区域物化 |
| [GRiever](https://aclanthology.org/2025.emnlp-industry.174/) | 如何低延迟执行多跳图检索 | 否 | 以运行资源为目标 | 否 | 强调低资源 graph-based retriever 和 partial-triple retrieval | 假定 passage/triple indices 已存在，不选择昂贵 KG 的物化范围 |
| [LogicRAG](https://ojs.aaai.org/index.php/AAAI/article/view/40278) | query time 构建怎样的逻辑 DAG | 否 | 隐式在线成本 | 否 | 直接质疑预构建全局图的必要性 | 每题临时构建推理结构，属于 no-prebuilt-graph 路线，不做 workload amortization |
| [CrossAug](https://arxiv.org/abs/2605.28004) | 哪些子图值得 LLM 补全 | 否；自监督图损坏 | 是，LLM 调用预算 | 是，但学的是结构缺失性 | 学习打分 + 预算内选择性 LLM 图构建、离线静态索引 | 不学 workload/QA 增益，不决定区域物化，不做 probe 与 held-out cross-fitting |
| [SAG](https://arxiv.org/abs/2608.12129) | query-time event-entity 动态超边（不建离线图） | 否 | 无离线构图预算 | 否 | 质疑离线全图的必要性，无图路线在 MuSiQue 已超 HippoRAG2 | 不学习任何可复用物理布局，增益全部来自在线 join 表示 |
| [PropRAG](https://arxiv.org/abs/2504.18070) | 全语料 proposition 图表示 | 否 | 构图成本低 | 否 | 全语料轻量构图 + 无 LLM 在线检索 | 不做选择性物化，region 与 workload 信号均不进入构图决策 |

从 GraphRAG 文献本身看，KET-RAG 与 G2ConS 已经覆盖了“昂贵 KG 不必覆盖全语料”；EA-GraphRAG 与 Active RAG 已经覆盖
了“根据问题和预算分配检索资源”；2026-08 复核新增的 CrossAug 还覆盖了“学习打分 + 显式预算下选择性 LLM 图构建”，
但它的监督来自自监督图拓扑损坏、作用对象是已经全量建好的 base graph；SAG 与 PropRAG 则说明无图/轻量路线在
MuSiQue 上已很强（Recall@5 约 80/77），进一步压缩“全图值得建”的默认假设。因此 WARP-G 不能宣称首次提出 selective、
adaptive、budget-aware 或 cost-efficient GraphRAG，也不能宣称首次提出 learned benefit + budgeted graph construction。
仍未找到被上述工作覆盖的，是以下完整链路：

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

| 工作 | 原问题 | 与 WARP-G 的共同抽象 | 与 WARP-G 的根本区别 | 对 novelty claim 的影响 |
|---|---|---|---|---|
| [Query Workload-based RDF Graph Fragmentation and Allocation](https://arxiv.org/abs/1508.07845) | 根据 SPARQL workload 将 RDF 图切分并分配到机器 | query coaccess/frequent patterns 影响 graph partition | 输入已经是结构化 RDF，目标是减少 crossing matches 和通信，不衡量 QA graph gain | workload-aware graph partition 不是新概念 |
| [WawPart](https://arxiv.org/abs/2203.14888) | 按 workload 划分大型 KG，减少 distributed joins | 从查询集合提取关键访问特征，再聚类 query 和 graph | 不决定哪些原始文档值得进行昂贵 LLM KG extraction | 不能宣称 first workload-aware graph system |
| [WISK](https://arxiv.org/abs/2302.14287) | 为 spatial-keyword queries 学习 workload-aware index | 使用已知 query distribution 学习数据分区和索引结构 | 优化空间关键词查询成本，不预测 GraphRAG 对 evidence retrieval 的边际质量收益 | “workload + learned partition/index”本身不是新算法框架 |
| [Dynamic Materialized View Management using GNN](https://dbgroup.cs.tsinghua.edu.cn/ligl/papers/dynamic-view-icde23.pdf) | 对动态 SQL workload 预测 view benefit，并在空间预算下维护 MVs | 从 workload 学习候选物化对象的 benefit，再受预算选择 | benefit 是查询执行时间下降；候选是 SQL views，而不是必须真实构图才能测量的 regional KGs | “learned benefit + budgeted materialization”已有直接先验 |
| [Workload-Aware Materialization of Junction Trees](https://arxiv.org/abs/2110.03475) | 为概率查询选择 junction-tree 物化结果 | 根据 workload 选择可复用结构以加速未来查询 | 面向 Bayesian inference，并给出专用优化算法和近似分析 | workload-aware materialization 术语及总体目标不能声称首次提出 |
| [Materialized View Selection for Regular Path Queries](https://doi.org/10.1145/3654955) | 在存储预算下为 graph RPQ workload 选择共享子查询视图 | 频率、收益、物化成本和预算共同决定选择 | 优化已有图上的 path-query execution，不构建文本 GraphRAG，也不优化答案/evidence 质量 | WARP-G 的选择问题属于已有 physical-design 家族 |

这些工作说明，WARP-G 的单个组成部分——Leiden、LightGBM、probe sampling、greedy benefit/cost ranking——都不应独立
包装为算法首创。合理的新颖性来自把数据库 physical-design 视角引入昂贵 GraphRAG indexing，并解决该场景特有的
区域效用观测、图后端构建、跨区域损失、查询路由和 QA 质量—成本联合评测问题。

### 新颖性结论与安全声明

截至 2026-08-26 复核（新增 CrossAug、SAG、PropRAG、HCG-RAG、MeshRAG 及两篇评测/审计），没有检索到同时实现
“真实历史 QA workload、原始文档共访问分区、少量真实 GraphRAG probes、区域边际收益学习、全局预算区域图物化、
held-out 查询路由评测”的公开工作。这支持的是**问题定义与系统闭环的新颖性**，而不是每个优化组件的新颖性。
需注意 CrossAug 已实现“学习打分 + 预算内选择性图增强”：若审稿人以此质疑，WARP-G 的区分点必须落在
workload/QA-gain 监督信号、区域物化与 Base routing 上，而非“learned + budgeted”本身。

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
sensemaking。WARP-G 当前用的是 HippoRAG2 后端和 QA evidence retrieval，不使用社区摘要；Full Graph 只代表
HippoRAG2 全图，不能写成 Microsoft GraphRAG。

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
| 同后端 budget–quality 主表 | KET-RAG-style、G2ConS-style、四个选择器消融、WARP-G | 共享 corpus、HippoRAG2、Base、reranker、预算分母，可归因比较。 |
| 无图/全图参考表 | BM25、Dense、Hybrid、HippoRAG2 graph-only、Base + Full Graph | 给出零图底座和全量图边界。 |
| 独立官方端到端表 | LinearRAG、LightRAG | 官方实现真实可跑，但构建单元和成本定义不同；报告 EM/F1、wall time、模型/API 配置。 |
| 可选官方端到端候选（未接入） | PropRAG、SAG | 均有官方代码、成本低、MuSiQue 强；若时间允许可作为第三/四个官方对照，接入前需锁定 commit 并沿用现有 1,000-query 协议。 |
| 后续机制消融 | EA-GraphRAG-style query gate、k-core partition | 与 WARP-G 正交且能直接检验 routing/partition 假设，但不能写成官方复现。 |
| 相关工作、不宜直接塞入当前主表 | RAPTOR、KGP、DALK、RoG、ToG、Graph-CoT、G-Retriever 等 | 输入图、任务、训练或 online agent 成本与当前 shared-corpus QA 协议差异太大。 |

## 8. 官方端到端实验运行方式

先下载锁定的作者仓库：

```bash
python scripts/prepare_official_baselines.py
```

LinearRAG 和 LightRAG 的依赖跨度较大，应按各自官方 `requirements.txt` / `pyproject.toml` 建立独立环境；两边都需要
`OPENAI_API_KEY`。使用兼容代理时应同时设置 `OPENAI_BASE_URL`（LinearRAG 官方实现读取）和
`OPENAI_API_BASE`（LightRAG 官方实现读取）。LinearRAG 还需安装官方指定的 spaCy 模型。准备好环境后：

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

运行四个数据集的两个官方系统：

```bash
python scripts/run_official_suite.py
```

每份 JSON 保存官方仓库 URL/commit、输入文件 SHA-256、文档和问题数量、实际构建/查询 wall time、逐题 prediction、
标准化 EM/F1。该入口只运行正式的完整 1,000-query 实验，没有 smoke test、mock backend 或失败降级路径。
