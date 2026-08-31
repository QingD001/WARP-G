# WARP-G：研究动机、问题定义、创新点与 Contributions

更新日期：2026-08-23。本文档定义 WARP-G 当前代码实际实现和实验能够支持的论文主张。它不是对未来功能的设想；如果
代码、数据划分或评测协议变化，本文档也应同步更新。

## 1. 一句话定位

WARP-G 将昂贵 GraphRAG 索引的部署范围视为一个 **workload-aware regional physical-design problem**：它根据历史
design workload 将共享语料划分为共同访问区域，用少量真实 GraphRAG probes 学习区域构图收益，并在全局构建预算下
只物化预期价值最高的区域图。

WARP-G 不发明新的 KG extraction 或 graph retrieval backend。当前图能力来自锁定版本的官方 HippoRAG2；WARP-G
负责决定昂贵图能力应该部署在哪里，以及如何在新查询到来时访问已经部署的区域图。

## 2. Motivation：为什么需要这个工作

### 2.1 Full GraphRAG 的前置成本与实际需求不匹配

传统 full-corpus GraphRAG 在看到真实查询之前，就对整个语料执行实体/关系抽取、embedding、索引和图存储。该成本随
corpus 扩展，但线上 workload 通常并不均匀：热点主题被反复访问，许多冷门文档很少被查询；有些区域的多跳关系能明显
改善证据召回，另一些区域仅靠 BM25 + Dense 已足够，构图甚至可能引入噪声。

因此，“所有文档都使用同等昂贵的图索引”隐含了两个未经验证的假设：

1. 不同语料区域具有相近的查询需求；
2. GraphRAG 相对强 Base Retriever 的边际收益在不同区域相近且总为正。

WARP-G 的核心 motivation 是直接检验并利用这两个假设的失效。

### 2.2 只按 corpus importance 选择文档仍然忽略真实需求

KET-RAG、G2ConS 等方法已经说明，可以根据 chunk/concept centrality 只构建一个核心 KG，再用廉价结构维持全语料覆盖。
但 corpus 中“结构上重要”的内容不一定是当前部署 workload 经常访问或最需要图推理的内容。一个低中心性但高频、Base
经常失败的主题，可能比一个高 PageRank 但几乎无人查询的主题更值得承担构图成本。

所以 WARP-G 不是再提出一种 corpus-only salience score，而是把 observed workload demand、GraphRAG marginal gain
和 construction cost 同时纳入物化决策。

### 2.3 区域价值不能在不构图时直接观察

某个区域是否值得构图，严格来说只有在该区域真实构建 GraphRAG、运行相关查询并和 Base 比较后才能知道。对所有区域都
这样测量，会先花掉接近 Full Graph 的成本，使选择策略失去意义。因此需要少量 probe regions、廉价构图前特征和一个
能够外推到未 probe regions 的 benefit predictor。

### 2.4 成本节省必须在 held-out workload 上成立

如果使用同一批 query 同时设计区域和报告效果，策略可能只是在记忆 benchmark。WARP-G 使用五折 cross-fitting，每折
只允许 800 条 design queries 影响 partition、probe、predictor 和 selection，另 200 条 queries 只用于测试；最终每条
query 恰好作为一次 held-out test。这样测量的是历史 workload 对未来同分布需求的迁移能力，而不是训练集拟合程度。

## 3. Problem：问题是什么

### 3.1 输入、输出和约束

给定：

- 共享文档语料 `D`；
- 历史监督式 design workload `Q_design`，包含 query 和 gold evidence；
- 不使用 gold evidence 的 held-out workload `Q_test`；
- 覆盖整个语料的共享 Base Retriever `H`；
- 固定 GraphRAG backend `G`，当前为 HippoRAG2；
- 全局离线图构建预算 `B`；
- 检索效用指标 `M`，主指标为 `CompleteEvidence@10`。

design workload 与语义近邻共同诱导文档分区：

```text
R = {r_1, r_2, ..., r_n},
r_i ∩ r_j = ∅,
union(R) = D.
```

为区域 `r_i` 构建图索引的估算成本记为 `c_i`。系统需要选择区域集合 `S ⊆ R`：

```text
Σ(r_i ∈ S) c_i ≤ B.
```

部署后，test query 先经过 Base Retriever；只有被 Base 命中的、且属于 `S` 的区域才调用 regional graph。最终目标是最大化
held-out workload 的期望检索效用：

```text
S* = argmax_{S ⊆ R} E_{q ~ Q_future}[M(q; H, G_S)]
     subject to Σ(r_i ∈ S) c_i ≤ B.
```

这里的 `Q_future` 用每折的 `Q_test` 近似。测试 query 不允许参与 partition、feature statistics、probe labels、模型拟合、
预算选择或参数调整。

### 3.2 当前方法使用的可部署近似

一般集合效用 `M(q; H, G_S)` 可能包含区域间互补或冲突，精确求解并不现实。当前 WARP-G 先估计每个区域相对 Base 的
独立边际收益 `g_i`，再计算：

```text
score_i = query_frequency_i × max(predicted_gain_i, 0) / estimated_graph_cost_i.
```

区域按 score 贪心选择，累计成本不得超过 `B`。这是一种可部署 heuristic，不是一般非加性集合效用问题的最优算法。
代码通过 probe-region pair interaction 测量独立收益假设偏离程度；若交互显著，论文必须把该算法描述为近似策略。

### 3.3 研究范围

当前问题被严格界定为：

- supervised workload-aware physical design；
- 静态共享 corpus 和重复/同分布 workload；
- 固定的一种 GraphRAG representation 在哪些区域物化；
- 离线 construction budget 与线上 retrieval cost 分别记账；
- 以 evidence retrieval 和固定 reader 的 answer EM/F1 为结果。

当前不解决：完全无标注日志、快速 workload drift、动态文档更新、跨 backend 自动选择、在线 agent planning、每题临时
KG extraction、图结构本身的联合学习，以及具有最优性保证的一般组合优化。

## 4. Research gap：已有方法缺少什么

| 既有方向 | 已经解决 | 没有解决的 WARP-G 问题 |
|---|---|---|
| Full GraphRAG / HippoRAG2 | 如何为完整 corpus 构图和执行多跳图检索 | 没有决定哪些区域值得承担构图成本 |
| KET-RAG / G2ConS | 根据 corpus centrality 低成本构建 partial/core KG | 不利用 observed workload，也不通过真实区域 probe 学习 QA 边际收益 |
| EA-GraphRAG / Active RAG routing | 每个 query 是否需要图或检索 | 不改变已经付出的离线全图 construction cost |
| Dynamic/on-demand GraphRAG | 查询时按需要补图或构建推理结构 | 不学习可由未来 workload 复用的 regional physical layout |
| Workload-aware graph partition | 根据结构化查询降低已有 KG 的 join/通信成本 | 不从原始文本选择性构建昂贵 GraphRAG，也不优化 evidence/answer quality |
| Learned indexes/materialized views | 学习 benefit 并在空间预算下选择物化对象 | 没有 GraphRAG 特有的昂贵效用观测、regional KG、跨区域事实和 Base routing 问题 |

WARP-G 位于这些方向的交叉点。论文的研究缺口不应写成“过去没人研究高效 GraphRAG”，而应写成：过去的高效 GraphRAG
主要优化图表示或按 corpus salience 选择内容，adaptive RAG 主要分配在线检索资源，数据库 physical design 则不衡量
文本 GraphRAG 对 QA 的边际质量收益；尚缺少针对昂贵 GraphRAG indexing 的 workload-aware regional physical design。

## 5. Innovation：当前工作的创新点

| 创新点 | 当前实现 | 新颖性来自哪里 | 已知组成，不能单独声称首创 |
|---|---|---|---|
| GraphRAG physical-design formulation | 将 regional KG 视为预算约束下的可选物理结构 | 把 workload demand、QA graph utility 和 construction budget 放进同一 GraphRAG 部署问题 | workload-aware index/view selection 的总体思想 |
| Workload-aware regionalization | 用 Base top-k query coaccess 与 semantic kNN 建图，再用 Leiden 形成物化单元 | region 反映“未来问题可能共同需要哪些文档”，不是只反映 corpus 语义社区 | coaccess graph、semantic kNN、Leiden |
| Probe-and-predict graph utility | 分层选择少量区域，真实构建 HippoRAG2，测量相对 Base 的 CompleteEvidence gain，再用廉价特征预测其他区域 | GraphRAG benefit 是昂贵且不可直接观测的，通过少量真实部署样本估计 | supervised regression、LightGBM、feature importance |
| Demand–gain–cost joint selection | 用 frequency、正边际收益和 graph cost 共同排序 | 明确区分“经常被访问”“需要图”“构图便宜”三个条件 | greedy benefit/cost ranking、knapsack heuristic |
| Regional materialization and routing | 每个选中 region 建隔离的官方 HippoRAG2 index；查询只访问 Base 命中的已部署区域 | 把离线布局决策落实为真实可查询的 GraphRAG artifact，并保持全语料 Base 覆盖 | Base routing、RRF、CrossEncoder |
| Cross-fitted quality–cost evaluation | 每折完整重做 physical design，合并 1,000 条 held-out 结果，并分账 deployment/design/online cost | 防止测试 query 参与布局，同时测量 first-run 和 amortized deployment trade-off | cross-validation、paired bootstrap/randomization |

最核心的创新是前四项构成的闭环。单独换用 LightGBM、Leiden 或 HippoRAG2 不构成论文贡献；如果实验不能证明联合策略优于
Frequency-only、Gain-only、Cost-only、KET-RAG 和 G2ConS，则“workload-aware learned physical design”的核心论点不成立。

## 6. Contributions：论文可以主张的贡献

### 6.1 中文版本

1. **问题定义。** 提出昂贵 GraphRAG 索引的 workload-aware regional physical-design 问题：在共享 Base Retriever
   和全局构建预算下，选择应当物化 GraphRAG 的语料区域，使未来 workload 的证据检索质量最大化。
2. **方法。** 提出 WARP-G，通过 query coaccess 形成区域，利用少量真实 GraphRAG probes 学习区域边际收益，并联合
   workload frequency、predicted gain 与 construction cost 选择和部署 regional graph indices。
3. **系统。** 在不重写图算法的前提下，把每个区域映射为隔离的官方 HippoRAG2 索引，并使用共享 Base routing、RRF 和
   CrossEncoder 将区域物化策略转化为可运行的 end-to-end GraphRAG 系统。
4. **评测。** 建立五折 cross-fitted、matched-budget 的评测协议，在四个数据集上与 KET-RAG、G2ConS、区域选择消融、
   零图 Base 和 Full Graph 比较，同时报告 deployment、design-search、first-run 和 online cost。
5. **经验发现。** 如果正式结果支持，则可报告 GraphRAG 的边际收益在语料区域间显著异质，以及 workload-aware selective
   materialization 能在较低实际构建成本下接近或超过 full/corpus-only alternatives。该条必须在结果出来后填写具体数值，
   不能在实验前作为既成事实。

### 6.2 可用于论文的英文版本

> 1. We formulate expensive GraphRAG indexing as a workload-aware regional physical-design problem: given a shared
>    base retriever and a global construction budget, the system selects corpus regions on which graph indices should
>    be materialized to maximize retrieval utility on future queries.
> 2. We propose WARP-G, which derives regions from query co-access patterns, estimates regional marginal graph utility
>    from a small number of real GraphRAG probes, and jointly accounts for workload frequency, predicted gain, and
>    construction cost when selecting regional materializations.
> 3. We implement WARP-G as a backend-preserving physical-design layer over the official HippoRAG2 implementation,
>    with deterministic base routing and isolated regional graph artifacts, enabling controlled comparison against
>    full-graph and corpus-centric selective-indexing approaches.
> 4. We introduce a cross-fitted, matched-budget evaluation protocol that separates design and test queries and reports
>    deployment, design-search, first-run, and online retrieval costs together with evidence and answer quality.

结果贡献应在正式实验完成后单独补充，例如：

> Across [datasets], WARP-G achieves [quality/cost result] and consistently outperforms [baselines] at matched realized
> construction budgets.

方括号内容不得在没有正式结果时预填。

## 7. 可使用的 novelty claim 与边界

截至 2026-08-23，`related_work.md` 的详细检索没有发现同时覆盖以下完整链路的公开工作：

```text
historical QA workload
  -> workload coaccess regions
  -> selective real GraphRAG probes
  -> regional marginal-utility prediction
  -> budgeted regional graph materialization
  -> Base-routed evaluation on held-out queries
```

因此可以使用带限定语的主张：

> To our knowledge, WARP-G is the first framework to study workload-aware regional physical design for expensive
> GraphRAG indexing, combining selective graph-utility probing, regional benefit prediction, and budget-constrained
> graph materialization for future workloads.

不能使用以下更宽泛的声明：

- first cost-efficient GraphRAG；
- first selective or partial GraphRAG construction；
- first adaptive/budget-aware RAG；
- first workload-aware graph partitioning；
- first learned index or materialized-view selector；
- provably optimal graph materialization。

WARP-G 的新颖性等级应描述为“新的 GraphRAG 问题定义、完整方法闭环和系统评测”，而不是“所有技术组件均为原创”。

## 8. 论文故事线

Introduction 可以按以下逻辑展开：

1. GraphRAG 能改善多跳证据检索，但 full-corpus KG extraction 带来高额、与查询无关的前置成本；
2. 真实 workload 在语料空间中不均匀，而且图相对强 Base 的收益也可能不均匀或为负；
3. 现有低成本 GraphRAG 主要按 corpus centrality 选择内容，online adaptive RAG 则无法收回已经支付的全图构建成本；
4. 因此关键问题不是再设计一个图检索器，而是决定**图应该在哪里被物化**；
5. WARP-G 用 workload coaccess regions、少量真实 probes、benefit prediction 和 budget selection 回答这个问题；
6. cross-fitted matched-budget 实验检验区域收益异质性、可预测性和实际 quality–cost 优势。

## 9. 使 contributions 成立所需的证据

| 主张 | 必须报告的证据 |
|---|---|
| 区域 graph gain 不均匀 | probe gain 分布、正/负收益比例、跨 fold/dataset 稳定性 |
| gain 可以预测 | leave-one-region-out MAE/RMSE/Spearman、learning curve、feature ablation |
| 三信号联合有效 | Random、Frequency-only、Gain-only、Cost-only 与 WARP 的同预算比较 |
| 比 corpus-only selective GraphRAG 更好 | matched-backend KET-RAG/G2ConS 的实际 cost–quality frontier |
| 节省不是记账假象 | actual deployment cost、design-search cost、first-run cost、online cost 和 quality-cost AUC |
| 没有 test leakage | 每折重新 partition/probe/train/select，test queries 只进入最终 evaluation |
| 区域化没有破坏关键关系 | Full Graph 对照、pair interaction、any/complete gold-region routing recall、partition ablation |
| 结果可推广 | HotpotQA、2WikiMultiHopQA、MuSiQue 和 single-hop PopQA control 的一致趋势 |

如果 WARP-G 不能稳定优于 Frequency-only 或 G2ConS，贡献应降级为“GraphRAG regional utility 的实证分析与负面结果”，
不能继续声称 learned workload-aware selection 有效。反之，如果这些证据成立，当前 formulation、system 和 evaluation 三层
贡献足以形成一篇边界清楚的 GraphRAG efficiency/physical-design 工作。
