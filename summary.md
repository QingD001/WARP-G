# WARP-G 研究目的与代码架构速览

> 面向熟悉图论与机器学习、但对 RAG/GraphRAG 领域较新的读者。目标是读完本文即可理解整个项目在做什么、
> 为什么这样做,以及代码各层分别承担什么职责。

## 一、领域背景

**RAG(检索增强生成)**解决一个朴素问题:LLM 记不住私有语料,所以回答问题前先做一步**检索**——从语料里
捞出相关文档,塞进 prompt 再生成答案。科研评测的标准口径是 **evidence retrieval** : 每个问题标注了
"gold evidence 文档"(正确答案所依赖的原始证据),检索的目标就是把这些文档捞进 top-k(指标:
Evidence Recall@k、Complete Evidence@k)。

**GraphRAG 是这条线的高级版**:普通检索只算"文档↔问题"的相似度(BM25 词匹配 / 向量余弦),但**多跳问题**
需要跨文档推理(如"这部电影的导演的父亲的职业"),单文档相似度捞不出证据链。GraphRAG 的思路是:离线把语料
抽成**知识图谱**(LLM 逐文档做 OpenIE,抽出 实体-关系-实体 三元组),检索时在图上游走(PPR 等),把推理结构
带出来。代表系统就是本项目锁定的 **HippoRAG2**(基于 PPR 的个性化图检索)。

**项目的出发点**:建图极贵——每个文档都要过 LLM 抽三元组,一个语料 ≈ 9-10M tokens;而且
**所有文档都建**意味着:冷门文档、单跳就能答的区域也在付同样的钱。而真实 query 的访问是高度不均匀的。

## 二、WARP-G 的研究目的

**"用历史 query 流学会'哪里值得',别给整个语料建图,只给值得的区域建图。"**

形式化地说:

$$S^* = \arg\max_{S \subseteq \mathcal{R}} \mathbb{E}_{q \sim Q_{\text{future}}}\big[ M(q; H, G_S) \big] \quad \text{s.t.} \sum_{r \in S} c_r \le B$$

- $\mathcal{R}$:把语料 $D$ 划分成的**区域**(物化单元);
- $G_S$:只在选中区域构建的 HippoRAG2 图;未选中的区域靠共享的 Base 检索(BM25+向量)兜底;
- $c_r$、$B$:构图成本与预算。

**困难在于 $M$ 不可观测**:一个区域的图收益,只有真建图、跑 query、和 Base 对比才知道——全测一遍就花掉了
等于全量建图的钱,选择就失去意义。所以方法是 **probe-and-predict**(探针+预测):

1. 分层抽少量区域(probe)真建图,测出真实增益标签 $y_r$;
2. 用 9 个**构图前廉价特征** $\phi(r)$(文档数、query 频率、Base recall、failure rate、熵、dispersion…)
   训练 LightGBM 预测 $\hat g_r$;
3. 按 `query频率 × relu(预测增益) / 估算成本` 贪心选区域,直到预算花完。

**五个研究问题**:收益是否真的区域异质(RQ1)、能否预测(RQ2)、三信号联合公式是否赢过所有单信号消融和两个已发表的稀疏构图 baseline KET-RAG/G2ConS(RQ3)、分区/路由的稳健性(RQ4)、省下的钱会不会被探测和学习成本吃掉(RQ5)。

**竞争力**:数据库领域做 workload-aware 物化视图选择是几十年的老问题(related_work.md 里就有GNN 视图选择的先例),但"昂贵 LLM 构图 + QA 质量收益"这个场景没人做过——收益对象从"查询延迟"变成了"检索质量",观测成本从免费变成了"建一次图就要花钱"。本项目的新颖性在**问题定义和闭环**,不在任何单个组件
(Leiden、LightGBM、贪心 knapsack 全是现成的)。

## 三、与图论 / 机器学习背景的对应关系

| 已有背景 | 在 WARP-G 里对应 |
|---|---|
| 图论/社区发现 | **共访问图** : 节点=文档，边=同被一条 query 的 top-k 命中的共现次数 $w_{query}$ + 语义余弦边 $\lambda w_{sem}$;Leiden 做社区划分 → regions。这就是 "workload-aware" 的载体:分区不是按语义,是按**查询习惯** |
| 有监督学习 | LightGBM 回归(9 维特征 → 区域增益),但标签**极贵**(建图+跑 query),所以样本只有 ~20% 区域——这是小样本元学习/实验设计场景 |
| 实验设计/主动学习 | probe 分层抽样：沿 workload 密度、failure、cost 三轴轮转桶抽样,保证训练集代表性 |
| 组合优化 | 0/1 knapsack 的贪心近似(按 score 排序填预算);区域间收益**非可加**(有交互),所以明确不声称最优,只声称可部署启发式 |
| 因果/统计推断 | 五折 cross-fitting(800 design / 200 held-out),test query 严格不进任何设计环节;配对 bootstrap CI、符号翻转 randomization test(10000 次)、Holm 校正 |
| 系统/会计 | 成本四账本分账(deployment / design-search / first-run / online),token 双轨记账(逻辑 vs 物理)——这是论文 RQ5 的证据基础 |

## 四、代码架构

4034 行,分八层,依赖方向是**单向向下**的(schema 最底层):

```text
warp/run.py ─────────────── 论文实验入口(读 YAML→五折→全部方法→统计→JSON artifact)
      │
warp/pipeline.py ────────── 核心编排器 WARPG:fit(设计阶段) / evaluate(测试阶段)
      │
├─ warp/data/      ── JSONL 加载 + 4 个 benchmark schema 适配 + 五折划分(sha256 轮询)
├─ warp/retrieval/ ── 共享 Base:BM25、Dense(NV-Embed-v2)、FAISS HNSW、RRF 融合、CrossEncoder 重排
├─ warp/partition/ ── 共访问图构建 + Leiden 分区                    ← 图论主场
├─ warp/advisor/   ── 特征提取 / probe / LightGBM 预测 / 预算选择   ← ML 主场
├─ warp/graph/     ── HippoRAG2 官方后端适配层(不重写算法,只做 region 隔离 + 成本统计)
├─ warp/baselines/ ── KET-RAG、G2ConS 两个已发表方法的 matched-backend 适配
├─ warp/eval/      ── ER/CE@k、EM/F1、bootstrap/randomization/Holm、成本聚合
└─ warp/models.py  ── 数据 schema:Document/Query/Region/SearchResult/ConstructionCost/RegionFeatures
```

**pipeline 的数据流**(fit 阶段,只吃 design queries):

```text
① Base 索引(BM25+Dense+RRF) ──→ ② 共访问图 + Leiden → regions
③ 9 维特征(用 Base top-k 和 gold evidence)     ④ 估算每个 region 的 token 成本
⑤ 分层选 probe regions → 真建图 → 跑 query → 增益标签 y_r
⑥ LightGBM 拟合 → 预测所有 region 的 ĝ_r
⑦ 选择器按 score 贪心选(5 个区域方法共享设计状态 + 2 个 global baseline 独立构建)
```

evaluate 阶段:test query → Base 路由(查 top-20 命中哪些 region)→ 只查已物化区域的图 → RRF 融合 →
CrossEncoder 重排 → 指标;成本每个 method/budget/trial 单独记账。

**给图论/ML 背景读者的上手路径**:

1. 先读 [warp/partition/coaccess_graph.py](warp/partition/coaccess_graph.py)(74 行)和
   [leiden.py](warp/partition/leiden.py)(66 行)——纯图论,半小时能看懂;
2. 再读 [warp/advisor/](warp/advisor/)(features→probe→predictor→selector)——纯 ML,这是论文的核心贡献;
3. [warp/eval/statistics.py](warp/eval/statistics.py)(71 行)——统计口径,直接决定论文数字可不可信;
4. retrieval/ 和 graph/ 是领域 glue,了解接口即可,细节留给审查报告;
5. [pipeline.py](warp/pipeline.py)(400 行)最后读,它是把 ①-⑦ 串起来的线。

一句话概括此前代码审查的结论:**图论和 ML 部分本身是对的;问题集中在"账本"(成本记账)和
"特征-标签对齐"(有监督学习的基本功),这两类均已修复。**

## 五、什么是消融实验(ablation)

**消融 = "砍掉一个组件,看效果掉多少"。** 一个方法往往是多个组件叠出来的,消融实验就是控制变量法:
每次只移除一个组件(其余保持不变),对比完整方法 vs 残缺方法的差距——差距越大,说明这个组件贡献越大。

**为什么必须做**:如果不做,审稿人会说"你效果好,但不知道是哪部分起作用的"。尤其本项目用了大量现成组件
(Leiden、LightGBM、贪心)——**新颖性声明完全押在"三个信号联合使用"上**,而不是任何一个组件本身。所以
消融是生死线,不是加分项。

本项目有**三层消融**,各回答不同问题:

| 消融 | 具体做法 | 回答的问题 |
|---|---|---|
| **① 选择器消融**(最重要) | 同一个 score 公式 `频率×增益/成本`,每次去掉一个信号:Frequency-only(只留频率)、Gain-only(只留增益)、Cost-only(只留成本)、Random(全去掉) | "三信号联合 > 任何单信号吗?"——research_positioning.md 原文:**如果 WARP 不能稳定赢 Frequency-only,核心论点不成立**,论文只能降级成"负面结果分析" |
| **② 特征消融** | 9 维特征逐一删掉重训 LightGBM,看 LOOCV MAE 的变化 | "预测器靠哪些特征?"(feature importance 的补充) |
| **③ 分区消融** | 同预算下换分区方式:combined vs 只 query 边 / 只语义边 / 随机分区 | "workload-aware 分区是否必要?" |

注意消融的**纪律**:只能改一个变量。此前审查发现的 "Frequency-only 用 cost 破平局" 就是消融污染——嘴上说
移除了 cost 信号,实际上平局时 cost 还在起作用。这类问题会直接毁掉结论的可信度,已修复。

## 六、大实验看什么:work 不 work 的判断标准

"work 不 work"分两层:(a) **工程上能跑通、数字自洽**(成本账本对得上、无泄漏);(b) **科学主张成立**。
前者靠 smoke 验证,后者看下面的指标。

### 最该盯的四个"生死指标"(按检查顺序)

**1. Probe 增益分布(RQ1)——第一个要看,决定项目死活**

每个被 probe 的区域真建图后,测得 `y_r = 有图后的 CE@10 − Base 的 CE@10`。看:

- **增益是不是明显异质?**(有的区域大幅正增益,有的接近 0 甚至负)
- **正增益区域占比多少?**

如果增益分布几乎是平的、接近 0,整个"选择性物化"的前提(收益不均匀)就不成立,后面全都不用看。
**这是跑完第一折就该看的数字。**

**2. Predictor 的 Spearman / MAE(RQ2)——第二个生死指标**

LightGBM 的 leave-one-region-out 预测质量。**Spearman 相关系数**是核心:预测收益排序 vs 真实收益排序的
秩相关。如果 Spearman ≈ 0,说明"收益不可预测",probe-and-predict 方法论失败(方法本身仍可靠频率+成本选,
但 "learned" 这个卖点没了)。MAE/RMSE 是辅助口径。

**3. 预算-质量曲线(RQ3)——论文的主图**

横轴 = 实际部署成本(占全图成本的分数),纵轴 = Complete Evidence@10,画 7 条曲线。看:

- **WARP 曲线是否压在 4 个消融和 KET-RAG/G2ConS 之上**;
- 在低预算段(0.1-0.2)能否以**一小部分成本逼近 Full Graph 的水平线**(效率叙事);
- 对应的显著性:**paired significance 表里 Holm 校正后的 p 值是否 < 0.05**。

**4. 路由上限(RQ4)——界定天花板**

any/complete gold-region recall:Base 的 top-20 命中 gold 证据所在区域的比例。如果这个数很低(比如 60%),
说明**上限就低**——区域图再好也救不了 Base 没路由到的 query,论文必须如实讨论这个限制。

### 辅助指标(支撑性证据)

| 指标 | 在哪看 | 作用 |
|---|---|---|
| Evidence Recall@5/10 | quality_cost_curve | CE 的宽松版(部分命中也给分),CE 全 0 时区分度靠它 |
| Answer EM/F1 | reader_evaluation | 端到端答案质量,证明检索增益能转化为答案增益 |
| quality-cost AUC | quality_cost_auc | 把曲线压成一个数,方法间比效率 |
| 交互分布 | design_trials | probe 区域对的两两交互——非零说明收益非可加,"贪心近似"的免责声明要写上 |
| 成本账本 | first-run / online cost | RQ5:省钱是真的省,不是把成本藏到探测和学习里 |
| 分区消融表 | partition_ablation_summary | 排序稳健性(combined vs 其他三种) |

### 一句话总结判断标准

> **Work = ① 增益异质且可预测(Spearman 明显 >0)→ ② 曲线图上 WARP 显著压住所有单信号消融和两个
> baseline(校正后 p<0.05)→ ③ 低预算段接近 Full Graph → ④ 账本显示省的钱是实打实的。**

四件事同时成立,paper 才立得住。任何一环断了,按 research_positioning.md 的预先约定,贡献等级就要相应降级——
本项目在设计上就诚实地把"什么结果算成功、什么算失败"提前写死了,这是它严谨的地方。
