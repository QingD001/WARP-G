# WARP-G 改进计划

## 1. 总览

### 1.1 方法判断

- 问题成立：全量 GraphRAG 构图昂贵；KET-RAG / G2ConS 等也在做选择性索引，但多从 corpus importance 出发。
- WARP-G 的差异化：**workload-conditioned utility**——先按 query co-access 形成可独立构图计费的区域，再预测区域图边际收益并选择性物化。
- 不要把贡献说成「又一种 clustering / 新划分算法」。

### 1.2 核心定位（保持）

> **GraphRAG does not need to be a monolithic index.**

把

`Corpus → 一张昂贵全局图`

改成

`Corpus → Workload regions → 可独立物化的图单元`

像数据库物理设计一样：**哪些区域值得建图，由 workload 与边际收益决定。**

成本叙事与当前实验协议对齐：

- 各方法完整跑完 pipeline（**无 token 预算截断 / 无 knapsack**）
- 主报 **Total Tokens** 与 **Token Efficiency**
- 不再讲 matched-budget 曲线

### 1.3 当前最大风险

不是 idea 不够，而是两个逻辑漏洞：

1. **硬分区**切断 multi-hop 图路径  
2. **路由绑死 Base Top-k**：Base miss 时图永远没有纠正机会  

先修这两点，比换 embedding、调 Leiden 更重要。

---

## 2. 优先级总表

| 优先级 | 问题 | 改动 |
| ------ | ---- | ---- |
| **评测-P0** | 主表 k 档过粗；缺多步对照 | Recall/CE @2/3/5/10；单步 + IRCoT 多步 |
| **工程-P0** | 阅读器重复检索 | 检索结果缓存，直接送 QA |
| **方法-P0** | Base miss → 进不了正确 region | 独立 region router |
| **方法-P0** | 硬分区切断多跳 | boundary replication 和/或 cheap bridge |
| **方法-P1** | 区域效用当独立标量 | interaction-aware / marginal selection（无 knapsack） |
| **方法-P1** | co-access 原始计数产生 hub | rank decay + 文档 IDF |
| **方法-P1** | 只有最小 region 尺寸 | 最大估计成本 + 递归再切 |
| **方法-P2** | probe 样本效率低 | active probing（可多轮） |
| **方法-P2** | workload 静态 | 时间衰减 + 增量再物化（后置） |

---

## 3. 评测与流水线改动

### 3.1 多档检索指标

- Evidence Recall（及 CompleteEvidence，若继续作主指标）统一报告：**@2 / @3 / @5 / @10**
- 单步、多步两套结果都要有完整 k 档

### 3.2 单步 + 多步检索（IRCoT 风格）

同一系统同时统计：

| 模式 | 定义 |
| ---- | ---- |
| 单步 | 现有 Base / Base+区域图一次检索 |
| 多步 | 迭代「检索 → 推理/扩展 → 再检索」；固定最大步数与停止条件；与基线公平对齐 |

#### 多步 log（硬要求）

必须可复现整条轨迹，**不能只留最后一轮 top-k**。  
每条 query、每一步至少落盘：

- step 编号、停止原因
- 本步原始 query / 改写 query
- 本步检索方法、命中 region、路由来源（`base-hit` / `region-router` / `bridge`）
- 本步 top-k：doc id、分数、来源（`base` / `graph` / `bridge`）
- 本步推理/扩展文本（若有 LLM）
- 本步及累计的 gold hit、Recall@2/3/5/10（及 CE 若启用）
- token / 调用（input、output、cache）与时间戳

### 3.3 检索完直接问答（禁止再检索）

**现状问题**：主表 `evaluate_retrieval` 搜一次；阅读器阶段对同一方法再调 `search`，重复消耗在线 tokens/时间。

**目标流程**：

```text
一次检索 → 缓存 top-k
    ├─ 计算检索指标（Recall/CE @k）
    └─ 同一批文档直接送固定 HippoRAG2 QA → EM / F1
```

- `--skip-reader` 仍可关掉问答
- 开启阅读器时，**不得为答题重新跑图检索**

---

## 4. 方法改动详情

### 4.1 P0：独立 region router

**现状**

```text
调用 region i  ⟺  Base Top-k(q) ∩ region_i ≠ ∅
```

Base 没打中该区文档时，该区图永远不会被调用。  
而 GraphRAG 最有价值的场景，往往正是 Base 找不到正确证据的时候。

**目标**

```text
R(q) = R_base-hit(q) ∪ TopK_region-router(q)
```

可选实现：

- 区域表示：`z_r = pool{e(d) : d ∈ r}`，query-region 相似度取 TopK
- 或用 design 期 `(q, r)` co-access 监督学习 `P(r|q)`

该改动优先于调 Leiden / λ。

### 4.2 P0：跨区连通（boundary / bridge）

**现状**：`r_i ∩ r_j = ∅`，每个 passage 只属一区。成本记账干净，但 multi-hop 链

`p1 → e1 → p2 → e2 → p3`

若 `p1,p2 ∈ R_A` 而 `p3 ∈ R_B`，两个 HippoRAG2 实例无法继续 PPR / 遍历。

**方案 A：硬核心 + 边界复制（轻量）**

- 核心文档仍互斥
- 对跨区 co-access 最强的 boundary passages 做少量复制（如 top 5%）
- 复制率消融：`ρ ∈ {0, 2%, 5%, 10%}`
- 看 replication cost vs Complete-Gold / Recall

**方案 B：区内贵 KG + 区间廉价 bridge（更推荐作对照故事）**

- 区内：HippoRAG2（昂贵 OpenIE）
- 区间：dense / entity string overlap / concept / co-access 等廉价边
- 路径：`R_A → cheap bridge → R_B`，不为全库做昂贵抽取
- 便于与 G2ConS 的 lightweight concept graph 对照

### 4.3 P1：interaction-aware selection

**现状启发式**

```text
score_i = query_freq_i × max(predicted_gain_i, 0) / estimated_cost_i
```

隐含假设：区域增益近似可加。但 multi-hop 上常有互补：

```text
Utility({R1,R2}) ≠ Utility({R1}) + Utility({R2})
```

**目标形式（概念）**

```text
U(S) = Σ_q w_q [ M(q; Base+S) − M(q; Base) ] − λ C(S)
Δ(r|S) = U(S∪{r}) − U(S)
```

- 问的是「在已选集合 S 下，下一个区的边际增益」
- 求解可用 **greedy**，不必上复杂组合优化
- 与现有 probe 二阶交互诊断对齐
- **落地时不做预算 knapsack**；`λ C(S)` 若保留，只作正则/报告，不截断主实验

### 4.4 P1：co-access 边权去偏

**现状**：Base Top-20 做完全 clique，共现一次 `w += 1`，易把 generic 文档做成 hub。

**建议边权**（按 query 累计）：

```text
w_q(i,j) = 1 / (log(1+rank_i) log(1+rank_j))
         × log(|Q|/df(i)) × log(|Q|/df(j))
```

直觉：两个**排名靠前且不常见**的文档共现，比两个到处出现的 generic 文档共现，更能代表真实 workload affinity。  
可用 Jaccard / PMI 作消融。

### 4.5 P1：区域最大成本 + 递归再切

**现状**：只有 `min_region_size ≥ 5`，没有上限。热门主题可能并成巨大区，物化成本过高。

**目标**：`C_estimated(r) ≤ C_max`（按估计构图成本，而非单纯文档数）

```text
Leiden
  → 若 C_est(r) > C_max：对该区递归 Leiden
  → 直到满足成本上限
```

使 region 成为真正可物化的 physical unit。  
（旧「小预算选不起」叙述改为：避免单区成本失控、便于选择性物化。）

### 4.6 P2：Active probing

探针流程有价值：少建几区 → 观测真实 gain → 预测其余区。  
不要浪费在低效抽样上。

```text
score_probe(r) = α·uncertainty(r) + β·diversity(r) + γ·demand(r)
```

- 高不确定区值得探
- 避免探针扎堆同一类区
- 高频区的真实 label 更值钱
- 可选 2–3 轮：`Probe → Train → Probe → Train`

### 4.7 P2：动态 workload（后置）

时间衰减 + incremental rematerialization；当前主实验可后置。

---

## 5. 建议落地顺序

1. Recall/CE @2/3/5/10 + **阅读器复用检索结果**（不重复检索）  
2. 定 IRCoT 多步协议 + **全量 step log** 格式与落盘  
3. 独立 region router  
4. boundary replication 和/或 cheap bridge  
5. interaction-aware selection（无 knapsack）  
6. co-access 去偏、最大区域成本、active probing  

若只能先做方法侧三件：**router → bridge/boundary → interaction-aware selection**。

---

## 6. 与当前 PopQA 结果的关系

避免改完就预期主表暴涨：

| 观察 | 含义 |
| ---- | ---- |
| 探针增益多数为负、主表贴 Base | **进区后**区域图边际收益本身很平 |
| 只改 router / bridge | 不保证救 PopQA；解决的是结构漏洞 |
| router / bridge / 多步 | 主验证场更应看 **MuSiQue / 2Wiki** 等多跳数据 |
| 多档 Recall + 单步/多步 | 用来判断是「真平」还是「单步 @10 看不出来」 |

---

## 7. 相关参考

- KET-RAG（选择性多粒度索引）：<https://arxiv.org/abs/2502.09304>
- G2ConS（concept selection + lightweight graph）：<https://arxiv.org/abs/2510.24120>
- HippoRAG2（相对更低 offline indexing 消耗）：<https://arxiv.org/abs/2502.14802>
