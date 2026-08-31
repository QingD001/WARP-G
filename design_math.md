# WARP-G：最终研究设计（数学形式化版）

> 本文为 design.md 的数学形式化版本：所有实验口径、约束与统计决策用公式精确化，自然语言只保留动机与解释。
> 公式采用 LaTeX 记法（`$...$` 行内、`$$...$$` 独立行），可直接转写论文。所有符号定义见第 0 节。

## 0. 记号与对象

| 记号 | 含义 |
|---|---|
| $D = \{d_1, \dots, d_N\}$ | 共享语料，$N$ 篇文档，ID 与内容唯一 |
| $Q = \{q_1, \dots, q_M\}$ | 完整 query 集（HippoRAG2 官方发布，$M = 1000$） |
| $\Gamma(q) \subseteq D$ | query $q$ 的 gold evidence 文档集（须满足 $\Gamma(q) \subseteq D$） |
| $a(q)$ | query $q$ 的 gold answer（别名列表） |
| $H$ | 共享 Base Retriever：BM25 + NV-Embed-v2 Dense → RRF → CrossEncoder |
| $\mathrm{TopK}_H(q, k)$ | Base 对 $q$ 返回的 top-$k$ 文档 |
| $\mathcal{R} = \{r_1, \dots, r_n\}$ | 区域划分：$\bigcup_i r_i = D$，$r_i \cap r_j = \varnothing\ (i \ne j)$ |
| $G_r$ | region $r$ 的物化图索引（官方 HippoRAG2，隔离 artifact） |
| $G_S = \{G_r : r \in S\}$ | 物化集合 $S$ 对应的图集 |
| $M(q; H, G_S)$ | 检索效用，主指标 $\mathrm{CE@10}$ |
| $\hat c_r$ | region $r$ 的估算构图成本（token proxy） |
| $B$ | 全局离线构建预算 |
| $k, k_{\mathrm{cand}}, k_{\mathrm{route}}, k_{\mathrm{co}}, k_{\mathrm{sem}}$ | 检索深度 / 候选深度 / 路由深度 / 共访问深度 / 语义邻居数 |
| $\lambda \ge 0$ | 语义边权系数（semantic_lambda） |

**五折交叉拟合**。按稳定哈希排序后轮询分配（与 `split_manifest.json` 和代码实现一致）：

$$h(q) = \mathrm{sha256}(\mathrm{seed} \| q.\mathrm{id}),\qquad \mathrm{seed} = 42$$

$$Q_{\mathrm{test}}^{(f)} = \left\{ q : \mathrm{rank}_h(q) \bmod 5 = f - 1 \right\},\qquad Q_{\mathrm{design}}^{(f)} = Q \setminus Q_{\mathrm{test}}^{(f)},\quad f \in \{1, \dots, 5\}$$

$$\left| Q_{\mathrm{design}}^{(f)} \right| = 800,\quad \left| Q_{\mathrm{test}}^{(f)} \right| = 200,\quad \bigsqcup_{f=1}^{5} Q_{\mathrm{test}}^{(f)} = Q$$

每条 query 恰好作为一次 held-out test；每折独立重跑完整 physical-design 流程，最终统计在合并后的 1000 条逐题结果上计算。

## 1. 研究问题

### 1.1 问题形式化

WARP-G 研究共享语料上的 workload-aware regional graph materialization。所有方法拥有完全相同的
BM25、NV-Embed-v2 和 CrossEncoder。昂贵的 HippoRAG2 图是附加物理结构，不是基础数据库。

给定共享语料 $D$、历史设计 workload $Q_{\mathrm{design}}$、共享 Base $H$、区域划分 $\mathcal{R}$、成本估计 $\hat c$ 与预算 $B$，选择：

$$S^* = \underset{S \subseteq \mathcal{R}}{\arg\max}\ \mathbb{E}_{q \sim Q_{\mathrm{future}}}\left[ M\left(q; H, G_S\right) \right]\qquad \mathrm{s.t.}\quad \sum_{r \in S} \hat c_r \le B$$

其中 $Q_{\mathrm{future}}$ 以每折 $Q_{\mathrm{test}}^{(f)}$ 近似；test queries 不参与 partition、features、probe、predictor 训练或预算选择。
论文只研究同一种 Graph representation 应该在哪些 regions 物化，不引入 Tree、Summary、Agent、RL 或在线更新。

### 1.2 RQ1–RQ5 的形式化

- **RQ1（收益异质性）**：定义区域边际收益

$$\Delta_r = \mathbb{E}_{q \sim Q_{\mathrm{design}}}\left[ M(q; H, G_r) - M(q; H) \right]$$

检验 $H_0$：$\Delta_r$ 跨区域近似均匀。报告 $\{\Delta_r\}$ 的分布、$\hat P(\Delta_r > 0)$、跨 fold/dataset 稳定性。

- **RQ2（收益可预测性）**：能否根据构图前特征 $\phi(r)$（第 5 节）学习 $f_\theta : \mathbb{R}^9 \to \mathbb{R}$，使 $\hat g_r = f_\theta(\phi(r))$ 逼近 $\Delta_r$；以 leave-one-region-out MAE/RMSE/Spearman 评估（第 6 节）。

- **RQ3（预算内优势）**：对每个预算点 $b$ 与每个 baseline $k$，检验

$$\bar M_{\mathrm{WARP}}(B_b) > \bar M_k(B_b),\qquad b \in \{0, 0.1, 0.2, 0.4, 0.6, 1.0\}$$

$$k \in \{\mathrm{KET\text{-}RAG}, \mathrm{G2ConS}, \mathrm{Random}, \mathrm{Freq}, \mathrm{Gain}, \mathrm{Cost}\}$$

以 query-level paired randomization 检验并做 Holm correction；同时报告 WARP-G 以更低实际成本接近 Full Graph 的 frontier。

- **RQ4（排序稳定性与路由上限）**：设 $\Pi = \{\pi_{\mathrm{combined}}, \pi_{\mathrm{query}}, \pi_{\mathrm{semantic}}, \pi_{\mathrm{random}}\}$ 为分区模式，比较各模式下收益排序的稳定性

$$\rho_{\pi, \pi'} = \mathrm{Spearman}\left( \{\hat g_r^{(\pi)}\}, \{\hat g_r^{(\pi')}\} \right)$$

并报告 any/complete gold-region routing recall（第 8 节）对系统上限的界定。

- **RQ5（成本守恒）**：construction savings 是否被策略搜索成本或在线区域图调用抵消，即

$$C_{\mathrm{first}}(\mathrm{WARP}) \overset{?}{<} C_{\mathrm{deploy}}(\mathrm{Full}),\qquad C_{\mathrm{online}}(\mathrm{WARP}) \overset{?}{\approx} C_{\mathrm{online}}(\mathrm{Base})$$

first-run 与 online 成本分别报告（第 9 节）。

## 2. 数据边界

每个数据集使用 HippoRAG2 官方发布的共享 corpus 和完整 1,000 条 query。所有配置在正式运行前冻结，
不使用 query 自动调参。固定 `seed=42` 做五折交叉拟合：

- 每折 800 条 design query 用于 partition、features、probe labels、predictor 和 materialization selection；
- 另 200 条 held-out query 只用于 retrieval、reader 和 routing evaluation；
- 五折分别重新执行完整 physical-design 流程；
- 每条 query 恰好作为一次 held-out test，最终统计在合并后的 1,000 条逐题结果上计算。

监督信号来自 design-fold gold evidence：

$$\mathrm{recall}(q, k) = \frac{\left| \mathrm{TopK}(q, k) \cap \Gamma(q) \right|}{\left| \Gamma(q) \right|}$$

$$\mathrm{fail}(q, k) = \mathbf{1}\left[ \mathrm{recall}(q, k) < 1 \right],\qquad \mathrm{multidoc}(q) = \mathbf{1}\left[ \left| \Gamma(q) \right| \ge 2 \right]$$

因此方法被明确界定为 **supervised workload-aware physical design**，不宣称适用于完全无标注的线上日志。

数据集为 HotpotQA、2WikiMultiHopQA、MuSiQue 和 PopQA。前三个测量多跳完整证据，PopQA 是
single-hop control。所有文档 ID 和内容必须唯一，所有 gold evidence 必须存在于共享 corpus。

## 3. Base 与统一排序路径

全语料 Base 为：

$$\mathrm{TopK}_H(q, k) = \mathrm{Rerank}_{\mathrm{CE}}\Big( \mathrm{RRF}\big( \mathrm{BM25}(q, \cdot),\ \mathrm{Dense}(q, \cdot) \big),\ k_{\mathrm{cand}} \Big)[1{:}k]$$

各组件：

$$\mathrm{BM25}(q, d) = \sum_{t \in q} \mathrm{IDF}(t) \cdot \frac{f(t, d)\,(k_1 + 1)}{f(t, d) + k_1\left(1 - b + b\, \frac{|d|}{\mathrm{avgdl}}\right)}$$

$$\mathrm{Dense}(q, d) = \cos(e_q, e_d) = \frac{\langle e_q, e_d \rangle}{\|e_q\|\,\|e_d\|},\qquad e_{\cdot} = \mathrm{NV\text{-}Embed\text{-}v2}(\cdot)$$

$$\mathrm{RRF}(d) = \sum_{s \in \mathcal{S}} \frac{1}{k_{\mathrm{rr}} + \mathrm{rank}_s(d)},\qquad \mathcal{S} \subseteq \{\mathrm{bm25}, \mathrm{dense}, \mathrm{graph}\}$$

$$s_{\mathrm{final}}(q, d) = \mathrm{CE}\left( q,\ \mathrm{content}(d) \right)$$

Base、区域 probe、WARP-G、corpus-level baselines 和 Full Graph 使用相同 $k_{\mathrm{cand}}$、RRF 参数、reranker
和最终 $k$。因此零图预算下，每个选择方法严格退化为同一个 Base，不会把 reranker 收益误记成图收益。

## 4. Workload-aware partition

对每折 $f$，在 $Q_{\mathrm{design}}^{(f)}$ 上构建共访问图 $\mathcal{G}_{\mathrm{co}} = (D, \mathcal{E}, w)$。

**Query coaccess 边**：

$$w_{\mathrm{query}}(d_i, d_j) = \#\Big\{ q \in Q_{\mathrm{design}}^{(f)} : d_i, d_j \in \mathrm{TopK}_H(q, k_{\mathrm{co}}) \Big\}$$

**语义边**（FAISS HNSW 为每篇文档取 top-$k_{\mathrm{sem}}$ 语义邻居）：

$$w_{\mathrm{sem}}(d_i, d_j) = \cos(e_{d_i}, e_{d_j})$$

**组合边权**：

$$w(d_i, d_j) = w_{\mathrm{query}}(d_i, d_j) + \lambda\, w_{\mathrm{sem}}(d_i, d_j)$$

**分区**：

$$\mathcal{R}^{(f)} = \mathrm{Leiden}\left( \mathcal{G}_{\mathrm{co}}^{(f)} \right),\qquad \forall r \in \mathcal{R}^{(f)}: |r| \ge \mathrm{min\_region\_size}$$

小于 `min_region_size` 的社区按照与其他社区的边权和进行合并，最终生成稳定编号的 $\{r_1, \dots, r_n\}$。

**分区消融**（$\Pi$ 四种模式，在相同 region size multiset 下比较）：

| 模式 | 边权 |
|---|---|
| combined（正式） | $w_{\mathrm{query}} + \lambda\, w_{\mathrm{sem}}$ |
| query-only | $w_{\mathrm{query}}$ |
| semantic-only | $\lambda\, w_{\mathrm{sem}}$ |
| random | 保持 $\{|r_1|, \dots, |r_n|\}$ 不变，文档随机分配（seeded） |

Embedding dispersion 只采样固定数量文档对（`dispersion_pairs`），不产生二次内存。

## 5. Region features

$$\phi(r) = \left( \phi_1(r), \dots, \phi_9(r) \right) \in \mathbb{R}^9$$

记 region-query routing 关系：$Q_r = \left\{ q \in Q_{\mathrm{design}} : \mathrm{TopK}_H(q, k_{\mathrm{co}}) \cap r \ne \varnothing \right\}$。

| # | 特征 | 定义 |
|---|---|---|
| 1 | num_docs | $\phi_1(r) = |r|$ |
| 2 | num_tokens | $\phi_2(r) = \sum_{d \in r} |d|$ |
| 3 | query_freq | $\phi_3(r) = |Q_r|$ |
| 4 | base_recall | $\phi_4(r) = \mathrm{mean}_{q \in Q_r}\ \mathrm{recall}(q, k)$ |
| 5 | failure_rate | $\phi_5(r) = \mathrm{mean}_{q \in Q_r}\ \mathrm{fail}(q, k)$ |
| 6 | avg_retrieval_entropy | $\phi_6(r) = \mathrm{mean}_{q \in Q_r}\ H(p_q),\quad H(p) = -\sum_i p_i \log p_i$ |
| 7 | multi_doc_rate | $\phi_7(r) = \mathrm{mean}_{q \in Q_r}\ \mathrm{multidoc}(q)$ |
| 8 | embedding_dispersion | $\phi_8(r) = \mathrm{mean}_{(i,j) \in \mathrm{SampledPairs}(r)}\ \big( 1 - \cos(e_{d_i}, e_{d_j}) \big)$ |
| 9 | coaccess_density | $\phi_9(r) = \dfrac{\sum_{(i,j) \in \mathcal{E}(r)} w(i,j)}{\binom{|r|}{2}}$ |

论文直接报告 feature leave-one-out ablation（逐维移除重训后的 $\Delta\mathrm{MAE}$）、LightGBM feature importance
和 probe learning curve。

## 6. Probe labels 与 predictor

Probe regions 沿 workload density、cost 和 failure 轴分层抽取，且至少包含六个有 workload 覆盖的
regions：

$$\mathcal{P}^{(f)} \subset \mathcal{R}^{(f)},\qquad \left| \mathcal{P}^{(f)} \right| \ge 6$$

每个 probe region 使用正式 HippoRAG2 构建 $G_r$。标签走与最终系统完全一致的 candidate generation、RRF 和
CrossEncoder 路径：

$$y_r = \frac{1}{|Q_r|} \sum_{q \in Q_r} \Big[ \mathrm{CE@10}\left( q; H, G_r \right) - \mathrm{CE@10}\left( q; H \right) \Big]$$

同时保存普通 evidence recall gain $\Delta \mathrm{ER@}k$。

**Predictor**（LightGBM，只用 probe labels 拟合；已观测 label 覆盖对应区域预测）：

$$\hat g_r = f_\theta\big( \phi(r) \big),\qquad \theta^* = \arg\min_\theta \sum_{r \in \mathcal{P}} \ell\left( y_r,\ f_\theta(\phi(r)) \right) + \Omega(\theta)$$

**Leave-one-region-out 验证**（$\hat g_r^{(\setminus r)}$ 表示去掉 $r$ 后训练的预测）：

$$\mathrm{MAE} = \frac{1}{|\mathcal{P}|} \sum_{r \in \mathcal{P}} \left| y_r - \hat g_r^{(\setminus r)} \right|$$

$$\mathrm{RMSE} = \sqrt{ \frac{1}{|\mathcal{P}|} \sum_{r \in \mathcal{P}} \left( y_r - \hat g_r^{(\setminus r)} \right)^2 },\qquad \rho = \mathrm{Spearman}\Big( \{y_r\},\ \{\hat g_r^{(\setminus r)}\} \Big)$$

**Probe learning curve**：对嵌套 probe 子集 $\mathcal{P}_1 \subset \cdots \subset \mathcal{P}_m = \mathcal{P}$ 报告 $\mathrm{MAE}(|\mathcal{P}_j|)$。

**区域间交互**（区域图集合的真实效用不被假设为严格可加）：

$$U(S) = \mathbb{E}_{q}\left[ \mathrm{CE@10}\left( q; H, G_S \right) \right]$$

$$\mathrm{Interaction}(r_i, r_j) = U(\{r_i, r_j\}) - U(\{r_i\}) - U(\{r_j\}) + U(\varnothing)$$

代码对 probe region pairs 直接测量并报告分布。如果交互不可忽略，论文只能把独立 gain 排序描述为
**可部署近似**，不能声称求解了一般集合效用最优化。

## 7. WARP selection 与 baselines

**WARP score**：

$$s_r = \frac{\phi_3(r) \cdot \max(\hat g_r,\ 0)}{\hat c_r}$$

**选择**（区域不可拆分，贪心；预测非正收益区域在任何预算下都不物化）：

$$S_B = \mathrm{Greedy}\Big( \left\{ (r, s_r, \hat c_r) \right\}_{r \in \mathcal{R}};\ B \Big):\quad \text{按 } s_r \downarrow \text{ 依次选取，跳过 } \hat g_r \le 0 \text{ 的区域，满足 } \sum_{r \in S_B} \hat c_r \le B$$

因此 $B_{1.0}$ 的 WARP 是"全部预测正收益区域"，不等于 Full Graph（Full Graph 作为独立上界参考）。

**选择器消融**（复用完全相同的 $\mathcal{R}$、成本估计和检索后端，只替换区域排序公式）：

| 方法 | 排序键 | 被移除的信号 |
|---|---|---|
| WARP-G | $s_r = \phi_3(r) \cdot \max(\hat g_r, 0) / \hat c_r$ | — |
| Random-region | $s_r \sim \mathcal{U}[0,1)$（seeded） | 全部 |
| Frequency-only | $s_r = \phi_3(r)$ | gain、cost |
| Gain-only | $s_r = \max(\hat g_r, 0)$ | frequency、cost |
| Cost-only | $s_r = -\hat c_r$ | frequency、gain |

**正式比较方法**：

- BM25、Dense 与 Hybrid；
- Random-region、Frequency-only、Gain-only 与 Cost-only：依次检验随机选择、workload frequency、预测收益
  和低成本偏好的单独作用；
- KET-RAG：lexical/semantic KNN PageRank core chunks、HippoRAG2 KG skeleton、全语料 keyword bipartite
  retrieval；
- G2ConS：sentence-level concept embeddings、semantic-filtered co-occurrence concept graph、PageRank
  core chunks、HippoRAG2 core-KG 和 dual-path retrieval；
- HippoRAG2 graph-only；
- Base + corpus-wide Full HippoRAG2。

KET-RAG 与 G2ConS 的昂贵 KG 都使用与 WARP-G 相同的 HippoRAG2 builder，避免 Graph backend 能力差异。
它们的 keyword/concept index embedding、时间、节点、边和存储全部计入成本；轻量结构先占用同一 token
proxy 预算，只有剩余部分可用于 core KG。预算不足以构造该结构时，对应点就是共享 Base。

**预算点**：

$$B_b = b \cdot T_{\mathrm{full}},\qquad b \in \{0, 0.1, 0.2, 0.4, 0.6, 1.0\}$$

其中 $T_{\mathrm{full}}$ 为 corpus-wide Full Graph 的 token proxy。

四个 region-selection controls 是 selector ablation，不是四套独立构图系统：它们复用同一折已经完成的 WARP
physical-design state，只替换最后的区域排序公式。LinearRAG 与 LightRAG 另用锁定的作者官方仓库，在相同完整
corpus 和 1,000 queries 上运行独立 end-to-end 表；由于构图单元和成本维度不相同，不强行映射到本节 token-proxy
预算曲线。完整方法差异、官方代码状态和 commit 见 `related_work.md` 与 `configs/official_baselines.yaml`。

## 8. Query routing

Test query 先运行 Base，取 top-`routing_k` 文档并查表得到 regions：

$$C_q = \mathrm{TopK}_H(q, k_{\mathrm{route}}),\qquad R(q) = \left\{ r \in \mathcal{R} : r \cap C_q \ne \varnothing \right\}$$

$$G_{\mathrm{active}}(q) = G_{S \cap R(q)}$$

只查询已物化且被 Base 命中的区域图。

**路由诊断指标**（界定系统上限）：

$$\mathrm{ARR} = \mathrm{mean}_{q \in Q}\ \mathbf{1}\Big[ \exists\, r \in R(q) : r \cap \Gamma(q) \ne \varnothing \Big]$$

$$\mathrm{CRR} = \mathrm{mean}_{q \in Q}\ \mathbf{1}\Big[ \forall\, e \in \Gamma(q),\ \exists\, r \in R(q) : e \in r \Big]$$

$$\forall q:\quad M(q; H, G_S) \le M\big( q; H, G_{\mathcal{R} \cap R(q)} \big)$$

Base 没有命中正确区域时，区域图不能修复该 query。

## 9. 成本口径

**构建成本向量**（多维，不把异质单位压缩成一个不可解释的分数）：

$$c_r = \left( t_{\mathrm{in}}, t_{\mathrm{out}}, t_{\mathrm{emb}}, \tau, n_{\mathrm{node}}, n_{\mathrm{edge}}, s_{\mathrm{storage}}, \mathrm{USD} \right)$$

$$\mathrm{USD}(c_r) = t_{\mathrm{in}}\, p_{\mathrm{in}} + t_{\mathrm{out}}\, p_{\mathrm{out}} + t_{\mathrm{emb}}\, p_{\mathrm{emb}},\qquad T_r = t_{\mathrm{in}} + t_{\mathrm{out}} + t_{\mathrm{emb}}\ \text{(token proxy)}$$

成本严格分账：

1. **Deployment**（最终保留图及 KET/G2ConS 轻量结构的冷构建成本）：

$$C_{\mathrm{deploy}} = \sum_{r \in S_B} c_r + C_{\mathrm{lightweight}}$$

2. **Design search**（获得 probe labels 的图构建成本；打标阶段的在线图检索 token 单独记录、归属同一口径）：

$$C_{\mathrm{design}} = \sum_{r \in \mathcal{P}} c_r$$

3. **方法专属非图设计时间**：$T_{\mathrm{method}}$ = partition、feature、predictor 等非图设计 wall seconds；
4. **First-run**（物理并集口径，probe 图被物化复用时只计一次）：

$$C_{\mathrm{first}} = C_{\mathrm{deploy}} + \sum_{r \in \mathcal{P} \setminus S_B} c_r$$
5. **Online retrieval**：$C_{\mathrm{online}}(m, b, \mathrm{trial})$，每个 method/budget/trial 独立的 query-time tokens、calls 和 wall time。

原始维度包括 LLM input/output tokens、embedding tokens、wall time、nodes、edges、storage 和按配置中
固定价格快照计算的 USD。不同 token 类型不会只以单一总数呈现；token sum 仅作为预构建预算 proxy。

## 10. 评价与统计

**Retrieval 指标**：

$$\mathrm{ER@}k(q) = \frac{\left| \mathrm{TopK}(q, k) \cap \Gamma(q) \right|}{\left| \Gamma(q) \right|},\qquad \mathrm{CE@}k(q) = \mathbf{1}\Big[ \Gamma(q) \subseteq \mathrm{TopK}(q, k) \Big],\qquad k \in \{5, 10\}$$

$$\bar M = \frac{1}{M} \sum_{q \in Q} M(q),\qquad M = 1000$$

**Reader**（固定同一 HippoRAG2 QA LLM，只改变检索证据，不改变 reader）：

$$\mathrm{EM}(q) = \mathbf{1}\Big[ \mathrm{norm}\left( \hat a(q) \right) = \mathrm{norm}\left( a(q) \right) \Big],\qquad \mathrm{F1}(q) = \mathrm{token\text{-}level\ F1}\left( \hat a(q),\ a(q) \right)$$

预算为 $\{0, 0.1, 0.2, 0.4, 0.6, 1.0\} \times$ Full Graph token proxy。全部正式实验固定 `seed=42`。

**统计推断**：

- **Bootstrap 95% CI**：对 query 级结果重采样，$\mathrm{CI}_{95\%} = \left[ \hat Q_{0.025},\ \hat Q_{0.975} \right]$；
  paired 比较在逐题差分 $\delta(q) = M_A(q) - M_B(q)$ 上做 paired bootstrap；
- **Paired randomization**（$N_{\mathrm{perm}} = 10000$ 次随机翻转配对符号）：

$$p = \frac{1}{N_{\mathrm{perm}}} \sum_{\pi} \mathbf{1}\Big[ \left| \bar\delta^{(\pi)} \right| \ge \left| \bar\delta_{\mathrm{obs}} \right| \Big],\qquad \bar\delta = \frac{1}{M} \sum_{q \in Q} \delta(q)$$

- **Holm correction**：对同一指标下的全部方法比较族应用；
- **Quality-cost AUC**（主图使用实际 deployment cost，梯形近似积分）：

$$\mathrm{AUC} = \int \bar M(c)\, \mathrm{d}c \approx \sum_i \frac{\bar M(c_{i+1}) + \bar M(c_i)}{2}\, \Delta c_i$$

另画 first-run cost 和 online cost。

## 11. 可复现性

Artifact $\mathcal{A}$ 自描述，包含：

$$\mathcal{A} = \Big( \mathrm{config},\ \{\mathrm{version}_i\},\ \mathrm{sha256}(D),\ \mathrm{sha256}(Q),\ \mathrm{commit}_{\mathrm{HippoRAG}},\ \mathrm{seed},\ \big\{ M(q) \big\}_{q \in Q},\ \big\{ \mathrm{manifest}_r \big\},\ \big\{ c_r \big\} \Big)$$

Artifact 保存：完整 YAML、所有包版本、HippoRAG commit/API version、数据 SHA-256、Python/平台、
CUDA/cuDNN/GPU、固定 seed、每题指标、每次构建 manifest 和多维成本。正式运行要求 CUDA、
锁定模型 revision、官方 HippoRAG API、完整 usage metadata 和稳定 `Chunk.source_id`；任何不一致直接失败。
