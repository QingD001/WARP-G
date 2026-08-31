# WARP-G：最终研究设计

> 数学形式化版本见 `design_math.md`（LaTeX 公式版，内容口径一致）。

## 1. 研究问题

WARP-G 研究共享语料上的 workload-aware regional graph materialization。所有方法拥有完全相同的
BM25、NV-Embed-v2 和 CrossEncoder。昂贵的 HippoRAG2 图是附加物理结构，不是基础数据库。

给定 corpus regions `R`、历史设计 workload `Q_train`、构建预算 `B` 和检索指标 `M`，目标是选择：

$$
\forall S\subseteq R,\sum_{i\in S} c(i)\le B
$$

使 held-out workload 上的 `E[M(q; S)]` 最大。论文只研究同一种 Graph representation 应该在哪些
regions 物化，不引入 Tree、Summary、Agent、RL 或在线更新。

研究问题固定为：

- RQ1：HippoRAG2 相对 Base 的收益是否在 corpus regions 之间显著不均匀？
- RQ2：能否根据构图前可获得的 supervised workload/corpus features 预测区域收益？
- RQ3：在相同构建预算下，WARP-G 是否优于 KET-RAG、G2ConS 和四个 region-selection controls，
  并以更低成本接近 Full Graph？
- RQ4：收益排序对 partition method 的变化是否稳定，query routing 对系统上限有多大影响？
- RQ5：construction savings 是否会被策略搜索成本或在线区域图调用抵消？

## 2. 数据边界

每个数据集使用 HippoRAG2 官方发布的共享 corpus 和完整 1,000 条 query。所有配置在正式运行前冻结，
不使用 query 自动调参。固定 `seed=42` 做五折交叉拟合：

- 每折 800 条 design query 用于 partition、features、probe labels、predictor 和 materialization selection；
- 另 200 条 held-out query 只用于 retrieval、reader 和 routing evaluation；
- 五折分别重新执行完整 physical-design 流程；
- 每条 query 恰好作为一次 held-out test，最终统计在合并后的 1,000 条逐题结果上计算。

`base_recall`、`failure_rate` 和 `multi_doc_rate` 使用 design-fold gold evidence，因此方法被明确界定为
supervised workload-aware physical design，不宣称适用于完全无标注的线上日志。

数据集为 HotpotQA、2WikiMultiHopQA、MuSiQue 和 PopQA。前三个测量多跳完整证据，PopQA 是
single-hop control。所有文档 ID 和内容必须唯一，所有 gold evidence 必须存在于共享 corpus。

## 3. Base 与统一排序路径

全语料 Base 为：

```text
BM25 + NV-Embed-v2 -> RRF -> pinned BGE CrossEncoder -> top-k
```

Base、区域 probe、WARP、corpus-level baselines 和 Full Graph 使用相同 `candidate_k`、RRF、reranker
和最终 `k`。因此零图预算下，每个选择方法严格退化为同一个 Base，不会把 reranker 收益误记成图收益。

## 4. Workload-aware partition

对每个 train query 取得 Base top-`coaccess_k`。共同出现的文档形成 query coaccess edge：

```text
w_query(i,j) = #{q: d_i,d_j ∈ TopK_base(q)}
```

使用 FAISS HNSW 为每篇文档取得少量 semantic neighbors：

```text
w(i,j) = w_query(i,j) + λ × cosine(i,j)
```

Leiden 在该稀疏图上生成 regions。正式消融包括：combined、query-only、semantic-only 和在相同
region size multiset 下随机分配文档。Embedding dispersion 只采样固定数量文档对，不产生二次内存。

## 5. Region features

构图前特征为：

- `num_docs`、`num_tokens`；
- `query_freq`；
- `base_recall`、`failure_rate`；
- `avg_retrieval_entropy`；
- `multi_doc_rate`；
- `embedding_dispersion`；
- `coaccess_density`。

论文直接报告 feature leave-one-out ablation、LightGBM feature importance 和 probe learning curve。

## 6. Probe labels 与 predictor

Probe regions 沿 workload density、cost 和 failure 轴分层抽取，且至少包含六个有 workload 覆盖的
regions。每个 probe graph 使用正式 HippoRAG2 构建。

标签走与最终系统完全一致的 candidate generation、RRF 和 CrossEncoder。主要目标为：

```text
y_i = mean_q [CompleteEvidence@10(Base + Graph_i) - CompleteEvidence@10(Base)]
```

同时保存普通 evidence recall gain。LightGBM 只使用 probe labels 拟合；已观测 label 覆盖对应区域
预测。对 probe regions 做 leave-one-region-out，报告 MAE、RMSE、Spearman 和逐区域预测。

区域图集合的真实效用不被假设为严格可加。代码对 probe region pairs 直接测量：

```text
interaction(i,j) = U({i,j}) - U({i}) - U({j}) + U(∅)
```

主论文报告交互分布。如果交互不可忽略，论文只能把独立 gain 排序描述为可部署近似，不能声称求解了
一般集合效用最优化。

## 7. WARP selection 与 baselines

WARP score：

```text
score_i = query_freq_i × max(predicted_gain_i, 0) / estimated_graph_cost_i
```

区域不可拆分，按 score 选择且不超过 estimated graph budget。预测收益 ≤ 0 的区域在任何预算下都被排除
（score 中的 `max(·,0)` 与选择器的显式跳过），因此 WARP 在 budget=1.0 时只物化全部预测正收益区域，
不等于 Full Graph；Full Graph 作为独立上界参考。

正式比较方法：

- BM25、Dense 与 Hybrid；
- Random-region、Frequency-only、Gain-only 与 Cost-only：复用完全相同的 regions、成本估计和检索后端，
  依次检验随机选择、workload frequency、预测收益和低成本偏好的单独作用；
- KET-RAG：lexical/semantic KNN PageRank core chunks、HippoRAG2 KG skeleton、全语料 keyword
  bipartite retrieval；
- G2ConS：sentence-level concept embeddings、semantic-filtered co-occurrence concept graph、PageRank
  core chunks、HippoRAG2 core-KG 和 dual-path retrieval；
- HippoRAG2 graph-only；
- Base + corpus-wide Full HippoRAG2。

KET-RAG 与 G2ConS 的昂贵 KG 都使用与 WARP 相同的 HippoRAG2 builder，避免 Graph backend 能力差异。
它们的 keyword/concept index embedding、时间、节点、边和存储全部计入成本；轻量结构先占用同一 token
proxy 预算，只有剩余部分可用于 core KG。预算不足以构造该结构时，对应点就是共享 Base。

四个 region-selection controls 是 selector ablation，不是四套独立构图系统：它们复用同一折已经完成的 WARP
physical-design state，只替换最后的区域排序公式。LinearRAG 与 LightRAG 另用锁定的作者官方仓库，在相同完整
corpus 和 1,000 queries 上运行独立 end-to-end 表；由于构图单元和成本维度不相同，不强行映射到本节 token-proxy
预算曲线。完整方法差异、官方代码状态和 commit 见 `related_work.md` 与 `configs/official_baselines.yaml`。

## 8. Query routing

Test query 先运行 Base，取 top-`routing_k` 文档并查表得到 regions。只查询已物化且被 Base 命中的图。
论文同时报告 any-gold-region recall 和 complete-gold-region recall。该指标界定系统上限：Base 没有命中
正确区域时，区域图不能修复该 query。

## 9. 成本口径

成本严格分账：

1. `deployment_cost`：最终保留图及 KET/G2ConS 轻量结构的冷构建成本；
2. `design_search_cost`：WARP/benefit predictor 获得 probe labels 的图构建成本；probe 打标阶段的在线
   图检索 token 单独记录并归属 design 口径；
3. `method_specific_design_wall_seconds`：partition、feature、predictor 等非图设计时间；
4. `first_run_cost`：按物理并集记账（deployment ∪ probe），probe 图被物化复用时只计一次；
5. `online_retrieval_cost`：每个 method/budget/trial 独立的 query-time tokens、calls 和 wall time。

原始维度包括 LLM input/output tokens、embedding tokens、wall time、nodes、edges、storage 和按配置中
固定价格快照计算的 USD。不同 token 类型不会只以单一总数呈现；token sum 仅作为预构建预算 proxy。

## 10. 评价与统计

Retrieval 指标：Evidence Recall@5/10、Complete Evidence@5/10。Reader 固定为同一 HippoRAG2 QA LLM，
报告 Answer EM/F1。

预算为 `{0, 0.1, 0.2, 0.4, 0.6, 1.0}` × Full Graph token proxy。全部正式实验固定 `seed=42`。
每个方法通过 query-level paired bootstrap 报告 95% CI。WARP 与所有同预算 baselines 做 paired
randomization test，并对同一指标的比较应用 Holm correction。

主图使用实际 deployment cost，而不是 requested proxy budget。另画 first-run cost 和 online cost，
并报告按实际 deployment cost 积分的 quality-cost AUC。

## 11. 可复现性

Artifact 保存：完整 YAML、所有包版本、HippoRAG commit/API version、数据 SHA-256、Python/平台、
CUDA/cuDNN/GPU、固定 seed、每题指标、每次构建 manifest 和多维成本。正式运行要求 CUDA、
锁定模型 revision、官方 HippoRAG API、完整 usage metadata 和稳定 `Chunk.source_id`；任何不一致直接失败。
