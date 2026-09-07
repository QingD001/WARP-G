# WARP-G：最终研究设计

## 研究问题

WARP-G 研究共享语料上的 workload-aware regional graph materialization。所有方法拥有完全相同的 BM25、NV-Embed-v2 和 CrossEncoder。昂贵的 HippoRAG2 图是附加物理结构，不是基础数据库。

给定 corpus regions R、历史设计 workload Q_train 和检索指标 M，目标是让各方法按照自身策略完整执行构图与检索 pipeline，并统计总消耗 tokens。研究旨在评估：

``Total_Tokens(method) = Construction_Tokens + Online_Retrieval_Tokens``

使 held-out workload 上的 E[M(q; S)] 最大化，并在此基础上评估 Token Efficiency。论文只研究同一种 Graph representation 应该在哪些 regions 物化，不引入 Tree、Summary、Agent、RL 或在线更新。

研究问题固定为：

- RQ1：HippoRAG2 相对 Base 的收益是否在 corpus regions 之间显著不均匀？
- RQ2：能否根据构图前可获得的 supervised workload/corpus features 预测区域收益？
- RQ3：各方法按自身 pipeline 完整跑完后，WARP-G 是否在总消耗 tokens 和 Token Efficiency（检索效果/总消耗 tokens）上优于 KET-RAG、G2ConS 和四个 region-selection controls？
- RQ4：收益排序对 partition method 的变化是否稳定，query routing 对系统上限有多大影响？
- RQ5：construction savings 是否会被策略搜索成本或在线区域图调用抵消？

## 数据边界

每个数据集使用 HippoRAG2 官方发布的共享 corpus 和完整 1,000 条 query。所有配置在正式运行前冻结， 不使用 query 自动调参。

- 取 800 条 design query 用于 partition、features、probe labels、predictor 和 materialization selection；
- 另 200 条 held-out query 只用于 retrieval、reader 和 routing evaluation；
- 执行完整 physical-design 流程；
- 每条 query 恰好作为一次 held-out test，design 阶段（那 800 条）和 test 阶段（那 200 条）分开统计。

base_recall、failure_rate 和 multi_doc_rate 使用 design-fold gold evidence，且只统计落在该 region 内的 gold，因此方法被明确界定为 supervised workload-aware physical design，不宣称适用于完全无标注的线上日志。

数据集为 HotpotQA、2WikiMultiHopQA、MuSiQue 和 PopQA。前三个测量多跳完整证据，PopQA 是 single-hop control。所有文档 ID 和内容必须唯一，所有 gold evidence 必须存在于共享 corpus。

## Base 与统一排序路径

全语料 Base 为：

``BM25 + NV-Embed-v2 -> RRF -> pinned BGE CrossEncoder -> top-k``

Base、区域 probe、WARP、corpus-level baselines 和 Full Graph 使用相同 candidate_k、RRF、reranker 和最终 k。因此在不构建任何图的情况下，每个选择方法严格退化为同一个 Base，不会把 reranker 收益误记成图收益。

## Workload-aware partition

对每个 train query 取得 Base top-coaccess_k。共同出现的文档形成 query coaccess edge：

``w_query(i,j) = #{q: d_i,d_j ∈ TopK_base(q)}``

使用 FAISS HNSW 为每篇文档取得少量 semantic neighbors：

``w(i,j) = w_query(i,j) + λ × cosine(i,j)``

Leiden 在该稀疏图上生成 regions。正式消融包括：combined、query-only、semantic-only 和在相同 region size multiset 下随机分配文档。Embedding dispersion 只采样固定数量文档对，不产生二次内存。

## Region features

构图前特征为：

- num_docs、num_tokens；
- query_freq；
- base_recall、failure_rate（该 region 内 gold 被 Base 找回的比例 / 是否未找全）；
- avg_retrieval_entropy；
- multi_doc_rate（该 region 内是否至少有两篇 gold）；
- embedding_dispersion；
- coaccess_density。

上述三个证据特征只看区内 gold；probe 标签与正式评测仍用整题 CompleteEvidence，不按 region 截断。论文直接报告 feature leave-one-out ablation、LightGBM feature importance 和 probe learning curve。

## Probe labels 与 predictor

Probe regions 沿 workload density、cost 和 failure 轴分层抽取，且至少包含六个有 workload 覆盖的 regions。每个 probe graph 使用正式 HippoRAG2 构建。

标签走与最终系统完全一致的 candidate generation、RRF 和 CrossEncoder。主要目标为：

``y_i = mean_q [CompleteEvidence@10(Base + Graph_i) - CompleteEvidence@10(Base)]``

同时保存普通 evidence recall gain。LightGBM 只使用 probe labels 拟合；已观测 label 覆盖对应区域 预测。对 probe regions 做 leave-one-region-out，报告 MAE、RMSE、Spearman 和逐区域预测。

区域图集合的真实效用不被假设为严格可加。代码对 probe region pairs 直接测量：

``interaction(i,j) = U({i,j}) - U({i}) - U({j}) + U(∅)``

主论文报告交互分布。如果交互不可忽略，论文只能把独立 gain 排序描述为可部署近似，不能声称求解了 一般集合效用最优化。

## WARP selection 与 baselines

WARP score：

``score_i = query_freq_i × max(predicted_gain_i, 0) / estimated_graph_cost_i``

区域不可拆分，按 score 选择。

正式比较方法：

- BM25、Dense 与 Hybrid；

- Random-region、Frequency-only、Gain-only 与 Cost-only：复用完全相同的 regions、成本估计和检索后端， 依次检验随机选择、workload frequency、预测收益和低成本偏好的单独作用（这些都是选择器的消融实验，并没有重新构图）

- KET-RAG：lexical/semantic KNN PageRank core chunks、HippoRAG2 KG skeleton、全语料 keyword bipartite retrieval；

- G2ConS：sentence-level concept embeddings、semantic-filtered co-occurrence concept graph、PageRank core chunks、HippoRAG2 core-KG 和 dual-path retrieval；

- LightRAG（EMNLP 2025）— 有官方代码，已接入

  - 论文：[LightRAG](https://arxiv.org/abs/2410.05779)
  - 官方代码：[HKUDS/LightRAG](https://github.com/HKUDS/LightRAG)
  - 本仓库锁定 commit：`d49112fb7548ee14cb727d43bd68e34da0a2c942`

  LightRAG 建立实体—关系图，并提供 local/global/hybrid/mix 多粒度检索。它的主要目标是轻量、增量、通用的完整 GraphRAG 系统。

- LinearRAG（ICLR 2026）— 有官方代码，已接入

  - 论文：[LinearRAG](https://arxiv.org/abs/2510.10114)
  - 官方代码：[DEEP-PolyU/LinearRAG](https://github.com/DEEP-PolyU/LinearRAG)
  - 本仓库锁定 commit：`bcc94e66c221f798801255efba09311d6fbcd8d6`

  LinearRAG 用轻量实体识别和语义连接构建 relation-free Tri-Graph，强调线性构建复杂度和零 LLM 构图 token，再通过实体激活和全局重要性聚合取回 passage。

- HippoRAG2 graph-only；

- Base + corpus-wide Full HippoRAG2。

KET-RAG 与 G2ConS 的 KG 都使用与 WARP 相同的 HippoRAG2 builder，避免 Graph backend 能力差异。它们的 keyword/concept index embedding、时间、节点、边和存储全部计入成本。各方法不设预算上限，完整执行各自 pipeline，并统计各自总消耗 tokens。

四个 region-selection controls 是只针对 warp 项目的选择器的消融实验。它们复用同一折已经完成的 WARP physical-design state，只替换最后的区域排序公式。

LinearRAG 与 LightRAG 另用锁定的作者官方仓库，在相同完整 corpus 和 1,000 queries 上运行独立 end-to-end 表；由于构图单元和成本维度不相同，不强行映射到本节 token-proxy 预算曲线。完整方法差异、官方代码状态和 commit 见 related_work.md 与 configs/official_baselines.yaml。

## Query routing

Test query 先运行 Base，取 top-routing_k 文档并查表得到 regions。只查询已物化且被 Base 命中的图。 论文同时报告 any-gold-region recall 和 complete-gold-region recall。该指标界定系统上限：Base 没有命中 正确区域时，区域图不能修复该 query。

## 成本口径

成本严格分账并统计总消耗 tokens：

1. deployment_cost：最终保留图及 KET/G2ConS 轻量结构的冷构建成本（包含构图阶段 LLM 提取实体、关系等所有输入输出 tokens）；
2. design_search_cost：WARP/benefit predictor 获得 probe labels 的图构建成本；
3. method_specific_design_wall_seconds：partition、feature、predictor 等非图设计时间；
4. first_run_cost：deployment + 方法专属 design cost；
5. online_retrieval_cost：仅统计 test set（即数据划分中 200 条 held-out query）上所有 query 的 LLM 调用 tokens（输入上下文 + 输出）及 wall time；design set（800 条 query）不计入 online retrieval cost。

原始维度包括 LLM input/output tokens、embedding tokens、wall time、nodes、edges、storage 和按配置中 固定价格快照计算的 USD。不同 token 类型不会只以单一总数呈现。

新增 Token Efficiency 指标（暂时不作为正式指标使用）：

``Token Efficiency = 检索效果得分 / 总消耗 Tokens``

- 检索效果得分均在 test set 上计算，使用 Complete Evidence@10 或 Answer EM；
- 总消耗 tokens 包含 deployment cost 和 online retrieval cost；
- 可以报告两个版本：包含设计成本和不包含设计成本。
- online 部分仅统计 test set（200 条 held-out query），design set（800 条 query）的 LLM 消耗计入 deployment_cost / design_search_cost，不计入 online_retrieval_cost。

## 评价与统计

Retrieval 指标：Evidence Recall@5/10、Complete Evidence@5/10。Reader 固定为同一 HippoRAG2 QA LLM， 报告 Answer EM/F1。

各方法完整跑完后，统计总消耗 tokens 和 Token Efficiency。所有成本指标中的 online_retrieval_cost 仅统计 test set（200 条 held-out query）上的消耗，design set（800 条 query）不计入。全部正式实验固定 seed=42。 每个方法通过 query-level paired bootstrap 报告 95% CI。WARP 与所有同预算 baselines 做 paired randomization test，并对同一指标的比较应用 Holm correction。

主图改为对比各方法的 Token Efficiency 柱状图。另画 first-run cost 和 online cost， 并报告按实际 deployment cost 积分的 quality-cost AUC。

## 可复现性

Artifact 保存：完整 YAML、所有包版本、HippoRAG commit/API version、数据 SHA-256、Python/平台、 CUDA/cuDNN/GPU、固定 seed、每题指标、每次构建 manifest 和多维成本。正式运行要求 CUDA、 锁定模型 revision、官方 HippoRAG API、完整 usage metadata 和稳定 Chunk.source_id；任何不一致直接失败。