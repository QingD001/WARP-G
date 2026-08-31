# related_papers

2026-08-26 复核下载的论文 PDF,用于补充 `related_work.md`。以 arXiv 页面 / ACL Anthology 为准,下载版本可能有后续更新。

| 文件 | 论文 | 链接 | 与 WARP-G 的关系 |
|---|---|---|---|
| crossaug.pdf | Beyond Chunk-Local Extraction: Cross-Chunk Graph Augmentation for GraphRAG | [arXiv:2605.28004](https://arxiv.org/abs/2605.28004) | **最接近的新工作**:自监督 GNN 打分 + 预算内选择性 LLM 图补全;不学 workload/QA 增益。有官方代码 [DonFinliani/CrossAug](https://github.com/DonFinliani/CrossAug) |
| sag.pdf | SAG: SQL-Retrieval Augmented Generation with Query-Time Dynamic Hyperedges | [arXiv:2608.12129](https://arxiv.org/abs/2608.12129) | 无图路线最强公开结果(MuSiQue Recall@5 80.36),需在 discussion 回应 |
| proprag.pdf | PropRAG: Guiding Retrieval with Beam Search over Proposition Paths | [arXiv:2504.18070](https://arxiv.org/abs/2504.18070) | 全语料轻量构图(MuSiQue Recall@5 77.3),官方端到端候选 |
| g2cons.pdf | Graph-Guided Concept Selection for Efficient RAG | [arXiv:2510.24120](https://arxiv.org/abs/2510.24120) | 本仓库 G2ConS baseline 的原始论文(无官方代码) |
| ket_rag.pdf | KET-RAG: A Cost-Efficient Multi-Granular Indexing Framework for Graph-RAG | [arXiv:2502.09304](https://arxiv.org/abs/2502.09304) | 本仓库 KET-RAG baseline 的原始论文(有官方代码) |
| ea_graphrag.pdf | EA-GraphRAG(Use Graph When It Needs) | [arXiv:2602.03578](https://arxiv.org/abs/2602.03578) | per-query 在线路由,与离线物化互补 |
| hcg_rag.pdf | Structure Over Scale: Schema-Constrained Causal Graphs for RAG | [arXiv:2607.22592](https://arxiv.org/abs/2607.22592) | schema 约束降低单位构图成本 |
| meshrag.pdf | Collision to Cognition: Hash-Driven Graph Construction for Efficient RAG | [ACL 2026](https://aclanthology.org/2026.acl-long.1156/) | 零 token 构图路线 |
| commercial_tax.pdf | The Commercial Tax: Rent-vs-Own Blind Spots in Multi-Hop Retrieval Benchmarks | [arXiv:2608.16096](https://arxiv.org/abs/2608.16096) | NV-Embed-v2 license 审计;本仓库也用该模型,注意合规披露 |
| reasoning_bottleneck.pdf | The Reasoning Bottleneck in Graph-RAG: Structured Prompting and Context Compression for Multi-Hop QA | [arXiv:2603.14045](https://arxiv.org/abs/2603.14045) | KET-RAG 独立评测;检索高 coverage 但 reasoning 是瓶颈 |
