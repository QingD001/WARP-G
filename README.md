# WARP-G

WARP-G 是一个面向科研实验的 workload-aware regional graph materialization 系统。它在共享语料上先建立
BM25 + NV-Embed-v2 基础检索，再根据训练 workload 把文档划分为 regions，少量构建 HippoRAG2 probe graph，
用 LightGBM 预测每个 region 的构图收益，最后在预算约束下只物化最值得构建的区域图。

正式实验固定使用 `seed=42`。主预算曲线比较 KET-RAG、G2ConS、Random-region、Frequency-only、Gain-only、
Cost-only 和 WARP-G；BM25、Dense、Hybrid、HippoRAG2 graph-only 和 Base + Full Graph 作为标准参考方法。
LinearRAG 与 LightRAG 使用锁定的作者官方实现运行独立端到端对照。详细研究假设与评测口径见
[design.md](design.md)，所有已讨论工作的介绍、差异和官方代码状态见 [related_work.md](related_work.md)。

## 执行流程

```text
HippoRAG2 corpus + 1,000 released queries
            │
            ▼
deterministic 5-fold cross-fitting
            │
            ▼
BM25 + Dense + RRF 基础索引
            │
            ▼
train-query coaccess graph + semantic kNN
            │
            ▼
Leiden region partition
            │
            ▼
cheap region features + stratified graph probes
            │
            ▼
LightGBM benefit prediction
            │
            ▼
budgeted regional materialization
            │
            ├── WARP-G 与四个 region-selector ablations
            ├── KET-RAG / G2ConS matched-backend comparison
            └── retrieval、reader、cost、significance artifacts
```

## 项目结构

```text
WARP-G/
├── configs/paper/          四个数据集的正式实验配置
├── scripts/                数据准备、整套实验执行和结果导出脚本
├── warp/
│   ├── advisor/            特征、probe、收益预测和预算选择
│   ├── baselines/          KET-RAG 与 G2ConS
│   ├── data/               数据 schema 适配与加载
│   ├── eval/               检索、reader、统计和成本评测
│   ├── graph/              官方 HippoRAG2 构图与检索适配
│   ├── partition/          共访问图和 Leiden 分区
│   └── retrieval/          BM25、Dense、ANN、RRF 和 CrossEncoder
├── design.md               论文研究设计与实验口径
├── related_work.md         相关工作、差异与官方代码状态
└── pyproject.toml          安装依赖、包信息和命令入口
```

## 每个 Python 文件的职责

### 顶层核心文件

#### `warp/__init__.py`

包的最外层公开接口。声明当前版本，并直接导出 `Document`、`Query`、`Region` 和 `SearchResult`，让其他代码
不必了解这些数据模型具体放在哪个模块。

#### `warp/models.py`

定义整个项目跨模块传递的数据结构：

- `Document`：稳定文档 ID、标题、正文和 metadata；`content` 统一生成标题加正文。
- `Query`：问题、gold evidence IDs、答案和 metadata。
- `SearchResult`：统一检索结果，包括文档 ID、分数、来源、排名和 region ID。
- `Region`：一组被共同物化的文档。
- `DatasetBundle`：共享 corpus 和当前折严格分离的 design/test queries。
- `ConstructionCost`：LLM tokens、embedding tokens、耗时、图规模、存储和美元成本。
- `RegionFeatures`：LightGBM 使用的九维构图前特征及固定向量顺序。

这是其他模块共同依赖的 schema 层，不执行检索或实验。

#### `warp/pipeline.py`

WARP-G 的核心编排器。`WARPConfig` 定义检索深度、probe 比例、partition 模式和固定 seed；`WARPG` 串联完整
设计与部署流程：

1. 拟合 Base 检索器；
2. 从 train workload 构建共访问图并进行 Leiden 分区；
3. 提取区域特征并执行 graph probes；
4. 训练 LightGBM benefit predictor；
5. 测量 probe region 的二阶交互；
6. 按预算选择和物化 regions；
7. 执行 Base、WARP-G、Full Graph 和 graph-only 检索；
8. 输出 routing、预测、probe、interaction 和成本审计信息。

test queries 只会进入评测和 routing diagnostics，不参与分区、特征、probe 或 predictor 训练。

#### `warp/run.py`

正式论文实验入口，也是 `warp-g` 命令实际调用的文件。它负责：

- 读取并验证 YAML；
- 固定 `seed=42` 创建 WARP-G、HippoRAG2、Base 和 CrossEncoder；
- 在六个预算点运行 KET-RAG、G2ConS、四个 region-selection controls 和 WARP-G；
- 运行 BM25、Dense、Hybrid、HippoRAG2 和 Full Graph 参考实验；
- 执行 partition ablation 和固定 reader evaluation；
- 记录 deployment、design-search、first-run 和 online costs；
- 完整执行五个 800/200 folds，并合并全部 1,000 条 held-out query；
- 计算 query-level bootstrap CI、paired randomization、Holm correction 和 quality-cost AUC；
- 保存数据哈希、包版本、CUDA、GPU 和 HippoRAG commit 等复现元数据；
- 最终写出一个自描述 JSON artifact。

#### `warp/utils.py`

无状态通用工具集合：文本 tokenization、稳定字符串哈希、cosine、min-max normalization、JSON/JSONL 读取、
JSON 写入和固定大小 batching。它不包含实验策略。

### `warp/retrieval/`：基础检索与统一排序

#### `warp/retrieval/__init__.py`

检索子包的公开出口，集中导出 BM25、Dense、Hybrid、CrossEncoder、RRF 和统一融合函数。

#### `warp/retrieval/base.py`

定义 `Retriever` Protocol，约束所有基础检索器必须提供 `fit(documents)` 和
`search(query, k, doc_ids)`。它只描述接口，不包含具体算法。

#### `warp/retrieval/bm25.py`

实现 corpus-level BM25。`fit` 计算文档长度、词频、文档频率和 IDF；`search` 支持全语料搜索以及显式
`doc_ids` 子集搜索，返回统一 `SearchResult`。

#### `warp/retrieval/dense.py`

封装 HippoRAG 共享 passage encoder 的 Dense Retriever。它批量生成并缓存文档向量，提供文档向量查找和
cosine dense retrieval，同时向 partition/features/baselines 暴露一致的 embedding 空间。

#### `warp/retrieval/ann.py`

使用 FAISS HNSW 构建 cosine ANN index，为每个文档返回 top-k semantic neighbors。该文件用于替代全量
两两相似度矩阵，主要服务于共访问图补边和 KET-RAG semantic KNN。

#### `warp/retrieval/hybrid.py`

实现共享排序路径：

- `reciprocal_rank_fusion`：按 rank 融合 BM25、Dense 和图检索结果，并支持不同通路权重。
- `HybridRetriever`：BM25 + Dense 的全语料 Base。
- `fuse_and_rerank`：先生成统一候选，再交给 CrossEncoder，供 Base、probe、WARP-G 和所有 baseline 共用。

#### `warp/retrieval/reranker.py`

定义 `Reranker` Protocol，并实现固定 revision 的 `CrossEncoderReranker`。它把 query 与候选文档内容送入
BGE CrossEncoder，按照模型分数产生最终 top-k。

### `warp/partition/`：workload-aware region 划分

#### `warp/partition/__init__.py`

导出 `CoaccessGraph`、`CoaccessGraphBuilder` 和 `RegionPartitioner`。

#### `warp/partition/coaccess_graph.py`

用 train queries 构建文档共访问图。每个 query 的 Base top-k 文档两两形成 query coaccess edges；FAISS
semantic neighbors 形成低权重语义边。输出同时保存总边、两种来源的边和每题 Base 结果，供后续特征提取
与消融复用。

#### `warp/partition/leiden.py`

把 `CoaccessGraph` 转换为 igraph，并调用 Leiden community detection 得到 regions。小于
`min_region_size` 的社区按照与其他社区的边权进行合并，最终生成稳定编号的 `Region` 对象。

### `warp/advisor/`：区域收益建模与选择

#### `warp/advisor/__init__.py`

导出区域特征、probe、收益预测和预算选择的公开类。

#### `warp/advisor/features.py`

`RegionFeatureExtractor` 在真正构图前计算九维 cheap features：文档数、token 数、query frequency、Base
recall、failure rate、retrieval entropy、multi-document rate、embedding dispersion 和 coaccess density。
它还建立 region-query routing 关系，并缓存 probe 使用的 Base candidates。

#### `warp/advisor/probe.py`

定义 `EvidenceRecall` 与 `CompleteEvidence` probe utility、`ProbeOutcome` 和 `RegionProber`。它对 workload
覆盖区域做分层抽样，实际构建少量 HippoRAG2 regional graphs，并使用与正式部署完全相同的 RRF +
CrossEncoder 路径测量每个 region 相对 Base 的真实增益。

#### `warp/advisor/predictor.py`

实现 LightGBM `BenefitPredictor`。它只用已 probe regions 的真实标签拟合，然后预测所有未 probe regions
的图收益。该文件同时提供 leave-one-region-out validation、feature ablation、feature importance 和 probe
learning curve。

#### `warp/advisor/selector.py`

在相同 regions 和预算上实现 WARP-G 公式与四个选择器消融：

```text
query_frequency × max(predicted_gain, 0) / estimated_graph_cost
```

它把 region 视为不可拆分对象，保证累计估算成本不超过给定预算；Random-region、Frequency-only、Gain-only 和
Cost-only 分别移除 workload/gain/cost 信号，用于定位 WARP-G 收益来源。

### `warp/graph/`：HippoRAG2 后端

#### `warp/graph/__init__.py`

集中导出图抽象接口和官方 HippoRAG2 builder/retriever。

#### `warp/graph/builder.py`

定义 `RegionalGraph`，保存 region、artifact 路径、实测构建成本、后端实例和 metadata；同时定义
`GraphBuilder` Protocol，要求图后端实现预构建成本估计、区域构图和 corpus-wide Full Graph 构建。

#### `warp/graph/retriever.py`

定义 `GraphRetriever` Protocol，统一图搜索、在线统计快照和增量成本接口，使 pipeline 不依赖某个具体图
检索实现。

#### `warp/graph/hipporag2.py`

官方 HippoRAG2 的严格适配层，是图相关代码的主体：

- `HippoRAG2Config` 映射固定的 LLM、embedding、PPR、synonymy 和价格参数；
- 校验安装的 HippoRAG API/version 是否与锁定 commit 一致；
- 把每个 region 映射为隔离的官方 HippoRAG index；
- 使用稳定 `source_id` 将官方 chunk 结果映射回 WARP 文档 ID；
- 共享 LLM/embedding 模型权重，但隔离 region artifacts；
- 统计逻辑/物理 LLM tokens、embedding tokens、时间、图规模、存储和估算 USD；
- `HippoRAG2GraphRetriever` 执行正式图检索并测量每个实验单元的在线成本；
- `_HippoRAGPassageEncoder` 将官方 embedding backend 适配为 Dense Retriever 所需接口。

该文件不重新实现 HippoRAG 图算法，区域图与 Full Graph 都调用锁定版本的官方后端。

### `warp/baselines/`：公开论文 baseline

#### `warp/baselines/__init__.py`

导出 `GlobalBaselineFactory` 和 `GlobalGraphBaseline`。

#### `warp/baselines/global_graph.py`

实现两个 corpus-level 论文 baseline：

- KET-RAG：lexical/semantic KNN、chunk PageRank、core chunk selection、keyword bipartite retrieval 和
  HippoRAG2 core KG。
- G2ConS：sentence-level concept embeddings、semantic-filtered co-occurrence、Dice edge weights、concept
  PageRank、core chunk selection、concept graph expansion 和 HippoRAG2 core KG。

`LightweightGraphIndex` 使用 FAISS 检索 query concepts 并沿轻量图扩展到文档；`GlobalBaselineFactory` 在统一
预算下构造轻量结构与 core KG；`GlobalGraphBaseline` 将 Base、轻量结构和 graph results 按固定权重 RRF，
最后进入共享 CrossEncoder。轻量 embedding、构图耗时、节点、边和存储全部计入 deployment cost。

### `warp/data/`：数据加载

#### `warp/data/__init__.py`

公开通用 split loader 和正式实验使用的 deterministic cross-fitting loader。

#### `warp/data/base.py`

规范化 JSON/JSONL 数据加载器。它将常见文档和 query 字段映射为 `Document`/`Query`，保留未知 metadata，
并按 `sha256(seed:query_id)` 的稳定顺序生成五个 design/test folds。

#### `warp/data/benchmarks.py`

HotpotQA、2WikiMultiHopQA、MuSiQue 和 PopQA 的 schema 适配层。它提取不同格式中的 supporting evidence
IDs 和答案，强制每个 split 使用同一共享 corpus，并返回统一 `DatasetBundle`。

### `warp/eval/`：评测、统计与成本

#### `warp/eval/__init__.py`

导出成本聚合、检索评测、答案指标和 reader evaluation 公共函数。

#### `warp/eval/construction_cost.py`

逐维累加多个 `ConstructionCost`。token、时间、图规模、存储和美元成本分别求和，不把异质单位压缩成一个
不可解释的分数。

#### `warp/eval/retrieval.py`

计算每题和整体的 `Evidence Recall@5/10`、`Complete Evidence@5/10`，保存 per-query metrics，并对每个指标
执行 paired bootstrap 95% confidence interval。

#### `warp/eval/statistics.py`

实现论文使用的统计量：均值、paired bootstrap interval、paired randomization p-value，以及 predictor 的
MAE、RMSE 和带并列排名处理的 Spearman correlation。

#### `warp/eval/qa.py`

实现答案 normalization、Exact Match 和 token-level F1，用于统一计算生成式 reader 的答案质量。

#### `warp/eval/reader.py`

把任意检索方法返回的 evidence 送入同一个官方 HippoRAG2 QA prompt/LLM，保证比较时只改变检索证据，不改变
reader。输出 Answer EM/F1、每题 prediction 和 reader token usage。

### `scripts/`：实验脚本

#### `scripts/prepare_benchmark.py`

通用的显式 split 转换工具；四个正式 HippoRAG2 配置不调用它，而是使用 `prepare_hipporag2.py` 生成完整
query set 并由 runner 做五折交叉拟合。

#### `scripts/run_paper_suite.py`

依次启动 HotpotQA、2Wiki、MuSiQue 和 PopQA 四份正式配置。每个数据集使用独立 Python 进程，便于上一份
数据集结束后释放 GPU 模型与图内存；任意进程失败都会直接终止整套运行。

#### `scripts/export_paper_results.py`

读取四个正式 JSON artifacts，将嵌套结果展开成论文绘图和制表使用的 tidy CSV：baseline、quality-cost
trials/summary/AUC、partition ablation、reader 和 Holm-corrected paired significance。

#### `scripts/prepare_official_baselines.py`

按 `configs/official_baselines.yaml` 下载 LinearRAG 与 LightRAG 作者仓库，并检出实验锁定 commit；不修改官方算法。

#### `scripts/run_official_baseline.py` 与 `scripts/run_official_suite.py`

把当前完整 corpus 和 1,000 条 queries 送入锁定的作者官方 API，分别或成套运行独立 end-to-end 对照，记录输入哈希、
官方 commit、构建/查询 wall time、逐题答案和 EM/F1。它们不混入同后端 region-budget 曲线。

#### `scripts/export_official_results.py`

把八份 LinearRAG/LightRAG 官方实验 artifact 汇总为 `official_end_to_end.csv`。

## 配置文件

`configs/paper/` 包含四份正式配置：

- `hotpotqa.yaml`
- `2wiki.yaml`
- `musique.yaml`
- `popqa.yaml`

每份配置显式指定 corpus/query set、五折交叉拟合、WARP 参数、HippoRAG2 参数、固定模型 revision、六个预算点、
`seed=42`、partition ablation 和 reader methods。代码不会在正式运行时自动搜索或改变这些参数。

## 安装

需要 Python 3.10+、CUDA、OpenAI API 凭证和 Hugging Face 模型访问权限：

```bash
pip install -e .
export OPENAI_API_KEY=<your-key>
export HF_HOME=<huggingface-cache-directory>
export CUDA_VISIBLE_DEVICES=0
```

正式后端为：

- graph/dense：`nvidia/NV-Embed-v2`；
- OpenIE/reader：`gpt-4o-mini-2024-07-18`；
- reranker：固定 revision 的 `BAAI/bge-reranker-v2-m3`；
- graph implementation：锁定 commit `c617143f01477243992a63b2e2151cc003dd3b21` 的 HippoRAG2。

## 数据格式

Corpus JSONL：

```json
{"id":"doc-1","title":"Title","text":"Passage text"}
```

Query JSONL：

```json
{"id":"q-1","query":"Question?","gold_doc_ids":["doc-1","doc-2"],"answer":["alias"]}
```

### 使用 HippoRAG2 官方发布数据

HippoRAG2 发布了本项目四个 benchmark 的 corpus/query pairs。下载固定 revision 后运行转换脚本：

```bash
python3 -m pip install -U huggingface_hub
mkdir -p data/raw/hipporag2
hf download osunlp/HippoRAG_2 \
  hotpotqa.json hotpotqa_corpus.json \
  2wikimultihopqa.json 2wikimultihopqa_corpus.json \
  musique.json musique_corpus.json \
  popqa.json popqa_corpus.json \
  --repo-type dataset \
  --revision 5ec05b38deecc3318bb432c69865959c56058990 \
  --local-dir data/raw/hipporag2

python3 scripts/prepare_hipporag2.py
```

转换器会生成配置文件已经指向的 `data/processed/{hotpotqa,2wiki,musique,popqa}`，为重复标题/内容建立稳定
passage ID，验证所有 gold evidence，并写入 `queries.jsonl` 与 `split_manifest.json`。正式 runner 使用固定
`seed=42` 的五折交叉拟合：每折 800 条 query 只用于 physical design，另 200 条只用于 held-out evaluation；
五折合并后，HippoRAG2 发布的 1,000 条 query 每条恰好被测试一次。

## 运行正式实验

运行全部数据集：

```bash
python3 scripts/run_paper_suite.py
```

运行单个数据集：

```bash
python3 -m warp.run \
  --config configs/paper/hotpotqa.yaml \
  --output outputs/paper/hotpotqa.json
```

导出论文 CSV：

```bash
python3 scripts/export_paper_results.py
```

正式运行会构建真实 HippoRAG2 indexes、调用配置中的 LLM、加载 GPU embedding/reranker，并执行完整 reader
evaluation，因此需要提前准备数据、模型权限、CUDA 环境和 API 凭证。
