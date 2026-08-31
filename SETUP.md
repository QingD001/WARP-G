# 环境配置指南

> 状态(2026-08-26 检测):GPU RTX 5060 Laptop 8GB ✓;Python 3.12.8 ✓;数据已就绪 ✓;
> torch 为 CPU 版 ✗;lightgbm/faiss/sentence-transformers/igraph/hipporag 等缺失 ✗;
> C 盘余 33GB,D 盘余 316GB;无 OPENAI_API_KEY;直连 GitHub 失败。
>
> 按以下顺序配置,每一步的备选方案都已列出。

## 1. GitHub 访问(装 HippoRAG 锁定 commit)

依赖锁定上游 commit:`c617143f01477243992a63b2e2151cc003dd3b21`(版本 `2.0.0a4`)。

**路径 A — git 代理 + VCS 安装**:

```bash
git config --global url."https://ghproxy.com/https://github.com/".insteadOf "https://github.com/"
python -m pip install "hipporag @ git+https://github.com/OSU-NLP-Group/HippoRAG.git@c617143f01477243992a63b2e2151cc003dd3b21"
```

**路径 B — 本地 clone 后安装**(commit 校验已支持此方式,会读 clone 目录的 `.git` HEAD):

```bash
git clone https://github.com/OSU-NLP-Group/HippoRAG.git   # 经代理或镜像
cd HippoRAG && git checkout c617143f01477243992a63b2e2151cc003dd3b21
python -m pip install -e ./HippoRAG
```

⚠️ **不要用 zip 下载解压安装**——没有 `.git`,运行时 commit 校验会拒绝。

## 2. Python 依赖

```bash
# 1) 换 CUDA 版 torch(RTX 5060 是 Blackwell 架构,需 cu128;先卸 CPU 版)
python -m pip uninstall -y torch torchvision torchaudio
python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
# 国内慢可换镜像: --index-url https://mirror.sjtu.edu.cn/pytorch-wheels/cu128

# 2) 装 warp 包本身(--no-deps 跳过 git 依赖,hipporag 已在第 1 步装好)
python -m pip install -e . --no-deps

# 3) 其余依赖
python -m pip install lightgbm sentence-transformers igraph leidenalg datasets tiktoken faiss-cpu PyYAML
```

## 3. 模型下载(放 D 盘,共 ~18GB)

```bash
export HF_HOME=D:/hf-cache
export HF_ENDPOINT=https://hf-mirror.com        # 国内镜像
huggingface-cli download nvidia/NV-Embed-v2                     # ~16GB,dense + 构图 embedding
huggingface-cli download BAAI/bge-reranker-v2-m3 \
  --revision b5160aeac3c6c8fe7beaaaf04c9e0142826b58d1           # ~2.3GB,CrossEncoder
```

## 4. LLM API(构图 OpenIE + reader 都必须要)

- `OPENAI_API_KEY`:gpt-4o-mini-2024-07-18 的凭证(或中转站 key);
- 用中转站时,把四份 `configs/paper/*.yaml` 的 `llm_base_url` 改成中转地址;
- 官方 baseline 脚本另需:`OPENAI_BASE_URL`(LinearRAG)、`OPENAI_API_BASE`(LightRAG)。

**费用预估**(gpt-4o-mini):HippoRAG2 全图索引 ≈ 9-10M input tokens/数据集,另有 probe 构图、
每次图检索的 rerank_filter LLM 调用、reader(1000 题 × 8 方法 × 4 数据集)。粗估:
**单数据集完整实验 $30–100;四数据集 $150–400**。先跑一折验证,再放量。

## 5. 环境变量汇总

```bash
export CUDA_VISIBLE_DEVICES=0
export OPENAI_API_KEY=<your-key>
export HF_HOME=D:/hf-cache
export HF_ENDPOINT=https://hf-mirror.com
```

## 6. 配置文件的必要修改

- 四份 `configs/paper/*.yaml` 的 `graph.artifact_root` 目前是相对路径(会写进 C 盘),
  建议改为 `D:/warp-outputs/indexes/<dataset>`(C 盘只剩 33GB,产物放不下);
- 用中转站时同步改 `llm_base_url`。

## 7. 显存现实与实验策略

- **NV-Embed-v2 是 7.8B 模型,bf16 ≈ 16GB,本机 8GB 显存装不下**;bge-reranker-v2-m3(2.3GB)没问题,
  BM25/FAISS 走 CPU 没问题;
- **本机 smoke**:把 config 里 `embedding_model_name` 临时换成小模型(如 `BAAI/bge-m3`)验证全 pipeline
  机制,只用于 smoke,不进入正式协议;
- **正式实验**:租卡(如 AutoDL 4090/3090 24GB,约 ¥1-2/h),或确认上游支持 4bit 加载(装好库后测)。

## 8. 装好后的验证清单(由 Claude 执行)

```bash
python -c "import torch; assert torch.cuda.is_available(), 'CUDA 不可用'"
python -c "from warp.graph.hipporag2 import _installed_upstream_commit; print(_installed_upstream_commit())"
# 期望输出 c617143f01477243992a63b2e2151cc003dd3b21
python -c "import warp.pipeline, warp.run"    # 全部依赖可导入
```

随后按序进行:

1. **上游 API 逐点 smoke**:验证审查报告中列出的 ~20 处 HippoRAG 适配假设
   (infer 返回三元组、doc_scores 元素类型、source_id 键、BaseConfig 字段名、PopQA dataset=None 等);
2. **单折 mini 实验**(小 embedding 模型):数据加载 → Base → 分区 → 特征 → probe → predictor;
3. **看 probe gain 分布与 Spearman**——这是判断项目 work 不 work 的第一个生死指标(见 summary.md §6)。
