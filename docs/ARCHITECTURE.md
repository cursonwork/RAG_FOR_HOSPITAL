# RAG for Hospital — 技术架构全景文档

> **版本**: 0.1.0 ｜ **语言**: Python 3.13 ｜ **包管理**: uv ｜ **部署**: Docker Compose

---

## 1. 项目概述

面向医疗系统的企业级 RAG（Retrieval-Augmented Generation）问答系统。核心目标：将 PDF 医学文献转化为结构化知识库，通过检索增强生成回答高质量的医学问题。

### 1.1 关键数字

| 指标 | 数值 |
|------|------|
| 代码量 | ~9,000 行 Python（28 个源文件） |
| 检索管线段数 | 4 段（查询改写 → 混合检索 → 重排序 → 意图路由） |
| 混合检索引擎 | BM25（客户端）+ Dense（Milvus Cosine） |
| RRF 融合常数 | k=60 |
| Embedding 维度 | 1024（BGE-M3） |
| Embedding 批大小 | 20 条/次（Ollama 并发限制） |
| 默认分块大小 | 2000 字符（约 2-3 个医学英文章落） |
| 重排序模型 | ms-marco-MiniLM-L-6-v2（ONNX，4MB） |
| Chunk 规模 | 54 chunks / 22 页论文（↓74% vs 旧版 209） |

### 1.2 依赖矩阵

| 类别 | 组件 | 用途 |
|------|------|------|
| LLM | DeepSeek V4（deepseek-chat）| 生成回答、意图分类、查询改写 |
| Embedding | Ollama + BGE-M3（1024 维）| 文本向量化 |
| 多模态 | DashScope Qwen 3.7 Plus | 医学图片描述生成 |
| 向量库 | Milvus 2.5.10（Standalone）| 稠密向量存储与近邻检索 |
| 关系库 | PostgreSQL 15 | 用户/会话/消息/分块/图片 五表持久化 |
| 对象存储 | MinIO（RELEASE.2023-03-20）| Milvus 底层对象存储 |
| 协调 | etcd v3.5.18 | Milvus 元数据协调 |
| 重排序 | FlashRank（ONNX Runtime）| Cross-encoder 精排（CPU） |
| BM25 | rank-bm25 0.2.2 | 客户端稀疏检索 |
| PDF 解析 | opendataloader-pdf（Java 17+）| 结构化 PDF→Markdown+JSON |
| UI | Streamlit 1.58 | Web 交互界面 |
| API | FastAPI + Uvicorn | REST 接口 |
| 编排 | Docker Compose | 全家桶一键启动 |

---

## 2. 架构全景

```
┌─────────────────────────────────────────────────────────────────────┐
│                         用户界面层                                   │
│              Streamlit (app.py)  │  CLI (main.py)  │  FastAPI (api.py) │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                    ┌────────▼────────┐
                    │   意图路由层      │
                    │  intent.py      │
                    │  medical_qa     │
                    │  drug_query     │
                    │  diagnosis      │
                    └────────┬────────┘
                             │
              ┌──────────────▼──────────────┐
              │         RAG 链层             │
              │  conversation.py (多轮)      │
              │  rag_chain.py (单次)         │
              └──────────────┬──────────────┘
                             │
     ╔═══════════════════════╧═══════════════════════╗
     ║           检索管线（四段式）                    ║
     ║                                               ║
     ║  ① query_rewriter.py  查询改写                ║
     ║  ② hybrid_search.py   BM25+Dense RRF 融合     ║
     ║  ③ reranker.py        FlashRank 精排          ║
     ║  ④ intent.py           意图路由 → Prompt      ║
     ╚═══════════════════════╤═══════════════════════╝
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
  ┌─────▼──────┐   ┌─────────▼──────┐   ┌───────▼──────┐
  │ Embedding  │   │   LLM 层       │   │  图片管线     │
  │ embeddings │   │   llm.py       │   │  image_pipe-  │
  │ .py        │   │   DeepSeek V4  │   │  line.py      │
  │ Ollama     │   │   temp=0.3     │   │  Qwen 多模态   │
  │ BGE-M3     │   │   streaming    │   │  二阶段并发    │
  └─────┬──────┘   └─────────┬──────┘   └───────┬──────┘
        │                    │                    │
  ┌─────▼────────────────────▼────────────────────▼──────┐
  │                    持久化层                           │
  │  database.py (PG 连接池 + 自动建表 + 五表 CRUD)       │
  │  vector_store.py (Milvus 手写封装 + 混合检索委托)      │
  │  chat_history.py (PG 版 LangChain 兼容历史管理)       │
  └─────────────────────────────────────────────────────┘
```

---

## 3. PDF 导入流程（Ingestion Pipeline）

### 3.1 整体流程

```
PDF 文件
  │
  ├─→ opendataloader-pdf（Java 引擎，XY-Cut++ 阅读顺序排序）
  │     │
  │     ├─→ 结构化 Markdown（标题/段落/表格/列表保留）
  │     └─→ ODL JSON（kids[] 数组，每元素含 type/heading_level/page/bbox）
  │
  ├─→ 失败时降级：PyMuPDF get_text()（逐页提取，"## 第N页" 格式）
  │
  ▼
text_splitter.py：ODL JSON section-aware 分块
  │
  ├─→ Phase 1：按 heading level 2-4 切 section，level 1 取为 section 名
  ├─→ Phase 1.5：合并过短 section（<100 字符）到下一 section
  ├─→ Phase 2：section 内按段落边界组合到 chunk_size 上限
  └─→ 超大单元素递归降级分割
  │
  ▼
chunk_id → PG chunks 表（原文存储）→ BGE-M3 Embed → Milvus
  │
  ▼
图片管线（两阶段并发）
  ├─→ Phase 1：PyMuPDF 提取图片 → 过滤 <5KB 噪声
  │      → 写 PG images 表（含 image_data BYTEA）
  │      → 写 Milvus 占位向量（caption 做 embedding）
  └─→ Phase 2：ThreadPoolExecutor(8) 并发调 Qwen 多模态
         → Qwen 生成医学描述
         → UPDATE PG images.description
         → DELETE 旧向量 + INSERT 新向量（Milvus 不支持 update vector）
```

### 3.2 技术选型：为什么用 opendataloader-pdf 而不用 PyMuPDF？

| 维度 | PyMuPDF get_text() | opendataloader-pdf |
|------|-------------------|-------------------|
| 双栏文档 | 文本顺序混乱，左右栏交错 | XY-Cut++ 视觉阅读顺序，正确还原 |
| 表格 | 无结构，丢失行列关系 | Markdown 表格格式完整保留 |
| 标题层级 | 丢失，所有文字扁平化 | heading_level 明确标注（1-5+） |
| 页眉页脚 | 混入正文 | 自动识别过滤 |
| 部署要求 | 纯 Python | 需要 Java 17+（Docker 内置） |

opendataloader-pdf 基于 Apache Tika 引擎，通过 XY-Cut++ 算法根据元素在页面上的视觉位置排序，保证阅读顺序正确。输出 JSON 的 `kids[]` 数组中每个元素都携带 `bounding box`、`type`、`heading_level`、`page number` 等信息，这是 section-aware 分块的前置条件。

---

## 4. 分块策略（Chunking Strategy）

### 4.1 v1 vs v2 对比

| 维度 | v1（旧） | v2（当前） |
|------|---------|-----------|
| 分块方式 | Markdown `#/##/###` 字符串匹配 | ODL JSON kids[] type/heading_level 遍历 |
| chunk_size | 512 字符（硬切） | 2000 字符（段落边界组合） |
| section_title 准确率 | 0%（全是论文标题） | 100%（来自 heading 元素） |
| chunk 数量（22 页论文） | 209 | 54（↓74%） |
| 句子截断 | 频繁（512 字符随机断） | 极少（段落边界断开） |
| 表格处理 | 拆散（表格被当正文切开） | 完整保留（表格作为一个元素） |

### 4.2 ODL JSON section-aware 分块详细算法

```
输入：ODL JSON 的 kids[] 数组（每个 kid 含 type/heading_level/content/page/bbox）

1. 按 heading 边界分组
   - heading_level ∈ {2,3,4} → 创建新 section
   - heading_level = 1 → 提取为 section 标题（用于第一个 section）
   - heading_level ≥ 5 → 视为噪音（作者名等），归入当前 section

2. 合并 tiny sections
   - 总文本量 < 100 字符的 section 合并到下一个 section

3. 段落边界组合
   - 每个 section 内，逐元素累加文本
   - 累加到 chunk_size（2000）上限时 emit chunk
   - 以元素为单位，不在元素中间切断

4. 超大元素降级
   - 单个元素 > chunk_size → RecursiveCharacterTextSplitter 递归分割
```

### 4.3 Markdown 降级分块

当 PDF 无法通过 opendataloader 解析时（如纯文本 PDF），降级为：
- `MarkdownHeaderTextSplitter`（headers_to_split_on: `#`, `##`, `###`）
- 超出 chunk_size 的 section 子分割用 `RecursiveCharacterTextSplitter`（separators: `["\n\n", "\n", "。", "；", "！", "？", "，", " ", ""]`）

---

## 5. Embedding 层

### 5.1 技术选型：为什么用 BGE-M3？

| 维度 | BGE-M3 | text-embedding-ada-002（OpenAI） | all-MiniLM-L6-v2 |
|------|--------|-------------------------------|-------------------|
| 维度 | 1024 | 1536 | 384 |
| 语言支持 | 多语言（中英均优） | 仅英文最优 | 仅英文 |
| 医疗领域 | MTEB 医学子集表现优异 | 通用领域优化 | 通用短文本 |
| 部署 | Ollama 本地（离线可用） | 需 API（联网 + 付费） | 本地 CPU |
| 成本 | 免费 | $0.0001/1K tokens | 免费 |

BGE-M3 是 BAAI 的多语言通用 Embedding 模型，在 MTEB 基准的医学相关子任务（BioASQ、TREC-COVID）中表现突出。选择 1024 维而非更小模型，是为保证医学术语的稠密语义区分能力。

### 5.2 Ollama 批大小限制

```python
# embeddings.py
OpenAIEmbeddings(
    model="bge-m3",
    chunk_size=20,          # ← 关键：每批 20 条
    check_embedding_ctx_length=False,  # 禁用 tiktoken 校验
)
```

**踩坑记录**：Ollama bge-m3 tokenizer 在大批量文本时 `connection reset by peer`。经测试，20 条/批是稳定上限。如果要提高吞吐，可以考虑多线程并发发送多个 batch。

### 5.3 tiktoken 校验问题

LangChain 的 `OpenAIEmbeddings` 默认用 tiktoken 校验输入长度，但 bge-m3 不在 tiktoken 支持的模型列表中。必须设置 `check_embedding_ctx_length=False` 否则报错。

---

## 6. 向量存储（Vector Store）

### 6.1 Milvus Collection 设计

```
Collection: hospital_knowledge
├── 向量字段：1024 维 float，COSINE 距离
├── auto_id: True（Milvus 自动生成主键）
└── 元数据字段：
    ├── chunk_id: VARCHAR    ← 关联 PG chunks 表
    ├── image_id: VARCHAR    ← 关联 PG images 表（图片 chunk 才有）
    ├── source: VARCHAR      ← 来源文件
    ├── page: INT            ← 页码
    ├── section: VARCHAR     ← 章节标题
    └── chunk_type: VARCHAR  ← "text" | "image"
```

### 6.2 技术选型：为什么用 Milvus？

| 维度 | Milvus | Qdrant | Weaviate | Chroma | FAISS |
|------|--------|--------|----------|--------|-------|
| 分布式 | 原生支持 | 原生支持 | 原生支持 | 单机 | 单机（库） |
| 持久化 | MinIO/S3 | 内置 RocksDB | 内置 | 内置 SQLite | 需手动 |
| 生态成熟度 | CNCF 毕业 | 活跃但年轻 | 活跃 | 轻量入门 | FAIR 维护 |
| 中文社区 | 活跃（Zilliz 国内团队） | 一般 | 一般 | 较少 | 较少 |
| gRPC 性能 | 优（C++ 内核） | Rust 内核 | Go 内核 | Python | C++ |
| 原生 BM25 | 2.5+ 支持但不完善 | 支持 | 支持 | 不支持 | 不支持 |

选择 Milvus 的核心原因：
1. **CNCF 毕业项目**，生产环境验证充分
2. **国内团队维护**，中文社区活跃，文档完善
3. **Docker Compose 一键部署**（etcd + MinIO + Milvus Standalone）
4. **Cosine 距离**——医学文本 Embedding 用 Cosine 比 L2 更合适（长度不敏感）

### 6.3 为什么 BM25 不放在 Milvus 里？

Milvus 2.5.10 原生支持 `SPARSE_FLOAT_VECTOR` + `FunctionType.BM25`，schema 可建、insert 可跑，但 **query 端 `RunAnalyzer` RPC 未实现**：`MilvusClient` 无法将原始 query text 编码为 sparse vector。

**折中方案**：客户端 `rank_bm25` —— 全量 chunk 文本常驻内存（分词后 ~1MB/chunk），RRF 融合 BM25 + Milvus Dense。54 chunks 规模完全够用，未来 >10 万 chunks 时等 Milvus 补齐再迁移。

---

## 7. 混合检索（Hybrid Search）—— 全文 + 语义

### 7.1 架构

```
用户查询
    │
    ├─→ BM25SparseRetriever.search(query, k=20)
    │     │
    │     ├─ rank_bm25.BM25Okapi（全量 chunk 在内存中分词）
    │     ├─ 分词：content.lower().split()（空格分词，适配英文医学文本）
    │     └─ 返回 top-20 + bm25_score
    │
    ├─→ MilvusStore.similarity_search(query, k=40)
    │     │
    │     ├─ BGE-M3 Embed → 1024 维向量
    │     ├─ Milvus COSINE 搜索
    │     └─ 返回 top-40 + distance score
    │
    ▼
_rrf_fusion() — Reciprocal Rank Fusion
    │
    ├─ RRF 公式：score = Σ 1/(k + rank_i + 1)
    ├─ k_rrf = 60（标准常数）
    ├─ 按 chunk_id 去重 → 按 RRF 分数降序
    └─ 返回 top-20
```

### 7.2 技术选型：为什么用 RRF 而不用加权求和？

| 维度 | RRF（Reciprocal Rank Fusion） | 加权求和 |
|------|------------------------------|---------|
| 分数归一化 | 不需要（只用排名） | 需要（BM25 分和 Cosine 分分布不同） |
| 超参数 | 仅 k_rrf（标准 60） | 至少需要 1 个权重系数 |
| 对分数分布的鲁棒性 | 完全无视绝对分数 | 受异常高分文档影响 |
| 实现复杂度 | 极简单（O(n)） | 需要分数标准化管道 |

RRF 的核心优势是**不需要分数归一化**。BM25 分数和 Cosine 分数来自完全不同的分布，直接用加权求和需要一个复杂的归一化管道（min-max / z-score / 等）。RRF 只关注排名位置，天然免疫分数尺度差异。

### 7.3 技术选型：为什么用 BM25 而不用 TF-IDF？

| 维度 | BM25 | TF-IDF |
|------|------|--------|
| 文档长度归一化 | 内置（k1, b 参数） | 无（长文档天然高分） |
| 词频饱和 | 有（对数生长 → 饱和） | 无（线性生长） |
| 医学文本适配 | k1 默认 1.2 适合中等长度文档 | 对 PDF 长文本不友好 |
| 使用 | rank-bm25 库开箱即用 | 需自己实现或 scikit-learn |

BM25 的文档长度归一化对医学 PDF 至关重要——不同论文长度差异可达 10 倍（2 页 vs 20 页），没有归一化的 TF-IDF 会让长文档系统性地获得更高的相关性分数。

---

## 8. 重排序（Reranking）—— Cross-encoder 精排

### 8.1 技术选型：为什么用 FlashRank？

| 维度 | FlashRank | Cross-encoder（HuggingFace） | Cohere Rerank API | ColBERT |
|------|-----------|------------------------------|--------------------|---------|
| 模型大小 | 4MB（ONNX） | 400MB+（PyTorch） | 云 API | 2-4GB |
| 推理速度 | ~100ms/20 docs（CPU） | ~2s/20 docs（CPU） | ~500ms + 网络延迟 | gRPC 服务 |
| GPU 需求 | 无 | 强烈建议 | 无（云端） | 强烈建议 |
| 部署 | 纯 pip install | 需 PyTorch + GPU 驱动 | 需 API key | 需 GPU |
| 离线可用 | 是 | 是（但慢） | 否 | 需本地服务 |
| 模型 | ms-marco-MiniLM-L-6-v2 | 可选多种 | 黑盒 | 学术模型 |

FlashRank 是为**无 GPU 生产环境**设计的——它将 HuggingFace 的 cross-encoder 模型量化为 ONNX 格式（4MB），在 CPU 上重排 20 个文档仅需 ~100ms。这在医学 RAG 场景中足够：检索已经缩小到 20 个候选，Cross-encoder 只需要在 20 个中精排 5 个。

### 8.2 模型细节

```
模型：ms-marco-MiniLM-L-6-v2
格式：ONNX Runtime（量化后 4MB）
输入：(query, document) pair
输出：relevance score（0-1）
缓存：~/.cache/flashrank/（首次自动下载）
降级：下载失败 → 去重 + 截断（无精排）
```

### 8.3 降级策略

```python
def _dedup_and_slice(documents):
    """去重（前 120 字符模糊匹配）+ 截断到 top_n"""
    seen = set()
    unique = []
    for d in documents:
        key = d.page_content[:120]
        if key not in seen:
            seen.add(key)
            unique.append(d)
    return unique[:self._top_n]
```

---

## 9. 查询改写（Query Rewriting）—— 四策略路由

### 9.1 策略矩阵

```
输入：用户问题 + 对话历史
          │
    ┌─────▼──────┐
    │ 多轮对话？   │── 是 ──→ 历史补全改写（DeepSeek temp=0）
    │ history≥2   │           解决代词/省略指代："这个药的副作用呢？"
    └─────┬──────┘           → "二甲双胍的副作用有哪些？"
          │ 否
    ┌─────▼──────┐
    │ 短问题？    │── 是 ──→ 多查询扩展（3 个同义变体）
    │ <25 chars   │           原始 + 3 variant = 4 个查询
    └─────┬──────┘
          │ 否
    ┌─────▼──────┐
    │ 长问题？    │── 是 ──→ Step-back 回退（生成上位问题）
    │ >120 chars  │           "结直肠癌基质微环境在分子水平上通过
    └─────┬──────┘            免疫细胞浸润和ECM重塑影响预后的详细机制？"
          │ 否                 → "结直肠癌预后影响因素有哪些？"
    ┌─────▼──────┐
    │ 直通        │── 25-120 chars → 直接使用原始查询
    └────────────┘
```

### 9.2 为什么短问题要扩展？

医学短查询（"糖尿病"、"高血压"）在纯 Dense 检索中会产生高度集中的语义匹配——返回的 chunk 可能全部来自同一论文的同一 section，缺乏 diversity。多查询扩展生成 3 个不同角度的变体（如 Synonyms、Treatment、Diagnosis），每个变体独立检索后合并去重，显著提高召回覆盖率。

### 9.3 为什么长问题要回退？

医学长查询通常来自用户用自然语言描述的病例详情——包含大量 Dense 向量不关心的细枝末节（"上周三开始的"、"吃了布洛芬之后"）。Step-back 提取上位问题，用更概括的医学术语检索，匹配到与核心概念相关的 chunk。

---

## 10. 意图识别（Intent Classification）

### 10.1 三种路由

| 意图 | System Prompt | 触发场景 | Prompt 风格 |
|------|--------------|---------|------------|
| `medical_qa` | MEDICAL_QA_PROMPT | 疾病机制/病理/解剖/流行病学 | 医学顾问：引用文献 + 数据不足标记 + 免责声明 |
| `drug_query` | DRUG_QUERY_PROMPT | 药名/适应症/不良反应/禁忌 | 药学专家：通用名/适应症/剂量/禁忌/不良反应 |
| `diagnosis` | DIAGNOSIS_ASSIST_PROMPT | 症状/检查结果推断疾病 | 临床诊断顾问：鉴别诊断/建议检查/引用证据 |

### 10.2 为什么用 LLM 做意图路由？

医疗场景的 queries 用关键词匹配难以区分——"胸痛的原因"既可能是 medical_qa（病理），也可能是 diagnosis（问症状推断疾病），还可能是 drug_query（问用什么药治胸痛）。只有 LLM 能区分意图的细微差异。

成本：~170 tokens/次（intent prompt + 1 token 输出），可忽略不计。

---

## 11. LLM 生成层

### 11.1 技术选型：为什么用 DeepSeek V4？

| 维度 | DeepSeek V4 | GPT-4o | Claude Opus 4 | 本地模型（Qwen 72B） |
|------|------------|--------|---------------|---------------------|
| 价格 | $0.5/M input tokens | $2.5/M input tokens | $15/M input tokens | 免费（硬件成本） |
| 中文医学 | 强（中文原生） | 中等 | 中等 | 较强 |
| 长上下文 | 64K | 128K | 200K | 32K-128K |
| API 兼容 | OpenAI 兼容 | 原生 OpenAI | 需适配 | Ollama 兼容 |
| 流式输出 | 支持 | 支持 | 支持 | 支持 |

DeepSeek V4 的中文医学能力源于其训练数据包含大量中文学术文献，且在 medical benchmark 上表现与 GPT-4o 接近但成本仅为 1/5。

### 11.2 生成配置

```python
ChatOpenAI(
    model="deepseek-chat",
    temperature=0.3,       # 低温度保证医学回答稳定性
    streaming=True,        # 流式输出提升用户体验
    base_url="https://api.deepseek.com/v1",
)
```

---

## 12. PostgreSQL 持久化设计

### 12.1 五表 ER 关系

```
users (id PK)
  │
  └─── sessions (id PK, user_id FK → users)
         │
         └─── messages (id PK, session_id FK → sessions CASCADE)

chunks (id PK)
  │
  └─── images (id PK, chunk_id FK → chunks)
```

### 12.2 表详情

| 表 | 主键 | 核心字段 | CASCADE |
|----|------|---------|---------|
| `users` | `id` SERIAL | `username` UNIQUE, `created_at` | 需手动关联 |
| `sessions` | `id` VARCHAR(64) | `user_id` FK, `mode`, `title`, `updated_at` | `DELETE` 时级联 messages |
| `messages` | `id` SERIAL | `session_id` FK, `role`, `content` TEXT, `created_at` | 由 sessions CASCADE |
| `chunks` | `id` VARCHAR(64) | `content` TEXT, `source`, `page`, `section_title`, `chunk_type` | — |
| `images` | `id` VARCHAR(64) | `chunk_id` FK, `image_data` BYTEA, `description`, `caption`, `bbox_*` | — |

### 12.3 关键技术细节

- **BYTEA → memoryview**：SQLAlchemy 查询 PG `BYTEA` 列返回 `memoryview`，`st.image()` 无法处理。`get_image()` 中显式 `bytes(result["image_data"])`。
- **Schema 迁移**：`src/migrations.py` 提供 `@migration(version, description)` 装饰器，`database.py` 的 `_ensure_tables()` 会在首次连接时自动执行所有未应用迁移。

---

## 13. 图片管线（Image Pipeline）

### 13.1 为什么需要图片管线？

医学 PDF 中图片通常包含关键信息：病理切片、放射影像、统计图表、诊断流程图。纯文本检索会丢失图片中的信息。

### 13.2 两阶段并发架构

```
Phase 1（同步，导入时立即执行）：
  每一页 PDF:
    ├─ PyMuPDF page.get_images(full=True)
    ├─ 过滤：image_bytes < 5KB → skip（噪声/icon/logo）
    ├─ 寻找 caption：JSON elements 优先 → 空间最近文本块降级
    └─ 写入 PG images（含 image_data）+ Milvus 占位（caption embedding）

Phase 2（异步，Phase 1 完成后执行）：
  ThreadPoolExecutor(8 worker):
    ├─ 图片压缩：RGBA→RGB, max 800px, JPEG quality 80
    ├─ Base64 encode → Qwen 多模态 API
    ├─ Qwen 返回描述（图片类型/关键元素/临床意义）
    ├─ UPDATE PG images.description
    └─ DELETE 旧向量 + INSERT 新向量（Milvus 不支持 vector update）
```

### 13.3 为什么 Milvus 向量要 delete+insert？

Milvus 不支持更新已插入的向量——一旦写入，`chunk_id` 对应的向量就不能原地修改。因此图片描述更新需要先按 `chunk_id` 删除旧向量，再插入新向量。这个操作**不是原子的**——如果插入失败（虽然在 try/except 保护下），会导致该图片在检索中丢失。

---

## 14. 检索问答全流程（端到端）

### 14.1 完整链路

```
用户输入："二甲双胍对肾功能不全患者安全吗？"
    │
    ▼
┌────────────────────────────────────────────────────────────┐
│ 1. 查询改写（query_rewriter.py）                           │
│    ├─ 检测：单轮对话，52 chars → 直通模式                  │
│    └─ 输出：["二甲双胍对肾功能不全患者安全吗？"]             │
└────────────────────────────────────────────────────────────┘
    │
    ▼
┌────────────────────────────────────────────────────────────┐
│ 2. 混合检索（hybrid_search.py）                            │
│    ├─ BM25 路径：全量 chunk 分词匹配 → 返回 top-20         │
│    │   命中："metformin renal impairment safety..."         │
│    ├─ Dense 路径：BGE-M3 Embed → Milvus Cosine → top-40    │
│    │   命中：语义相近的 chunk（即使不含 "metformin"）       │
│    └─ RRF 融合（k=60）→ 去重 → top-20                      │
└────────────────────────────────────────────────────────────┘
    │
    ▼
┌────────────────────────────────────────────────────────────┐
│ 3. 重排序（reranker.py）                                   │
│    ├─ 去重：前 120 字符匹配                                │
│    ├─ FlashRank cross-encoder (query, doc) → score          │
│    └─ 精排 top-5（每个 doc 带 rerank_score）               │
└────────────────────────────────────────────────────────────┘
    │
    ▼
┌────────────────────────────────────────────────────────────┐
│ 4. 意图识别（intent.py）                                   │
│    ├─ LLM 分类："二甲双胍对肾功能不全患者安全吗？"          │
│    └─ → drug_query                                         │
└────────────────────────────────────────────────────────────┘
    │
    ▼
┌────────────────────────────────────────────────────────────┐
│ 5. 上下文构建（rag_chain.py:format_docs()）                │
│    ├─ PG 取原文：get_chunk(chunk_id) → 完整 chunk 文本      │
│    ├─ PG 取图片描述（如有图片 chunk）：get_image(image_id)   │
│    └─ 格式化为：[文献1] 来源: paper3.pdf 第5页  [相关度: ...│
│                 原文: Metformin is generally...            │
└────────────────────────────────────────────────────────────┘
    │
    ▼
┌────────────────────────────────────────────────────────────┐
│ 6. LLM 生成（DeepSeek V4）                                 │
│    ├─ System Prompt: DRUG_QUERY_PROMPT                     │
│    ├─ Context: Top-5 chunks 原文 + 图片描述                 │
│    ├─ Chat History: 前 3 轮对话（如有）                     │
│    ├─ 流式输出 → TokenLoggingCallback 记录 token 量         │
│    └─ 回答中引用 [文献1][文献3] 标注图片来源 [图1]          │
└────────────────────────────────────────────────────────────┘
    │
    ▼
┌────────────────────────────────────────────────────────────┐
│ 7. 后处理（app.py / api.py）                               │
│    ├─ _build_citation_maps()：构建引用映射                  │
│    ├─ _render_citations()：[文献1] → hover 浮窗显示原文     │
│    ├─ _render_images_after_answer()：[图1] → 直接渲染图片   │
│    └─ answer 写入 PG messages 表                            │
└────────────────────────────────────────────────────────────┘
```

### 14.2 如何消除双重检索

**旧架构**（app.py 双重检索）：
```
_pre_retrieve() → 独立调用 retriever → 构建 citation_map
    ↓（检索 1）
chain.invoke() → 内部再检索一次 → 生成回答
    ↓（检索 2，浪费！）
```

**新架构**（消除后）：
```
chain.invoke() → 检索一次（_retrieve_and_rank）
    ↓
get_last_retrieved_docs() → 从 ContextVar 取结果 → 构建 citation_map
```

使用 `ContextVar`（而非模块级 `list`）确保多用户并发时不会互相污染。

---

## 15. 并发安全设计

### 15.1 ContextVar：每请求独立状态

```python
# rag_chain.py
_last_retrieved_docs: ContextVar[list | None] = ContextVar(
    "_last_retrieved_docs", default=None
)
```

`ContextVar` 是 Python 3.7+ 的协程/线程安全上下文变量——每个 `chain.invoke()` 调用栈拥有独立值。Streamlit 每个用户 session 运行在独立线程中，ContextVar 天然隔离。

### 15.2 单例双重检查锁

```python
# vector_store.py
_store: MilvusStore | None = None
_store_lock = threading.Lock()

def get_vector_store() -> MilvusStore:
    global _store
    if _store is None:           # 快速路径（无锁）
        with _store_lock:         # 慢速路径（持锁）
            if _store is None:    # 双重检查
                _store = MilvusStore()
    return _store
```

`get_bm25_retriever()` 和 `get_reranker()` 使用相同模式。

### 15.3 为什么不用连接池的 async？

整个代码库是同步的（无 `async/await`）。原因：
- Streamlit 是同步框架（不支持 async handler）
- Milvus `pymilvus.MilvusClient` 是同步的
- PostgreSQL 使用 SQLAlchemy 同步连接池
- 唯一的并发是图片管线的 `ThreadPoolExecutor`

如果需要高并发 API，FastAPI 层可以使用 `asyncio.to_thread()` 包装同步调用。

---

## 16. 部署架构

### 16.1 Docker Compose 全家桶

```
┌────────────────────────────────────────────────────┐
│                   docker compose up                 │
│                                                    │
│  ┌──────────┐  ┌──────────┐  ┌───────────────┐    │
│  │   etcd   │  │  minio   │  │     milvus    │    │
│  │  :2379   │  │:9000/9001│  │ :19530/:9091  │    │
│  │  协调    │  │ 对象存储  │  │   向量检索     │    │
│  └──────────┘  └──────────┘  └───────┬───────┘    │
│                                      │             │
│  ┌──────────┐              ┌─────────▼──────┐     │
│  │    pg    │              │      app       │     │
│  │  :5432   │◄─────────────│     :8501      │     │
│  │  关系库  │              │  Streamlit UI  │     │
│  └──────────┘              └────────────────┘     │
│                                                    │
│  ┌──────────────────────────────────────────┐     │
│  │  Ollama (host.docker.internal:11434)      │     │
│  │  宿主机运行，不在 Docker 内                │     │
│  └──────────────────────────────────────────┘     │
└────────────────────────────────────────────────────┘
```

### 16.2 三个入口

| 入口 | 命令 | 用途 |
|------|------|------|
| Web UI | `docker compose up` → :8501 | 最终用户交互 |
| CLI | `uv run python main.py ask -q "..."` | 运维/调试单问 |
| REST API | `uv run uvicorn api:app` → :8000 | 系统集成/自动化 |

### 16.3 FastAPI 端点

| 端点 | 方法 | 输入 | 输出 |
|------|------|------|------|
| `/health` | GET | — | 四组件健康状态 JSON |
| `/rag/ask` | POST | `{question, mode?, session_id?, top_k?}` | `{answer, session_id, intent}` |
| `/rag/retrieve` | POST | `{query, top_k?}` | `{results: [{content, source, page, section, chunk_id, score}]}` |
| `/ingest` | POST | `multipart/form-data` (PDF) | `{status, filename, size_bytes}` |

---

## 17. 评估体系

### 17.1 三层指标

| 层级 | 指标 | 说明 |
|------|------|------|
| **检索层** | Recall@k / Precision@k / NDCG@k / MRR / MAP / Hit@k | 评估 chunk 与查询的相关性，含 Bootstrap 95% CI |
| **生成层** | Faithfulness / Answer Relevance / Context Relevance / Hallucination Rate | LLM-as-judge 三提示词评分 |
| **端到端** | ROUGE-L / BLEU-1/4 / Semantic Similarity / Keyword Coverage | 回答与参考答案的字面/语义匹配 |

### 17.2 数据集

105 条手工标注 QA 对：7 篇论文 × 15 题，覆盖 6 种问题类型（factoid/numerical/comparative/multi_hop/summary/terminology）× 3 种难度。

### 17.3 管线对比

```bash
uv run python scripts/run_full_evaluation.py --compare
# 输出：
# baseline (Dense only) vs hybrid (BM25+Dense) vs hybrid+rerank
# 逐指标 @5/@10/@20 对比 + 逐论文切片 + 逐问题类型 + Zero-recall 诊断
```

---

## 18. 已知限制与改进方向

| # | 限制 | 影响 | 改进方向 |
|---|------|------|---------|
| 1 | BM25 索引全量驻内存，无增量更新 | 新增 chunk 需重启进程 | 增量索引或定时重建 |
| 2 | Milvus 向量不支持原地 update | 图片描述更新有短暂窗口期（向量不一致） | 等待 Milvus 支持或改用事务 |
| 3 | FlashRank 模型需 HuggingFace 下载 | 离线环境需手动预置模型 | 模型打包进 Docker image |
| 4 | opendataloader-pdf 需 Java 17+ | Docker image 体积 +200MB | 提供 PyMuPDF-only 精简镜像 |
| 5 | 同步代码库无 async | 高并发 API 场景有瓶颈 | FastAPI + `asyncio.to_thread()` 包装 |
| 6 | 无用户认证（仅有用户名） | 仅适合内网部署 | 接入 OAuth2 / LDAP |
| 7 | 评估数据集为内联 Python 代码 | 1728 行不可维护 | 外置为 JSON/YAML |
| 8 | 缺少单元测试和集成测试 | 重构风险高 | 持续补充测试覆盖 |

---

## 附录 A：配置速查表

所有配置通过 `.env` 文件或环境变量设置。完整列表见 `.env.example`。

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `DEEPSEEK_API_KEY` | (必填) | DeepSeek API 密钥 |
| `DEEpSEEK_MODEL` | `deepseek-chat` | 对话模型 |
| `OLLAMA_BASE_URL` | `http://localhost:11434/v1` | Ollama 服务地址 |
| `EMBEDDING_MODEL_NAME` | `bge-m3` | Embedding 模型 |
| `MILVUS_HOST` | `localhost` | Milvus 地址 |
| `MILVUS_COLLECTION_NAME` | `hospital_knowledge` | Collection 名称 |
| `EMBEDDING_DIMENSION` | `1024` | 向量维度 |
| `CHUNK_SIZE` | `2000` | 分块大小（字符） |
| `CHUNK_OVERLAP` | `100` | 分块重叠（字符） |
| `RETRIEVAL_TOP_K` | `5` | 最终返回 LLM 的文档数 |
| `HYBRID_ENABLED` | `true` | 启用混合检索 |
| `HYBRID_RETRIEVAL_TOP_K` | `20` | 混合检索初筛数量 |
| `RERANKER_ENABLED` | `true` | 启用重排序 |
| `RERANKER_MODEL` | `ms-marco-MiniLM-L-6-v2` | FlashRank 模型 |
| `RERANKER_TOP_N` | `5` | 重排序保留数量 |
| `QUERY_REWRITING_ENABLED` | `true` | 启用查询改写 |
| `PDF_PARSER` | `opendataloader` | PDF 解析引擎 |
| `ENABLE_IMAGE_UNDERSTANDING` | `true` | 启用图片管线 |
| `IMAGE_MAX_CONCURRENT` | `8` | 图片描述并发数 |
| `IMAGE_MIN_BYTES` | `5120` | 最小图片大小（5KB） |
