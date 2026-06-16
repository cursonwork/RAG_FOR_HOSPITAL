# RAG for Hospital — 项目上下文速览

## 项目概览

面向医疗系统的企业级 RAG 问答系统。核心链路：PDF 医学文献 → opendataloader-pdf 结构化解析 → ODL JSON section-aware 分块 → BGE-M3 向量化 → 查询改写 → 混合检索(BM25+Dense RRF) → FlashRank 重排序 → 意图识别 → DeepSeek V4 生成回答。

- **GitHub**: https://github.com/cursonwork/RAG_FOR_HOSPITAL
- **Python**: 3.13
- **包管理**: uv (pyproject.toml)
- **Java**: 17+ (opendataloader-pdf 依赖)

---

## 架构总览

```
用户界面层     Streamlit (app.py)  /  CLI (main.py)
                    │
意图路由层    intent.py (自动识别 medical_qa / drug_query / diagnosis)
                    │
RAG 链层      conversation.py (多轮)  /  rag_chain.py (单次)
                    │
────────────────── 检索管线（4 步） ──────────────────
  query_rewriter.py   → 查询改写（多轮补全/短问题扩展/长问题回退/直通）
  hybrid_search.py    → BM25 (rank_bm25) + BGE-M3 Dense → RRF 融合 top-20
  reranker.py         → FlashRank cross-encoder 精排 top-5（下载失败自动降级）
────────────────────────────────────────────────────────
                    │                              ↑
Embedding 层   embeddings.py (Ollama bge-m3)       text_splitter.py (ODL JSON section-aware + Markdown fallback)
                    │                              ↑
LLM 层         llm.py (DeepSeek V4)                document_loader.py (opendataloader-pdf + PyMuPDF fallback)
                    │
图片管线        image_pipeline.py (PyMuPDF 提取 → qwen 多模态描述 → 向量化)
                    │
持久化层        database.py (PG 连接池/建表)  +  chat_history.py (PG 历史)
                    │
基础设施        config.py  →  .env (API Key / 连接参数)
                logger.py  →  logs/app.log (RotatingFileHandler)
                prompts.py →  三种 System Prompt（意图路由自动选择）

评估工具         retrieval_eval.py (4 指标 + 20 条医学用例)
                scripts/eval_retrieval.py (baseline vs hybrid vs hybrid+rerank 对比)
                scripts/annotate_chunks.py (分块标注 PDF 生成器，彩色块叠加原文)
```

## 数据流

### PDF 导入流程
```
PDF → opendataloader-pdf（Java 引擎，XY-Cut++ 排序，表格/标题保留）
  → 结构化 Markdown + JSON（kids[] 含 type/heading_level/page/bbox）
  → text_splitter: ODL JSON section-aware grouper（heading level 2-4 切 section，段落边界组合到 chunk_size~2000）
  → chunk_id 写入 PG chunks 表（原文存储）
  → BGE-M3 embed → Milvus（metadata 存 chunk_id + source + section + page）

图片管线（两阶段并发）：
  PyMuPDF 提取图片 → 过滤 <5KB 噪声 → 写 PG images 表 + Milvus 占位（caption 做向量）
  → ThreadPoolExecutor(8) 并发调 qwen 多模态 → UPDATE PG description
  → 删除旧向量 + 插入新向量（Milvus 不支持 update vector）
```

### 问答流程（4 步检索管线）
```
用户输入 → query_rewriter.py（4 策略：多轮补全/短问题扩展/长问题回退/直通）
  → 每个变体 → hybrid_search（BM25 + Dense RRF 融合 top-20）→ 合并去重
  → reranker.py（FlashRank cross-encoder 精排 top-5，不可用时降级为去重+截断）
  → intent.py 意图识别（1次 LLM，~170 tokens）
  → 选择对应的 System Prompt（medical_qa / drug_query / diagnosis）
  → format_docs() 拼上下文（原文 + hover 引用标记 + rerank_score + 图片描述）
  → DeepSeek V4 流式生成 → TokenLoggingCallback 记录 token
  → app.py 后处理：文献引用 hover 浮窗 + 图片渲染（复用 chain 内检索结果，无双重检索）
```

### PostgreSQL 表结构（database.py:_ensure_tables() 自动建表）
```
users     (id SERIAL PK, username UNIQUE, created_at)
sessions  (id VARCHAR(64) PK, user_id FK→users, mode, title, created_at, updated_at)
messages  (id SERIAL PK, session_id FK→sessions CASCADE, role, content, created_at)
chunks    (id VARCHAR(64) PK, content TEXT, source, page, section_title, chunk_index, chunk_type, created_at)
images    (id VARCHAR(64) PK, chunk_id FK→chunks, image_data BYTEA, image_format, description, caption,
           source, page, bbox_x0/y0/x1/y1, created_at)
```

---

## 关键技术决策与踩坑记录

### 1. opendataloader-pdf 替代 PyMuPDF get_text()
**问题**: PyMuPDF `get_text()` 双栏混乱、表格无结构、页眉页脚混入、标题层级丢失。
**决策**: 用 opendataloader-pdf（Java 引擎，XY-Cut++ 阅读顺序，markdown 输出含表格/标题/列表），PyMuPDF 作为 fallback。`config.py:pdf_parser` 可切换。
**代价**: 需要 Java 17+（Docker 已包含）。

### 2. ODL JSON section-aware 分块（v2，替代旧 Markdown 分块）
**问题 v1**: 旧方案仅看 Markdown `#/##/###` 字符串匹配，所有 chunk 的 section_title 都是论文标题，且 512 字符硬切导致句子截断。
**决策 v2**: 遍历 opendataloader JSON `kids[]` 数组的 `type`/`heading_level` 字段——level 2-4 切新 section，level 1（Doctitle）取为 section 名，level 5+（作者名）归入当前 section。段落边界组合到 chunk_size 上限。chunk_size 512→2000。
**效果**: paper1 22 页从 209 chunks 降到 54 chunks（↓74%），section_title 正确率 0%→100%。
**代码**: `text_splitter.py:_split_by_odl_elements()`

### 3. 混合检索：客户端 BM25 + RRF 融合（非 Milvus 原生 BM25）
**问题**: Milvus 2.5.10 原生 BM25（SPARSE_FLOAT_VECTOR + FunctionType.BM25）schema 可建、insert 可跑，但 query 端 `RunAnalyzer` RPC 未实现——MilvusClient 无法将 raw query text 编码为 sparse vector。
**决策**: 客户端 `rank_bm25` 做 BM25——全量 chunk 文本常驻内存（分词后 ~1MB/chunk），RRF 融合 BM25 + Milvus 稠密向量。54 chunks 规模完全够用，未来 >10 万 chunks 时等 Milvus 补齐再迁移。
**代码**: `hybrid_search.py`

### 4. 重排序：FlashRank（ONNX CPU），不可用自动降级
**问题**: 无 GPU，huggingface 可能不可达（下载模型失败）。
**决策**: FlashRank `ms-marco-MiniLM-L-6-v2`（4MB ONNX），CPU 友好 <10ms/doc。`compress_documents()` 内 try/except——下载失败自动降级为去重+截断。部署时手动 `wget` 模型到 `~/.cache/flashrank/` 即可启用。
**代码**: `reranker.py`

### 5. 查询改写：4 策略路由
**问题**: 多轮对话中 retriever 只看到当前问题（如 "这个药的副作用呢？"），丢失历史上下文。
**决策**: 4 策略——多轮→历史补全指代；短问题(<25chars)→3 同义变体；长问题(>120chars)→步骤回退生成上位问题；正常长度→直通。全部用 DeepSeek temp=0，确定性输出。
**代码**: `query_rewriter.py`

### 6. Ollama Embedding 批大小限制
**问题**: Ollama bge-m3 tokenizer 大批量文本时 `connection reset by peer`。
**决策**: `chunk_size=20`，限制每次 API 调用 20 条文本。代码见 `embeddings.py`。

### 7. tiktoken 校验
**问题**: `langchain-openai` 的 `OpenAIEmbeddings` 默认用 tiktoken 校验输入长度，bge-m3 不在列表里。
**决策**: `check_embedding_ctx_length=False`。代码见 `embeddings.py`。

### 8. Token 统计：usage_metadata 不在 llm_output
**问题**: LangChain callback 的 `response.llm_output` 在 DeepSeek V4 响应中为 None。
**决策**: 从 `message.usage_metadata` 提取 input_tokens / output_tokens / total_tokens。代码见 `callbacks.py`。

### 9. PG BYTEA 返回 memoryview 而非 bytes
**问题**: SQLAlchemy 查询 PG `BYTEA` 列返回 `memoryview`，`st.image()` 无法处理。
**决策**: `get_image()` 中显式 `bytes(result["image_data"])`。代码见 `database.py`。

### 10. Prompt 模板里 {context} 用 str.replace 不可以用 .format()
**问题**: prompt 同时包含 `{context}` 和 `{question}`，.format(context=...) 会因 `{question}` 未传而 KeyError。
**决策**: `.replace("{context}", ctx)` 只替换已知占位符。代码见 `conversation.py:_route()`。

### 11. app.py 双重检索消除
**问题**: `_pre_retrieve()` 独立调用 retriever 构建 citation_map，chain.invoke 内又检索一次——每次用户提问检索两次。
**决策**: 删除 `_pre_retrieve()`。chain 检索完成后从 `rag_chain.get_last_retrieved_docs()` 取结果构建 citation_map。代码见 `app.py:_build_citation_maps()`。

---

## 文件清单

| 文件 | 职责 | 关键函数/类 |
|------|------|-----------|
| `src/config.py` | pydantic-settings，从 .env 加载所有配置 | `Settings` |
| `src/logger.py` | 日志系统：文件轮转(10MB×5) + 控制台，抑制第三方噪音 | `setup_logging()`, `get_logger()` |
| `src/embeddings.py` | Ollama bge-m3 封装 | `create_embeddings()` → `OpenAIEmbeddings` |
| `src/llm.py` | DeepSeek V4 封装 | `create_llm()` → `ChatOpenAI` |
| `src/document_loader.py` | PDF→Markdown（opendataloader-pdf 优先，PyMuPDF fallback）；metadata 存 odl_elements | `load_pdf()`, `load_pdfs()` |
| `src/text_splitter.py` | ODL JSON section-aware 分块（v2） + Markdown fallback | `create_semantic_splitter()`, `_split_by_odl_elements()`, `_split_by_markdown()` |
| `src/image_pipeline.py` | PDF 图片提取 + qwen 多模态描述（两阶段并发） | `save_image_placeholders()`, `fill_image_descriptions()` |
| `src/vector_store.py` | Milvus 手写封装 + hybrid_search（委托 hybrid_search.py） | `MilvusStore`, `similarity_search()`, `hybrid_search()`, `as_hybrid_retriever()` |
| `src/hybrid_search.py` | 客户端 BM25 (rank_bm25) + Dense RRF 融合 | `BM25SparseRetriever`, `hybrid_search()`, `_rrf_fusion()` |
| `src/reranker.py` | FlashRank cross-encoder 精排，下载失败自动降级 | `FlashRankReranker`, `get_reranker()` |
| `src/query_rewriter.py` | 4 策略查询改写（多轮补全/短问题扩展/长问题回退/直通） | `rewrite_query()` |
| `src/retrieval_eval.py` | 检索评估：Recall@k / Precision@k / NDCG@k / MRR + 20 条医学用例 | `run_eval()`, `compare_pipelines()`, `load_paper1_queries()` |
| `src/intent.py` | 意图识别：自动分诊 medical_qa / drug_query / diagnosis | `classify_intent()` |
| `src/rag_chain.py` | 单次问答链（含完整检索管线 + 意图路由） | `create_rag_chain()`, `_retrieve_and_rank()`, `format_docs()` |
| `src/conversation.py` | 多轮对话链（history 传入查询改写 + PG 持久化） | `create_conversational_chain()`, `_retrieve_with_history()` |
| `src/prompts.py` | 三种 System Prompt + get_system_prompt() | `get_system_prompt()`, `SYSTEM_PROMPTS` |
| `src/database.py` | PG 连接池 + 自动建表（5表） + chunks/images CRUD | `save_chunk()`, `get_chunk()`, `save_image()`, `get_image()`, `clear_all_chunks()` |
| `src/chat_history.py` | PG 版聊天历史（LangChain 兼容） | `PostgresChatMessageHistory` |
| `src/callbacks.py` | LLM 回调：Token 统计 + 响应内容日志 | `TokenLoggingCallback` |
| `app.py` | Streamlit 前端：多用户 + 会话管理 + 自动意图 + hover 引用 + 图片展示（无双重检索） | — |
| `main.py` | CLI 入口：`ingest --clear` / `ask` (mode 可选) / `serve` | `cmd_ingest()`, `cmd_ask()`, `cmd_serve()` |
| `Dockerfile` | Python 3.13 + Java 17 + 项目依赖 | — |
| `docker-compose.yml` | etcd + minio + milvus + pg + app 全家桶 | — |
| `scripts/annotate_chunks.py` | 分块标注 PDF 生成器：彩色块叠加原 PDF + 图例页 | — |
| `scripts/eval_retrieval.py` | 检索评估脚本：baseline vs hybrid vs hybrid+rerank 三线对比 | — |
| `.streamlit/config.toml` | 禁用文件监控 | `fileWatcherType = "none"` |

---

## 环境启动检查清单

1. **Docker**: `docker compose up -d etcd minio milvus pg`
2. **Ollama**: 确保 `ollama serve` 运行中，`ollama pull bge-m3` 已下载
3. **Java**: 本地需 JDK 17+（macOS: `brew install openjdk@17`），Docker 内已包含
4. **.env**: 项目根目录创建，填入 `DEEPSEEK_API_KEY` + `DASHSCOPE_API_KEY` + PG/Milvus 参数
5. **依赖**: `uv sync`
6. **导入文档**: `uv run python main.py ingest --path data/documents/ --clear`
7. **评估检索**: `uv run python scripts/eval_retrieval.py`
8. **CLI 问答**: `uv run python main.py ask --question "你的问题"`
9. **启动 Web**: `uv run python main.py serve` → http://localhost:8501

---

## 三种模式（自动意图识别）

系统自动识别意图，无需手动切换：

| 模式 | 触发条件 | 适用场景 |
|------|---------|---------|
| 医疗问答 | 疾病机制/病理/解剖/流行病学 | 通用医学知识 |
| 药物查询 | 药名/适应症/不良反应/禁忌 | 药物信息 |
| 辅助诊断 | 症状/检查结果推断疾病 | 鉴别诊断 |

CLI 可通过 `--mode` 手动覆盖自动识别。

---

## 检索管线配置

所有开关在 `.env` / `config.py` 中：

| 配置项 | 默认值 | 作用 |
|--------|--------|------|
| `HYBRID_ENABLED` | true | 启用 BM25 + Dense 混合检索 |
| `HYBRID_RETRIEVAL_TOP_K` | 20 | 混合检索初筛数量 |
| `RERANKER_ENABLED` | true | 启用 FlashRank 重排序 |
| `RERANKER_MODEL` | ms-marco-MiniLM-L-6-v2 | FlashRank 模型 |
| `RERANKER_TOP_N` | 5 | 重排序后保留数量 |
| `QUERY_REWRITING_ENABLED` | true | 启用 4 策略查询改写 |
| `RETRIEVAL_TOP_K` | 5 | 最终返回给 LLM 的文档数 |
| `CHUNK_SIZE` | 2000 | 分块最大字符数 |
| `CHUNK_OVERLAP` | 100 | 分块重叠字符数 |

---

## 日志系统

- 根 logger: `hospital_rag`，子模块: `hospital_rag.src.xxx`
- 文件: `logs/app.log`，10MB 轮转，保留 5 个历史
- 控制台: INFO+，文件: DEBUG+
- Token 统计: `TokenLoggingCallback.on_llm_end()` 记录每次 LLM 调用的 input_tokens / output_tokens / total_tokens
- 所有异常都用 `logger.exception()` 自动带 traceback

---

## 已知待改进

- [ ] `RunnableWithMessageHistory` 被 LangChain 标记为 deprecated，未来迁移 LangGraph
- [ ] FlashRank 模型需手动下载到 `~/.cache/flashrank/`（huggingface 不可达时自动降级为去重+截断）
- [ ] 多文档知识库（>10 篇论文）尚未充分测试检索效果
- [ ] Milvus 原生 BM25 待其补齐 query 端 text→sparse 编码后迁移（当前 rank_bm25 客户端方案 OK）
- [ ] PDF 表格的特殊提取可进一步优化
- [ ] 缺少单元测试和集成测试
- [ ] 用户认证目前仅用户名输入（无密码），仅适合内网部署
- [ ] 图片向量更新目前是 delete+insert（Milvus 不支持 update vector），大量图片时需优化
