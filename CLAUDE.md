# RAG for Hospital — 项目上下文速览

## 项目概览

面向医疗系统的企业级 RAG 问答系统。核心链路：PDF 医学文献 → opendataloader-pdf 结构化解析 → Markdown 语义分块 → BGE-M3 向量化 → Milvus 检索 → 意图识别 → DeepSeek V4 生成回答。

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
检索层         vector_store.py (MilvusClient 手写封装)
                    │                              ↑
Embedding 层   embeddings.py (Ollama bge-m3)       text_splitter.py (Markdown 标题语义分块)
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
```

## 数据流

### PDF 导入流程
```
PDF → opendataloader-pdf（Java 引擎，XY-Cut++ 排序，表格/标题保留）
  → 结构化 Markdown → MarkdownHeaderTextSplitter（按 #/##/### 切）
  → RecursiveCharacterTextSplitter（大段二次限制 ~512 字符）
  → chunk_id 写入 PG chunks 表（原文存储）
  → BGE-M3 embed → Milvus（metadata 存 chunk_id + source + section）

图片管线（两阶段并发）：
  PyMuPDF 提取图片 → 过滤 <5KB 噪声 → 写 PG images 表 + Milvus 占位（caption 做向量）
  → ThreadPoolExecutor(8) 并发调 qwen 多模态 → UPDATE PG description
  → 删除旧向量 + 插入新向量（Milvus 不支持 update vector）
```

### 问答流程
```
用户输入 → intent.py 意图识别（1次 LLM，~170 tokens）
  → 选择对应的 System Prompt（medical_qa / drug_query / diagnosis）
  → BGE-M3 embed → Milvus search(top_k=3) → PG 查原文/图片
  → format_docs() 拼上下文（原文 + hover 引用标记 + 图片描述）
  → DeepSeek V4 流式生成 → TokenLoggingCallback 记录 token
  → app.py 后处理：文献引用 hover 浮窗 + 图片渲染
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

### 2. Markdown 语义分块替代固定 512 字符
**问题**: 固定长度分块跨越章节边界，上下文碎片化。
**决策**: 两阶段——MarkdownHeaderTextSplitter（#/##/### 标题边界）→ RecursiveCharacterTextSplitter（大段二次限制 512）。每个 chunk 带 section_title 元数据。

### 3. 图片管线：两阶段并发
**问题**: 144 张图片串行调 qwen，需 70 分钟；API 波动致零星失败。
**决策**: 第一阶段快速占位（caption 做向量），第二阶段 ThreadPoolExecutor(max_workers=8) 并发生成描述 + UPDATE。~8 分钟完成。小图 <5KB 自动过滤。

### 4. 为什么不用 langchain-milvus 的 ORM 封装？
**问题**: `langchain-milvus` 内部用 `Collection` (ORM API)，需要 pymilvus `connections` 注册全局连接，与 MilvusClient 动态 alias 耦合。
**决策**: 手写 `MilvusStore(VectorStore)`，只用 `pymilvus.MilvusClient`。代码见 `vector_store.py`。

### 5. Ollama Embedding 批大小限制
**问题**: Ollama bge-m3 tokenizer 大批量文本时 `connection reset by peer`。
**决策**: `chunk_size=20`，限制每次 API 调用 20 条文本。代码见 `embeddings.py`。

### 6. tiktoken 校验
**问题**: `langchain-openai` 的 `OpenAIEmbeddings` 默认用 tiktoken 校验输入长度，bge-m3 不在列表里。
**决策**: `check_embedding_ctx_length=False`。代码见 `embeddings.py`。

### 7. RunnableParallel 链必须用 RunnableLambda 包装自定义函数
**问题**: `dict | function` 在 LangChain 1.x 中报 `TypeError`。
**决策**: 用 `RunnableParallel(...) | RunnableLambda(func)`。代码见 `rag_chain.py`、`conversation.py`。

### 8. Token 统计：usage_metadata 不在 llm_output
**问题**: LangChain callback 的 `response.llm_output` 在 DeepSeek V4 响应中为 None。
**决策**: 从 `message.usage_metadata` 提取 input_tokens / output_tokens / total_tokens。代码见 `callbacks.py`。

### 9. Milvus auto_id 不接受手动 id
**问题**: `auto_id=True` 时手动传 string uuid 会 `DataNotMatchException`。
**决策**: 不传 id 字段。代码见 `vector_store.py`。

### 10. PG BYTEA 返回 memoryview 而非 bytes
**问题**: SQLAlchemy 查询 PG `BYTEA` 列返回 `memoryview`，`st.image()` 无法处理。
**决策**: `get_image()` 中显式 `bytes(result["image_data"])`。代码见 `database.py`。

### 11. Prompt 模板里 {context} 用 str.replace 不可以用 .format()
**问题**: prompt 同时包含 `{context}` 和 `{question}`，.format(context=...) 会因 `{question}` 未传而 KeyError。
**决策**: `.replace("{context}", ctx)` 只替换已知占位符。代码见 `conversation.py:_route()`。

### 12. 图片向量更新不能 INSERT 已存在的 chunk_id
**问题**: `update_image_vectors` 调了 `save_chunk`（INSERT），PG 主键冲突。
**决策**: 新增 `update_chunk_content()` 用 UPDATE。代码见 `database.py`。

---

## 文件清单

| 文件 | 职责 | 关键函数/类 |
|------|------|-----------|
| `src/config.py` | pydantic-settings，从 .env 加载所有配置 | `Settings` |
| `src/logger.py` | 日志系统：文件轮转(10MB×5) + 控制台，抑制第三方噪音 | `setup_logging()`, `get_logger()` |
| `src/embeddings.py` | Ollama bge-m3 封装 | `create_embeddings()` → `OpenAIEmbeddings` |
| `src/llm.py` | DeepSeek V4 封装 | `create_llm()` → `ChatOpenAI` |
| `src/document_loader.py` | PDF→Markdown（opendataloader-pdf 优先，PyMuPDF fallback） | `load_pdf()`, `load_pdfs()`, `load_pdf_opendataloader()`, `load_pdf_pymupdf()` |
| `src/text_splitter.py` | Markdown 标题语义分块 + 二级大小限制 | `create_semantic_splitter()`, `create_text_splitter()` |
| `src/image_pipeline.py` | PDF 图片提取 + qwen 多模态描述（两阶段并发） | `save_image_placeholders()`, `fill_image_descriptions()`, `describe_image()` |
| `src/vector_store.py` | Milvus 手写封装（chunk_id 引用 PG）+ 图片占位/更新 | `MilvusStore`, `add_image_placeholders()`, `update_image_vectors()`, `delete_all()` |
| `src/intent.py` | 意图识别：自动分诊 medical_qa / drug_query / diagnosis | `classify_intent()` |
| `src/rag_chain.py` | 单次问答链（含意图路由） | `create_rag_chain(mode=None)`, `format_docs()` |
| `src/conversation.py` | 多轮对话链（PG 持久化 + 意图路由） | `create_conversational_chain(mode=None)` |
| `src/prompts.py` | 三种 System Prompt + get_system_prompt() | `get_system_prompt()`, `SYSTEM_PROMPTS` |
| `src/database.py` | PG 连接池 + 自动建表（5表） + chunks/images CRUD | `save_chunk()`, `get_chunk()`, `save_image()`, `get_image()`, `update_chunk_content()`, `update_image_description()`, `clear_all_chunks()` |
| `src/chat_history.py` | PG 版聊天历史（LangChain 兼容） | `PostgresChatMessageHistory` |
| `src/callbacks.py` | LLM 回调：Token 统计 + 响应内容日志 | `TokenLoggingCallback` |
| `app.py` | Streamlit 前端：多用户 + 会话管理 + 自动意图 + hover 引用 + 图片展示 | — |
| `main.py` | CLI 入口：`ingest --clear` / `ask` (mode 可选) / `serve` | `cmd_ingest()`, `cmd_ask()`, `cmd_serve()` |
| `Dockerfile` | Python 3.13 + Java 17 + 项目依赖 | — |
| `docker-compose.yml` | etcd + minio + milvus + pg + app 全家桶 | — |
| `.streamlit/config.toml` | 禁用文件监控 | `fileWatcherType = "none"` |

---

## 环境启动检查清单

1. **Docker**: `docker compose up -d etcd minio milvus pg`
2. **Ollama**: 确保 `ollama serve` 运行中，`ollama pull bge-m3` 已下载
3. **Java**: 本地需 JDK 17+（macOS: `brew install openjdk@17`），Docker 内已包含
4. **.env**: 项目根目录创建，填入 `DEEPSEEK_API_KEY` + `DASHSCOPE_API_KEY` + PG/Milvus 参数
5. **依赖**: `uv sync`
6. **导入文档**: `uv run python main.py ingest --path data/documents/ --clear`
7. **CLI 问答**: `uv run python main.py ask --question "你的问题"`
8. **启动 Web**: `uv run python main.py serve` → http://localhost:8501

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

## 日志系统

- 根 logger: `hospital_rag`，子模块: `hospital_rag.src.xxx`
- 文件: `logs/app.log`，10MB 轮转，保留 5 个历史
- 控制台: INFO+，文件: DEBUG+
- Token 统计: `TokenLoggingCallback.on_llm_end()` 记录每次 LLM 调用的 input_tokens / output_tokens / total_tokens
- 所有异常都用 `logger.exception()` 自动带 traceback

---

## 已知待改进

- [ ] `RunnableWithMessageHistory` 被 LangChain 标记为 deprecated，未来迁移 LangGraph
- [ ] 检索目前仅向量相似度，后续可加 BM25 混合检索 + Reranker
- [ ] PDF 表格的特殊提取可进一步优化
- [ ] 缺少测试用例
- [ ] 用户认证目前仅用户名输入（无密码），仅适合内网部署
- [ ] 图片向量更新目前是 delete+insert（Milvus 不支持 update vector），大量图片时需优化
