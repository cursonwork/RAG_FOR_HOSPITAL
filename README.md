# RAG for Hospital

面向医疗系统的企业级 RAG 问答系统（MVP 阶段）。

## 技术栈

| 组件 | 技术 |
|------|------|
| LLM | DeepSeek V4 (API) |
| Embedding | BAAI/bge-m3 (本地) |
| 向量库 | Milvus Lite |
| 框架 | LangChain 1.3+ |
| 界面 | Streamlit 1.58+ |

## 快速开始

### 1. 环境配置

```bash
cp .env.example .env
# 编辑 .env，填入 DEEPSEEK_API_KEY
```

### 2. 安装依赖

```bash
uv sync
```

### 3. 导入文档

```bash
# 将 PDF 放入 data/documents/ 后执行
uv run python main.py ingest --path data/documents/
```

首次运行会自动下载 BGE-M3 模型（约 2.2GB），请确保网络通畅。

### 4. 启动应用

```bash
uv run python main.py serve
```

浏览器打开 http://localhost:8501

### 5. CLI 问答（不启动界面）

```bash
uv run python main.py ask --question "高血压的一线治疗药物有哪些？"
```

## 三种模式

- **医疗问答** — 通用医学知识问答，引用文献来源
- **药物查询** — 药物信息查询（适应症、用法、禁忌等）
- **辅助诊断** — 鉴别诊断分析参考

## 项目结构

```
├── src/
│   ├── config.py           # 配置管理
│   ├── document_loader.py  # PDF 加载（PyMuPDF）
│   ├── text_splitter.py    # 中文分块策略
│   ├── embeddings.py       # BGE-M3 封装
│   ├── vector_store.py     # Milvus Lite 向量存储
│   ├── llm.py              # DeepSeek V4 封装
│   ├── rag_chain.py        # RAG 链
│   ├── conversation.py     # 多轮对话
│   └── prompts.py          # 系统提示词
├── app.py                  # Streamlit 前端
├── main.py                 # CLI 入口
└── data/
    ├── documents/          # PDF 存放目录
    └── vector_db/          # 向量库数据
```

## 免责声明

本系统仅供医学信息参考，不构成诊断或治疗建议。
