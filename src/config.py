from pathlib import Path

from dotenv import load_dotenv
from pydantic_settings import BaseSettings

from src.logger import setup_logging

load_dotenv()

# 初始化全局日志
setup_logging()


class Settings(BaseSettings):
    # DeepSeek API
    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com/v1"
    deepseek_model: str = "deepseek-chat"

    # DashScope (Qwen 多模态)
    dashscope_api_key: str = ""
    dashscope_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    dashscope_model: str = "qwen3.7-plus"

    # Ollama Embedding
    ollama_base_url: str = "http://localhost:11434/v1"
    embedding_model_name: str = "bge-m3"

    # Milvus (Docker)
    milvus_host: str = "localhost"
    milvus_port: int = 19530
    milvus_collection_name: str = "hospital_knowledge"
    embedding_dimension: int = 1024

    # PostgreSQL (Docker)
    pg_host: str = "localhost"
    pg_port: int = 5432
    pg_database: str = "hospital_rag"
    pg_user: str = "postgres"
    pg_password: str = "postgres"

    # PDF parser: "opendataloader" | "pymupdf"
    pdf_parser: str = "opendataloader"

    # Chunking — 2000 字符约等于 2-3 个完整英文段落，保证医学语义完整
    chunk_size: int = 2000
    chunk_overlap: int = 100

    # Retrieval
    retrieval_top_k: int = 5  # 最终返回给 LLM 的文档数

    # Hybrid search (Milvus 2.5+ native BM25)
    hybrid_enabled: bool = True
    hybrid_retrieval_top_k: int = 20  # BM25 + Dense 初筛数量

    # Reranker (FlashRank cross-encoder, CPU-friendly)
    reranker_enabled: bool = True
    reranker_model: str = "ms-marco-MiniLM-L-6-v2"
    reranker_top_n: int = 5

    # Query rewriting
    query_rewriting_enabled: bool = True

    # Image understanding (PyMuPDF 提取图片 → Qwen 多模态描述)
    enable_image_understanding: bool = True
    image_max_size: int = 800
    image_max_concurrent: int = 8
    image_min_bytes: int = 5120  # 5KB，过滤噪声小图

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()

# 确保数据目录存在
Path("data/documents").mkdir(parents=True, exist_ok=True)
