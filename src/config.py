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

    # PostgreSQL (Docker)
    pg_host: str = "localhost"
    pg_port: int = 5432
    pg_database: str = "hospital_rag"
    pg_user: str = "postgres"
    pg_password: str = "postgres"

    # PDF parser: "opendataloader" | "pymupdf"
    pdf_parser: str = "opendataloader"

    # Chunking
    chunk_size: int = 512
    chunk_overlap: int = 50

    # Retrieval
    retrieval_top_k: int = 3

    # Image understanding
    enable_image_understanding: bool = True
    image_max_size: int = 800

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()

# 确保数据目录存在
Path("data/documents").mkdir(parents=True, exist_ok=True)
