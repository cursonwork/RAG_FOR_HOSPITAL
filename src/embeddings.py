from langchain_openai import OpenAIEmbeddings

from src.config import settings
from src.logger import get_logger

logger = get_logger(__name__)


def create_embeddings() -> OpenAIEmbeddings:
    """通过 Ollama 的 OpenAI 兼容 API 创建 Embedding 实例。"""
    logger.info("初始化 Embedding 模型: %s (Ollama)", settings.embedding_model_name)
    return OpenAIEmbeddings(
        model=settings.embedding_model_name,
        base_url=settings.ollama_base_url,
        api_key="ollama",
        check_embedding_ctx_length=False,
        chunk_size=20,
    )
