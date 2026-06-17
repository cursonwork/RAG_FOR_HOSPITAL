"""健康检查模块：对各依赖服务做连通性探测。"""

from src.config import settings
from src.logger import get_logger

logger = get_logger(__name__)


def check_milvus() -> dict:
    """Ping Milvus。"""
    try:
        from pymilvus import MilvusClient

        client = MilvusClient(host=settings.milvus_host, port=settings.milvus_port)
        ok = client.has_collection(settings.milvus_collection_name)
        client.close()
        return {"status": "ok" if ok else "no_collection", "host": f"{settings.milvus_host}:{settings.milvus_port}"}
    except Exception as e:
        logger.warning("Milvus health check failed: %s", e)
        return {"status": "unreachable", "error": str(e)}


def check_postgres() -> dict:
    """Ping PostgreSQL。"""
    try:
        from sqlalchemy import text

        from src.database import get_engine

        engine = get_engine()
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return {"status": "ok", "host": f"{settings.pg_host}:{settings.pg_port}"}
    except Exception as e:
        logger.warning("PostgreSQL health check failed: %s", e)
        return {"status": "unreachable", "error": str(e)}


def check_ollama() -> dict:
    """Ping Ollama 嵌入服务。"""
    try:
        from langchain_openai import OpenAIEmbeddings

        emb = OpenAIEmbeddings(
            model=settings.embedding_model_name,
            base_url=settings.ollama_base_url,
            api_key="ollama",
            check_embedding_ctx_length=False,
        )
        emb.embed_query("health check")
        return {"status": "ok", "model": settings.embedding_model_name}
    except Exception as e:
        logger.warning("Ollama health check failed: %s", e)
        return {"status": "unreachable", "error": str(e)}


def check_deepseek() -> dict:
    """Ping DeepSeek API。"""
    try:
        from openai import OpenAI

        client = OpenAI(api_key=settings.deepseek_api_key, base_url=settings.deepseek_base_url)
        client.models.list()
        return {"status": "ok", "model": settings.deepseek_model}
    except Exception as e:
        logger.warning("DeepSeek health check failed: %s", e)
        return {"status": "unreachable", "error": str(e)}


def health_check() -> dict:
    """聚合所有依赖服务的健康状态。"""
    return {
        "milvus": check_milvus(),
        "postgres": check_postgres(),
        "ollama": check_ollama(),
        "deepseek": check_deepseek(),
    }
