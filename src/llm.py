from langchain_openai import ChatOpenAI

from src.config import settings
from src.logger import get_logger

logger = get_logger(__name__)


def create_llm(temperature: float = 0.3) -> ChatOpenAI:
    """创建 DeepSeek V4 LLM 实例（OpenAI 兼容接口）。"""
    logger.info("初始化 LLM: %s (temperature=%.1f)", settings.deepseek_model, temperature)
    return ChatOpenAI(
        model=settings.deepseek_model,
        api_key=settings.deepseek_api_key,
        base_url=settings.deepseek_base_url,
        temperature=temperature,
        streaming=True,
    )
