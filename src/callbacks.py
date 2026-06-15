"""LangChain 回调：Token 统计与响应内容日志。"""

from langchain_core.callbacks import BaseCallbackHandler

from src.logger import get_logger

logger = get_logger(__name__)


class TokenLoggingCallback(BaseCallbackHandler):
    """记录每次 LLM 调用的 token 消耗和响应内容。"""

    def on_llm_end(self, response, **kwargs) -> None:
        if response.generations:
            msg = response.generations[0][0].message
            usage = getattr(msg, "usage_metadata", None) or {}

            if usage:
                logger.info(
                    "Token 消耗: prompt=%d, completion=%d, total=%d",
                    usage.get("input_tokens", 0),
                    usage.get("output_tokens", 0),
                    usage.get("total_tokens", 0),
                )

            content = msg.content if hasattr(msg, "content") else ""
            if content:
                logger.debug("LLM 响应 (%d 字符): %s", len(str(content)), str(content)[:500])
