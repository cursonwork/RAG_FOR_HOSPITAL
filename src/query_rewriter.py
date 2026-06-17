"""查询改写器：提升检索召回率的查询预处理。

策略矩阵：
    - 多轮对话 → 基于历史补全省略/指代，生成 1 个独立可检索的问题
    - 单轮短问题 (<25 chars) → 生成 3 个同义/细化变体
    - 单轮正常 (25-120 chars) → 直通，不做改写
    - 单轮长问题 (>120 chars) → 步骤回退，生成 1 个更宽泛的上位问题 + 原问题

全部用 DeepSeek（temperature=0）一次调用完成，确定性输出。
"""

from src.config import settings
from src.llm import create_llm
from src.logger import get_logger

logger = get_logger(__name__)

# ── Prompt 模板 ──

STEP_BACK_PROMPT = """You are a medical research assistant. Given a detailed clinical question,
write a more general, broader question that captures the underlying medical concept.

Detailed question: {question}
Broader question:"""

MULTI_QUERY_PROMPT = """You are a medical search expert. Generate 3 alternative search queries
for the medical question below. Each variant should use different terminology, synonyms, or focus
on different aspects. Output ONLY the 3 queries, one per line, no numbering, no explanation.

Original: {question}
Variants:"""

HISTORY_REWRITE_PROMPT = """You are a medical conversation assistant. Given the conversation
history and the user's latest question, rewrite the latest question into a self-contained query
that can be understood without the conversation context. If the user uses pronouns (e.g., "it",
"this drug", "the patient"), replace them with the specific entities from the history.
Output ONLY the rewritten question, nothing else.

Conversation history:
{history}

Latest question: {question}
Rewritten question:"""


def _build_history_str(history: list) -> str:
    """将 LangChain 消息列表转为可读字符串。"""
    lines = []
    for msg in history[-6:]:  # 最近 3 轮对话
        role = "User" if msg.__class__.__name__ == "HumanMessage" else "Assistant"
        content = msg.content[:300] if hasattr(msg, "content") else str(msg)[:300]
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def rewrite_query(
    question: str,
    history: list | None = None,
) -> list[str]:
    """改写用户问题，返回 1-3 个检索用查询变体。

    Args:
        question: 原始用户问题
        history: 可选，LangChain 消息列表（多轮对话上下文）

    Returns:
        变体列表，至少包含原始问题
    """
    if not settings.query_rewriting_enabled:
        return [question]

    try:
        llm = create_llm(temperature=0.0)

        # ── 多轮对话：历史感知改写 ──
        if history and len(history) >= 2:
            history_str = _build_history_str(history)
            rewritten = llm.invoke(
                HISTORY_REWRITE_PROMPT.format(history=history_str, question=question)
            ).content.strip()
            if rewritten and rewritten != question and len(rewritten) >= 5:
                logger.debug("查询改写 (多轮): %s", rewritten)
                return [rewritten]
            return [question]

        # ── 单轮短问题：多查询扩展 ──
        if len(question) < 25:
            variants_str = llm.invoke(MULTI_QUERY_PROMPT.format(question=question)).content.strip()
            variants = [v.strip() for v in variants_str.split("\n") if v.strip()]
            # 过滤明显不好的变体
            variants = [v for v in variants if 5 <= len(v) <= 300]
            if len(variants) >= 2:
                all_qs = [question] + variants[:3]
                logger.debug("查询改写 (多查询): %d 个变体", len(all_qs))
                return all_qs
            return [question]

        # ── 单轮长问题：步骤回退 ──
        if len(question) > 120:
            broader = llm.invoke(STEP_BACK_PROMPT.format(question=question)).content.strip()
            if broader and broader != question and 10 <= len(broader) <= 300:
                logger.debug("查询改写 (回退): %s", broader)
                return [broader, question]
            return [question]

        # ── 单轮正常：直通 ──
        return [question]

    except Exception:
        logger.exception("查询改写失败，使用原始问题")
        return [question]
