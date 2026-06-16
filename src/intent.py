"""意图识别：根据用户输入自动判断属于医疗问答、药物查询还是辅助诊断。

使用 DeepSeek 做一次性轻量分类，max_tokens 限制在 10，极低延迟。
"""

from src.llm import create_llm
from src.logger import get_logger

logger = get_logger(__name__)

INTENT_PROMPT = """判断以下用户问题的意图类别，只回复一个标签，不要解释。

类别：
- medical_qa: 通用医学知识问答（疾病机制、病理、解剖、检查方法、流行病学等）
- drug_query: 药物相关查询（药品名称、适应症、用法用量、不良反应、禁忌、相互作用等）
- diagnosis: 辅助诊断/鉴别诊断（根据症状或检查结果推断可能的疾病）

用户问题：{question}

标签："""

INTENT_LABELS = {"medical_qa", "drug_query", "diagnosis"}


def classify_intent(question: str) -> str:
    """返回 'medical_qa' | 'drug_query' | 'diagnosis'。"""
    llm = create_llm(temperature=0)
    try:
        result = llm.invoke(INTENT_PROMPT.format(question=question))
        label = result.content.strip().lower() if hasattr(result, "content") else str(result).strip().lower()
        for key in INTENT_LABELS:
            if key in label:
                logger.debug("意图识别: %s → %s", question[:80], key)
                return key
        logger.debug("意图识别未匹配，默认 medical_qa: %s", label)
        return "medical_qa"
    except Exception:
        logger.exception("意图识别失败，默认 medical_qa")
        return "medical_qa"
