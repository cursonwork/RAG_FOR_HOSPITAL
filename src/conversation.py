from operator import itemgetter

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnableSerializable

from src.chat_history import PostgresChatMessageHistory
from src.intent import classify_intent
from src.llm import create_llm
from src.logger import get_logger
from src.prompts import get_system_prompt
from src.rag_chain import _retrieve_and_rank, format_docs

logger = get_logger(__name__)


def _load_history(session_id: str) -> list:
    """从 PG 加载会话历史，转为 LangChain message 对象列表。"""
    pg_history = PostgresChatMessageHistory(session_id=session_id)
    messages = []
    for msg in pg_history.messages:
        if msg.type == "human":
            messages.append(HumanMessage(content=msg.content))
        elif msg.type == "ai":
            messages.append(AIMessage(content=msg.content))
    logger.debug("会话 %s: 历史消息 %d 条", session_id, len(messages))
    return messages


def _save_turn(session_id: str, question: str, answer: str) -> None:
    """保存本轮对话到 PG。"""
    pg_history = PostgresChatMessageHistory(session_id=session_id)
    pg_history.add_user_message(question)
    pg_history.add_ai_message(answer)


def _make_rag_chain(mode: str | None = None):
    """构建核心 RAG 链（不含历史管理）。"""
    llm = create_llm()

    def _route(inputs: dict) -> str:
        question = inputs.get("question", "")
        history = inputs.get("history", [])
        ctx = inputs.get("context", "")

        intent = mode or classify_intent(question)
        if not mode:
            logger.info("意图路由: %s → %s", question[:80], intent)

        system_prompt = get_system_prompt(intent).replace("{context}", ctx)

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{question}"),
            ]
        )
        return prompt.invoke({"question": question, "history": history})

    def _retrieve_with_history(inputs: dict) -> str:
        question = inputs["question"]
        history = inputs.get("history", [])
        docs = _retrieve_and_rank(question, history)
        return format_docs(docs)

    chain = (
        RunnableParallel(
            context=RunnableLambda(_retrieve_with_history),
            question=itemgetter("question"),
            history=itemgetter("history"),
        )
        | RunnableLambda(_route)
        | llm
        | StrOutputParser()
    )

    return chain


class _ConversationalRagChain:
    """多轮对话包装器：加载历史 → 调用核心链 → 保存本轮。

    替代已废弃的 RunnableWithMessageHistory，保持相同的调用约定。
    """

    def __init__(self, chain: RunnableSerializable, mode: str | None = None) -> None:
        self._chain = chain
        self._mode = mode

    def invoke(self, inputs: dict, config: dict | None = None) -> str:
        configurable = (config or {}).get("configurable", {})
        session_id = configurable.get("session_id", "default")
        question = inputs.get("question", "")

        history = _load_history(session_id)
        result = self._chain.invoke({"question": question, "history": history})
        _save_turn(session_id, question, result)
        return result

    def stream(self, inputs: dict, config: dict | None = None):
        configurable = (config or {}).get("configurable", {})
        session_id = configurable.get("session_id", "default")
        question = inputs.get("question", "")

        history = _load_history(session_id)
        chunks = []
        for chunk in self._chain.stream({"question": question, "history": history}):
            chunks.append(chunk)
            yield chunk
        result = "".join(chunks)
        _save_turn(session_id, question, result)


def create_conversational_chain(mode: str | None = None) -> _ConversationalRagChain:
    """创建多轮对话 RAG 链，自动意图识别 + 查询改写 + 混合检索 + 重排序。

    mode=None 时自动识别意图；传入具体值则跳过识别直接使用。
    多轮对话中查询改写器会基于历史补全省略/指代。

    返回 _ConversationalRagChain 包装器，调用约定与旧版兼容：
        chain.invoke({"question": q}, config={"configurable": {"session_id": sid}})
    """
    return _ConversationalRagChain(_make_rag_chain(mode), mode)
