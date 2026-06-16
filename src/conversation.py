from operator import itemgetter

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda, RunnableParallel
from langchain_core.runnables.history import RunnableWithMessageHistory

from src.chat_history import PostgresChatMessageHistory
from src.config import settings
from src.intent import classify_intent
from src.llm import create_llm
from src.logger import get_logger
from src.prompts import get_system_prompt
from src.rag_chain import format_docs, _retrieve_and_rank
from src.vector_store import get_retriever

logger = get_logger(__name__)


def _get_session_history(session_id: str) -> BaseChatMessageHistory:
    history = PostgresChatMessageHistory(session_id=session_id)
    logger.debug("会话 %s: 历史消息 %d 条", session_id, len(history.messages))
    return history


def create_conversational_chain(mode: str | None = None):
    """创建多轮对话 RAG 链，自动意图识别 + 查询改写 + 混合检索 + 重排序。

    mode=None 时自动识别意图；传入具体值则跳过识别直接使用。
    多轮对话中查询改写器会基于历史补全省略/指代。
    """
    llm = create_llm()
    retriever = get_retriever()

    def _route(inputs: dict) -> str:
        """意图路由：识别意图 → 选择 System Prompt → 构建最终 prompt。"""
        question = inputs.get("question", "")
        history = inputs.get("history", [])
        ctx = inputs.get("context", "")

        if mode:
            intent = mode
        else:
            intent = classify_intent(question)
            logger.info("意图路由: %s → %s", question[:80], intent)

        system_prompt = get_system_prompt(intent).replace("{context}", ctx)

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}"),
        ])
        return prompt.invoke({"question": question, "history": history})

    def _retrieve_with_history(inputs: dict) -> str:
        """带历史感知的检索管线。"""
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

    return RunnableWithMessageHistory(
        chain,
        _get_session_history,
        input_messages_key="question",
        history_messages_key="history",
    )
