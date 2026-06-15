from operator import itemgetter

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory

from src.chat_history import PostgresChatMessageHistory
from src.llm import create_llm
from src.logger import get_logger
from src.prompts import SYSTEM_PROMPTS
from src.rag_chain import format_docs
from src.vector_store import get_retriever

logger = get_logger(__name__)


def _get_session_history(session_id: str) -> BaseChatMessageHistory:
    history = PostgresChatMessageHistory(session_id=session_id)
    logger.debug("会话 %s: 历史消息 %d 条", session_id, len(history.messages))
    return history


def create_conversational_chain(mode: str = "medical_qa"):
    """创建支持多轮对话的 RAG 链，历史由 PostgreSQL 持久化。"""
    logger.info("创建多轮对话链 (mode=%s)", mode)

    llm = create_llm()
    retriever = get_retriever()

    system_prompt = SYSTEM_PROMPTS.get(mode, SYSTEM_PROMPTS["medical_qa"])

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}"),
        ]
    )

    chain = (
        {
            "context": itemgetter("question") | retriever | format_docs,
            "question": itemgetter("question"),
            "history": itemgetter("history"),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return RunnableWithMessageHistory(
        chain,
        _get_session_history,
        input_messages_key="question",
        history_messages_key="history",
    )
