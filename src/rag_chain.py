from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough

from src.llm import create_llm
from src.logger import get_logger
from src.prompts import SYSTEM_PROMPTS

logger = get_logger(__name__)


def format_docs(docs: list) -> str:
    """将检索到的文档拼接为上下文字符串。"""
    parts = []
    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "未知")
        page = doc.metadata.get("page", "")
        header = f"[文献{i}] 来源: {source}" + (f" 第{page}页" if page else "")
        parts.append(f"{header}\n{doc.page_content}")

    logger.debug("格式化上下文: %d 个文档块, 总长度 %d 字符", len(docs), sum(len(p) for p in parts))
    return "\n\n---\n\n".join(parts)


def create_rag_chain(mode: str = "medical_qa"):
    """创建 RAG 问答链。"""
    from src.vector_store import get_retriever

    logger.info("创建 RAG 链 (mode=%s)", mode)

    llm = create_llm()
    retriever = get_retriever()

    prompt_template = SYSTEM_PROMPTS.get(mode, SYSTEM_PROMPTS["medical_qa"])
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"],
    )

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain
