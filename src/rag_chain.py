from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough, RunnableParallel

from src.intent import classify_intent
from src.llm import create_llm
from src.logger import get_logger
from src.prompts import get_system_prompt

logger = get_logger(__name__)


def format_docs(docs: list) -> str:
    """将检索到的文档拼接为上下文字符串，从 PG 取原文和图片引用。"""
    from src.database import get_chunk, get_image

    parts = []
    text_idx = 0
    image_idx = 0

    for doc in docs:
        chunk_id = doc.metadata.get("chunk_id", "")
        image_id = doc.metadata.get("image_id", "")
        chunk_type = doc.metadata.get("chunk_type", "text")
        source = doc.metadata.get("source", "未知")
        page = doc.metadata.get("page", "")

        if chunk_type == "image" and image_id:
            image_idx += 1
            image = get_image(image_id)
            desc = image["description"] if image else doc.page_content
            caption = image["caption"] if image else ""
            parts.append(
                f"[图{image_idx}] 来源: {source}" + (f" 第{page}页" if page else "")
                + f"\n图片描述: {desc}"
                + (f"\n原始说明: {caption}" if caption else "")
            )
        else:
            text_idx += 1
            chunk = get_chunk(chunk_id)
            original_text = chunk["content"] if chunk else doc.page_content
            section = chunk.get("section_title", "") if chunk else ""
            pn = chunk.get("page", page) if chunk else page

            header = f"[文献{text_idx}] 来源: {source}"
            if pn:
                header += f" 第{pn}页"
            if section:
                header += f" | 章节: {section}"
            header += f"\n原文: {original_text}"
            parts.append(header)

    logger.debug("格式化上下文: %d 个文档块 (文本%d, 图片%d), 总长度 %d 字符",
                 len(docs), text_idx, image_idx, sum(len(p) for p in parts))
    return "\n\n---\n\n".join(parts)


def create_rag_chain(mode: str | None = None):
    """创建 RAG 问答链。mode=None 时自动识别意图。"""
    from src.vector_store import get_retriever

    llm = create_llm()
    retriever = get_retriever()

    def _build_prompt(inputs: dict) -> dict:
        question = inputs.get("question", "")
        context = inputs.get("context", "")

        if mode:
            intent = mode
        else:
            intent = classify_intent(question)
            logger.info("意图路由: %s → %s", question[:80], intent)

        template = get_system_prompt(intent)
        prompt = PromptTemplate(template=template, input_variables=["context", "question"])
        return prompt.invoke({"context": context, "question": question})

    chain = (
        RunnableParallel(
            context=retriever | format_docs,
            question=RunnablePassthrough(),
        )
        | RunnableLambda(_build_prompt)
        | llm
        | StrOutputParser()
    )
    return chain
