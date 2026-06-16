from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough, RunnableParallel

from src.config import settings
from src.intent import classify_intent
from src.llm import create_llm
from src.logger import get_logger
from src.prompts import get_system_prompt

logger = get_logger(__name__)

# 存储最近一次检索结果，供 app.py 构建 citation_map
_last_retrieved_docs: list = []


def get_last_retrieved_docs() -> list:
    """返回最近一次检索到的文档列表（供 UI 层构建引用映射）。"""
    return _last_retrieved_docs


def _retrieve_and_rank(
    question: str,
    history: list | None = None,
    top_k: int | None = None,
) -> list:
    """统一的检索 + 重排序管线：

    查询改写 → 混合检索(BM25+Dense) → 合并去重 → 重排序 → top-k

    Args:
        question: 用户问题
        history: 可选，多轮对话历史
        top_k: 最终返回文档数，默认 settings.retrieval_top_k

    Returns:
        排好序的 Document 列表
    """
    from src.query_rewriter import rewrite_query
    from src.vector_store import get_vector_store
    from src.reranker import get_reranker

    top_k = top_k or settings.retrieval_top_k
    store = get_vector_store()

    # Step 1: 查询改写
    variants = rewrite_query(question, history)

    # Step 2: 每个变体混合检索
    all_docs: list = []
    seen_ids: set = set()

    for variant in variants:
        if settings.hybrid_enabled:
            docs = store.hybrid_search(variant, k=settings.hybrid_retrieval_top_k)
        else:
            docs = store.similarity_search(variant, k=settings.hybrid_retrieval_top_k)

        for doc in docs:
            cid = doc.metadata.get("chunk_id", "")
            if cid and cid not in seen_ids:
                seen_ids.add(cid)
                all_docs.append(doc)

    logger.debug(
        "多查询检索: %d 个变体 → %d 个候选文档 (去重后)",
        len(variants), len(all_docs),
    )

    # Step 3: 重排序
    if settings.reranker_enabled:
        reranker = get_reranker()
        all_docs = reranker.compress_documents(all_docs, question)

    result = all_docs[:top_k]
    global _last_retrieved_docs
    _last_retrieved_docs = result
    return result


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
        section = doc.metadata.get("section", "")
        rerank_score = doc.metadata.get("rerank_score")
        relevance = f" [相关度: {rerank_score:.3f}]" if rerank_score else ""

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
            pn = chunk.get("page", page) if chunk else page
            sec = chunk.get("section_title", section) if chunk else section

            header = f"[文献{text_idx}] 来源: {source}"
            if pn:
                header += f" 第{pn}页"
            if sec:
                header += f" | 章节: {sec}"
            header += f"{relevance}\n原文: {original_text}"
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

    def _retrieve_context(question: str) -> str:
        """完整检索管线：改写 → 混合检索 → 重排序 → 格式化。"""
        docs = _retrieve_and_rank(question)
        return format_docs(docs)

    chain = (
        RunnableParallel(
            context=RunnableLambda(lambda q: _retrieve_context(q)),
            question=RunnablePassthrough(),
        )
        | RunnableLambda(_build_prompt)
        | llm
        | StrOutputParser()
    )
    return chain
