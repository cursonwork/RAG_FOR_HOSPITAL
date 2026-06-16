from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from src.config import settings
from src.logger import get_logger

logger = get_logger(__name__)

# 中文友好的分割符优先级
CHINESE_SEPARATORS = [
    "\n\n",
    "\n",
    "。",
    "；",
    "！",
    "？",
    "，",
    " ",
    "",
]

# Markdown 标题层级 → 语义边界
HEADERS_TO_SPLIT_ON = [
    ("#", "h1"),
    ("##", "h2"),
    ("###", "h3"),
]


def create_text_splitter() -> RecursiveCharacterTextSplitter:
    """原始固定大小分割器（保留兼容）。"""
    logger.debug("创建文本分割器 chunk_size=%d overlap=%d", settings.chunk_size, settings.chunk_overlap)
    return RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=CHINESE_SEPARATORS,
        keep_separator=True,
    )


def create_semantic_splitter():
    """两阶段语义分割器：先按 Markdown 标题切，再对长段二次限制大小。"""
    logger.debug("创建语义分块器 chunk_size=%d overlap=%d", settings.chunk_size, settings.chunk_overlap)

    md_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=HEADERS_TO_SPLIT_ON,
        strip_headers=False,
    )
    size_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=CHINESE_SEPARATORS,
        keep_separator=True,
    )

    def split_documents(documents):
        all_chunks = []
        for doc in documents:
            source = doc.metadata.get("source", "")
            try:
                md_chunks = md_splitter.split_text(doc.page_content)
            except Exception:
                logger.debug("Markdown 分块失败，回退到纯文本分块: %s", source)
                md_chunks = [doc.page_content]

            chunk_index = 0
            for md_chunk in md_chunks:
                if isinstance(md_chunk, str):
                    text = md_chunk
                    section = ""
                else:
                    text = md_chunk.page_content
                    section = ""
                    for level in ["h1", "h2", "h3"]:
                        v = md_chunk.metadata.get(level)
                        if v:
                            section = v
                            break

                if len(text) <= settings.chunk_size + 100:
                    # 短段直接作为一个分块
                    chunk_index += 1
                    all_chunks.append(Document(
                        page_content=text,
                        metadata={
                            **doc.metadata,
                            "section_title": section,
                            "chunk_index": chunk_index,
                        },
                    ))
                else:
                    # 长段二次切分
                    subs = size_splitter.split_text(text)
                    for sub_text in subs:
                        chunk_index += 1
                        all_chunks.append(Document(
                            page_content=sub_text,
                            metadata={
                                **doc.metadata,
                                "section_title": section,
                                "chunk_index": chunk_index,
                            },
                        ))

        logger.info("语义分块: %d 个文档 → %d 个块", len(documents), len(all_chunks))
        return all_chunks

    return split_documents
