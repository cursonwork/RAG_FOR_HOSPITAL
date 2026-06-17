from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

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

# Markdown 标题层级 → 语义边界（旧方案 fallback 使用）
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


# ═══════════════════════════════════════════════════════════════════════════════
# 新版：基于 opendataloader JSON 元素的 section-aware 分块
# ═══════════════════════════════════════════════════════════════════════════════


def _get_element_text(el: dict) -> str:
    """从 ODL JSON 元素提取可显示文本。"""
    content = (el.get("content") or "").strip()
    if content:
        return content

    # 列表：从 list items 中抽取文本
    if el.get("type") == "list":
        items = el.get("list items", [])
        if items:
            return "\n".join(
                (item.get("content") or "").strip() for item in items if (item.get("content") or "").strip()
            )

    return ""


def _split_by_odl_elements(doc: Document, elements: list[dict]) -> list[Document]:
    """基于 opendataloader JSON 元素的 section-aware 分块。

    策略：
    1. 遍历 kids[], 遇到任意 heading 元素即开始新 section
    2. 将后续 paragraph / list / table 等归入当前 section
    3. 每个 section 内尽可能组合相邻元素到 chunk_size，只在元素边界切分
    4. 超大单个元素（如超长段落）回退到 RecursiveCharacterTextSplitter
    """
    source = doc.metadata.get("source", "")
    chunks: list[Document] = []
    chunk_index = 0

    # ── Phase 1: 按 heading 边界归组 ──
    # 策略：仅 heading level 2-4 创建新 section。
    # Level 1 为文档标题标签（取第一个长文本作 section 名），Level 5+ 为作者名等噪音。
    sections: list[tuple[str, list[dict]]] = []  # (heading_text, elements)
    current_heading = ""
    current_elems: list[dict] = []

    for el in elements:
        el_type = el.get("type", "")

        if el_type == "image":
            continue

        if el_type == "heading":
            content = (el.get("content") or "").strip()
            hl = el.get("heading level", 99)

            if 2 <= hl <= 4:
                # 真正的章节标题 → 开始新 section
                if current_elems:
                    sections.append((current_heading, current_elems))
                current_heading = content
                current_elems = [el]
            else:
                # Level 1（文档标题）或 Level 5+（作者名/噪音）→ 归入当前 section
                if hl == 1 and len(content) >= 50 and not current_heading:
                    current_heading = content
                current_elems.append(el)
        elif _get_element_text(el):
            current_elems.append(el)

    if current_elems:
        sections.append((current_heading, current_elems))

    # ── Phase 1.5: 合并微小 section（仅含标题无正文）入下一 section ──
    merged_sections: list[tuple[str, list[dict]]] = []
    i = 0
    while i < len(sections):
        heading, elems = sections[i]
        text = "\n\n".join(_get_element_text(el) for el in elems)
        if len(text) < 100 and i + 1 < len(sections):
            next_heading, next_elems = sections[i + 1]
            merged_sections.append((next_heading, elems + next_elems))
            i += 2
        else:
            merged_sections.append((heading, elems))
            i += 1
    sections = merged_sections

    # ── Phase 2: 每个 section 内按段落边界组合到 chunk_size ──
    size_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=CHINESE_SEPARATORS,
        keep_separator=True,
    )

    for section_title, elems in sections:
        section_chunks = _build_section_chunks(
            elems=elems,
            section_title=section_title,
            source=source,
            start_index=chunk_index,
            size_splitter=size_splitter,
        )
        chunks.extend(section_chunks)
        chunk_index += len(section_chunks)

    return chunks


def _build_section_chunks(
    elems: list[dict],
    section_title: str,
    source: str,
    start_index: int,
    size_splitter: RecursiveCharacterTextSplitter,
) -> list[Document]:
    """将 section 内元素组合成 chunk，按 chunk_size 软限制在元素边界切分。"""
    chunks: list[Document] = []
    current_elems: list[dict] = []
    current_len = 0

    for el in elems:
        text = _get_element_text(el)
        if not text:
            continue

        # 当前 chunk 已满 → 在元素边界 emit
        if current_len + len(text) > settings.chunk_size and current_elems:
            _emit_chunk(current_elems, section_title, source, start_index + len(chunks), chunks)
            current_elems = []
            current_len = 0

        # 单个元素超长（罕见：超长段落）→ 用 RecursiveCharacterTextSplitter 切
        if len(text) > settings.chunk_size:
            # 先 emit 已积攒的元素
            if current_elems:
                _emit_chunk(current_elems, section_title, source, start_index + len(chunks), chunks)
                current_elems = []
                current_len = 0

            # 回退到字符级切分
            page = el.get("page number", 0)
            subs = size_splitter.split_text(text)
            for sub_text in subs:
                chunks.append(
                    Document(
                        page_content=sub_text,
                        metadata={
                            "source": source,
                            "section_title": section_title,
                            "page": page,
                            "chunk_index": start_index + len(chunks),
                            "parser": "opendataloader-json",
                        },
                    )
                )
        else:
            current_elems.append(el)
            current_len += len(text)

    # Emit 剩余元素
    if current_elems:
        _emit_chunk(current_elems, section_title, source, start_index + len(chunks), chunks)

    return chunks


def _emit_chunk(
    elems: list[dict],
    section_title: str,
    source: str,
    chunk_index: int,
    chunks: list[Document],
) -> None:
    """将元素列表拼接为一个 Document 并追加到 chunks。"""
    text = "\n\n".join(_get_element_text(el) for el in elems)
    page = elems[0].get("page number", 0)
    element_ids = [el["id"] for el in elems if "id" in el]
    chunks.append(
        Document(
            page_content=text,
            metadata={
                "source": source,
                "section_title": section_title,
                "page": page,
                "chunk_index": chunk_index,
                "parser": "opendataloader-json",
                "element_ids": element_ids,
            },
        )
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 旧方案：Markdown 标题分块（fallback）
# ═══════════════════════════════════════════════════════════════════════════════


def _split_by_markdown(doc: Document) -> list[Document]:
    """基于 Markdown 标题的语义分块（无 JSON 时的回退方案）。"""
    source = doc.metadata.get("source", "")

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

    try:
        md_chunks = md_splitter.split_text(doc.page_content)
    except Exception:
        logger.debug("Markdown 分块失败，回退到纯文本分块: %s", source)
        subs = size_splitter.split_text(doc.page_content)
        return [
            Document(
                page_content=sub_text,
                metadata={
                    **doc.metadata,
                    "section_title": "",
                    "chunk_index": i + 1,
                },
            )
            for i, sub_text in enumerate(subs)
        ]

    all_chunks: list[Document] = []
    chunk_index = 0

    for md_chunk in md_chunks:
        if isinstance(md_chunk, str):
            text = md_chunk
            section = ""
        else:
            text = md_chunk.page_content
            # 取最深层的标题
            section = ""
            for level in ["h3", "h2", "h1"]:
                v = md_chunk.metadata.get(level)
                if v:
                    section = v
                    break

        if len(text) <= settings.chunk_size + 100:
            chunk_index += 1
            all_chunks.append(
                Document(
                    page_content=text,
                    metadata={
                        **doc.metadata,
                        "section_title": section,
                        "chunk_index": chunk_index,
                    },
                )
            )
        else:
            subs = size_splitter.split_text(text)
            for sub_text in subs:
                chunk_index += 1
                all_chunks.append(
                    Document(
                        page_content=sub_text,
                        metadata={
                            **doc.metadata,
                            "section_title": section,
                            "chunk_index": chunk_index,
                        },
                    )
                )

    return all_chunks


# ═══════════════════════════════════════════════════════════════════════════════
# 统一入口
# ═══════════════════════════════════════════════════════════════════════════════


def create_semantic_splitter():
    """两阶段语义分割器。

    优先使用 opendataloader JSON 元素进行 section-aware 分块（精确标题、段落边界、页码），
    JSON 不可用时回退到 Markdown 标题分块。
    """
    logger.debug("创建语义分块器 chunk_size=%d overlap=%d (ODL JSON 优先)", settings.chunk_size, settings.chunk_overlap)

    def split_documents(documents: list[Document]) -> list[Document]:
        all_chunks: list[Document] = []

        for doc in documents:
            elements = doc.metadata.get("odl_elements")
            source = doc.metadata.get("source", "")

            if elements:
                logger.debug("%s: 使用 ODL JSON 分块 (%d 个元素)", source, len(elements))
                chunks = _split_by_odl_elements(doc, elements)
            else:
                logger.debug("%s: 回退到 Markdown 分块", source)
                chunks = _split_by_markdown(doc)

            all_chunks.extend(chunks)

        logger.info("语义分块: %d 个文档 → %d 个块", len(documents), len(all_chunks))
        return all_chunks

    return split_documents
