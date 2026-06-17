import re
import tempfile
import uuid
from contextlib import suppress
from pathlib import Path

import streamlit as st

from src.callbacks import TokenLoggingCallback
from src.chat_history import PostgresChatMessageHistory
from src.config import settings
from src.conversation import create_conversational_chain
from src.database import (
    count_chunks,
    create_session,
    get_chunk,
    get_image,
    get_or_create_user,
    list_user_sessions,
)
from src.document_loader import load_pdf
from src.logger import get_logger
from src.vector_store import add_documents_to_store

logger = get_logger("app")

st.set_page_config(page_title="医疗RAG系统", page_icon="🏥", layout="wide")

# ── Global CSS ──
st.markdown(
    """
<style>
    /* 引用 hover tooltip */
    .citation {
        display: inline; position: relative; cursor: help;
        border-bottom: 1px dotted #aaa;
    }
    .citation .tooltip {
        display: none; position: absolute; bottom: 28px; left: 50%;
        transform: translateX(-50%); background: #1e1e1e; color: #f0f0f0;
        padding: 10px 14px; border-radius: 8px; font-size: 0.85rem;
        max-width: 420px; white-space: pre-wrap; word-break: break-word;
        z-index: 9999; box-shadow: 0 4px 16px rgba(0,0,0,0.5); line-height: 1.5;
    }
    .citation:hover .tooltip { display: block; }

    /* 检索结果卡片 */
    .result-card {
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 20px 24px;
        margin-bottom: 16px;
        background: #fff;
    }
    .result-card:hover {
        border-color: #1a73e8;
        box-shadow: 0 2px 8px rgba(26,115,232,0.12);
    }
    .result-header {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        margin-bottom: 12px;
    }
    .result-source {
        font-size: 0.82rem;
        color: #5f6368;
        line-height: 1.5;
    }
    .result-score {
        text-align: right;
        flex-shrink: 0;
        margin-left: 24px;
    }
    .result-rank {
        font-size: 0.78rem;
        color: #80868b;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .result-pct {
        font-size: 1.5rem;
        font-weight: 700;
        line-height: 1.2;
    }
    .score-high { color: #137333; }
    .score-mid  { color: #e37400; }
    .score-low  { color: #80868b; }

    /* 内容区 */
    .result-content {
        font-size: 0.92rem;
        line-height: 1.65;
        color: #202124;
        white-space: pre-wrap;
        word-break: break-word;
    }

    /* 模式切换 pill */
    .mode-switch {
        display: flex;
        gap: 4px;
        background: #f1f3f4;
        border-radius: 8px;
        padding: 3px;
    }

    /* 全局间距 */
    .block-container {
        padding-top: 2rem;
    }
</style>
""",
    unsafe_allow_html=True,
)

# ── Session state ──
_defaults = {
    "username": "",
    "user_id": None,
    "session_id": None,
    "messages": [],
    "chain": None,
    "mode": "retrieval",
    "citation_map": {},
    "image_map": {},
    "image_index": {},
}
for _key, _val in _defaults.items():
    if _key not in st.session_state:
        st.session_state[_key] = _val


# ═══════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════


def _load_messages_from_db(session_id: str) -> list[dict]:
    history = PostgresChatMessageHistory(session_id=session_id)
    msgs = []
    for m in history.messages:
        role = "user" if m.type == "human" else "assistant"
        msgs.append({"role": role, "content": m.content})
    return msgs


def _retrieve_for_display(question: str, k: int = 5) -> list:
    from src.rag_chain import _retrieve_and_rank

    docs = _retrieve_and_rank(question, top_k=k)
    logger.info(
        "检索结果 (%d 条): %s",
        len(docs),
        " | ".join(
            f"#{i + 1} [{d.metadata.get('source', '?')[:45]}] score={d.metadata.get('score', 0):.3f}"
            for i, d in enumerate(docs[:5])
        ),
    )
    return docs


def _render_result_card(doc, i: int) -> None:
    """渲染单条检索结果卡片。"""
    source = doc.metadata.get("source", "Unknown")
    page = doc.metadata.get("page", "")
    section = doc.metadata.get("section_title", doc.metadata.get("section", ""))
    chunk_type = doc.metadata.get("chunk_type", "text")
    image_id = doc.metadata.get("image_id", "")
    score = doc.metadata.get("score")
    rerank = doc.metadata.get("rerank_score")

    if rerank is not None:
        relevance = round(max(0, min(1, rerank)) * 100)
    elif score is not None:
        relevance = round(score * 100)
    else:
        relevance = 0

    if relevance >= 80:
        score_class = "score-high"
    elif relevance >= 50:
        score_class = "score-mid"
    else:
        score_class = "score-low"

    # ── 构建来源行 ──
    source_parts = [f"📄 {source}"]
    if page:
        source_parts.append(f"p.{page}")
    if section:
        source_parts.append(section)
    source_line = "  ·  ".join(source_parts)

    # ── HTML 卡片 ──
    html = f"""
    <div class="result-card">
        <div class="result-header">
            <div class="result-source">{source_line}</div>
            <div class="result-score">
                <div class="result-rank">#{i + 1}</div>
                <div class="result-pct {score_class}">{relevance}%</div>
            </div>
        </div>
    """

    if chunk_type == "image" and image_id:
        # 图片 chunk：下面用 st.image 渲染
        pass
    else:
        text = doc.page_content
        if len(text) > 400:
            truncated = text[:400] + "..."
            html += f'<div class="result-content">{truncated}</div>'
        else:
            html += f'<div class="result-content">{text}</div>'

    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)

    # 图片渲染（must be outside HTML）
    if chunk_type == "image" and image_id:
        try:
            img_info = get_image(image_id)
            if img_info and img_info.get("image_data"):
                st.image(
                    bytes(img_info["image_data"]),
                    caption=img_info.get("caption", f"Source: {source}"),
                    use_container_width=True,
                )
                desc = img_info.get("description", "")
                if desc:
                    with st.expander("AI description"):
                        st.caption(desc[:500])
        except Exception:
            st.caption(doc.page_content[:200])

    # 长文本展开
    elif chunk_type != "image" and len(doc.page_content) > 400:
        show = st.checkbox(
            f"Show full text ({len(doc.page_content)} chars)",
            key=f"full_{i}",
        )
        if show:
            st.markdown(doc.page_content)


def _build_citation_maps(docs: list) -> None:
    citation_map, image_map, image_index = {}, {}, {}
    text_idx, img_idx = 0, 0
    for doc in docs:
        chunk_id = doc.metadata.get("chunk_id", "")
        image_id = doc.metadata.get("image_id", "")
        chunk_type = doc.metadata.get("chunk_type", "text")

        if chunk_type == "image" and image_id:
            img_idx += 1
            img_info = get_image(image_id)
            if img_info:
                image_map[image_id] = img_info
                image_index[str(img_idx)] = image_id
        elif chunk_id:
            text_idx += 1
            chunk = get_chunk(chunk_id)
            if chunk:
                citation_map[chunk_id] = {
                    "content": chunk["content"],
                    "source": chunk["source"],
                    "page": chunk["page"],
                    "section": chunk.get("section_title", ""),
                    "index": text_idx,
                }
    st.session_state.citation_map = citation_map
    st.session_state.image_map = image_map
    st.session_state.image_index = image_index


def _render_citations(text: str) -> str:
    def _replacer(match):
        num = match.group(1)
        for _cid, info in st.session_state.citation_map.items():
            if str(info.get("index", "")) == num:
                src = info.get("source", "")
                page = f" p.{info['page']}" if info.get("page") else ""
                section = f" | {info['section']}" if info.get("section") else ""
                content = info.get("content", "")[:300]
                tooltip = f"{src}{page}{section}\n\n{content}"
                tooltip = tooltip.replace("&", "&amp;").replace('"', "&quot;").replace("<", "&lt;")
                return (
                    f'<span class="citation">[{num}] {src}{page}{section}<span class="tooltip">{tooltip}</span></span>'
                )
        return match.group(0)

    return re.sub(r"\[文献(\d+)\]", _replacer, text)


def _render_images_after_answer(text: str) -> None:
    for num, img_id in st.session_state.image_index.items():
        if f"[图{num}]" in text:
            img_info = st.session_state.image_map.get(img_id)
            if img_info and img_info.get("image_data"):
                with suppress(Exception):
                    st.image(
                        img_info["image_data"],
                        caption=f"[图{num}] {img_info['source']} p.{img_info['page']} | {img_info.get('description', '')[:100]}",
                        use_container_width=True,
                    )


# ═══════════════════════════════════════════
# Sidebar
# ═══════════════════════════════════════════

with st.sidebar:
    # ── User ──
    username = st.text_input(
        "Username",
        value=st.session_state.username,
        placeholder="Enter username...",
        label_visibility="collapsed",
    )
    if username and username != st.session_state.username:
        st.session_state.username = username
        st.session_state.user_id = get_or_create_user(username)
        new_id = uuid.uuid4().hex[:16]
        create_session(new_id, st.session_state.user_id)
        st.session_state.session_id = new_id
        st.session_state.messages = []
        st.session_state.chain = create_conversational_chain()
        st.rerun()

    if not st.session_state.user_id:
        st.info("Enter a username to begin")
        st.stop()

    st.divider()

    # ── Mode ──
    st.caption("MODE")
    mode_idx = 1 if st.session_state.mode == "chat" else 0
    mode = st.radio(
        "mode",
        ["🔍  Search", "💬  Chat"],
        index=mode_idx,
        horizontal=True,
        label_visibility="collapsed",
        key="sidebar_mode",
    )
    st.session_state.mode = "chat" if "Chat" in mode else "retrieval"

    st.divider()

    # ── Sessions ──
    _, col_new = st.columns([3, 1])
    with col_new:
        if st.button("＋", help="New session", use_container_width=True):
            new_id = uuid.uuid4().hex[:16]
            create_session(new_id, st.session_state.user_id)
            st.session_state.session_id = new_id
            st.session_state.messages = []
            st.session_state.chain = create_conversational_chain()
            st.rerun()

    sessions = list_user_sessions(st.session_state.user_id)
    for s in sessions[-8:]:  # show last 8
        sid, _, title, _, updated_at, first_q = s
        label = title or first_q or "New session"
        label = label[:24] + ("..." if len(label) > 24 else "")
        active = sid == st.session_state.session_id
        if st.button(
            label,
            key=f"s_{sid}",
            type="primary" if active else "secondary",
            use_container_width=True,
        ):
            st.session_state.session_id = sid
            st.session_state.messages = _load_messages_from_db(sid)
            st.session_state.chain = create_conversational_chain()
            st.rerun()

    st.divider()

    # ── Knowledge base ──
    with st.expander("📚  Knowledge base", expanded=False):
        uploaded_files = st.file_uploader(
            "Upload PDF",
            type=["pdf"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )
        if uploaded_files and st.button("Import", type="primary", use_container_width=True):
            with st.status("Processing...", expanded=True):
                all_docs = []
                for uf in uploaded_files:
                    st.write(f"Parsing: {uf.name}")
                    suffix = Path(uf.name).suffix or ".pdf"
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                        tmp.write(uf.read())
                        doc = load_pdf(tmp.name)
                        if doc:
                            all_docs.append(doc)
                    Path(tmp.name).unlink(missing_ok=True)
                add_documents_to_store(all_docs)
                st.session_state.chain = create_conversational_chain()
                st.write(f"Done — {count_chunks()} chunks")

        try:
            st.metric("Chunks", count_chunks())
        except Exception:
            st.metric("Chunks", 0)
        st.caption(f"Model: {settings.deepseek_model}")
        st.caption(f"Embedding: {settings.embedding_model_name}")


# ═══════════════════════════════════════════
# Main area
# ═══════════════════════════════════════════

st.title("🏥 Medical RAG")

# ── Chat history (only for chat mode) ──
if st.session_state.mode == "chat":
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant":
                st.markdown(msg.get("raw_content", msg["content"]), unsafe_allow_html=True)
            else:
                st.markdown(msg["content"])

# ── Search / chat input ──
is_retrieval = st.session_state.mode == "retrieval"
placeholder = "Search the knowledge base..." if is_retrieval else "Ask a medical question..."

if question := st.chat_input(placeholder):
    logger.info("Query (user=%s, mode=%s): %s", st.session_state.username, st.session_state.mode, question[:200])

    if is_retrieval:
        # ═══ Retrieval mode ═══
        with st.spinner("Searching..."):
            docs = _retrieve_for_display(question)

        if not docs:
            st.warning("No results found.")
        else:
            st.caption(f"{len(docs)} results")
            for i, doc in enumerate(docs):
                _render_result_card(doc, i)

    else:
        # ═══ Chat mode ═══
        if st.session_state.chain is None:
            st.session_state.chain = create_conversational_chain()

        st.chat_message("user").markdown(question)
        st.session_state.messages.append({"role": "user", "content": question})

        with st.chat_message("assistant"), st.spinner("Thinking..."):
            try:
                result = st.session_state.chain.invoke(
                    {"question": question},
                    config={
                        "configurable": {"session_id": st.session_state.session_id},
                        "callbacks": [TokenLoggingCallback()],
                    },
                )
                from src.rag_chain import get_last_retrieved_docs

                _build_citation_maps(get_last_retrieved_docs())
                rendered = _render_citations(result)
                st.markdown(rendered, unsafe_allow_html=True)
                _render_images_after_answer(result)
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": result,
                        "raw_content": rendered,
                    }
                )
            except Exception as e:
                logger.exception("Chat error")
                st.error(str(e))

# ── Footer ──
st.divider()
st.caption("⚠️ For medical reference only. Not a substitute for professional medical advice.")
