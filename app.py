import re
import tempfile
import uuid
from pathlib import Path

import streamlit as st

from src.callbacks import TokenLoggingCallback
from src.chat_history import PostgresChatMessageHistory
from src.config import settings
from src.conversation import create_conversational_chain
from src.database import (
    count_chunks,
    create_session,
    delete_session,
    get_chunk,
    get_image,
    get_or_create_user,
    list_user_sessions,
)
from src.document_loader import load_pdf
from src.logger import get_logger
from src.vector_store import add_documents_to_store, get_retriever

logger = get_logger("app")

st.set_page_config(page_title="医疗RAG系统", page_icon="🏥", layout="wide")

# ── 样式（hover 引用浮窗） ──
st.markdown("""
<style>
    .citation {
        display: inline;
        position: relative;
        cursor: help;
        border-bottom: 1px dotted #aaa;
    }
    .citation .tooltip {
        display: none;
        position: absolute;
        bottom: 28px;
        left: 50%%;
        transform: translateX(-50%%);
        background: #1e1e1e;
        color: #f0f0f0;
        padding: 10px 14px;
        border-radius: 8px;
        font-size: 0.85rem;
        max-width: 420px;
        white-space: pre-wrap;
        word-break: break-word;
        z-index: 9999;
        box-shadow: 0 4px 16px rgba(0,0,0,0.5);
        line-height: 1.5;
    }
    .citation:hover .tooltip {
        display: block;
    }
</style>
""", unsafe_allow_html=True)

# ── 会话状态 ──
_defaults = {
    "username": "",
    "user_id": None,
    "session_id": None,
    "messages": [],
    "chain": None,
    # 当前轮次的检索结果映射
    "citation_map": {},    # chunk_id → {content, source, page, section}
    "image_map": {},       # image_id → {image_data, image_format, description, caption, source, page}
    "image_index": {},     # [图N] → image_id
}
for _key, _val in _defaults.items():
    if _key not in st.session_state:
        st.session_state[_key] = _val


def _load_messages_from_db(session_id: str) -> list[dict]:
    history = PostgresChatMessageHistory(session_id=session_id)
    msgs = []
    for m in history.messages:
        role = "user" if m.type == "human" else "assistant"
        msgs.append({"role": role, "content": m.content})
    return msgs


def _pre_retrieve(question: str, top_k: int = 5) -> None:
    """检索并构建引用映射，同时返回上下文字符串供展示。"""
    retriever = get_retriever()
    docs = retriever.invoke(question)

    citation_map = {}
    image_map = {}
    image_index = {}

    text_idx = 0
    img_idx = 0

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
    """后处理：文献引用 → hover 浮窗。图片引用收集到 session_state 稍后渲染。"""
    def _cite_replacer(match):
        num = match.group(1)
        for cid, info in st.session_state.citation_map.items():
            if str(info.get("index", "")) == num:
                src = info.get("source", "未知")
                page = info.get("page", "")
                section = info.get("section", "")
                content = info.get("content", "")
                preview = content[:300] + ("..." if len(content) > 300 else "")
                page_str = f" 第{page}页" if page else ""
                section_str = f" | {section}" if section else ""
                tooltip = f"{src}{page_str}{section_str}\n\n{preview}"
                tooltip = tooltip.replace("&", "&amp;").replace('"', "&quot;").replace("<", "&lt;").replace(">", "&gt;")
                return (
                    f'<span class="citation">[文献{num}] 来源: {src}{page_str}{section_str}'
                    f'<span class="tooltip">{tooltip}</span></span>'
                )
        return match.group(0)

    return re.sub(r"\[文献(\d+)\]", _cite_replacer, text)


def _render_images_after_answer(text: str) -> None:
    """在 markdown 后面渲染匹配到的图片。"""
    for num, img_id in st.session_state.image_index.items():
        if f"[图{num}]" in text:
            img_info = st.session_state.image_map.get(img_id)
            if img_info and img_info.get("image_data"):
                try:
                    st.image(
                        img_info["image_data"],
                        caption=f"[图{num}] 来源: {img_info['source']} 第{img_info['page']}页 | {img_info.get('description', '')[:100]}",
                        use_container_width=True,
                    )
                except Exception:
                    logger.exception("渲染图片失败 [图%s]", num)


# ═══════════════════════════════════════════
# 侧边栏
# ═══════════════════════════════════════════

with st.sidebar:
    st.header("👤 用户")

    username = st.text_input(
        "用户名",
        value=st.session_state.username,
        placeholder="输入用户名开始使用...",
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
        logger.info("用户登录: %s (user_id=%d, session=%s)",
                    username, st.session_state.user_id, new_id)
        st.rerun()

    if not st.session_state.user_id:
        st.info("请输入用户名以开始使用")
        st.stop()

    st.divider()

    # ── 会话管理 ──
    st.header("💬 会话")

    if st.button("➕ 新建会话", use_container_width=True):
        new_id = uuid.uuid4().hex[:16]
        create_session(new_id, st.session_state.user_id)
        st.session_state.session_id = new_id
        st.session_state.messages = []
        st.session_state.chain = create_conversational_chain()
        st.rerun()

    sessions = list_user_sessions(st.session_state.user_id)

    for s in sessions:
        sid, mode, title, _created, updated_at, first_q = s
        is_active = sid == st.session_state.session_id

        label = title or first_q or "新会话"
        if len(label) > 22:
            label = label[:22] + "..."

        ts = updated_at.strftime("%m-%d %H:%M")

        col_btn, col_del = st.columns([6, 1])
        with col_btn:
            btn_type = "primary" if is_active else "secondary"
            if st.button(
                f"{label}  — {ts}",
                key=f"sess_{sid}",
                type=btn_type,
                use_container_width=True,
            ):
                st.session_state.session_id = sid
                st.session_state.messages = _load_messages_from_db(sid)
                st.session_state.chain = create_conversational_chain()
                st.rerun()
        with col_del:
            if st.button("🗑️", key=f"del_{sid}", help="删除此会话"):
                delete_session(sid)
                if is_active:
                    remaining = list_user_sessions(st.session_state.user_id)
                    if remaining:
                        st.session_state.session_id = remaining[0][0]
                        st.session_state.messages = _load_messages_from_db(remaining[0][0])
                    else:
                        new_id = uuid.uuid4().hex[:16]
                        create_session(new_id, st.session_state.user_id)
                        st.session_state.session_id = new_id
                        st.session_state.messages = []
                    st.session_state.chain = create_conversational_chain()
                st.rerun()

    st.divider()

    # ── 文档管理 ──
    st.header("📚 知识库")

    uploaded_files = st.file_uploader(
        "上传医学文献 PDF",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files:
        if st.button("导入知识库", type="primary", use_container_width=True):
            logger.info("===== Web 文档导入 =====")
            with st.status("正在处理文档...", expanded=True) as status:
                all_docs = []
                for uf in uploaded_files:
                    st.write(f"📄 解析: {uf.name}")
                    logger.info("上传文件: %s (%.1f KB)", uf.name, uf.size / 1024)
                    suffix = Path(uf.name).suffix or ".pdf"
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                        tmp.write(uf.read())
                        doc = load_pdf(tmp.name)
                        if doc is not None:
                            all_docs.append(doc)
                    Path(tmp.name).unlink(missing_ok=True)

                st.write(f"🔢 共 {len(all_docs)} 个文档，正在分块写入向量库...")
                logger.info("共加载 %d 个文档，开始入库", len(all_docs))
                try:
                    add_documents_to_store(all_docs)
                    st.session_state.chain = create_conversational_chain()
                    total = count_chunks()
                    logger.info("Web 导入完成，知识库共 %d 个分块", total)
                    status.update(
                        label=f"✅ 导入完成！共 {total} 个分块",
                        state="complete",
                    )
                except Exception:
                    logger.exception("Web 导入失败")
                    st.error("导入失败，请查看日志 logs/app.log")

    st.divider()
    try:
        total_chunks = count_chunks()
    except Exception:
        total_chunks = 0
    st.metric("已索引分块数", total_chunks)

    st.divider()
    st.caption(f"PDF解析: {settings.pdf_parser}")
    st.caption(f"Embedding: {settings.embedding_model_name}")
    st.caption(f"LLM: {settings.deepseek_model}")
    st.caption(f"图片理解: {'on' if settings.enable_image_understanding else 'off'} ({settings.dashscope_model})")
    st.caption("向量库: Milvus | 会话: PostgreSQL")
    st.caption("意图识别: automatic")

# ═══════════════════════════════════════════
# 主区域：对话界面
# ═══════════════════════════════════════════

st.header("🏥 医疗 RAG 智能问答")
st.caption("系统自动识别问题类型（医疗问答 / 药物查询 / 辅助诊断），无需手动切换")

st.divider()

# ── 消息展示 ──
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            # 渲染带 hover 引用的回答
            st.markdown(msg.get("raw_content", msg["content"]), unsafe_allow_html=True)
        else:
            st.markdown(msg["content"])

# ── 输入区 ──
if question := st.chat_input("请输入您的医疗问题..."):
    logger.info("用户提问 (user=%s, session=%s): %s",
                st.session_state.username,
                st.session_state.session_id, question[:200])

    if st.session_state.chain is None:
        st.session_state.chain = create_conversational_chain()

    st.chat_message("user").markdown(question)
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("assistant"):
        # 1. 预检索，建立引用映射
        _pre_retrieve(question)

        with st.spinner("检索中..."):
            try:
                result = st.session_state.chain.invoke(
                    {"question": question},
                    config={
                        "configurable": {"session_id": st.session_state.session_id},
                        "callbacks": [TokenLoggingCallback()],
                    },
                )
                logger.info("生成回答: %d 字符", len(result))

                # 2. 后处理：hover 引用
                rendered = _render_citations(result)
                st.markdown(rendered, unsafe_allow_html=True)
                # 3. 渲染匹配到的图片
                _render_images_after_answer(result)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result,          # 纯文本（用于历史展示）
                    "raw_content": rendered,    # HTML 版本（用于当前渲染）
                })
            except Exception as e:
                logger.exception("问答异常")
                st.error(f"出错了：{e}")

# ── 底部声明 ──
st.divider()
st.caption("⚠️ 本系统仅供医学信息参考，不构成诊断或治疗建议。如有健康问题请咨询执业医师。")
