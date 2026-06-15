import tempfile
import uuid
from pathlib import Path

import streamlit as st

from src.callbacks import TokenLoggingCallback
from src.chat_history import PostgresChatMessageHistory
from src.config import settings
from src.conversation import create_conversational_chain
from src.database import (
    create_session,
    delete_session,
    get_or_create_user,
    list_user_sessions,
    update_session_mode,
)
from src.document_loader import load_pdf
from src.logger import get_logger
from src.vector_store import add_documents_to_store

logger = get_logger("app")

st.set_page_config(page_title="医疗RAG系统", page_icon="🏥", layout="wide")

# ── 样式 ──
st.markdown("""
<style>
    .stChatMessage { padding: 1rem; border-radius: 10px; margin-bottom: 0.5rem; }
</style>
""", unsafe_allow_html=True)

# ── 会话状态初始化 ──
_defaults = {
    "username": "",
    "user_id": None,
    "session_id": None,
    "messages": [],
    "mode": "medical_qa",
    "chain": None,
    "doc_count": 0,
}
for _key, _val in _defaults.items():
    if _key not in st.session_state:
        st.session_state[_key] = _val

MODE_EMOJI = {"medical_qa": "🩺", "drug_query": "💊", "diagnosis": "🩻"}


def _load_messages_from_db(session_id: str) -> list[dict]:
    """从 PostgreSQL 加载会话消息，转为 Streamlit 展示格式。"""
    history = PostgresChatMessageHistory(session_id=session_id)
    msgs = []
    for m in history.messages:
        role = "user" if m.type == "human" else "assistant"
        msgs.append({"role": role, "content": m.content})
    return msgs


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
        create_session(new_id, st.session_state.user_id, st.session_state.mode)
        st.session_state.session_id = new_id
        st.session_state.messages = []
        st.session_state.chain = create_conversational_chain(st.session_state.mode)
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
        create_session(new_id, st.session_state.user_id, st.session_state.mode)
        st.session_state.session_id = new_id
        st.session_state.messages = []
        st.session_state.chain = create_conversational_chain(st.session_state.mode)
        st.rerun()

    sessions = list_user_sessions(st.session_state.user_id)

    for s in sessions:
        sid, mode, title, _created, updated_at, first_q = s
        is_active = sid == st.session_state.session_id

        label = title or first_q or "新会话"
        if len(label) > 22:
            label = label[:22] + "..."

        emoji = MODE_EMOJI.get(mode, "")
        ts = updated_at.strftime("%m-%d %H:%M")

        col_btn, col_del = st.columns([6, 1])
        with col_btn:
            btn_type = "primary" if is_active else "secondary"
            if st.button(
                f"{emoji} {label}  — {ts}",
                key=f"sess_{sid}",
                type=btn_type,
                use_container_width=True,
            ):
                st.session_state.session_id = sid
                st.session_state.mode = mode
                st.session_state.messages = _load_messages_from_db(sid)
                st.session_state.chain = create_conversational_chain(mode)
                st.rerun()
        with col_del:
            if st.button("🗑️", key=f"del_{sid}", help="删除此会话"):
                delete_session(sid)
                if is_active:
                    remaining = list_user_sessions(st.session_state.user_id)
                    if remaining:
                        st.session_state.session_id = remaining[0][0]
                        st.session_state.mode = remaining[0][1]
                        st.session_state.messages = _load_messages_from_db(
                            remaining[0][0]
                        )
                    else:
                        new_id = uuid.uuid4().hex[:16]
                        create_session(new_id, st.session_state.user_id, "medical_qa")
                        st.session_state.session_id = new_id
                        st.session_state.messages = []
                        st.session_state.mode = "medical_qa"
                    st.session_state.chain = create_conversational_chain(
                        st.session_state.mode
                    )
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
                    with tempfile.NamedTemporaryFile(
                        delete=False, suffix=suffix
                    ) as tmp:
                        tmp.write(uf.read())
                        docs = load_pdf(tmp.name)
                        all_docs.extend(docs)
                    Path(tmp.name).unlink(missing_ok=True)

                st.write(f"🔢 共 {len(all_docs)} 页，正在分块写入向量库...")
                logger.info("共提取 %d 页，开始入库", len(all_docs))
                try:
                    add_documents_to_store(all_docs)
                    st.session_state.doc_count += len(all_docs)
                    st.session_state.chain = create_conversational_chain(
                        st.session_state.mode
                    )
                    logger.info("Web 导入完成，知识库共 %d 页", st.session_state.doc_count)
                    status.update(
                        label=f"✅ 导入完成！共处理 {len(all_docs)} 页文档",
                        state="complete",
                    )
                except Exception:
                    logger.exception("Web 导入失败")
                    st.error("导入失败，请查看日志 logs/app.log")

    st.divider()
    st.metric("已索引文档页数", st.session_state.doc_count)

    st.divider()
    st.caption(f"Embedding: {settings.embedding_model_name}")
    st.caption(f"LLM: {settings.deepseek_model}")
    st.caption("向量库: Milvus | 会话: PostgreSQL")

# ═══════════════════════════════════════════
# 主区域：对话界面
# ═══════════════════════════════════════════

mode_labels = {
    "medical_qa": ("🩺 医疗问答", "tag-qa"),
    "drug_query": ("💊 药物查询", "tag-drug"),
    "diagnosis": ("🩻 辅助诊断", "tag-diag"),
}

cols = st.columns(len(mode_labels))
for i, (mode_key, (label, _)) in enumerate(mode_labels.items()):
    with cols[i]:
        is_active = st.session_state.mode == mode_key
        btn_type = "primary" if is_active else "secondary"
        if st.button(label, key=f"mode_{mode_key}", type=btn_type, use_container_width=True):
            if st.session_state.mode != mode_key:
                logger.info("切换模式: %s -> %s", st.session_state.mode, mode_key)
                st.session_state.mode = mode_key
                if st.session_state.session_id:
                    update_session_mode(st.session_state.session_id, mode_key)
                st.session_state.chain = create_conversational_chain(mode_key)
                st.rerun()

st.divider()

# ── 消息展示 ──
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── 输入区 ──
if question := st.chat_input("请输入您的医疗问题..."):
    logger.info("用户提问 (user=%s, mode=%s, session=%s): %s",
                st.session_state.username, st.session_state.mode,
                st.session_state.session_id, question[:200])

    if st.session_state.chain is None:
        st.session_state.chain = create_conversational_chain(st.session_state.mode)

    st.chat_message("user").markdown(question)
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("assistant"):
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
                logger.debug("回答内容: %s", result[:500])
                st.markdown(result)
                st.session_state.messages.append(
                    {"role": "assistant", "content": result}
                )
            except Exception as e:
                logger.exception("问答异常")
                st.error(f"出错了：{e}")

# ── 底部声明 ──
st.divider()
st.caption("⚠️ 本系统仅供医学信息参考，不构成诊断或治疗建议。如有健康问题请咨询执业医师。")
