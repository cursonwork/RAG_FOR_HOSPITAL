import tempfile
from pathlib import Path

import streamlit as st

from src.config import settings
from src.conversation import create_conversational_chain, _store
from src.document_loader import load_pdf
from src.logger import get_logger
from src.vector_store import add_documents_to_store

logger = get_logger("app")

st.set_page_config(page_title="医疗RAG系统", page_icon="🏥", layout="wide")

# ── 样式 ──
st.markdown("""
<style>
    .stChatMessage { padding: 1rem; border-radius: 10px; margin-bottom: 0.5rem; }
    .mode-tag { display: inline-block; padding: 2px 8px; border-radius: 4px;
                font-size: 0.75rem; font-weight: 600; margin-right: 6px; }
    .tag-qa { background: #e3f2fd; color: #1565c0; }
    .tag-drug { background: #e8f5e9; color: #2e7d32; }
    .tag-diag { background: #fff3e0; color: #e65100; }
</style>
""", unsafe_allow_html=True)

# ── 会话状态初始化 ──
if "messages" not in st.session_state:
    st.session_state.messages = []
if "mode" not in st.session_state:
    st.session_state.mode = "medical_qa"
if "chain" not in st.session_state:
    st.session_state.chain = create_conversational_chain("medical_qa")
if "doc_count" not in st.session_state:
    st.session_state.doc_count = 0

SESSION_ID = "default"

# ── 侧边栏：文档管理 ──
with st.sidebar:
    st.header("📚 知识库管理")

    uploaded_files = st.file_uploader(
        "上传医学文献 PDF",
        type=["pdf"],
        accept_multiple_files=True,
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
    st.caption(f"向量库: Milvus")

# ── 主区域：对话界面 ──
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
        if st.button(label, key=mode_key, type=btn_type, use_container_width=True):
            logger.info("切换模式: %s", mode_key)
            st.session_state.mode = mode_key
            st.session_state.chain = create_conversational_chain(mode_key)
            st.session_state.messages = []
            _store.pop(SESSION_ID, None)
            st.rerun()

st.divider()

# ── 消息展示 ──
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── 输入区 ──
if question := st.chat_input("请输入您的医疗问题..."):
    logger.info("用户提问 (mode=%s): %s", st.session_state.mode, question[:200])
    st.chat_message("user").markdown(question)
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("assistant"):
        with st.spinner("检索中..."):
            try:
                chain = st.session_state.chain
                result = chain.invoke(
                    {"question": question},
                    config={"configurable": {"session_id": SESSION_ID}},
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
