"""医疗 RAG 系统 CLI 入口。

用法:
    # 导入文档（清空旧数据）
    python main.py ingest --path data/documents/ --clear

    # 追加导入
    python main.py ingest --path data/documents/

    # 单次问答
    python main.py ask --question "如何用深度学习技术对病理切片进行组织成分分析？"

    # 启动 Web 界面
    python main.py serve
"""

import argparse

from src.logger import get_logger

logger = get_logger("main")


def cmd_ingest(path: str, clear: bool = False) -> None:
    """导入 PDF 文档到知识库。"""
    from pathlib import Path

    from src.database import clear_all_chunks, count_chunks
    from src.document_loader import load_pdfs
    from src.vector_store import add_documents_to_store, get_vector_store

    logger.info("===== 开始文档导入: %s =====", path)

    pdf_dir = Path(path)
    if pdf_dir.is_file():
        pdf_paths = [str(pdf_dir)]
    else:
        pdf_paths = sorted(str(p) for p in pdf_dir.glob("*.pdf"))

    if not pdf_paths:
        logger.warning("未找到 PDF 文件: %s", path)
        print(f"未找到 PDF 文件: {path}")
        return

    print(f"找到 {len(pdf_paths)} 个 PDF 文件")
    logger.info("找到 %d 个 PDF 文件", len(pdf_paths))

    if clear:
        print("清空旧数据...")
        logger.info("清空旧数据")
        store = get_vector_store()
        store.delete_all()
        clear_all_chunks()

    try:
        docs = load_pdfs(pdf_paths)
        print(f"共加载 {len(docs)} 个文档")
        logger.info("共加载 %d 个文档", len(docs))

        add_documents_to_store(docs)
        total_chunks = count_chunks()
        print(f"知识库导入完成，共 {total_chunks} 个分块")
        logger.info("===== 文档导入完成，共 %d 个分块 =====", total_chunks)
    except Exception:
        logger.exception("文档导入失败")
        raise


def cmd_ask(question: str, mode: str | None = None) -> None:
    """单次问答。mode=None 时自动识别意图。"""
    from src.rag_chain import create_rag_chain
    from src.callbacks import TokenLoggingCallback

    logger.info("===== CLI 问答 =====")
    logger.info("用户输入 (mode=%s): %s", mode or "auto", question)

    chain = create_rag_chain(mode)
    mode_label = mode or "自动识别"
    print(f"\n检索中（模式: {mode_label}）...\n")

    try:
        answer = chain.invoke(question, config={"callbacks": [TokenLoggingCallback()]})
        logger.info("生成回答: %d 字符", len(answer))
        logger.debug("回答内容: %s", answer)
        print(answer)
        print()
    except Exception:
        logger.exception("问答失败")
        raise


def cmd_serve() -> None:
    """启动 Streamlit Web 界面。"""
    import sys
    from streamlit.web import cli as stcli

    logger.info("启动 Streamlit Web 界面")
    sys.argv = ["streamlit", "run", "app.py"]
    stcli.main()


def main() -> None:
    parser = argparse.ArgumentParser(description="医疗 RAG 系统")
    sub = parser.add_subparsers(dest="command")

    p_ingest = sub.add_parser("ingest", help="导入PDF文档")
    p_ingest.add_argument("--path", required=True, help="PDF 文件或目录路径")
    p_ingest.add_argument("--clear", action="store_true", help="清空旧数据后重新导入")

    p_ask = sub.add_parser("ask", help="单次问答")
    p_ask.add_argument("--question", required=True, help="问题")
    p_ask.add_argument(
        "--mode", choices=["medical_qa", "drug_query", "diagnosis"],
        default=None,
        help="手动指定模式，不指定则自动识别",
    )

    sub.add_parser("serve", help="启动 Web 界面")

    args = parser.parse_args()

    logger.debug("CLI 参数: %s", args)

    if args.command == "ingest":
        cmd_ingest(args.path, clear=args.clear)
    elif args.command == "ask":
        cmd_ask(args.question, args.mode)
    elif args.command == "serve":
        cmd_serve()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
