"""医疗 RAG 系统 CLI 入口。

用法:
    # 导入 PDF 文档（清空旧数据）
    python main.py ingest --path data/documents/ --clear

    # 导入 Markdown 文档
    python main.py ingest-md --path data/md_documents/ --clear

    # 一次性导入所有（PDF + MD）
    python main.py ingest-all --clear

    # 运行爬虫获取医学数据
    python main.py crawl --type pubmed --max 20
    python main.py crawl --type dailymed --max 20
    python main.py crawl --type synthetic --consultations 50 --cases 20
    python main.py crawl --all

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
    pdf_paths = [str(pdf_dir)] if pdf_dir.is_file() else sorted(str(p) for p in pdf_dir.glob("*.pdf"))

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


def cmd_ingest_md(path: str, clear: bool = False) -> None:
    """导入 Markdown 文档到知识库。"""
    from pathlib import Path

    from src.database import clear_all_chunks, count_chunks
    from src.document_loader import load_markdown_files
    from src.vector_store import add_documents_to_store, get_vector_store

    logger.info("===== 开始 Markdown 导入: %s =====", path)

    md_dir = Path(path)
    md_paths = [str(md_dir)] if md_dir.is_file() else sorted(str(p) for p in md_dir.rglob("*.md"))

    if not md_paths:
        logger.warning("未找到 Markdown 文件: %s", path)
        print(f"未找到 Markdown 文件: {path}")
        return

    print(f"找到 {len(md_paths)} 个 Markdown 文件")
    logger.info("找到 %d 个 Markdown 文件", len(md_paths))

    if clear:
        print("清空旧数据...")
        store = get_vector_store()
        store.delete_all()
        clear_all_chunks()

    try:
        docs = load_markdown_files(md_paths)
        print(f"共加载 {len(docs)} 个文档")
        logger.info("共加载 %d 个 Markdown 文档", len(docs))

        add_documents_to_store(docs)
        total_chunks = count_chunks()
        print(f"Markdown 导入完成，共 {total_chunks} 个分块")
        logger.info("===== Markdown 导入完成，共 %d 个分块 =====", total_chunks)
    except Exception:
        logger.exception("Markdown 导入失败")
        raise


def cmd_ingest_all(
    documents_dir: str = "data/documents", md_dir: str = "data/md_documents", clear: bool = False
) -> None:
    """同时导入 PDF 和 Markdown 文档。"""
    from pathlib import Path

    if clear:
        from src.database import clear_all_chunks
        from src.vector_store import get_vector_store

        print("清空旧数据...")
        store = get_vector_store()
        store.delete_all()
        clear_all_chunks()

    # 导入 PDF
    pdf_path = Path(documents_dir)
    if pdf_path.exists() and list(pdf_path.glob("*.pdf")):
        cmd_ingest(documents_dir, clear=False)
    else:
        print("未找到 PDF 文件，跳过")

    # 导入 Markdown
    md_path = Path(md_dir)
    if md_path.exists() and list(md_path.rglob("*.md")):
        cmd_ingest_md(md_dir, clear=False)
    else:
        print("未找到 Markdown 文件，跳过")

    from src.database import count_chunks

    total_chunks = count_chunks()
    print(f"全部导入完成，共 {total_chunks} 个分块")


def cmd_crawl(crawl_type: str = "all", **kwargs) -> None:
    """运行爬虫/生成器获取数据。"""
    if crawl_type == "all":
        _crawl_all(**kwargs)
    elif crawl_type == "pubmed":
        _crawl_pubmed(kwargs.get("max", 20))
    elif crawl_type == "dailymed":
        _crawl_dailymed(kwargs.get("max", 20))
    elif crawl_type == "synthetic":
        _crawl_synthetic(
            consultations=kwargs.get("consultations", 50),
            textbook=kwargs.get("textbook", 20),
            symposium=kwargs.get("symposium", 20),
            cases=kwargs.get("cases", 20),
        )
    else:
        print(f"未知爬虫类型: {crawl_type}")
        print("可选: pubmed, dailymed, synthetic, all")


def _crawl_pubmed(max_items: int) -> None:
    from src.crawlers.pubmed import PubMedCrawler

    print(f"PubMed 爬虫启动，目标 {max_items} 篇论文...")
    crawler = PubMedCrawler()
    paths = crawler.crawl(max_items=max_items)
    print(f"PubMed 爬虫完成: 下载 {len(paths)} 篇论文")


def _crawl_dailymed(max_items: int) -> None:
    from src.crawlers.dailymed import DailyMedCrawler

    print(f"DailyMed 爬虫启动，目标 {max_items} 份说明书...")
    crawler = DailyMedCrawler()
    paths = crawler.crawl(max_items=max_items)
    print(f"DailyMed 爬虫完成: 下载 {len(paths)} 份说明书")


def _crawl_synthetic(consultations: int, textbook: int, symposium: int, cases: int) -> None:
    from src.crawlers.synthetic import (
        CaseGenerator,
        ConsultationGenerator,
        DrugManualGenerator,
        PaperGenerator,
        SymposiumGenerator,
        TextbookGenerator,
    )

    print("合成数据生成开始:")
    print(f"  问诊记录: {consultations}")
    print(f"  教材章节: {textbook}")
    print(f"  座谈报告: {symposium}")
    print(f"  病例报告: {cases}")

    results = {}

    gen = PaperGenerator()
    paths = gen.crawl(max_items=20)
    results["合成论文"] = len(paths)
    print(f"  合成论文完成: {len(paths)}")

    gen_drug = DrugManualGenerator()
    paths = gen_drug.crawl(max_items=25)
    results["药品手册"] = len(paths)
    print(f"  药品手册完成: {len(paths)}")

    gen = ConsultationGenerator()
    paths = gen.crawl(max_items=consultations)
    results["问诊记录"] = len(paths)
    print(f"  问诊记录完成: {len(paths)}")

    gen = TextbookGenerator()
    paths = gen.crawl(max_items=textbook)
    results["教材章节"] = len(paths)
    print(f"  教材章节完成: {len(paths)}")

    gen = SymposiumGenerator()
    paths = gen.crawl(max_items=symposium)
    results["座谈报告"] = len(paths)
    print(f"  座谈报告完成: {len(paths)}")

    gen = CaseGenerator()
    paths = gen.crawl(max_items=cases)
    results["病例报告"] = len(paths)
    print(f"  病例报告完成: {len(paths)}")

    total = sum(results.values())
    print(f"\n合成数据生成完成，共 {total} 个文件")


def _crawl_all(**kwargs) -> None:
    from src.crawlers.generator import run_all_crawlers

    pubmed = kwargs.get("pubmed", 20)
    dailymed = kwargs.get("dailymed", 20)
    consultations = kwargs.get("consultations", 50)
    textbook = kwargs.get("textbook", 20)
    symposium = kwargs.get("symposium", 20)
    cases = kwargs.get("cases", 20)

    print("===== 全量数据获取开始 =====")
    print(f"  PubMed 论文: {pubmed}")
    print(f"  DailyMed 说明书: {dailymed}")
    print(f"  问诊记录: {consultations}")
    print(f"  教材章节: {textbook}")
    print(f"  座谈报告: {symposium}")
    print(f"  病例报告: {cases}")
    print()

    results = run_all_crawlers(
        pubmed_count=pubmed,
        dailymed_count=dailymed,
        consultations_count=consultations,
        textbook_count=textbook,
        symposium_count=symposium,
        cases_count=cases,
    )

    total = sum(results.values())
    print(f"\n===== 全量数据获取完成，共 {total} 个文件 =====")
    for k, v in results.items():
        print(f"  {k}: {v}")


def cmd_ask(question: str, mode: str | None = None) -> None:
    """单次问答。mode=None 时自动识别意图。"""
    from src.callbacks import TokenLoggingCallback
    from src.rag_chain import create_rag_chain

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

    p_ingest = sub.add_parser("ingest", help="导入 PDF 文档")
    p_ingest.add_argument("--path", required=True, help="PDF 文件或目录路径")
    p_ingest.add_argument("--clear", action="store_true", help="清空旧数据后重新导入")

    p_ingest_md = sub.add_parser("ingest-md", help="导入 Markdown 文档")
    p_ingest_md.add_argument("--path", required=True, help="Markdown 文件或目录路径")
    p_ingest_md.add_argument("--clear", action="store_true", help="清空旧数据后重新导入")

    p_ingest_all = sub.add_parser("ingest-all", help="同时导入 PDF 和 Markdown 文档")
    p_ingest_all.add_argument("--documents", default="data/documents", help="PDF 目录 (默认 data/documents)")
    p_ingest_all.add_argument("--md", default="data/md_documents", help="Markdown 目录 (默认 data/md_documents)")
    p_ingest_all.add_argument("--clear", action="store_true", help="清空旧数据后重新导入")

    p_crawl = sub.add_parser("crawl", help="爬取/生成医学数据")
    p_crawl.add_argument("--type", choices=["pubmed", "dailymed", "synthetic", "all"], default="all", help="爬虫类型")
    p_crawl.add_argument("--max", type=int, default=20, help="最大数量 (pubmed/dailymed)")
    p_crawl.add_argument("--consultations", type=int, default=50, help="问诊记录数量")
    p_crawl.add_argument("--textbook", type=int, default=20, help="教材章节数量")
    p_crawl.add_argument("--symposium", type=int, default=20, help="座谈报告数量")
    p_crawl.add_argument("--cases", type=int, default=20, help="病例报告数量")
    p_crawl.add_argument("--pubmed", type=int, default=20, help="PubMed 论文数量 (--type all)")
    p_crawl.add_argument("--dailymed", type=int, default=20, help="DailyMed 数量 (--type all)")

    p_ask = sub.add_parser("ask", help="单次问答")
    p_ask.add_argument("--question", required=True, help="问题")
    p_ask.add_argument(
        "--mode",
        choices=["medical_qa", "drug_query", "diagnosis"],
        default=None,
        help="手动指定模式，不指定则自动识别",
    )

    sub.add_parser("serve", help="启动 Web 界面")

    args = parser.parse_args()

    logger.debug("CLI 参数: %s", args)

    if args.command == "ingest":
        cmd_ingest(args.path, clear=args.clear)
    elif args.command == "ingest-md":
        cmd_ingest_md(args.path, clear=args.clear)
    elif args.command == "ingest-all":
        cmd_ingest_all(
            documents_dir=args.documents,
            md_dir=args.md,
            clear=args.clear,
        )
    elif args.command == "crawl":
        cmd_crawl(
            crawl_type=args.type,
            max=args.max,
            consultations=args.consultations,
            textbook=args.textbook,
            symposium=args.symposium,
            cases=args.cases,
            pubmed=args.pubmed,
            dailymed=args.dailymed,
        )
    elif args.command == "ask":
        cmd_ask(args.question, args.mode)
    elif args.command == "serve":
        cmd_serve()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
