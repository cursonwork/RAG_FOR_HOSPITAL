"""爬虫编排器：统一运行所有爬虫和生成器。"""

from src.logger import get_logger

logger = get_logger(__name__)


def run_all_crawlers(
    pubmed_count: int = 0,
    dailymed_count: int = 0,
    paper_count: int = 20,
    drug_manual_count: int = 25,
    consultations_count: int = 50,
    textbook_count: int = 20,
    symposium_count: int = 20,
    cases_count: int = 20,
    skip_real: bool = True,
) -> dict[str, int]:
    """运行所有爬虫和生成器。

    skip_real 默认 True——PubMed/DailyMed 在中国大陆网络可能不可达。
    """
    results: dict[str, int] = {}

    # ── Real crawlers (skip by default—GFW may block NCBI) ──
    if not skip_real:
        if pubmed_count > 0:
            try:
                from src.crawlers.pubmed import PubMedCrawler

                pubmed = PubMedCrawler()
                paths = pubmed.crawl(max_items=pubmed_count)
                results["pubmed_papers"] = len(paths)
            except Exception:
                logger.exception("PubMed 爬虫失败")
                results["pubmed_papers"] = 0

        if dailymed_count > 0:
            try:
                from src.crawlers.dailymed import DailyMedCrawler

                dailymed = DailyMedCrawler()
                paths = dailymed.crawl(max_items=dailymed_count)
                results["drug_manuals_dailymed"] = len(paths)
            except Exception:
                logger.exception("DailyMed 爬虫失败")
                results["drug_manuals_dailymed"] = 0

    # ── Synthetic: papers ──
    if paper_count > 0:
        try:
            from src.crawlers.synthetic import PaperGenerator

            gen = PaperGenerator()
            paths = gen.crawl(max_items=paper_count)
            results["synthetic_papers"] = len(paths)
        except Exception:
            logger.exception("论文生成失败")
            results["synthetic_papers"] = 0

    # ── Synthetic: drug manuals ──
    if drug_manual_count > 0:
        try:
            from src.crawlers.synthetic import DrugManualGenerator

            gen = DrugManualGenerator()
            paths = gen.crawl(max_items=drug_manual_count)
            results["drug_manuals"] = len(paths)
        except Exception:
            logger.exception("药品手册生成失败")
            results["drug_manuals"] = 0

    # ── Synthetic: consultations ──
    if consultations_count > 0:
        try:
            from src.crawlers.synthetic import ConsultationGenerator

            gen = ConsultationGenerator()
            paths = gen.crawl(max_items=consultations_count)
            results["consultations"] = len(paths)
        except Exception:
            logger.exception("问诊生成失败")
            results["consultations"] = 0

    # ── Synthetic: textbook ──
    if textbook_count > 0:
        try:
            from src.crawlers.synthetic import TextbookGenerator

            gen = TextbookGenerator()
            paths = gen.crawl(max_items=textbook_count)
            results["textbook_chapters"] = len(paths)
        except Exception:
            logger.exception("教材生成失败")
            results["textbook_chapters"] = 0

    # ── Synthetic: symposium ──
    if symposium_count > 0:
        try:
            from src.crawlers.synthetic import SymposiumGenerator

            gen = SymposiumGenerator()
            paths = gen.crawl(max_items=symposium_count)
            results["symposium_reports"] = len(paths)
        except Exception:
            logger.exception("座谈生成失败")
            results["symposium_reports"] = 0

    # ── Synthetic: cases ──
    if cases_count > 0:
        try:
            from src.crawlers.synthetic import CaseGenerator

            gen = CaseGenerator()
            paths = gen.crawl(max_items=cases_count)
            results["case_reports"] = len(paths)
        except Exception:
            logger.exception("病例生成失败")
            results["case_reports"] = 0

    total = sum(results.values())
    logger.info("=== 全部爬虫/生成完成，共 %d 个文件 ===", total)
    for k, v in results.items():
        logger.info("  %s: %d", k, v)

    return results
