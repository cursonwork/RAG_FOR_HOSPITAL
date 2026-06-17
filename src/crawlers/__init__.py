"""医学数据爬虫与合成数据生成器。"""

from src.crawlers.base import BaseCrawler
from src.crawlers.dailymed import DailyMedCrawler
from src.crawlers.generator import run_all_crawlers
from src.crawlers.pubmed import PubMedCrawler
from src.crawlers.synthetic import (
    CaseGenerator,
    ConsultationGenerator,
    DrugManualGenerator,
    PaperGenerator,
    SymposiumGenerator,
    TextbookGenerator,
)

__all__ = [
    "BaseCrawler",
    "PubMedCrawler",
    "DailyMedCrawler",
    "ConsultationGenerator",
    "TextbookGenerator",
    "SymposiumGenerator",
    "CaseGenerator",
    "PaperGenerator",
    "DrugManualGenerator",
    "run_all_crawlers",
]
