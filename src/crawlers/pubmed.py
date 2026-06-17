"""PubMed Central OA 文献爬虫 — 通过 NCBI E-utilities API 下载免费全文 PDF。"""

import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

from src.crawlers.base import BaseCrawler
from src.logger import get_logger

logger = get_logger(__name__)

NCBI_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
PMC_PDF_BASE = "https://www.ncbi.nlm.nih.gov/pmc/articles"


class PubMedCrawler(BaseCrawler):
    """从 PubMed Central OA 子集搜索并下载医学文献 PDF。"""

    # 搜索主题覆盖临床医学各领域
    SEARCH_QUERIES = [
        "colorectal cancer deep learning pathology",
        "breast cancer AI diagnosis histopathology",
        "lung cancer CT screening machine learning",
        "diabetes management clinical decision support",
        "cardiovascular disease risk prediction AI",
        "medical image segmentation deep learning",
        "electronic health records natural language processing",
        "drug safety surveillance pharmacovigilance AI",
        "infectious disease epidemiology machine learning",
        "radiology report generation large language model",
        "dermatology AI skin lesion classification",
        "pathology whole slide image computational",
        "genomic medicine precision oncology",
        "emergency medicine triage machine learning",
        "surgical outcome prediction artificial intelligence",
        "mental health NLP clinical notes",
        "pediatric diagnosis expert system",
        "stroke imaging AI detection",
        "chronic kidney disease prediction model",
        "hepatocellular carcinoma deep learning",
        "endoscopy AI polyp detection",
        "ophthalmology retinal image deep learning",
        "neurology MRI brain tumor segmentation",
        "immunotherapy response prediction biomarker",
        "thyroid nodule ultrasound AI classification",
    ]

    def __init__(self, output_dir: str = "data/documents"):
        super().__init__(output_dir, request_interval=0.4)

    def crawl(self, max_items: int = 20) -> list[Path]:
        """搜索并下载 PMC OA 论文 PDF。"""
        logger.info("PubMed Crawler 开始，目标 %d 篇", max_items)
        downloaded: list[Path] = []
        seen_ids: set[str] = set()

        for query in self.SEARCH_QUERIES:
            if len(downloaded) >= max_items:
                break

            pmc_ids = self._search_pmc(query, retmax=5)
            for pmc_id in pmc_ids:
                if pmc_id in seen_ids:
                    continue
                if len(downloaded) >= max_items:
                    break
                seen_ids.add(pmc_id)

                path = self._download_pdf(pmc_id)
                if path:
                    downloaded.append(path)
                    logger.info("已下载 [%d/%d]: %s", len(downloaded), max_items, path.name)

        logger.info("PubMed Crawler 完成，成功下载 %d 篇", len(downloaded))
        return downloaded

    def _search_pmc(self, query: str, retmax: int = 5) -> list[str]:
        """搜索 PubMed Central OA 文章，返回 PMC ID 列表。"""
        # 加 open access 过滤
        full_query = f"({query}) AND openaccess[filter]"
        params = urllib.parse.urlencode(
            {
                "db": "pmc",
                "term": full_query,
                "retmax": retmax,
                "retmode": "xml",
                "sort": "relevance",
            }
        )
        url = f"{NCBI_BASE}/esearch.fcgi?{params}"

        try:
            self._rate_limit()
            with urllib.request.urlopen(url, timeout=30) as resp:
                root = ET.fromstring(resp.read())
            return [e.text for e in root.findall(".//Id") if e.text and e.text.startswith("PMC")]
        except Exception as e:
            logger.warning("PMC 搜索失败: %s — %s", query[:60], e)
            return []

    def _download_pdf(self, pmc_id: str) -> Path | None:
        """下载 PMC 文章的 PDF。"""
        pdf_url = f"{PMC_PDF_BASE}/{pmc_id}/pdf/main.pdf"
        output_path = self.output_dir / f"pubmed_{pmc_id}.pdf"

        if output_path.exists():
            logger.debug("PDF 已存在，跳过: %s", pmc_id)
            return output_path

        return self._retry(self._do_download, pdf_url, output_path, pmc_id)

    def _do_download(self, pdf_url: str, output_path: Path, pmc_id: str) -> Path | None:
        """执行下载，验证 PDF 文件头。"""
        self._rate_limit()

        req = urllib.request.Request(
            pdf_url, headers={"User-Agent": "Mozilla/5.0 (compatible; MedicalRAG/1.0; mailto:research@example.com)"}
        )

        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = resp.read()
        except urllib.error.HTTPError as e:
            logger.warning("PDF 下载失败 %s: HTTP %d", pmc_id, e.code)
            return None
        except Exception as e:
            logger.warning("PDF 下载失败 %s: %s", pmc_id, e)
            return None

        # 验证 PDF 文件头
        if not data.startswith(b"%PDF"):
            logger.warning("非 PDF 文件，跳过 %s (大小=%d)", pmc_id, len(data))
            return None

        if len(data) < 10000:
            logger.warning("PDF 太小 %s (%d bytes)，可能不完整，跳过", pmc_id, len(data))
            return None

        output_path.write_bytes(data)
        logger.debug("下载完成 %s (%d KB)", pmc_id, len(data) // 1024)
        return output_path
