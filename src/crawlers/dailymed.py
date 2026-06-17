"""FDA DailyMed 药品说明书爬虫 — REST API → SPL XML → 结构化 Markdown。"""

import json
import re
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

from src.crawlers.base import BaseCrawler
from src.logger import get_logger

logger = get_logger(__name__)

DAILYMED_SPLS = "https://dailymed.nlm.nih.gov/dailymed/services/v2/spls.json"
DAILYMED_SPL_XML = "https://dailymed.nlm.nih.gov/dailymed/services/v2/spls/{setid}.xml"
DAILYMED_DRUGS = "https://dailymed.nlm.nih.gov/dailymed/services/v2/drugnames.json"

# SPL XML 中常见的 section 标签 → 中文标题映射
SECTION_MAP = {
    "INDICATIONS & USAGE": "## 适应症",
    "INDICATIONS AND USAGE": "## 适应症",
    "DOSAGE & ADMINISTRATION": "## 用法用量",
    "DOSAGE AND ADMINISTRATION": "## 用法用量",
    "CONTRAINDICATIONS": "## 禁忌",
    "WARNINGS AND PRECAUTIONS": "## 注意事项",
    "WARNINGS": "## 警告",
    "ADVERSE REACTIONS": "## 不良反应",
    "DRUG INTERACTIONS": "## 药物相互作用",
    "USE IN SPECIFIC POPULATIONS": "## 特殊人群用药",
    "OVERDOSAGE": "## 药物过量",
    "CLINICAL PHARMACOLOGY": "## 临床药理学",
    "MECHANISM OF ACTION": "## 作用机制",
    "PHARMACOKINETICS": "## 药代动力学",
    "HOW SUPPLIED": "## 包装与贮藏",
    "STORAGE AND HANDLING": "## 包装与贮藏",
    "DESCRIPTION": "## 药品描述",
    "INFORMATION FOR PATIENTS": "## 患者须知",
    "NONCLINICAL TOXICOLOGY": "## 非临床毒理学",
    "CLINICAL STUDIES": "## 临床研究",
    "REFERENCES": "## 参考文献",
    "SPL UNCLASSIFIED": "## 其他信息",
}


class DailyMedCrawler(BaseCrawler):
    """从 FDA DailyMed API 获取药品说明书，转为结构化 Markdown。"""

    DRUG_NAMES = [
        "metformin",
        "atorvastatin",
        "omeprazole",
        "amlodipine",
        "metoprolol",
        "losartan",
        "gabapentin",
        "hydrochlorothiazide",
        "sertraline",
        "simvastatin",
        "azithromycin",
        "metronidazole",
        "ibuprofen",
        "acetaminophen",
        "prednisone",
        "levothyroxine",
        "albuterol",
        "fluoxetine",
        "lisinopril",
        "warfarin",
        "amoxicillin",
        "pantoprazole",
        "doxycycline",
        "montelukast",
        "tramadol",
        "ciprofloxacin",
        "clopidogrel",
        "rosuvastatin",
        "escitalopram",
        "furosemide",
    ]

    def __init__(self, output_dir: str = "data/md_documents/drug_manual"):
        super().__init__(output_dir, request_interval=0.3)

    def crawl(self, max_items: int = 20) -> list[Path]:
        """搜索药物并下载说明书。"""
        logger.info("DailyMed Crawler 开始，目标 %d 份说明书", max_items)
        downloaded: list[Path] = []

        for drug_name in self.DRUG_NAMES:
            if len(downloaded) >= max_items:
                break

            setids = self._search_drug(drug_name)
            if not setids:
                continue

            # 取第一个匹配结果
            path = self._fetch_and_save(setids[0], drug_name)
            if path:
                downloaded.append(path)
                logger.info("已下载 [%d/%d]: %s", len(downloaded), max_items, path.name)

        logger.info("DailyMed Crawler 完成，成功 %d 份", len(downloaded))
        return downloaded

    def _search_drug(self, name: str) -> list[str]:
        """搜索药品，返回 setid 列表。"""
        params = urllib.parse.urlencode(
            {
                "drug_name": name,
                "limit": 5,
            }
        )
        url = f"{DAILYMED_SPLS}?{params}"

        try:
            self._rate_limit()
            req = urllib.request.Request(url, headers={"Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read())
        except Exception as e:
            logger.warning("DailyMed 搜索失败 %s: %s", name, e)
            return []

        results = []
        for item in data.get("data", []):
            setid = item.get("setid", "")
            if setid:
                results.append(setid)
        return results

    def _fetch_and_save(self, setid: str, drug_name: str) -> Path | None:
        """获取 SPL XML 并转换为 Markdown。"""
        output_path = self.output_dir / f"drug_{drug_name}_{setid[:8]}.md"
        if output_path.exists():
            return output_path

        return self._retry(self._do_fetch, setid, drug_name, output_path)

    def _do_fetch(self, setid: str, drug_name: str, output_path: Path) -> Path | None:
        """获取并解析 SPL XML → Markdown。"""
        self._rate_limit()
        url = DAILYMED_SPL_XML.format(setid=setid)

        try:
            req = urllib.request.Request(url, headers={"Accept": "application/xml"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                raw = resp.read()
        except Exception as e:
            logger.warning("SPL XML 获取失败 %s: %s", setid[:8], e)
            return None

        try:
            md = self._spl_to_markdown(raw, drug_name)
        except Exception as e:
            logger.warning("SPL 解析失败 %s: %s", setid[:8], e)
            return None

        if not md or len(md) < 300:
            logger.warning("说明书内容过短 %s (%d chars)，跳过", drug_name, len(md))
            return None

        output_path.write_text(md, encoding="utf-8")
        return output_path

    def _spl_to_markdown(self, raw_xml: bytes, drug_name: str) -> str:
        """将 SPL XML 转换为结构化 Markdown。"""
        # 去掉命名空间以便 XPath 查询
        content = raw_xml.decode("utf-8", errors="replace")
        content = re.sub(r'xmlns="[^"]*"', "", content)
        content = re.sub(r'xmlns:xsi="[^"]*"', "", content)
        content = re.sub(r'xsi:schemaLocation="[^"]*"', "", content)

        root = ET.fromstring(content)

        lines = [f"# {drug_name.upper()} — FDA 药品说明书", ""]

        # 尝试提取通用名和商品名
        for el in root.iter():
            tag = el.tag.split("}")[-1] if "}" in el.tag else el.tag
            if tag == "name" and el.text:
                lines.append(f"**通用名**: {el.text.strip()}")
                lines.append("")
                break

        # 提取所有 section
        sections_found = 0
        seen_titles: set[str] = set()

        for el in root.iter():
            tag = el.tag.split("}")[-1] if "}" in el.tag else el.tag
            if tag != "section":
                continue

            # 尝试从 <code> 或 <title> 获取 section 标题
            section_title = ""
            for child in el.iter():
                ctag = child.tag.split("}")[-1] if "}" in child.tag else child.tag
                if ctag in ("code", "title") and child.text:
                    section_title = " ".join(child.text.split()).strip().upper()
                    break

            if not section_title:
                continue

            section_title_short = section_title.split(" ")[:6]
            section_title_short = " ".join(section_title_short)

            # 匹配已知标题
            matched = SECTION_MAP.get(section_title)
            if not matched:
                # 尝试前缀匹配
                for key, value in SECTION_MAP.items():
                    if section_title.startswith(key):
                        matched = value
                        break

            if not matched or matched in seen_titles:
                continue

            seen_titles.add(matched)

            # 提取 section 内所有文本
            paragraphs = []
            for para in el.iter():
                ptag = para.tag.split("}")[-1] if "}" in para.tag else para.tag
                if ptag in ("paragraph", "text"):
                    text = ET.tostring(para, encoding="unicode", method="text")
                    text = " ".join(text.split())
                    if len(text) > 20:
                        paragraphs.append(text)

            if paragraphs:
                lines.append(matched)
                lines.append("")
                for p in paragraphs[:5]:  # 最多 5 段
                    lines.append(p)
                    lines.append("")
                sections_found += 1

        if sections_found == 0:
            # 回退：提取所有文本
            full_text = ET.tostring(root, encoding="unicode", method="text")
            full_text = " ".join(full_text.split())
            lines.append("## 完整文本")
            lines.append("")
            # 分段落
            for i in range(0, len(full_text), 2000):
                lines.append(full_text[i : i + 2000])
                lines.append("")

        result = "\n".join(lines)
        return result
