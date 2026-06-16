import json
import shutil
import tempfile
from pathlib import Path

import fitz
from langchain_core.documents import Document

from src.config import settings
from src.logger import get_logger

logger = get_logger(__name__)


def load_pdf_opendataloader(file_path: str) -> Document | None:
    """使用 opendataloader-pdf 将 PDF 转为结构化 Markdown + JSON 元数据。"""
    import opendataloader_pdf

    file_name = Path(file_path).name
    stem = Path(file_path).stem
    output_dir = tempfile.mkdtemp(prefix="odl_")

    try:
        opendataloader_pdf.convert(
            input_path=file_path,
            output_dir=output_dir,
            format=["markdown", "json"],
            quiet=True,
        )
    except Exception:
        logger.exception("opendataloader-pdf 转换失败: %s，回退到 PyMuPDF", file_name)
        shutil.rmtree(output_dir, ignore_errors=True)
        return None

    md_path = Path(output_dir) / f"{stem}.md"
    json_path = Path(output_dir) / f"{stem}.json"

    if not md_path.exists():
        logger.warning("opendataloader 未生成 Markdown: %s，回退到 PyMuPDF", file_name)
        shutil.rmtree(output_dir, ignore_errors=True)
        return None

    md_text = md_path.read_text(encoding="utf-8")

    metadata: dict = {"source": file_name, "file_path": file_path, "parser": "opendataloader"}

    if json_path.exists():
        try:
            j = json.loads(json_path.read_text(encoding="utf-8"))
            metadata.update({
                "author": j.get("author") or "",
                "title": j.get("title") or "",
                "num_pages": j.get("number of pages", 0),
                "creation_date": j.get("creation date", ""),
            })
        except (json.JSONDecodeError, KeyError):
            logger.debug("JSON 元数据解析失败: %s", file_name)

    shutil.rmtree(output_dir, ignore_errors=True)
    logger.info("%s: opendataloader 提取完成 (%d 字符)", file_name, len(md_text))
    return Document(page_content=md_text, metadata=metadata)


def load_pdf_pymupdf(file_path: str) -> Document:
    """使用 PyMuPDF 逐页提取文本（回退方案）。"""
    file_name = file_path.rsplit("/", 1)[-1]
    doc = fitz.open(file_path)
    pages_text: list[str] = []
    empty_pages = 0

    for page_num, page in enumerate(doc, start=1):
        text = page.get_text()
        if not text.strip():
            empty_pages += 1
            continue
        pages_text.append(f"## 第{page_num}页\n\n{text}")

    doc.close()

    if empty_pages:
        logger.debug("%s: %d 页为空，跳过", file_name, empty_pages)

    full_text = "\n\n".join(pages_text)
    logger.info("%s: PyMuPDF 提取完成 (%d 字符, %d 页)", file_name, len(full_text), len(pages_text))
    return Document(
        page_content=full_text,
        metadata={
            "source": file_name,
            "file_path": file_path,
            "parser": "pymupdf",
        },
    )


def load_pdf(file_path: str) -> Document | None:
    """加载单个 PDF，优先使用 opendataloader-pdf。"""
    if settings.pdf_parser == "opendataloader":
        doc = load_pdf_opendataloader(file_path)
        if doc is not None:
            return doc
        logger.warning("opendataloader 失败，fallback 到 PyMuPDF: %s", file_path)
    return load_pdf_pymupdf(file_path)


def load_pdfs(file_paths: list[str]) -> list[Document]:
    """批量加载多个 PDF。"""
    logger.info("批量加载 %d 个 PDF 文件", len(file_paths))
    all_docs: list[Document] = []
    for path in file_paths:
        try:
            doc = load_pdf(path)
            if doc is not None:
                all_docs.append(doc)
        except Exception:
            logger.exception("加载 PDF 失败: %s", path)
    logger.info("批量加载完成，共 %d 个文档", len(all_docs))
    return all_docs
