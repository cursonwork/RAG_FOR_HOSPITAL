import fitz  # PyMuPDF
from langchain_core.documents import Document

from src.logger import get_logger

logger = get_logger(__name__)


def load_pdf(file_path: str) -> list[Document]:
    """使用 PyMuPDF 加载单个 PDF，逐页提取文本并附加元数据。"""
    logger.info("开始解析 PDF: %s", file_path)
    doc = fitz.open(file_path)
    documents: list[Document] = []
    file_name = file_path.rsplit("/", 1)[-1]

    empty_pages = 0
    for page_num, page in enumerate(doc, start=1):
        text = page.get_text()
        if not text.strip():
            empty_pages += 1
            continue
        documents.append(
            Document(
                page_content=text,
                metadata={
                    "source": file_name,
                    "page": page_num,
                    "file_path": file_path,
                },
            )
        )

    doc.close()
    if empty_pages:
        logger.debug("%s: %d 页为空，跳过", file_name, empty_pages)
    logger.info("%s: 成功提取 %d 页文本", file_name, len(documents))
    return documents


def load_pdfs(file_paths: list[str]) -> list[Document]:
    """批量加载多个 PDF。"""
    logger.info("批量加载 %d 个 PDF 文件", len(file_paths))
    all_docs: list[Document] = []
    for path in file_paths:
        try:
            all_docs.extend(load_pdf(path))
        except Exception:
            logger.exception("加载 PDF 失败: %s", path)
    logger.info("批量加载完成，共提取 %d 页", len(all_docs))
    return all_docs
