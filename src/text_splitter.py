from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import settings
from src.logger import get_logger

logger = get_logger(__name__)

# 中文友好的分割符优先级
CHINESE_SEPARATORS = [
    "\n\n",
    "\n",
    "。",
    "；",
    "！",
    "？",
    "，",
    " ",
    "",
]


def create_text_splitter() -> RecursiveCharacterTextSplitter:
    logger.debug("创建文本分割器 chunk_size=%d overlap=%d", settings.chunk_size, settings.chunk_overlap)
    return RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=CHINESE_SEPARATORS,
        keep_separator=True,
    )
