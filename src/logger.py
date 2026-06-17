import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "app.log"

_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s:%(funcName)s:%(lineno)d | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(level: int = logging.DEBUG) -> None:
    logger = logging.getLogger("hospital_rag")
    logger.setLevel(level)

    if logger.handlers:
        return

    # 文件：轮转，单文件 10MB，保留 5 个历史
    file_handler = RotatingFileHandler(
        LOG_FILE,
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(logging.Formatter(_FORMAT, _DATE_FORMAT))
    logger.addHandler(file_handler)

    # 控制台：INFO 及以上
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(_FORMAT, _DATE_FORMAT))
    logger.addHandler(console_handler)

    # 抑制第三方库的 DEBUG 日志
    for noisy in ("pymilvus", "httpx", "openai", "urllib3", "grpc", "sentence_transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str = "hospital_rag") -> logging.Logger:
    if name == "hospital_rag":
        return logging.getLogger(name)
    return logging.getLogger(f"hospital_rag.{name}")
