"""爬虫基础设施：抽象基类 + 公共工具。"""

import random
import time
from abc import ABC, abstractmethod
from pathlib import Path

from src.logger import get_logger

logger = get_logger(__name__)


class BaseCrawler(ABC):
    """爬虫/数据生成器抽象基类。"""

    def __init__(self, output_dir: str, request_interval: float = 0.5):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.request_interval = request_interval
        self._last_request = 0.0

    @abstractmethod
    def crawl(self, max_items: int = 20) -> list[Path]:
        """执行爬取，返回输出文件路径列表。"""

    def _rate_limit(self):
        """简单的速率限制。"""
        elapsed = time.monotonic() - self._last_request
        if elapsed < self.request_interval:
            jitter = random.uniform(0, self.request_interval * 0.5)
            time.sleep(self.request_interval - elapsed + jitter)
        self._last_request = time.monotonic()

    def _retry(self, func, *args, max_retries: int = 3, **kwargs):
        """带指数退避的重试。"""
        last_exc = None
        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exc = e
                wait = 2**attempt + random.uniform(0, 1)
                logger.warning("重试 %d/%d: %s，等待 %.1fs", attempt + 1, max_retries, e, wait)
                time.sleep(wait)
        raise last_exc
