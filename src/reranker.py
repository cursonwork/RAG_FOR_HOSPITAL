"""FlashRank 重排序器 — 无 GPU 的 cross-encoder 精排。

FlashRank 使用 ONNX Runtime 运行 sub-14MB 的 miniLM 模型，
在 CPU 上对 20 个文档重排仅需 ~100ms，适合无 GPU 的生产环境。

首次使用会自动下载模型到 ~/.cache/flashrank/。
如网络受限无法下载 huggingface 模型，重排序自动降级（仅去重 + 截断）。

用法:
    from src.reranker import get_reranker
    reranker = get_reranker()
    reranked = reranker.compress_documents(docs, query)  # → top-5
"""

from langchain_core.callbacks import Callbacks
from langchain_core.documents import Document, BaseDocumentCompressor

from src.config import settings
from src.logger import get_logger

logger = get_logger(__name__)


class FlashRankReranker(BaseDocumentCompressor):
    """基于 FlashRank 的精排序器，CPU 友好。

    模型下载失败时自动降级为去重 + 截断。
    """

    def __init__(self, model_name: str | None = None, top_n: int | None = None) -> None:
        self._model_name = model_name or settings.reranker_model
        self._top_n = top_n or settings.reranker_top_n
        self._ranker = None  # None=未加载, False=加载失败
        self._model_available = True

    @property
    def _lazy_ranker(self):
        """延迟加载 FlashRank Ranker。首次使用时下载模型。"""
        if self._ranker is None:
            logger.info("加载 FlashRank 模型: %s", self._model_name)
            try:
                from flashrank import Ranker
                self._ranker = Ranker(model_name=self._model_name)
            except Exception:
                logger.warning(
                    "FlashRank 模型下载失败。重排序已降级为去重+截断。"
                    " 如需启用重排序，请手动下载模型到 ~/.cache/flashrank/"
                )
                self._ranker = False
                self._model_available = False
        return self._ranker if self._ranker is not False else None

    def _dedup_and_slice(self, documents: list[Document]) -> list[Document]:
        """去重 + 截断（降级方案）。"""
        seen = set()
        unique = []
        for doc in documents:
            key = doc.page_content[:120]
            if key not in seen:
                seen.add(key)
                unique.append(doc)
        return unique[:self._top_n]

    def compress_documents(
        self,
        documents: list[Document],
        query: str,
        callbacks: Callbacks | None = None,
    ) -> list[Document]:
        """对文档精排，保留 top_n。"""
        if not documents:
            return []

        if len(documents) <= self._top_n:
            return documents

        # 去重
        deduped = self._dedup_and_slice(documents)
        if len(deduped) <= self._top_n:
            return deduped

        ranker = self._lazy_ranker
        if ranker is None:
            return deduped[:self._top_n]

        # 构建 FlashRank 输入
        passages = [
            {"id": i, "text": doc.page_content, "meta": {}}
            for i, doc in enumerate(deduped)
        ]

        try:
            results = ranker.rerank(query, passages)
        except Exception:
            logger.exception("FlashRank 重排序失败，降级返回")
            return deduped[:self._top_n]

        # 构建结果
        reranked: list[Document] = []
        for result in results[:self._top_n]:
            doc = deduped[result["id"]]
            doc.metadata["rerank_score"] = round(float(result["score"]), 4)
            doc.metadata["rerank_rank"] = len(reranked) + 1
            reranked.append(doc)

        logger.debug(
            "重排序: %d → %d 文档 (top score=%.4f)",
            len(deduped), len(reranked),
            reranked[0].metadata["rerank_score"] if reranked else 0,
        )
        return reranked


_reranker: FlashRankReranker | None = None


def get_reranker() -> FlashRankReranker:
    global _reranker
    if _reranker is None:
        _reranker = FlashRankReranker()
    return _reranker


def rerank(docs: list[Document], query: str, top_n: int | None = None) -> list[Document]:
    """快捷函数：对文档列表重排序。"""
    ranker = get_reranker()
    if top_n is not None:
        ranker._top_n = top_n
    return ranker.compress_documents(docs, query)
